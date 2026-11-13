"""Solver for the "make the unit tests pass" tasks.

Pipeline (cheapest first):

1. Locate the test file(s) and the library, run the tests on a scratch copy.
   The tests are really executed (``python -m unittest`` - it does not depend
   on a ``unittest.main()`` call in the file) and counted: a run is green only
   if every statically visible test ran, at least one of them was not skipped
   and none failed. Already green -> answer immediately.
2. Free infeasibility checks: the test file itself is broken (syntax error,
   import of a module that does not exist), two assertions demand different
   results for the same call, a failing test does not touch the library.
3. Classical mutation-based repair (see ``mutations.py``): single-edit
   variants of the functions exercised by the failing tests are validated
   against the tests. Zero tokens.
4. LLM repair: one compact request with the library, the failing tests and
   the failure messages; the model returns corrected functions only. Up to
   three attempts while the per-task token cap allows, each validated by the
   tests. The model may also declare the task INFEASIBLE.

The library in the task workspace is replaced (atomically) only by a version
that passed the tests on a scratch copy, and the installed result is verified
once more on a fresh copy with the untouched test files; on failure the
original library is restored. The test files are never modified.
"""
from __future__ import annotations

import ast
import asyncio
import os
import queue
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .. import wsfiles
from ..answers import write_text_atomic
from ..budget import estimate_tokens
from ..llm import BudgetExceeded, LLMError
from ..sandbox import copy_workspace, remove_scratch, run_python, workspace_size
from .base import Outcome
from .mutations import function_mutants, render


# ----------------------------------------------------------------- test runs
@dataclass
class TestRun:
    passed: bool
    timed_out: bool = False
    statuses: Dict[str, str] = field(default_factory=dict)
    messages: Dict[str, str] = field(default_factory=dict)
    output: str = ""
    load_error: str = ""
    duration: float = 0.0
    ran: int = 0        # tests reported by the runner ("Ran N tests")
    executed: int = 0   # of them really executed (not skipped)

    @property
    def failing(self) -> Set[str]:
        return {k for k, v in self.statuses.items() if v in ("FAIL", "ERROR")}


_STATUS_RX = re.compile(
    r"^(\w+) \(([\w.<>]+)\)(?:\n[^\n]*)?? \.\.\. (ok|FAIL|ERROR|skipped[^\n]*|expected failure|unexpected success)\s*$",
    re.M)
_BLOCK_RX = re.compile(r"^(FAIL|ERROR): (\w+) \(([\w.<>]+)\)[^\n]*\n-{20,}\n(.*?)(?=^={20,}$|^-{20,}\nRan |\Z)", re.M | re.S)


def _test_key(method: str, dotted: str) -> str:
    if dotted == "<function>":
        return method
    parts = dotted.split(".")
    cls = parts[-2] if len(parts) >= 2 and parts[-1] == method else parts[-1]
    return "%s.%s" % (cls, method)


def parse_unittest_output(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    statuses: Dict[str, str] = {}
    messages: Dict[str, str] = {}
    text = _FUNC_CASE_RX.sub(r"\1 (<function>)", text)
    for m in _STATUS_RX.finditer(text):
        st = m.group(3)
        if st.startswith("skipped"):
            st = "skip"
        else:
            st = "ok" if st.startswith(("ok", "expected failure")) else ("FAIL" if st != "ERROR" else "ERROR")
        statuses[_test_key(m.group(1), m.group(2))] = st
    for m in _BLOCK_RX.finditer(text):
        key = _test_key(m.group(2), m.group(3))
        statuses[key] = m.group(1)
        lines = [ln for ln in m.group(4).strip().splitlines() if ln.strip()]
        messages[key] = "\n".join(lines[-3:])[:400]
    return statuses, messages


_FUNC_CASE_RX = re.compile(r"unittest\.case\.FunctionTestCase \((\w+)\)")
_RAN_RX = re.compile(r"^Ran (\d+) tests? in ", re.M)

# Runner for files with plain ``def test_x():`` functions (optionally next to
# TestCase classes) and for files with ``async def`` tests.
# * Functions that need arguments (pytest fixtures) cannot be executed without
#   pytest; that is reported instead of being silently skipped.
# * ``async def test_x()`` at module level is really awaited (``asyncio.run``):
#   unittest.FunctionTestCase would only create the coroutine and report OK.
# * An ``async def`` test method is executed by unittest only inside
#   ``IsolatedAsyncioTestCase``. In any other class its body would never run, so
#   the file is reported as unsupported instead of being counted as passed.
_FUNCTION_RUNNER = r'''
import asyncio, functools, importlib, inspect, sys, unittest
mod = importlib.import_module(sys.argv[1])
suite = unittest.TestSuite()
suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(mod))

def flat(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from flat(t)
        else:
            yield t

not_awaited = []
for t in flat(suite):
    method = getattr(t, getattr(t, "_testMethodName", ""), None)
    if method is not None and inspect.iscoroutinefunction(method) \
            and not isinstance(t, unittest.IsolatedAsyncioTestCase):
        not_awaited.append(t.id())
if not_awaited:
    sys.stderr.write("UNSUPPORTED: async test methods outside IsolatedAsyncioTestCase would not be awaited: %s\n"
                     % ", ".join(not_awaited))
    sys.exit(3)

def awaited(fn):
    @functools.wraps(fn)
    def run():
        return asyncio.run(fn())
    return run

needs_args = []
for name, fn in sorted(vars(mod).items()):
    if name.startswith("test") and inspect.isfunction(fn) and fn.__module__ == mod.__name__:
        required = [p for p in inspect.signature(fn).parameters.values()
                    if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if required:
            needs_args.append(name)
        elif inspect.iscoroutinefunction(fn):
            suite.addTest(unittest.FunctionTestCase(awaited(fn)))
        else:
            suite.addTest(unittest.FunctionTestCase(fn))
if needs_args:
    sys.stderr.write("UNSUPPORTED: test functions need fixtures: %s\n" % ", ".join(needs_args))
    sys.exit(3)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
'''


def test_layout(source: str) -> Tuple[str, int]:
    """(runner kind, number of statically visible tests) of a test file.

    kind: ``unittest`` (test methods in classes), ``functions`` (top-level
    ``test_*`` functions, possibly with classes, or any ``async def`` test - these
    go through the harness' own runner, which knows what is really awaited) or
    ``none``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "none", 0
    methods = functions = async_methods = 0
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            tests = [s for s in node.body
                     if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name.startswith("test")]
            methods += len(tests)
            async_methods += sum(1 for s in tests if isinstance(s, ast.AsyncFunctionDef))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            functions += 1
    if functions or async_methods:
        return "functions", methods + functions
    if methods:
        return "unittest", methods
    return "none", 0


def _run_one_test_file(directory: str, test_file: str, timeout: float, deadline: Optional[float]) -> TestRun:
    t0 = time.monotonic()
    path = wsfiles.safe_file(directory, test_file)
    source = _read(path) if path else None
    if source is None:
        return TestRun(passed=False, load_error="test file %s is missing" % test_file)
    kind, expected = test_layout(source)
    if kind == "none":
        return TestRun(passed=False, load_error="no tests found in %s (nothing to execute)" % test_file)
    stem = os.path.splitext(os.path.basename(test_file))[0]
    if kind == "unittest" and stem.isidentifier():
        args = ["-m", "unittest", "-v", stem]
    else:
        args = ["-c", _FUNCTION_RUNNER, stem]
    res = run_python(args, cwd=directory, timeout=timeout, deadline=deadline)
    out = res.output
    statuses, messages = parse_unittest_output(out)
    # ``python -m unittest`` reports a module that cannot be imported as a failed
    # pseudo-test; for us that is a load error of the file, not a failing test.
    broken = [k for k in statuses if k.startswith("_FailedTest.") or k.startswith("_ErrorHolder.")]
    load_failure = ""
    for k in broken:
        load_failure = load_failure or messages.get(k) or "the test module could not be loaded"
        statuses.pop(k, None)
        messages.pop(k, None)
    m = _RAN_RX.search(out)
    ran = int(m.group(1)) if m else 0
    skipped = sum(1 for v in statuses.values() if v == "skip")
    run = TestRun(passed=res.ok, timed_out=res.timed_out, statuses=statuses, messages=messages,
                  output=out[-6000:], duration=time.monotonic() - t0, ran=ran, executed=max(0, ran - skipped))
    if run.failing:
        run.passed = False
    if load_failure and not res.timed_out:
        run.passed = False
        run.load_error = load_failure[:800]
    elif not res.ok and not run.failing:
        run.passed = False
        run.load_error = "timeout" if res.timed_out else "\n".join(out.strip().splitlines()[-6:])[:800]
    elif res.ok:
        # Exit code 0 is not enough: the tests must really have been executed.
        problem = ""
        if ran == 0 or not statuses:
            problem = "no tests were executed"
        elif ran < expected:
            problem = "only %d of %d tests were run" % (ran, expected)
        elif run.executed == 0:
            problem = "all %d tests were skipped" % ran
        elif not re.search(r"^OK\b", out, re.M):
            problem = "the runner did not report OK"
        if problem:
            run.passed = False
            run.load_error = "%s: %s" % (test_file, problem)
    return run


def run_tests(directory: str, test_files, timeout: float, deadline: Optional[float] = None) -> TestRun:
    """Execute the required test file(s); green only if every file is green."""
    files = [test_files] if isinstance(test_files, str) else list(test_files)
    if len(files) == 1:
        return _run_one_test_file(directory, files[0], timeout, deadline)
    total = TestRun(passed=True)
    for f in files:
        r = _run_one_test_file(directory, f, timeout, deadline)
        total.passed = total.passed and r.passed
        total.timed_out = total.timed_out or r.timed_out
        total.duration += r.duration
        total.ran += r.ran
        total.executed += r.executed
        total.output = (total.output + "\n" + r.output)[-6000:]
        for k, v in r.statuses.items():
            total.statuses[multi_key(f, k)] = v
        for k, v in r.messages.items():
            total.messages[multi_key(f, k)] = v
        if r.load_error and not total.load_error:
            total.load_error = r.load_error
    return total


def multi_key(test_file: str, key: str) -> str:
    return "%s::%s" % (os.path.splitext(os.path.basename(test_file))[0], key)


def is_improvement(new: TestRun, old: TestRun) -> bool:
    if new.timed_out or new.load_error:
        return False
    if new.passed:
        return True
    if old.statuses and new.statuses and not old.load_error:
        return new.failing < old.failing and set(new.statuses) >= set(old.statuses)
    return False


# ----------------------------------------------------------- static analysis
@dataclass
class Unit:
    """A function (or method) of the library that can be mutated."""
    qualname: str
    node: ast.AST
    start: int  # first line index (0-based, including decorators)
    end: int    # line index after the last line
    indent: int


def parse_units(source: str) -> Optional[Dict[str, Unit]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    units: Dict[str, Unit] = {}

    def add(node, prefix: str) -> None:
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        units[prefix + node.name] = Unit(prefix + node.name, node, first, node.end_lineno, node.col_offset)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, "")
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(sub, node.name + ".")
    return units


def apply_variants(source: str, units: Dict[str, Unit], chosen: Dict[str, str]) -> str:
    lines = source.splitlines(keepends=True)
    for name in sorted(chosen, key=lambda n: units[n].start, reverse=True):
        u = units[name]
        text = chosen[name]
        if not text.endswith("\n"):
            text += "\n"
        lines[u.start:u.end] = [text]
    return "".join(lines)


@dataclass
class TestInfo:
    syntax_error: str = ""
    lib_aliases: Set[str] = field(default_factory=set)
    direct_names: Dict[str, str] = field(default_factory=dict)  # local name -> lib attribute
    refs: Dict[str, Set[str]] = field(default_factory=dict)      # "Class.test" -> lib attributes used
    sources: Dict[str, str] = field(default_factory=dict)        # "Class.test" -> source text
    contradictions: List[str] = field(default_factory=list)   # proven only (see analyze_tests)
    suspicions: List[str] = field(default_factory=list)       # same call, different results, purity unknown
    imports: Set[str] = field(default_factory=set)


def _literal(node: ast.AST) -> Tuple[bool, str]:
    try:
        return True, repr(ast.literal_eval(node))
    except Exception:
        return False, ""


def stateless_functions(lib_source: str) -> Set[str]:
    """Top-level functions of the library that evidently have no state.

    A function qualifies when it has no decorators, only constant defaults, no
    ``global`` / ``nonlocal`` / ``yield``, and touches neither module-level
    variables, imported modules, nor I/O / reflection builtins, and calls only
    other qualifying functions. Only for such functions "the same call must
    return two different values" is a real contradiction; a counter such as
    ``next_id()`` legitimately returns 1 and then 2.
    """
    try:
        tree = ast.parse(lib_source)
    except SyntaxError:
        return set()
    funcs: Dict[str, ast.AST] = {}
    state_names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                state_names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ClassDef):
            state_names.add(node.name)
        else:
            for x in ast.walk(node):
                if isinstance(x, ast.Name):
                    state_names.add(x.id)
    risky = {"input", "open", "next", "iter", "globals", "locals", "vars", "setattr", "getattr", "delattr",
             "exec", "eval", "compile", "id", "hash", "__import__", "object", "super", "type"}
    good: Set[str] = set()
    for name, fn in funcs.items():
        if fn.decorator_list:
            continue
        defaults = list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d is not None]
        if any(not isinstance(d, ast.Constant) for d in defaults):
            continue
        ok = True
        for x in ast.walk(fn):
            if isinstance(x, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await, ast.Import,
                              ast.ImportFrom, ast.ClassDef, ast.Lambda)):
                ok = False
            elif isinstance(x, ast.Name) and (x.id in state_names or x.id in risky):
                ok = False
            elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x is not fn:
                ok = False
            if not ok:
                break
        if ok:
            good.add(name)
    changed = True
    while changed:  # a function is only as stateless as the functions it uses
        changed = False
        for name in list(good):
            used = {x.id for x in ast.walk(funcs[name]) if isinstance(x, ast.Name) and x.id in funcs}
            if not used <= good:
                good.discard(name)
                changed = True
    return good


def analyze_tests(test_source: str, lib_module: str, lib_source: Optional[str] = None) -> TestInfo:
    """Static facts about a test file. ``contradictions`` are filled only when
    ``lib_source`` is given and the function involved is evidently stateless."""
    info = TestInfo()
    pure = stateless_functions(lib_source) if lib_source is not None else set()
    try:
        tree = ast.parse(test_source)
    except SyntaxError as e:
        info.syntax_error = "%s (line %s)" % (e.msg, e.lineno)
        return info
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                info.imports.add(a.name.split(".")[0])
                if a.name == lib_module:
                    info.lib_aliases.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                info.imports.add(node.module.split(".")[0])
            if node.module == lib_module:
                for a in node.names:
                    info.direct_names[a.asname or a.name] = a.name

    # Module-level helpers / fixtures of the test file: a test using one of them may
    # depend on the library indirectly, so it is never classified as "independent".
    module_names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("test"):
            module_names.add(node.name)
        elif isinstance(node, ast.ClassDef) and not any(
                isinstance(s, ast.FunctionDef) and s.name.startswith("test") for s in node.body):
            module_names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for tgt in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for x in ast.walk(tgt):
                    if isinstance(x, ast.Name):
                        module_names.add(x.id)

    def lib_refs(n: ast.AST) -> Set[str]:
        out: Set[str] = set()
        for x in ast.walk(n):
            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in info.lib_aliases:
                out.add(x.attr)
            elif isinstance(x, ast.Name) and x.id in info.direct_names:
                out.add(info.direct_names[x.id])
            elif isinstance(x, ast.Name) and x.id in module_names:
                out.add("<helper:%s>" % x.id)
            elif isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "self" \
                    and not x.attr.startswith(("assert", "fail", "subTest", "skipTest")):
                out.add("<self.%s>" % x.attr)
        return out

    def call_key(n: ast.AST) -> Optional[str]:
        """Canonical text of a library call with literal arguments."""
        if not isinstance(n, ast.Call) or not lib_refs(n.func):
            return None
        parts = []
        for a in n.args:
            ok, r = _literal(a)
            if not ok:
                return None
            parts.append(r)
        for k in n.keywords:
            ok, r = _literal(k.value)
            if not ok or k.arg is None:
                return None
            parts.append("%s=%s" % (k.arg, r))
        return "%s(%s)" % (ast.unparse(n.func), ", ".join(parts))

    def direct_function(n: ast.AST) -> Optional[str]:
        """Library function called directly (``lib.f(...)`` or ``f(...)``), else None."""
        f = n.func if isinstance(n, ast.Call) else None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in info.lib_aliases:
            return f.attr
        if isinstance(f, ast.Name) and f.id in info.direct_names:
            return info.direct_names[f.id]
        return None

    expectations: Dict[str, Dict[str, str]] = {}  # call -> {expected repr: test}
    forbidden: Dict[str, Set[str]] = {}
    call_func: Dict[str, Optional[str]] = {}
    lines = test_source.splitlines()

    def visit_test(key: str, fn: ast.AST, helpers: Set[str]) -> None:
        info.refs[key] = lib_refs(fn) | helpers
        info.sources[key] = "\n".join(lines[fn.lineno - 1:fn.end_lineno])
        for x in ast.walk(fn):
            if not (isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)):
                continue
            name = x.func.attr
            if name in ("assertEqual", "assertEquals", "assertNotEqual", "assertIs") and len(x.args) >= 2:
                a, b = x.args[0], x.args[1]
                for call, lit in ((a, b), (b, a)):
                    ck = call_key(call)
                    ok, r = _literal(lit)
                    if ck and ok:
                        call_func[ck] = direct_function(call)
                        if name == "assertNotEqual":
                            forbidden.setdefault(ck, set()).add(r)
                        else:
                            expectations.setdefault(ck, {})[r] = key
                        break
            elif name in ("assertTrue", "assertFalse") and x.args:
                ck = call_key(x.args[0])
                if ck:
                    call_func["bool:" + ck] = direct_function(x.args[0])
                    expectations.setdefault("bool:" + ck, {})[name] = key

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            helpers: Set[str] = set()
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name in ("setUp", "setUpClass", "tearDown"):
                    helpers |= lib_refs(sub)
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name.startswith("test"):
                    visit_test("%s.%s" % (node.name, sub.name), sub, helpers)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            visit_test(node.name, node, set())

    for ck, exp in expectations.items():
        found = []
        if len(exp) > 1:
            found.append("%s is expected to equal both %s" % (ck, " and ".join(sorted(exp))))
        for r in exp:
            if r in forbidden.get(ck, ()):
                found.append("%s must and must not equal %s" % (ck, r))
        # A contradiction is proven only for a stateless function; otherwise the
        # two results may simply come from two different states (a counter).
        (info.contradictions if call_func.get(ck) in pure else info.suspicions).extend(found)
    return info


# -------------------------------------------------------------------- context
@dataclass
class CodingContext:
    workspace: str
    test_file: str
    lib_file: str
    lib_module: str
    lib_source: str
    test_source: str
    info: TestInfo
    baseline: Optional[TestRun] = None
    done_word: str = "DONE"
    scratch: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)          # the required set, in run order
    test_sources: Dict[str, str] = field(default_factory=dict)   # file -> source as analysed

    def cleanup(self) -> None:
        for s in self.scratch:
            remove_scratch(s)
        self.scratch = []


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def locate_all_tests(instruction: str, workspace: str) -> Tuple[List[str], Optional[str], str]:
    """Returns (required test files, library file, reason if something is missing).

    A test file named by the instruction is the required set. Without a named
    file every ``test_*.py`` / ``*_test.py`` of the workspace is required - one
    of several is never picked silently.
    """
    test, lib, why = locate_files(instruction, workspace)
    if test is None and why == "no test file identified":
        py = wsfiles.top_level_names(workspace, (".py",))
        cands = [f for f in py if f.startswith("test_") or f.endswith("_test.py")]
        if len(cands) > 1:
            test, lib, why = locate_files(instruction, workspace, assume_test=cands[0])
            return (cands if test else []), lib, why
    return ([test] if test else []), lib, why


def locate_files(instruction: str, workspace: str,
                 assume_test: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
    """Returns (test file, library file, reason if something is missing)."""
    if not os.path.isdir(workspace):
        return None, None, "workspace is not readable"
    py = wsfiles.top_level_names(workspace, (".py",))  # regular files inside the workspace only
    ins = instruction or ""
    test = None
    m = re.search(r"python3?\s+(?:-m\s+(?:unittest|pytest)\s+)?`?([\w./-]+\.py)", ins)
    if m:
        test = os.path.basename(m.group(1))
    elif re.search(r"python3?\s+-m\s+unittest\s+([\w.]+)", ins):
        test = re.search(r"python3?\s+-m\s+unittest\s+([\w.]+)", ins).group(1).split(".")[0] + ".py"
    if test is None:
        cands = [f for f in py if f.startswith("test_") or f.endswith("_test.py")]
        test = cands[0] if len(cands) == 1 else assume_test
    if test is None:
        return None, None, "no test file identified"
    if test not in py:
        return None, None, "test file %s is missing" % test

    lib = None
    m = re.search(r"(?:bugs?|errors?|mistakes?|defects?|problems?|fix\w*|repair\w*|edit\w*|modify\w*|correct\w*)"
                  r"(?:\s+\w+){0,3}?\s+(?:in|of|to)?\s*`?([\w./-]+\.py)", ins, re.I)
    for cand in ([os.path.basename(m.group(1))] if m else []) + re.findall(r"`?([\w-]+\.py)`?", ins):
        cand = os.path.basename(cand)
        if cand != test and not cand.startswith("test_"):
            lib = cand
            break
    if lib is None:
        src = _read(os.path.join(workspace, test)) or ""
        mods = re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", src, re.M)
        local = [x + ".py" for x in mods if x + ".py" in py and x + ".py" != test]
        lib = local[0] if local else None
    if lib is None:
        return test, None, "library file not identified"
    if lib not in py:
        return test, None, "library file %s is missing" % lib
    return test, lib, ""


class ScratchPool:
    """A few private copies of the workspace for parallel test runs."""

    # All scratch copies of one pool together stay below this size (/tmp is a 2 GB tmpfs).
    BYTES_LIMIT = 192 * 1024 * 1024

    def __init__(self, ctx: CodingContext, size: int):
        self.ctx = ctx
        self.q: "queue.Queue[str]" = queue.Queue()
        self.size = 0
        self.dirs: List[str] = []
        per_copy = max(1, workspace_size(ctx.workspace))
        size = max(1, min(size, self.BYTES_LIMIT // per_copy))
        for _ in range(size):
            d = copy_workspace(ctx.workspace)
            if d:
                ctx.scratch.append(d)
                self.dirs.append(d)
                self.q.put(d)
                self.size += 1

    def evaluate(self, lib_source: str, timeout: float, deadline: Optional[float] = None) -> TestRun:
        d = self.q.get()
        try:
            try:
                with open(os.path.join(d, self.ctx.lib_file), "w", encoding="utf-8", newline="\n") as f:
                    f.write(lib_source)
            except OSError:
                return TestRun(passed=False, load_error="cannot write scratch copy")
            return run_tests(d, self.ctx.test_files, timeout, deadline)
        finally:
            self.q.put(d)

    def close(self) -> None:
        """Free the scratch copies as soon as the work that needed them is over."""
        for d in self.dirs:
            remove_scratch(d)
            if d in self.ctx.scratch:
                self.ctx.scratch.remove(d)
        self.dirs = []


# ------------------------------------------------------------------ analysis
def prepare(task, settings) -> Tuple[Optional[Outcome], Optional[CodingContext]]:
    """Baseline run and free checks. Returns (final outcome or None, context)."""
    tests, lib, why = locate_all_tests(task.instruction, task.workspace)
    if not tests or lib is None:
        if "missing" in why:
            return Outcome.infeasible(why), None
        return Outcome.unknown(why), None
    test = tests[0]
    sources = {t: _read(wsfiles.safe_file(task.workspace, t) or "") for t in tests}
    lib_path = wsfiles.safe_file(task.workspace, lib)
    lib_src = _read(lib_path) if lib_path else None
    if lib_src is None or any(s is None for s in sources.values()):
        return Outcome.unknown("task files are not readable"), None
    test_src = sources[test]
    module = os.path.splitext(lib)[0]
    info = analyze_tests(test_src, module, lib_src)
    for extra in tests[1:]:  # several required files: merge the analysis under file-qualified keys
        other = analyze_tests(sources[extra], module, lib_src)
        info.syntax_error = info.syntax_error or other.syntax_error
        info.lib_aliases |= other.lib_aliases
        info.direct_names.update(other.direct_names)
        info.imports |= other.imports
        info.contradictions += other.contradictions
        info.suspicions += other.suspicions
        for k, v in other.refs.items():
            info.refs[multi_key(extra, k)] = v
        for k, v in other.sources.items():
            info.sources[multi_key(extra, k)] = v
    if len(tests) > 1:
        for k in [k for k in info.refs if "::" not in k]:
            info.refs[multi_key(test, k)] = info.refs.pop(k)
            info.sources[multi_key(test, k)] = info.sources.pop(k)
    ctx = CodingContext(task.workspace, test, lib, module, lib_src, test_src, info,
                        test_files=list(tests), test_sources=sources)
    m = re.search(r"write ONLY the word\s+[`'\"]?(\w+)", task.instruction or "", re.I)
    if m:
        ctx.done_word = m.group(1)

    if info.syntax_error:
        return Outcome.infeasible("the test file has a syntax error and may not be edited: " + info.syntax_error), ctx

    if any(test_layout(s)[0] == "none" for s in sources.values()):
        # Nothing that could be executed and counted: success cannot be shown,
        # neither by us nor by a model, so no tokens are spent on it.
        return Outcome.unknown("no executable tests found in the test file"), None

    pool = ScratchPool(ctx, 1)
    if pool.size == 0:
        return Outcome.unknown("cannot create a scratch copy of the workspace"), ctx
    try:
        base = pool.evaluate(lib_src, settings.test_timeout_s)
    finally:
        pool.close()
    ctx.baseline = base
    if base.passed:
        return Outcome.solved(ctx.done_word, "tests already pass (%d run, %d executed)" % (base.ran, base.executed),
                              tier="deterministic"), ctx
    if "UNSUPPORTED:" in base.output or (not base.failing and not base.timed_out and base.load_error
                                         and re.search(r"no tests were executed|tests were skipped|tests were run",
                                                       base.load_error)):
        return Outcome.unknown("tests cannot be executed and counted: " + base.load_error[:160]), None

    if info.contradictions:
        return Outcome.infeasible("contradictory tests: " + "; ".join(info.contradictions[:2])), ctx

    if base.load_error and not base.timed_out:
        m = re.search(r"(?:ModuleNotFoundError|ImportError): No module named '([\w.]+)'", base.load_error)
        if m:
            missing = m.group(1).split(".")[0]
            try:
                lib_imports = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "").split(".")[0]
                               for n in ast.walk(ast.parse(lib_src)) if isinstance(n, (ast.Import, ast.ImportFrom))}
            except SyntaxError:
                lib_imports = set()
            if missing != module and missing in info.imports and missing not in lib_imports:
                return Outcome.infeasible("the tests import a module that does not exist: " + missing), ctx

    # A failing test that never touches the library cannot be fixed from the library.
    if base.failing and (info.lib_aliases or info.direct_names):
        for key in sorted(base.failing):
            if key in info.refs and not info.refs[key]:
                return Outcome.infeasible("failing test %s does not depend on %s" % (key, lib)), ctx
    return None, ctx


# ------------------------------------------------------------ mutation repair
def _target_units(units: Dict[str, Unit], run: TestRun, info: TestInfo) -> List[str]:
    names: Set[str] = set()
    for key in run.failing:
        names |= info.refs.get(key, set())
    targets: Set[str] = set()
    for n in names:
        for q in units:
            if q == n or q.startswith(n + "."):
                targets.add(q)
    if not targets:
        return list(units)
    # transitive closure over helper functions called from the targets
    changed = True
    while changed:
        changed = False
        for q in list(targets):
            for x in ast.walk(units[q].node):
                name = x.id if isinstance(x, ast.Name) else (x.attr if isinstance(x, ast.Attribute) else None)
                if not name:
                    continue
                for other in units:
                    if other not in targets and (other == name or other.endswith("." + name)):
                        targets.add(other)
                        changed = True
    return sorted(targets, key=lambda q: units[q].start)


def _evaluate_in_windows(ex, pool: "ScratchPool", jobs, make_source, timeout: float, deadline: float):
    """Evaluate ``jobs`` a few at a time and yield ``(job, TestRun)``.

    Nothing is queued ahead: new work is submitted only while the deadline has
    not passed and the consumer keeps asking, so at most ``2 * pool.size`` test
    runs are ever outstanding and each of them is itself cut at the deadline.
    """
    pending = {}
    it = iter(jobs)
    exhausted = False
    try:
        while True:
            while not exhausted and len(pending) < 2 * pool.size and time.monotonic() < deadline:
                job = next(it, None)
                if job is None:
                    exhausted = True
                    break
                pending[ex.submit(pool.evaluate, make_source(job), timeout, deadline)] = job
            if not pending:
                return
            done, _ = wait(list(pending), timeout=timeout + 10, return_when=FIRST_COMPLETED)
            if not done and time.monotonic() >= deadline + timeout + 10:
                return
            for fut in done:
                job = pending.pop(fut)
                try:
                    yield job, fut.result()
                except Exception:  # noqa: BLE001
                    continue
    finally:
        for fut in pending:
            fut.cancel()


def mutation_repair(ctx: CodingContext, settings, time_limit: float = 120.0,
                    run_deadline: Optional[float] = None) -> Optional[str]:
    """Search single-edit fixes validated by the tests. Returns new source or None."""
    base = ctx.baseline
    if base is None or parse_units(ctx.lib_source) is None:
        return None
    deadline = time.monotonic() + time_limit
    if run_deadline is not None:
        deadline = min(deadline, run_deadline)
    if deadline - time.monotonic() < 3:
        return None
    pool = ScratchPool(ctx, settings.workers)
    if pool.size == 0:
        return None
    try:
        return _mutation_search(ctx, settings, pool, base, deadline)
    finally:
        pool.close()


def _mutation_search(ctx: CodingContext, settings, pool: "ScratchPool", base: TestRun,
                     deadline: float) -> Optional[str]:
    quick = max(3.0, min(settings.test_timeout_s, 6.0 * max(base.duration, 0.3)))
    if base.timed_out:
        quick = min(settings.test_timeout_s, 5.0)  # probably an endless loop: fail fast
    source, run = ctx.lib_source, base
    evaluated = 0

    with ThreadPoolExecutor(max_workers=pool.size) as ex:
        for _round in range(5):
            if run.passed:
                break
            units = parse_units(source)
            if not units:
                return None
            targets = _target_units(units, run, ctx.info)
            picks: Dict[str, Tuple[int, str, str, TestRun]] = {}
            seen_texts: Set[str] = set()
            for rank in (0, 1, 2, 3):
                jobs = []
                for q in targets:
                    u = units[q]
                    original = render(u.node, u.indent)
                    for r, desc, node in function_mutants(u.node, max_rank=rank):
                        if r != rank:
                            continue
                        try:
                            text = render(node, u.indent)
                            ast.parse(text.strip() if not u.indent else "if 1:\n" + text)
                        except Exception:
                            continue
                        if text == original or (q, text) in seen_texts:
                            continue
                        seen_texts.add((q, text))
                        jobs.append((q, r, desc, text))
                if not jobs:
                    continue
                if time.monotonic() > deadline or evaluated >= 4000:
                    break
                jobs = jobs[:4000 - evaluated]
                for (q, r, desc, text), res in _evaluate_in_windows(
                        ex, pool, jobs, lambda job: apply_variants(source, units, {job[0]: job[3]}), quick,
                        deadline):
                    evaluated += 1
                    if is_improvement(res, run):
                        cur = picks.get(q)
                        score = (len(res.failing) if not res.passed else -1)
                        if cur is None or score < (len(cur[3].failing) if not cur[3].passed else -1):
                            picks[q] = (r, desc, text, res)
                        if res.passed:
                            break  # a complete fix: no need to look at the rest of this rank
                if picks:
                    break
            if not picks:
                return None
            chosen = {q: p[2] for q, p in picks.items()}
            new_source = apply_variants(source, units, chosen)
            new_run = pool.evaluate(new_source, quick, deadline + quick) if len(chosen) > 1 \
                else next(iter(picks.values()))[3]
            if not is_improvement(new_run, run):
                # the individual fixes interfere: keep only the best single one
                q = min(picks, key=lambda k: (len(picks[k][3].failing) if not picks[k][3].passed else -1, picks[k][0]))
                chosen = {q: picks[q][2]}
                new_source = apply_variants(source, units, chosen)
                new_run = picks[q][3]
            for q in chosen:
                ctx.notes.append("mutation %s: %s" % (q, picks[q][1]))
            source, run = new_source, new_run
    ctx.notes.append("mutants evaluated: %d" % evaluated)
    if not run.passed:
        return None
    # final confirmation with the normal timeout
    final = pool.evaluate(source, settings.test_timeout_s)
    return source if final.passed else None


# ---------------------------------------------------------------- finalising
def commit_fix(task, ctx: CodingContext, new_source: str, settings) -> bool:
    """Install the verified library into the task workspace and prove the result.

    The file is replaced atomically. The installed state is then copied to a
    fresh scratch directory together with the *untouched* test files and the
    tests are executed there once more; if that run is not green the original
    library is put back, so an unverified change is never left behind.
    """
    for name, analysed in ctx.test_sources.items():
        if _read(wsfiles.safe_file(ctx.workspace, name) or "") != analysed:
            ctx.notes.append("test file %s changed during the run" % name)
            return False
    path = wsfiles.safe_file(ctx.workspace, ctx.lib_file)
    if path is None:
        return False
    changed = new_source != ctx.lib_source
    if changed and not write_text_atomic(path, new_source):
        ctx.notes.append("the library file could not be replaced (permissions)")
        return False
    verified = False
    fresh = copy_workspace(ctx.workspace)
    if fresh is not None:
        try:
            unchanged = all(_read(os.path.join(fresh, n)) == s for n, s in ctx.test_sources.items())
            installed = (_read(os.path.join(fresh, ctx.lib_file)) or "") == new_source
            if unchanged and installed:
                verified = run_tests(fresh, ctx.test_files, settings.test_timeout_s).passed
        finally:
            remove_scratch(fresh)
    if not verified and changed:
        write_text_atomic(path, ctx.lib_source)  # roll back
        ctx.notes.append("installed fix failed the final verification and was rolled back")
    return verified


def solve_free(task, settings, time_limit: float = 120.0,
               run_deadline: Optional[float] = None) -> Tuple[Outcome, Optional[CodingContext]]:
    """Everything that costs no tokens. UNKNOWN means "needs the LLM"."""
    outcome, ctx = prepare(task, settings)
    if outcome is not None:
        if outcome.status == "solved" and ctx is not None:
            # green without changes: still confirm test files / installed state
            if not commit_fix(task, ctx, ctx.lib_source, settings):
                return Outcome.unknown("tests passed once but the confirmation run did not"), None
        return outcome, ctx
    if settings.enable_mutation_repair:
        try:
            fixed = mutation_repair(ctx, settings, time_limit, run_deadline)
        except Exception as e:  # noqa: BLE001
            ctx.notes.append("mutation repair error: %s" % e)
            fixed = None
        if fixed is not None:
            if commit_fix(task, ctx, fixed, settings):
                return Outcome.solved(ctx.done_word, "; ".join(ctx.notes), tier="mutation"), ctx
            ctx.notes.append("fix found but could not be installed into the workspace")
    return Outcome.unknown("tests fail: " + ", ".join(sorted(ctx.baseline.failing))[:200] if ctx.baseline
                           else "tests fail"), ctx


# ------------------------------------------------------------------ LLM repair
SYSTEM_PROMPT = (
    "You fix bugs in a small Python module so that its unittest file passes. "
    "Reply with ONLY one ```python code block containing the complete corrected definitions of the "
    "functions you change - no explanations, no tests, no unchanged functions. "
    "The tests must not be modified. If the tests cannot all pass with any correct implementation "
    "(they contradict each other or are broken), reply with exactly: INFEASIBLE"
)


def extract_code(reply: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.S | re.I)
    if not blocks:
        m = re.search(r"```(?:python|py)?[ \t]*\n(.*)\Z", reply, re.S | re.I)  # unterminated block
        if m:
            blocks = [m.group(1)]
    code = "\n\n".join(b.strip("\n") for b in blocks) if blocks else reply
    for cand in (code, reply):
        try:
            ast.parse(cand)
            if re.search(r"^\s*(?:def|class|import|from|[A-Za-z_]\w*\s*=)", cand, re.M):
                return cand
        except SyntaxError:
            continue
    return None


def merge_patch(source: str, patch: str) -> Optional[str]:
    """Replace same-named top-level definitions of ``source`` by those of ``patch``."""
    try:
        ptree = ast.parse(patch)
        stree = ast.parse(source)
    except SyntaxError:
        try:
            ast.parse(patch)
            return patch if patch.endswith("\n") else patch + "\n"  # original is broken: take the reply as the file
        except SyntaxError:
            return None
    plines = patch.splitlines()
    slines = source.splitlines()

    def span(node) -> Tuple[int, int]:
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
        return first, node.end_lineno

    existing = {}
    for node in stree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            existing[node.name] = span(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            existing[node.targets[0].id] = span(node)
    replacements = []
    appended: List[str] = []
    imports: List[str] = []
    for node in ptree.body:
        a, b = span(node)
        text = "\n".join(plines[a:b])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if text.strip() not in source:
                imports.append(text)
            continue
        else:
            continue  # stray expressions / prints are ignored
        if name in existing:
            replacements.append((existing[name], text))
        else:
            appended.append(text)
    for (a, b), text in sorted(replacements, key=lambda r: r[0][0], reverse=True):
        slines[a:b] = text.splitlines()
    out = "\n".join(imports + slines)
    if appended:
        out = out.rstrip("\n") + "\n\n\n" + "\n\n\n".join(appended)
    out = out.rstrip("\n") + "\n"
    try:
        ast.parse(out)
    except SyntaxError:
        return None
    return out


def build_prompt(ctx: CodingContext, run: TestRun, lib_source: str) -> str:
    failing = sorted(run.failing)
    parts = ["%s:\n```python\n%s\n```" % (ctx.lib_file, lib_source.strip()[:9000])]
    if failing and all(k in ctx.info.sources for k in failing):
        tests = "\n\n".join(ctx.info.sources[k] for k in failing)
        parts.append("FAILING tests from %s (it does `import %s`%s):\n```python\n%s\n```" % (
            ", ".join(ctx.test_files or [ctx.test_file]), ctx.lib_module,
            "".join(" as %s" % a for a in sorted(ctx.info.lib_aliases) if a != ctx.lib_module)[:20],
            tests[:5000]))
        msgs = "\n".join("%s: %s" % (k.split(".")[-1], run.messages.get(k, "").splitlines()[-1] if run.messages.get(k) else "failed")
                         for k in failing)
        parts.append("Failure messages:\n" + msgs[:1500])
    else:
        all_tests = "\n\n".join(s.strip() for s in ctx.test_sources.values()) or ctx.test_source.strip()
        parts.append("%s (FAILING):\n```python\n%s\n```" % (", ".join(ctx.test_files or [ctx.test_file]),
                                                          all_tests[:6000]))
        parts.append("Test output (tail):\n" + (run.load_error or run.output[-1200:]))
    return "\n\n".join(parts)


def estimate_llm_cost(ctx: CodingContext) -> float:
    """Planning estimate (weighted tokens) of one LLM repair attempt."""
    run = ctx.baseline
    prompt = build_prompt(ctx, run, ctx.lib_source) if run else ctx.lib_source + ctx.test_source
    return (estimate_tokens(SYSTEM_PROMPT + prompt) * 0.1 + _output_allowance(ctx, run)) * 1.3


def _output_allowance(ctx: CodingContext, run: Optional[TestRun]) -> int:
    units = parse_units(ctx.lib_source) or {}
    size = 0
    if run is not None and units:
        for q in _target_units(units, run, ctx.info):
            u = units[q]
            size += estimate_tokens("\n".join(ctx.lib_source.splitlines()[u.start:u.end]))
    if size == 0:
        size = estimate_tokens(ctx.lib_source)
    return int(max(160, min(1200, size * 1.6 + 80)))


async def solve_llm(task, ctx: CodingContext, llm, settings, trace=None) -> Outcome:
    run = ctx.baseline
    if run is None:
        return Outcome.unknown("no baseline test run")
    pool = ScratchPool(ctx, 1)
    if pool.size == 0:
        return Outcome.unknown("cannot create a scratch copy of the workspace")
    try:
        return await _llm_repair(task, ctx, llm, settings, trace, pool, run)
    finally:
        pool.close()


# Constructs a library fix has no reason to contain; they are how a "fix" could
# make the tests pass without fixing anything (patching the runner, exiting early).
_SUSPICIOUS_PATCH = re.compile(
    r"\bunittest\b|\bsys\s*\.\s*(?:exit|modules|settrace)|\bos\s*\.\s*_exit\b|\batexit\b|\bbuiltins\b|"
    r"__import__|\bimportlib\b|\bsubprocess\b|\bsocket\b|\binspect\b|test_\w+\.py")


async def _llm_repair(task, ctx: CodingContext, llm, settings, trace, pool: "ScratchPool", run: TestRun) -> Outcome:
    source = ctx.lib_source
    allowance = _output_allowance(ctx, run)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(ctx, run, source)}]
    # Up to three attempts; every attempt must fit into what is left of the
    # per-task cap, so the third one happens only if the first two were cheap.
    plan = [(settings.model_fast, allowance), (settings.model_fast, allowance),
            (settings.model_strong, min(2000, allowance + 400))]
    last_reason = "no attempt made"
    for attempt, (model, wanted) in enumerate(plan, 1):
        adm = llm.admission(task.task_id, messages, wanted, minimum=max(120, int(wanted * 0.5)))
        max_tokens = adm["allowed"]
        if max_tokens <= 0:
            # Postponed for lack of budget - this says nothing about feasibility.
            progress = "failing tests %d -> %d" % (len(ctx.baseline.failing) if ctx.baseline else -1, len(run.failing))
            last_reason = ("attempt %d not started: %s (task spent %s of cap %s%s, free global budget %s, "
                           "input %s, wanted %d, minimum %d; %s)" % (
                               attempt, adm["limit"], adm["task_spent"], adm["task_cap"],
                               " set by HARNESS_TASK_CAP" if adm["task_cap_explicit"] else "", adm["global_free"],
                               adm["input_cost"], adm["wanted"], adm["minimum"], progress))
            if trace:
                trace.event(task.task_id, "attempt_denied", attempt=attempt, progress=progress, **adm)
            break
        try:
            reply = await llm.chat(task.task_id, messages, model=model, max_tokens=max_tokens,
                                   purpose="coding_repair_%d" % attempt)
        except BudgetExceeded as e:
            last_reason = "budget: %s" % e
            break
        except LLMError as e:
            last_reason = "llm unavailable: %s" % str(e)[:120]
            break
        text = reply.text
        if re.fullmatch(r"\W*INFEASIBLE\W*", text.strip(), re.I) or (
                "INFEASIBLE" in text.upper() and "```" not in text and "def " not in text):
            return Outcome.infeasible("model judged the tests unsatisfiable", tier="llm")
        code = extract_code(text)
        feedback = None
        if code is None:
            feedback = "Your reply contained no valid Python code block."
        elif _SUSPICIOUS_PATCH.search(code) and not _SUSPICIOUS_PATCH.search(ctx.lib_source):
            feedback = "Fix the functions themselves; do not touch the test runner, imports of test tools or exits."
        else:
            candidate = merge_patch(source, code)
            if candidate is None:
                feedback = "Your code could not be merged into the module (syntax error)."
            else:
                res = await asyncio.to_thread(pool.evaluate, candidate, settings.test_timeout_s)
                if res.passed:
                    ok = await asyncio.to_thread(commit_fix, task, ctx, candidate, settings)
                    if ok:
                        return Outcome.solved(ctx.done_word, "LLM repair, attempt %d (%s)" % (attempt, model),
                                              tier="llm")
                    return Outcome.unknown("fix found but could not be written to the workspace")
                if is_improvement(res, run):
                    source, run = candidate, res  # keep partial progress
                still = ", ".join(k.split(".")[-1] for k in sorted(res.failing)) or "the test run"
                detail = "; ".join("%s: %s" % (k.split(".")[-1], (res.messages.get(k) or "").splitlines()[-1][:160])
                                   for k in sorted(res.failing) if res.messages.get(k))
                feedback = "Still failing: %s. %s" % (still, detail or (res.load_error or "")[-300:])
        last_reason = feedback or "tests still fail"
        if trace:
            trace.event(task.task_id, "coding_attempt_failed", attempt=attempt, feedback=feedback)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(ctx, run, source) +
                     "\n\nA previous fix attempt did not work. " + (feedback or "") +
                     "\nReturn the corrected functions again (only one python code block) or INFEASIBLE."}]
    return Outcome.unknown("LLM repair failed: " + last_reason)
