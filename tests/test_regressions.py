"""Regression tests for the defects found by the independent audit (13.11.2026).

Every test reproduces one confirmed defect and fails on the pre-fix code.
No network and no real proxy: the LLM is replaced by local stubs / a local
HTTP server. Run with ``python -m unittest discover -s tests -v``.
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_harness import answers, wsfiles  # noqa: E402
from agent_harness.budget import Ledger, weighted_cost  # noqa: E402
from agent_harness.config import Settings  # noqa: E402
from agent_harness.llm import BudgetExceeded, LLMClient, LLMUnavailable, Reply, classify  # noqa: E402
from agent_harness.sandbox import copy_workspace, remove_scratch, run_python  # noqa: E402
from agent_harness.solvers import (coding_solver, data_solver, llm_fallback, math_solver,  # noqa: E402
                                   search_solver)
from agent_harness.tasks import load_task  # noqa: E402

SENTINEL = "AUDIT_GRADING_SENTINEL"


def make_task(tmp: pathlib.Path, workspace: bool, **extra) -> pathlib.Path:
    meta = dict(task_id="t1", task_type="math", instruction="Compute seven squared.", reward=1,
                expected=SENTINEL, check="exact_int", feasible=True, expect="free", note=SENTINEL)
    meta.update(extra)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "task.json").write_text(json.dumps(meta), encoding="utf-8")
    if workspace:
        (tmp / "workspace").mkdir(exist_ok=True)
    return tmp


def symlink_or_skip(test: unittest.TestCase, target: pathlib.Path, link: pathlib.Path) -> None:
    try:
        os.symlink(str(target), str(link), target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as e:  # Windows without the privilege
        test.skipTest("symlinks are not available here: %s" % e)


class GradingMetadataIsolation(unittest.TestCase):
    """3.1: nothing from task.json but the four documented fields reaches a prompt or a child."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))

    def test_no_workspace_prompt_and_scratch_are_clean(self):
        d = make_task(self.tmp / "a", workspace=False)
        (d / "notes.txt").write_text("seven squared is asked\n", encoding="utf-8")
        (d / "answer.txt").write_text("OLD\n", encoding="utf-8")
        t = load_task(str(d))
        self.assertEqual(t.workspace, str(d))  # tasks without workspace/ are still supported
        prompt = json.dumps(llm_fallback.build_messages(t))
        self.assertNotIn(SENTINEL, prompt)
        self.assertNotIn("task.json", prompt)
        self.assertIn("notes.txt", prompt)
        scratch = copy_workspace(t.workspace)
        try:
            names = sorted(os.listdir(scratch))
            self.assertEqual(names, ["notes.txt"])
        finally:
            remove_scratch(scratch)

    def test_changing_grading_fields_changes_nothing(self):
        prompts, outcomes = [], []
        for i, (exp, chk) in enumerate([("389", "exact_int"), ("TOTALLY_DIFFERENT", "other"), (None, None)]):
            d = make_task(self.tmp / ("g%d" % i), workspace=False, task_type="data_analysis",
                          instruction="The file sales.csv has columns region,units. "
                                      "How many units were sold in region 'west'?",
                          expected=exp, check=chk, note="n%d" % i)
            (d / "sales.csv").write_text("region,units\nwest,5\neast,7\n", encoding="utf-8")
            t = load_task(str(d))
            prompts.append(json.dumps(llm_fallback.build_messages(t)).replace(str(d), "<dir>"))
            o = data_solver.solve(t.instruction, t.workspace)
            outcomes.append((o.status, o.answer))
        self.assertEqual(len(set(prompts)), 1)
        self.assertEqual(set(outcomes), {("solved", "5")})

    def test_nested_data_is_kept(self):
        d = make_task(self.tmp / "n", workspace=True)
        (d / "workspace" / "data" / "deep").mkdir(parents=True)
        (d / "workspace" / "data" / "deep" / "x.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        t = load_task(str(d))
        self.assertIn("data/deep/x.csv", [r for r, _p in wsfiles.list_files(t.workspace)])
        scratch = copy_workspace(t.workspace)
        try:
            self.assertTrue(os.path.isfile(os.path.join(scratch, "data", "deep", "x.csv")))
        finally:
            remove_scratch(scratch)

    def test_csv_column_named_expected_is_not_banned(self):
        d = make_task(self.tmp / "c", workspace=True)
        (d / "workspace" / "sales.csv").write_text("region,expected\nwest,5\nwest,6\n", encoding="utf-8")
        o = data_solver.solve("The file sales.csv has columns region,expected. "
                              "What is the total expected in region 'west'?", str(d / "workspace"))
        self.assertEqual((o.status, o.answer), ("solved", "11"), o.reason)

    def test_path_traversal_is_rejected(self):
        d = make_task(self.tmp / "p", workspace=True)
        (d / "secret.csv").write_text("region,units\nwest,999\n", encoding="utf-8")
        ws = str(d / "workspace")
        self.assertIsNone(wsfiles.safe_file(ws, "../secret.csv"))
        self.assertIsNone(wsfiles.safe_file(ws, "../task.json"))
        self.assertIsNone(wsfiles.safe_file(ws, str(d / "secret.csv")))
        self.assertIsNone(wsfiles.safe_file(str(d), "task.json"))
        o = data_solver.solve("The file ../secret.csv has columns region,units. "
                              "How many units were sold in region 'west'?", ws)
        self.assertNotEqual(o.answer, "999")
        o = search_solver.solve("The directory ../ contains a small knowledge base of markdown files. "
                                "In which year was Ann Lee born?", ws)
        self.assertNotEqual(o.status, "solved")

    def test_symlink_out_of_the_workspace_is_ignored(self):
        d = make_task(self.tmp / "s", workspace=True)
        outside = self.tmp / "outside"
        outside.mkdir()
        (outside / "leak.csv").write_text("region,units\nwest,999\n", encoding="utf-8")
        symlink_or_skip(self, outside / "leak.csv", d / "workspace" / "sales.csv")
        symlink_or_skip(self, d / "task.json", d / "workspace" / "meta.json")
        symlink_or_skip(self, outside, d / "workspace" / "dir")
        ws = str(d / "workspace")
        self.assertEqual(wsfiles.list_files(ws), [])
        self.assertIsNone(wsfiles.safe_file(ws, "sales.csv"))
        scratch = copy_workspace(ws)
        try:
            self.assertEqual(os.listdir(scratch), [])
        finally:
            remove_scratch(scratch)
        self.assertNotIn(SENTINEL, json.dumps(llm_fallback.build_messages(load_task(str(d)))))
        o = data_solver.solve("The file sales.csv has columns region,units. "
                              "How many units were sold in region 'west'?", ws)
        self.assertNotEqual(o.answer, "999")


class ChildIsolation(unittest.TestCase):
    """3.1: generated code cannot read the task tree, other tasks or the key."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))
        d = make_task(self.tmp / "t", workspace=True)
        (d / "workspace" / "data.txt").write_text("hello\n", encoding="utf-8")
        self.marker = self.tmp / "outside_marker.txt"
        self.marker.write_text("OWN_AUDIT_MARKER", encoding="utf-8")
        self.task_dir = d
        self.scratch = copy_workspace(str(d / "workspace"))

    def tearDown(self):
        remove_scratch(self.scratch)

    def run_code(self, code: str):
        return run_python(["-c", code], self.scratch, 20)

    def test_own_scratch_is_usable(self):
        r = self.run_code("print(open('data.txt').read().strip()); open('out.txt','w').write('x'); "
                          "import json, math, csv, re; print(math.sqrt(16))")
        self.assertTrue(r.ok, r.output)
        self.assertEqual(r.stdout.split(), ["hello", "4.0"])

    def test_cannot_read_outside_marker_or_task_json(self):
        for target in (self.marker, self.task_dir / "task.json"):
            r = self.run_code("print(open(%r).read())" % str(target))
            self.assertFalse(r.ok, r.output)
            self.assertNotIn("OWN_AUDIT_MARKER", r.output)
            self.assertNotIn(SENTINEL, r.output)

    def test_cannot_list_or_modify_the_task_directory(self):
        r = self.run_code("import os; print(os.listdir(%r))" % str(self.task_dir))
        self.assertFalse(r.ok, r.output)
        r = self.run_code("open(%r, 'w').write('{}')" % str(self.task_dir / "task.json"))
        self.assertFalse(r.ok, r.output)
        self.assertIn(SENTINEL, (self.task_dir / "task.json").read_text(encoding="utf-8"))
        r = self.run_code("import os; os.remove(%r)" % str(self.marker))
        self.assertFalse(r.ok, r.output)
        self.assertTrue(self.marker.exists())

    def test_environment_has_no_key_and_no_network(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "KEY_SENTINEL", "LLM_BASE_URL": "http://x.invalid/v1"}):
            r = self.run_code("import os; print(sorted(k for k in os.environ if 'LLM' in k or 'HARNESS' in k))")
            self.assertEqual(r.stdout.strip(), "[]", r.output)
            r = self.run_code("import socket; s = socket.socket(); s.settimeout(3); s.connect(('127.0.0.1', 9)); "
                              "print('CONNECTED')")
            self.assertNotIn("CONNECTED", r.output)
            self.assertFalse(r.ok)

    def test_output_is_bounded_and_runaway_is_killed(self):
        t0 = time.monotonic()
        r = run_python(["-c", "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)"], self.scratch, 3)
        self.assertTrue(r.timed_out)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertLessEqual(len(r.stdout), 300 * 1024)
        self.assertTrue(r.truncated)


class AnswerWriting(unittest.TestCase):
    """3.2: atomic, verified, and only <task_dir>/answer.txt counts."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))
        self.task = load_task(str(make_task(self.tmp / "t", workspace=True)))

    def test_goes_through_replace_and_leaves_no_temp_files(self):
        target = self.tmp / "t" / "answer.txt"
        with patch("agent_harness.answers.os.replace", wraps=os.replace) as rep:
            self.assertTrue(answers.write_answer(self.task, "42"))
        self.assertGreaterEqual(rep.call_count, 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "42\n")
        self.assertEqual((self.tmp / "t" / "workspace" / "answer.txt").read_text(encoding="utf-8"), "42\n")
        self.assertEqual([p.name for p in (self.tmp / "t").iterdir() if p.name.startswith(".tmp")], [])

    def test_failure_before_replace_keeps_the_old_file(self):
        target = self.tmp / "t" / "answer.txt"
        target.write_text("OLD\n", encoding="utf-8")
        with patch("agent_harness.answers.os.replace", side_effect=PermissionError("denied")):
            self.assertFalse(answers.write_answer(self.task, "NEW"))
        self.assertEqual(target.read_text(encoding="utf-8"), "OLD\n")  # never truncated
        self.assertEqual([p.name for p in (self.tmp / "t").iterdir() if p.name.startswith(".tmp")], [])

    def test_workspace_copy_alone_is_not_success(self):
        root = os.path.abspath(str(self.tmp / "t"))
        real = answers.write_text_atomic

        def only_workspace(path, text):
            return False if os.path.dirname(os.path.abspath(path)) == root else real(path, text)

        with patch("agent_harness.answers.write_text_atomic", side_effect=only_workspace):
            self.assertFalse(answers.write_answer(self.task, "42"))

    def test_old_file_does_not_confirm_a_new_answer(self):
        target = self.tmp / "t" / "answer.txt"
        target.write_text("41\n", encoding="utf-8")
        with patch("agent_harness.answers.os.replace", side_effect=lambda a, b: os.unlink(a)):
            self.assertFalse(answers.write_text_atomic(str(target), "42\n"))  # replace "succeeded" but did nothing

    @unittest.skipUnless(os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() != 0,
                         "needs POSIX permissions and a non-root user")
    def test_unwritable_directory_is_reported_not_faked(self):
        from agent_harness import pipeline
        from agent_harness.trace import Trace
        tdir = self.tmp / "t"
        other = load_task(str(make_task(self.tmp / "u", workspace=True, task_id="t2")))
        os.chmod(tdir, 0o555)
        try:
            st = pipeline.TaskState(self.task)
            pipeline._record(st, coding_solver.Outcome.solved("42"), Trace(None))
            self.assertEqual(st.status, "unsolved")
            self.assertFalse(st.written)
            self.assertIn("could not be written", st.reason)
            st2 = pipeline.TaskState(other)
            pipeline._record(st2, coding_solver.Outcome.solved("7"), Trace(None))
            self.assertEqual(st2.status, "solved")  # other results are unaffected
        finally:
            os.chmod(tdir, 0o755)


def coding_ws(tmp: pathlib.Path, files: dict) -> str:
    tmp.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (tmp / name).write_text(text, encoding="utf-8")
    return str(tmp)


class Task:
    def __init__(self, workspace, instruction="Fix the bugs in mylib.py until all tests pass."):
        self.task_id, self.task_type, self.instruction = "c1", "coding", instruction
        self.workspace = self.task_dir = workspace
        self.reward = 1


BAD_ADD = "def add(a, b):\n    return a - b\n"
GOOD_ADD = "def add(a, b):\n    return a + b\n"
TEST_NO_MAIN = ("import unittest\nimport mylib\n\nclass T(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(mylib.add(2, 3), 5)\n")


class CodingTestExecution(unittest.TestCase):
    """3.3: a coding task is green only if its tests were really executed."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))
        self.settings = Settings(enable_llm=False)

    def run_in_scratch(self, files: dict, tests):
        ws = coding_ws(self.tmp / "w", files)
        scratch = copy_workspace(ws)
        try:
            return coding_solver.run_tests(scratch, tests, 20)
        finally:
            remove_scratch(scratch)

    def test_file_without_unittest_main_is_executed(self):
        r = self.run_in_scratch({"mylib.py": BAD_ADD, "test_mylib.py": TEST_NO_MAIN}, "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertEqual(r.statuses, {"T.test_add": "FAIL"})
        r = self.run_in_scratch({"mylib.py": GOOD_ADD, "test_mylib.py": TEST_NO_MAIN}, "test_mylib.py")
        self.assertTrue(r.passed)
        self.assertEqual((r.ran, r.executed), (1, 1))

    def test_zero_tests_is_not_success(self):
        r = self.run_in_scratch({"mylib.py": BAD_ADD, "test_mylib.py": "import mylib\nprint('nothing here')\n"},
                                "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertIn("no tests", r.load_error)
        ws = coding_ws(self.tmp / "z", {"mylib.py": BAD_ADD, "test_mylib.py": "import mylib\n"})
        outcome, ctx = coding_solver.solve_free(Task(ws), self.settings)
        self.assertEqual(outcome.status, "unknown")
        self.assertIsNone(ctx)  # nothing verifiable: not sent to the LLM either

    def test_fully_skipped_suite_is_not_success(self):
        src = ("import unittest\nimport mylib\n\n@unittest.skip('later')\nclass T(unittest.TestCase):\n"
               "    def test_add(self):\n        self.assertEqual(mylib.add(2, 3), 5)\n")
        r = self.run_in_scratch({"mylib.py": BAD_ADD, "test_mylib.py": src}, "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertIn("skipped", r.load_error)
        outcome, _ctx = coding_solver.solve_free(Task(coding_ws(self.tmp / "s", {"mylib.py": BAD_ADD,
                                                                                 "test_mylib.py": src})),
                                                 self.settings)
        self.assertNotEqual(outcome.status, "solved")

    def test_partially_loaded_suite_is_not_success(self):
        # the second class is not a TestCase: unittest would silently run 1 of 2 tests
        src = TEST_NO_MAIN + "\nclass Extra:\n    def test_more(self):\n        assert mylib.add(1, 1) == 2\n"
        r = self.run_in_scratch({"mylib.py": GOOD_ADD, "test_mylib.py": src}, "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertIn("only 1 of 2", r.load_error)

    def test_failing_import_error_and_timeout_are_kept(self):
        r = self.run_in_scratch({"mylib.py": BAD_ADD, "test_mylib.py": TEST_NO_MAIN + "\nimport nothing_here\n"},
                                "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertIn("nothing_here", r.load_error + r.output)
        r = self.run_in_scratch({"mylib.py": "def add(a, b):\n    return a +\n", "test_mylib.py": TEST_NO_MAIN},
                                "test_mylib.py")
        self.assertFalse(r.passed)
        ws = coding_ws(self.tmp / "slow", {"mylib.py": "def add(a, b):\n    while True:\n        pass\n",
                                           "test_mylib.py": TEST_NO_MAIN})
        scratch = copy_workspace(ws)
        try:
            t0 = time.monotonic()
            r = coding_solver.run_tests(scratch, "test_mylib.py", 2)
            self.assertTrue(r.timed_out)
            self.assertFalse(r.passed)
            self.assertLess(time.monotonic() - t0, 15)
        finally:
            remove_scratch(scratch)

    def test_plain_test_functions_are_executed(self):
        src = "import mylib\n\ndef test_add():\n    assert mylib.add(2, 3) == 5\n"
        r = self.run_in_scratch({"mylib.py": BAD_ADD, "test_mylib.py": src}, "test_mylib.py")
        self.assertFalse(r.passed)
        self.assertEqual(r.statuses, {"test_add": "FAIL"})
        r = self.run_in_scratch({"mylib.py": GOOD_ADD, "test_mylib.py": src}, "test_mylib.py")
        self.assertTrue(r.passed)

    def test_several_test_files_are_all_required(self):
        second = ("import unittest\nimport mylib\n\nclass U(unittest.TestCase):\n"
                  "    def test_mul(self):\n        self.assertEqual(mylib.mul(2, 3), 6)\n")
        lib = BAD_ADD + "\n\ndef mul(a, b):\n    return a + b\n"
        ws = coding_ws(self.tmp / "m", {"mylib.py": lib, "test_a.py": TEST_NO_MAIN, "test_b.py": second})
        tests, lib_file, why = coding_solver.locate_all_tests("Fix mylib.py so that the tests pass.", ws)
        self.assertEqual((tests, lib_file), (["test_a.py", "test_b.py"], "mylib.py"), why)
        outcome, ctx = coding_solver.solve_free(Task(ws, "Fix mylib.py so that the tests pass."), self.settings)
        self.assertEqual(outcome.status, "solved", outcome.reason)
        fixed = (pathlib.Path(ws) / "mylib.py").read_text(encoding="utf-8")
        self.assertIn("a + b", fixed)
        self.assertIn("a * b", fixed)
        self.assertEqual((pathlib.Path(ws) / "test_a.py").read_text(encoding="utf-8"), TEST_NO_MAIN)

    def test_fix_is_verified_on_the_installed_version_and_tests_untouched(self):
        ws = coding_ws(self.tmp / "ok", {"mylib.py": BAD_ADD, "test_mylib.py": TEST_NO_MAIN})
        outcome, ctx = coding_solver.solve_free(Task(ws), self.settings)
        self.assertEqual((outcome.status, outcome.tier), ("solved", "mutation"), outcome.reason)
        self.assertEqual((pathlib.Path(ws) / "test_mylib.py").read_text(encoding="utf-8"), TEST_NO_MAIN)
        self.assertEqual(sorted(os.listdir(ws)), ["mylib.py", "test_mylib.py"])  # no temp files, no caches
        ctx.cleanup()
        self.assertEqual(ctx.scratch, [])

    def test_failed_final_verification_rolls_back(self):
        ws = coding_ws(self.tmp / "rb", {"mylib.py": BAD_ADD, "test_mylib.py": TEST_NO_MAIN})
        _o, ctx = coding_solver.prepare(Task(ws), self.settings)
        self.assertFalse(coding_solver.commit_fix(Task(ws), ctx, "def add(a, b):\n    return a * b\n", self.settings))
        self.assertEqual((pathlib.Path(ws) / "mylib.py").read_text(encoding="utf-8"), BAD_ADD)

    def test_stateful_function_is_not_a_contradiction(self):
        tests = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n    def test_counter(self):\n"
                 "        self.assertEqual(m.next_id(), 1)\n        self.assertEqual(m.next_id(), 2)\n")
        lib = "_n = 0\n\ndef next_id():\n    global _n\n    _n += 2\n    return _n\n"
        info = coding_solver.analyze_tests(tests, "mylib", lib)
        self.assertEqual(info.contradictions, [])
        ws = coding_ws(self.tmp / "st", {"mylib.py": lib, "test_mylib.py": tests})
        outcome, _ctx = coding_solver.solve_free(Task(ws), self.settings)
        self.assertNotEqual(outcome.status, "infeasible", outcome.reason)
        self.assertEqual(outcome.status, "solved", outcome.reason)  # `+= 2` -> `+= 1` is a single edit


class ParserRegressions(unittest.TestCase):
    """3.6: the three wrong verdicts of the audit and their neighbours."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))

    def test_python_word_does_not_drop_an_operation(self):
        q = ("A warehouse starts with 100 boxes. Using Python, it ships 5 boxes per day for 3 days. "
             "How many boxes does it have at the end?")
        o = math_solver.solve(q)
        self.assertNotEqual(o.answer, "100")
        self.assertIn(o.status, ("unknown", "solved"))
        if o.status == "solved":
            self.assertEqual(o.answer, "85")
        from agent_harness.solvers.base import problem_text
        self.assertIn("ships 5 boxes per day for 3 days", problem_text(q))  # kept for the LLM prompt too
        boiler = ("A warehouse starts with 10 boxes. Then it receives 5 more boxes. How many boxes does it have "
                  "at the end? Compute it with the bash tool (python3 -c). Work only inside the current directory.")
        self.assertNotIn("bash", problem_text(boiler))

    def test_bad_value_outside_the_selection_is_irrelevant(self):
        (self.tmp / "sales.csv").write_text("region,units\nwest,5\neast,unknown\n", encoding="utf-8")
        head = "The file sales.csv has columns region,units. "
        o = data_solver.solve(head + "How many units were sold in region 'west'?", str(self.tmp))
        self.assertEqual((o.status, o.answer), ("solved", "5"), o.reason)
        o = data_solver.solve(head + "How many units were sold in region 'east'?", str(self.tmp))
        self.assertNotEqual(o.status, "solved")  # the needed value itself is missing

    def test_bad_value_in_a_filter_column_is_not_ignored(self):
        (self.tmp / "s.csv").write_text("region,month,units\nwest,3,5\nwest,?,7\neast,x,1\n", encoding="utf-8")
        o = data_solver.solve("The file s.csv has columns region,month,units. "
                              "How many units were sold in region 'west' in month 3?", str(self.tmp))
        self.assertEqual(o.status, "unknown", "%s %s" % (o.answer, o.reason))  # membership of a row is undecidable
        (self.tmp / "s.csv").write_text("region,month,units\nwest,3,5\nwest,4,7\neast,x,1\n", encoding="utf-8")
        o = data_solver.solve("The file s.csv has columns region,month,units. "
                              "How many units were sold in region 'west' in month 3?", str(self.tmp))
        self.assertEqual((o.status, o.answer), ("solved", "5"), o.reason)  # the bad cell is in an excluded row

    def kb(self, cards: dict) -> str:
        kb = self.tmp / "kb"
        kb.mkdir(exist_ok=True)
        for name, body in cards.items():
            (kb / ("person_%s.md" % name)).write_text(body, encoding="utf-8")
        return str(self.tmp)

    def ask(self, ws: str, q: str):
        return search_solver.solve("The directory kb/ contains a small knowledge base of markdown files. " + q, ws)

    def test_mentor_direct_and_inverse(self):
        ws = self.kb({"ann": "# Ann Lee\nMentor: Bob Roe\n", "bob": "# Bob Roe\nMentor: Cy Poe\n",
                      "cy": "# Cy Poe\nMentor: Ann Lee\n"})  # a cycle: finite questions stay answerable
        for q, exp in {"Who does Ann Lee mentor?": "Cy Poe", "Who is Ann Lee the mentor of?": "Cy Poe",
                       "Who is the mentor of Ann Lee?": "Bob Roe", "Who is Ann Lee's mentor?": "Bob Roe",
                       "Who mentors Ann Lee?": "Bob Roe",
                       "Who is the mentor of Ann Lee's mentor?": "Cy Poe"}.items():
            o = self.ask(ws, q)
            self.assertEqual((o.status, o.answer), ("solved", exp), q + " -> " + o.reason)

    def test_inverse_with_several_or_no_matches_is_not_guessed(self):
        ws = self.kb({"ann": "# Ann Lee\nMentor: Bob Roe\n", "bob": "# Bob Roe\nBorn: 1970\n",
                      "cy": "# Cy Poe\nMentor: Bob Roe\n"})
        self.assertEqual(self.ask(ws, "Who does Bob Roe mentor?").status, "unknown")   # two candidates
        self.assertEqual(self.ask(ws, "Who does Ann Lee mentor?").status, "unknown")   # nobody: not proven here


class FaultServer:
    """Local OpenAI-compatible server that plays a script of responses."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.requests += 1
                step = outer.script.pop(0) if outer.script else {"status": 200}
                if step.get("sleep"):
                    time.sleep(step["sleep"])
                status = step.get("status", 200)
                if status == 200:
                    body = {"choices": [{"message": {"role": "assistant", "content": step.get("text", "ANSWER: 42")},
                                         "finish_reason": "stop"}]}
                    if "usage" in step:
                        if step["usage"] is not None:
                            body["usage"] = step["usage"]
                    else:
                        body["usage"] = {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
                else:
                    body = {"error": {"message": step.get("message", "synthetic error %d" % status)}}
                data = json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    for k, v in (step.get("headers") or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = "http://127.0.0.1:%d/v1" % self.server.server_port

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FastClient(LLMClient):
    BACKOFF_BASE_S = 0.05
    BACKOFF_CAP_S = 0.2
    COOLDOWN_CAP_S = 0.3


class LlmErrorPolicy(unittest.TestCase):
    """3.4 / 3.5 through the real HTTP transport against a local fault server."""

    def client(self, script, **kw):
        self.srv = FaultServer(script)
        self.addCleanup(self.srv.close)
        self.ledger = Ledger(kw.pop("budget", 10000))
        s = Settings(base_url=self.srv.url, api_key="local-test", llm_timeout_s=kw.pop("timeout", 10), **kw)
        c = FastClient(s, self.ledger)
        c._oj_ok = False  # plain HTTP transport; openJiuwen is exercised in the Docker mock run
        return c

    def call(self, c, task="t", max_tokens=50):
        return asyncio.run(c.chat(task, [{"role": "user", "content": "hello"}], model=c.s.model_fast,
                                  max_tokens=max_tokens))

    def test_429_then_success(self):
        c = self.client([{"status": 429}, {"status": 200}])
        self.assertEqual(self.call(c).text, "ANSWER: 42")
        self.assertEqual(self.srv.requests, 2)
        self.assertFalse(c.dead)

    def test_500_then_success_and_retry_after_is_honoured(self):
        c = self.client([{"status": 500, "headers": {"Retry-After": "1"}}, {"status": 200}])
        t0 = time.monotonic()
        self.assertEqual(self.call(c).text, "ANSWER: 42")
        self.assertGreaterEqual(time.monotonic() - t0, 0.9)
        self.assertEqual(self.srv.requests, 2)

    def test_persistent_errors_are_bounded_and_the_client_recovers(self):
        c = self.client([{"status": 503}] * 12 + [{"status": 200}])
        for _ in range(3):  # three logical calls fail completely ...
            with self.assertRaises(LLMUnavailable):
                self.call(c)
        self.assertEqual(self.srv.requests, 3 * c.MAX_ATTEMPTS)
        self.assertFalse(c.dead)            # ... and the client is still not switched off
        self.assertEqual(self.ledger.spent, 0.0)
        self.assertEqual(self.ledger.reserved, 0.0)
        self.assertEqual(self.call(c).text, "ANSWER: 42")  # it recovers after the cool-down

    def test_authorisation_failure_stops_at_once(self):
        c = self.client([{"status": 401}, {"status": 200}])
        with self.assertRaises(LLMUnavailable):
            self.call(c)
        self.assertEqual(self.srv.requests, 1)  # no second authorisation attempt
        self.assertTrue(c.dead)
        with self.assertRaises(LLMUnavailable):
            self.call(c)
        self.assertEqual(self.srv.requests, 1)

    def test_bad_request_is_not_repeated_and_does_not_stop_the_client(self):
        c = self.client([{"status": 404}, {"status": 200}])
        with self.assertRaises(LLMUnavailable):
            self.call(c)
        self.assertEqual(self.srv.requests, 1)
        self.assertFalse(c.dead)

    def test_plain_429_is_not_budget_exhaustion_but_explicit_message_is(self):
        self.assertEqual(classify(Exception("Error code: 429 - Too Many Requests")).kind, "rate")
        self.assertEqual(classify(Exception("Error code: 429 - token budget exhausted")).kind, "budget")
        c = self.client([{"status": 429, "message": "token budget exhausted"}] * 4)
        FastClient._hold_orig = FastClient._hold
        with patch.object(FastClient, "_hold", lambda self, s: FastClient._hold_orig(self, min(s, 0.05))):
            with self.assertRaises(LLMUnavailable):
                self.call(c)
        self.assertTrue(c.dead)
        self.assertEqual(self.srv.requests, 2)  # confirmed by a second signal, then no more requests

    def test_timeout_after_send_books_uncertain_spend_and_is_repeated_once(self):
        c = self.client([{"sleep": 3}, {"sleep": 3}, {"status": 200}], timeout=1)
        c.s.llm_timeout_s = 1
        with patch.object(LLMClient, "_send", self._short_send(c)):
            with self.assertRaises(LLMUnavailable):
                self.call(c)
        self.assertEqual(self.srv.requests, 2)
        notes = [r.note for r in self.ledger.calls]
        self.assertEqual(len(notes), 2)
        self.assertTrue(all("uncertain" in n for n in notes))
        self.assertGreater(self.ledger.spent, 0)
        self.assertEqual(self.ledger.reserved, 0.0)

    @staticmethod
    def _short_send(c):
        async def send(self, transport, messages, model, max_tokens, timeout):
            return await asyncio.wait_for(
                asyncio.to_thread(self._call_http_sync, messages, model, max_tokens, 1.0), 1.5)
        return send

    def test_missing_usage_is_not_free(self):
        c = self.client([{"status": 200, "usage": None}])
        r = self.call(c)
        self.assertGreater(r.cost, 0)
        self.assertGreater(self.ledger.spent, 0)
        self.assertIn("estimated", r.usage_note)

    def test_reasoning_semantics_come_from_the_usage_layout(self):
        nested = {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200,
                  "completion_tokens_details": {"reasoning_tokens": 20}}
        separate = {"prompt_tokens": 100, "completion_tokens": 100, "reasoning_tokens": 20, "total_tokens": 220}
        unknown = {"prompt_tokens": 100, "completion_tokens": 100, "reasoning_tokens": 20}
        c = self.client([{"status": 200, "usage": nested}, {"status": 200, "usage": separate},
                         {"status": 200, "usage": unknown}])
        self.assertAlmostEqual(self.call(c).cost, 110.0)   # a breakdown of completion_tokens: not added twice
        self.assertAlmostEqual(self.call(c).cost, 130.0)   # reported separately: added (audit case 110 vs 130)
        r = self.call(c)
        self.assertAlmostEqual(r.cost, 130.0)              # unknown layout: conservative
        self.assertIn("unknown", r.usage_note)
        self.assertAlmostEqual(weighted_cost(100, 100, 20), 110.0)

    def test_budget_caps_and_reservations_under_concurrency(self):
        c = self.client([{"status": 200, "sleep": 0.2}] * 10, budget=200)
        c.s.task_token_cap = 100000

        async def many():
            return await asyncio.gather(*[
                c.chat("t%d" % i, [{"role": "user", "content": "hello"}], model=c.s.model_fast, max_tokens=50)
                for i in range(8)], return_exceptions=True)

        results = asyncio.run(many())
        ok = [r for r in results if isinstance(r, Reply)]
        refused = [r for r in results if isinstance(r, BudgetExceeded)]
        self.assertEqual(len(ok) + len(refused), 8)
        self.assertEqual(len(ok), 3)             # 3 x ~51 fit into the soft limit 170, the 4th does not
        self.assertEqual(self.srv.requests, 3)   # refused calls never reach the proxy
        self.assertEqual(self.ledger.reserved, 0.0)

    def test_per_task_cap_shrinks_or_refuses_the_next_attempt(self):
        c = self.client([{"status": 200}] * 3)
        c.s.task_token_cap = 100
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(c.affordable_max_tokens("t", msgs, 60, minimum=10), 60)
        self.call(c, max_tokens=60)                                  # costs 5 + 20/10 = 7
        self.assertEqual(c.affordable_max_tokens("t", msgs, 2000, minimum=50), 92)  # 100 - 7 - 1 (prompt)
        self.assertEqual(c.affordable_max_tokens("t", msgs, 2000, minimum=500), 0)
        with self.assertRaises(BudgetExceeded):
            self.call(c, max_tokens=2000)

    def test_cancellation_releases_the_reservation_and_books_the_flight(self):
        c = self.client([{"sleep": 2, "status": 200}])

        async def run():
            job = asyncio.ensure_future(c.chat("t", [{"role": "user", "content": "hello"}], model=c.s.model_fast,
                                               max_tokens=50))
            await asyncio.sleep(0.5)
            job.cancel()
            return await asyncio.gather(job, return_exceptions=True)

        res = asyncio.run(run())
        self.assertIsInstance(res[0], asyncio.CancelledError)
        self.assertEqual(self.ledger.reserved, 0.0)
        self.assertTrue(any("uncertain" in r.note for r in self.ledger.calls))

    def test_tls_verification_is_never_switched_off(self):
        import ssl
        c = self.client([])
        ctx = c._ssl()
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(ctx.check_hostname)
        self.assertEqual(classify(ssl.SSLCertVerificationError("certificate verify failed")).kind, "network")
        src = (ROOT / "src" / "agent_harness" / "llm.py").read_text(encoding="utf-8")
        self.assertNotIn("CERT_NONE", src.split('"""', 2)[2])
        self.assertNotIn("verify_ssl=False", src)

    def test_default_models_are_non_reasoning(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ("HARNESS_MODEL_FAST", "HARNESS_MODEL_STRONG"):
                os.environ.pop(k, None)
            s = Settings()
        self.assertTrue(s.model_fast.endswith("-noreason"))
        self.assertTrue(s.model_strong.endswith("-noreason"))


class TimeAndScratch(unittest.TestCase):
    """3.7: deadlines reach the mutation search; scratch copies are released."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="reg_"))

    def test_mutation_search_stops_at_its_deadline(self):
        # every test run hangs: the search must give up at the time limit, not after the whole queue
        lib = "\n\n".join("def f%d(a, b):\n    x = a + b * 2 - 1\n    y = x // 3 + a %% 5\n    return x - y" % i
                          for i in range(6))
        lib += "\n\n\ndef spin(a):\n    while a >= 0:\n        a += 1\n    return a\n"
        tests = ("import unittest\nimport mylib\n\nclass T(unittest.TestCase):\n    def test_x(self):\n"
                 "        self.assertEqual(mylib.spin(1) + mylib.f0(1, 2) + mylib.f1(1, 2) + mylib.f2(1, 2)"
                 " + mylib.f3(1, 2) + mylib.f4(1, 2) + mylib.f5(1, 2), -1)\n")
        ws = coding_ws(self.tmp / "w", {"mylib.py": lib, "test_mylib.py": tests})
        settings = Settings(enable_llm=False, test_timeout_s=3, workers=4)
        before = set(os.listdir(tempfile.gettempdir()))
        t0 = time.monotonic()
        outcome, ctx = coding_solver.solve_free(Task(ws), settings, time_limit=6.0)
        elapsed = time.monotonic() - t0
        self.assertNotEqual(outcome.status, "solved")
        self.assertLess(elapsed, 6.0 + 3 * 3 + 10, "mutation search ignored its deadline")
        if ctx is not None:
            ctx.cleanup()
        left = [n for n in set(os.listdir(tempfile.gettempdir())) - before if n.startswith("ws_")]
        self.assertEqual(left, [])  # scratch copies released

    def test_expired_run_deadline_skips_the_search(self):
        ws = coding_ws(self.tmp / "d", {"mylib.py": BAD_ADD, "test_mylib.py": TEST_NO_MAIN})
        t0 = time.monotonic()
        outcome, ctx = coding_solver.solve_free(Task(ws), Settings(enable_llm=False), 120.0, time.monotonic() - 1)
        self.assertEqual(outcome.status, "unknown")
        self.assertLess(time.monotonic() - t0, 15)
        self.assertEqual((pathlib.Path(ws) / "mylib.py").read_text(encoding="utf-8"), BAD_ADD)


if __name__ == "__main__":
    unittest.main()
