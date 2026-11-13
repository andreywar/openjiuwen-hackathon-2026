"""Mutation operators for classical generate-and-validate program repair.

Given the AST of one function, ``function_mutants`` yields single-edit
variants of it (operator swaps, off-by-one changes, swapped builtins, ...).
The caller validates every variant against the task's own unit tests, so a
variant is only ever accepted when the tests prove it.

No machine learning is involved - this is the textbook mutation-based repair
technique, and it costs zero LLM tokens.

Each mutant carries a rank: 0 = inversion of a typical injected bug
(``+`` <-> ``-``, ``<`` <-> ``>``, ``sorted`` <-> ``reversed`` ...),
1 = other small operator / constant edits, 2 = structural edits,
3 = long-shot edits. Lower ranks are tried first.
"""
from __future__ import annotations

import ast
import copy
from typing import Iterator, List, Tuple

Mutant = Tuple[int, str, ast.AST]  # (rank, description, mutated function node)

_CMP_OPS = [ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq]
_CMP_INVERSE = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE,
                ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
                ast.In: ast.NotIn, ast.NotIn: ast.In}
_BIN_OPS = [ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow]
_BIN_INVERSE = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
                ast.FloorDiv: ast.Mult}
_NAME_FAMILIES = [
    ["sorted", "reversed"], ["min", "max"], ["any", "all"], ["sum", "len"], ["sum", "max", "min"],
    ["int", "float", "round"], ["upper", "lower"], ["title", "capitalize", "upper", "lower", "swapcase"],
    ["startswith", "endswith"], ["strip", "lstrip", "rstrip"], ["find", "rfind"], ["split", "rsplit"],
    ["keys", "values", "items"], ["append", "extend"], ["ceil", "floor"], ["isupper", "islower"],
    ["isdigit", "isalpha", "isalnum"], ["add", "discard", "remove"], ["union", "intersection", "difference"],
    ["list", "sorted", "reversed", "set", "tuple"], ["max", "len"], ["min", "len"], ["abs", "int"],
    ["pop", "remove"], ["index", "count"], ["sort", "reverse"],
]
_INVERSE_NAME_PAIRS = {frozenset(p) for p in (
    ("sorted", "reversed"), ("min", "max"), ("any", "all"), ("upper", "lower"), ("startswith", "endswith"),
    ("lstrip", "rstrip"), ("ceil", "floor"), ("sort", "reverse"), ("isupper", "islower"))}


def _name_alternatives(name: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    seen = {name}
    for fam in _NAME_FAMILIES:
        if name in fam:
            for other in fam:
                if other not in seen:
                    seen.add(other)
                    rank = 0 if frozenset((name, other)) in _INVERSE_NAME_PAIRS else 1
                    out.append((rank, other))
    return out


def _nodes(func: ast.AST) -> List[ast.AST]:
    return list(ast.walk(func))


def _clone(func: ast.AST, index: int) -> Tuple[ast.AST, ast.AST]:
    """Deep-copy the function and return (copy, node at the same walk index)."""
    new = copy.deepcopy(func)
    return new, _nodes(new)[index]


def _is_int_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool)


def _plus(node: ast.AST, delta: int) -> ast.AST:
    if _is_int_const(node):
        return ast.Constant(value=node.value + delta)
    op = ast.Add() if delta > 0 else ast.Sub()
    return ast.BinOp(left=node, op=op, right=ast.Constant(value=abs(delta)))


def _replace_child(parent: ast.AST, field: str, idx, new: ast.AST) -> None:
    if idx is None:
        setattr(parent, field, new)
    else:
        getattr(parent, field)[idx] = new


def _expr_slots(node: ast.AST) -> Iterator[Tuple[str, object, ast.AST, str]]:
    """Child expression positions where an off-by-one edit is plausible."""
    if isinstance(node, ast.Slice):
        for f in ("lower", "upper"):
            child = getattr(node, f)
            if child is not None:
                yield f, None, child, "slice"
    elif isinstance(node, ast.Subscript):
        if not isinstance(node.slice, (ast.Slice, ast.Tuple)):
            yield "slice", None, node.slice, "index"
    elif isinstance(node, ast.Call):
        fname = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else "")
        if fname in ("range", "islice", "randint", "enumerate", "round", "pow"):
            for i, a in enumerate(node.args):
                yield "args", i, a, "range"
    elif isinstance(node, ast.Compare):
        yield "left", None, node.left, "cmp"
        for i, c in enumerate(node.comparators):
            yield "comparators", i, c, "cmp"
    elif isinstance(node, ast.BinOp):
        yield "left", None, node.left, "arith"
        yield "right", None, node.right, "arith"
    elif isinstance(node, ast.Return):
        if node.value is not None:
            yield "value", None, node.value, "return"
    elif isinstance(node, (ast.Assign, ast.AugAssign)):
        yield "value", None, node.value, "assign"


def function_mutants(func: ast.AST, max_rank: int = 3) -> Iterator[Mutant]:
    """Yield single-edit mutants of ``func`` (an ``ast.FunctionDef``)."""
    nodes = _nodes(func)
    local_names = sorted({n.id for n in nodes if isinstance(n, ast.Name)} |
                         {a.arg for n in nodes if isinstance(n, ast.arguments)
                          for a in (n.args + n.kwonlyargs + n.posonlyargs)})
    builtin_like = {"sorted", "reversed", "min", "max", "any", "all", "sum", "len", "int", "float", "round",
                    "range", "list", "set", "tuple", "abs", "str", "enumerate", "zip", "map", "filter", "print",
                    "dict", "isinstance", "bool", "ord", "chr", "divmod", "pow", "True", "False", "None"}
    value_names = [n for n in local_names if n not in builtin_like]

    for i, node in enumerate(nodes):
        # ---- comparison operators ------------------------------------
        if isinstance(node, ast.Compare):
            for j, op in enumerate(node.ops):
                inv = _CMP_INVERSE.get(type(op))
                targets = []
                if inv is not None:
                    targets.append((0, inv))
                if type(op) in _CMP_OPS:
                    targets += [(1, t) for t in _CMP_OPS if t is not type(op) and t is not inv]
                for rank, t in targets:
                    new, n = _clone(func, i)
                    n.ops[j] = t()
                    yield rank, "compare %s->%s" % (type(op).__name__, t.__name__), new

        # ---- arithmetic operators ------------------------------------
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BIN_OPS:
            inv = _BIN_INVERSE.get(type(node.op))
            for t in _BIN_OPS:
                if t is type(node.op):
                    continue
                new, n = _clone(func, i)
                n.op = t()
                yield (0 if t is inv else 1), "arith %s->%s" % (type(node.op).__name__, t.__name__), new
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                new, n = _clone(func, i)
                n.left, n.right = n.right, n.left
                yield 2, "swap operands", new

        # ---- boolean operators ---------------------------------------
        if isinstance(node, ast.BoolOp):
            new, n = _clone(func, i)
            n.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
            yield 0, "and<->or", new
        if isinstance(node, ast.UnaryOp):
            new, n = _clone(func, i)
            # replace the unary node by its operand everywhere it is referenced
            for parent in _nodes(new):
                for field, value in ast.iter_fields(parent):
                    if value is n:
                        setattr(parent, field, n.operand)
                    elif isinstance(value, list):
                        for k, item in enumerate(value):
                            if item is n:
                                value[k] = n.operand
            yield (0 if isinstance(node.op, ast.Not) else 1), "drop unary %s" % type(node.op).__name__, new

        # ---- conditions: negate --------------------------------------
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            if not (isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)):
                new, n = _clone(func, i)
                n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
                yield 1, "negate condition", new
        if isinstance(node, ast.comprehension):
            for j, cond in enumerate(node.ifs):
                if not (isinstance(cond, ast.UnaryOp) and isinstance(cond.op, ast.Not)):
                    new, n = _clone(func, i)
                    n.ifs[j] = ast.UnaryOp(op=ast.Not(), operand=n.ifs[j])
                    yield 1, "negate filter", new

        # ---- constants -----------------------------------------------
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                new, n = _clone(func, i)
                n.value = not v
                yield 0, "bool flip", new
            elif isinstance(v, int):
                alts = [(0 if v in (0, 1) else 1, 1 - v)] if v in (0, 1) else []
                alts += [(1, v + 1), (1, v - 1)]
                if v not in (0,):
                    alts.append((2, -v))
                if v not in (0, 1, 2):
                    alts += [(2, 0), (2, 1)]
                if v in (0, 1):
                    alts.append((2, 2))
                seen = {v}
                for rank, a in alts:
                    if a in seen:
                        continue
                    seen.add(a)
                    new, n = _clone(func, i)
                    n.value = a
                    yield rank, "const %r->%r" % (v, a), new
            elif isinstance(v, float):
                for a in (v + 1, v - 1, -v):
                    new, n = _clone(func, i)
                    n.value = a
                    yield 2, "const %r->%r" % (v, a), new
            elif isinstance(v, str) and max_rank >= 3:
                for a in ({" ": ["", ",", ", "], "": [" "], ",": [" ", ", "], ", ": [",", " "]}.get(v, [])):
                    new, n = _clone(func, i)
                    n.value = a
                    yield 3, "str const %r->%r" % (v, a), new

        # ---- swapped builtins / methods ------------------------------
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                for rank, other in _name_alternatives(node.func.id):
                    new, n = _clone(func, i)
                    n.func.id = other
                    yield rank, "call %s->%s" % (node.func.id, other), new
            elif isinstance(node.func, ast.Attribute):
                for rank, other in _name_alternatives(node.func.attr):
                    new, n = _clone(func, i)
                    n.func.attr = other
                    yield rank, "method %s->%s" % (node.func.attr, other), new
            fname = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            # sorted(x) <-> sorted(x, reverse=True)
            if fname in ("sorted", "sort"):
                new, n = _clone(func, i)
                kw = [k for k in n.keywords if k.arg == "reverse"]
                if kw:
                    n.keywords = [k for k in n.keywords if k.arg != "reverse"]
                else:
                    n.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(value=True)))
                yield 1, "toggle reverse=", new
            # f(x) -> x
            if isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords and \
                    node.func.id in ("sorted", "reversed", "abs", "list", "set", "int", "round", "str", "len"):
                new, n = _clone(func, i)
                for parent in _nodes(new):
                    for field, value in ast.iter_fields(parent):
                        if value is n:
                            setattr(parent, field, n.args[0])
                        elif isinstance(value, list):
                            for k, item in enumerate(value):
                                if item is n:
                                    value[k] = n.args[0]
                yield 2, "unwrap %s()" % node.func.id, new
            # swap the two arguments
            if len(node.args) == 2 and max_rank >= 2:
                new, n = _clone(func, i)
                n.args[0], n.args[1] = n.args[1], n.args[0]
                yield 2, "swap call args", new
            # wrap an argument: join(x) -> join(reversed(x)) / join(sorted(x))
            if max_rank >= 3 and len(node.args) == 1 and fname in ("join", "list", "tuple", "sum", "enumerate"):
                for w in ("reversed", "sorted"):
                    new, n = _clone(func, i)
                    n.args[0] = ast.Call(func=ast.Name(id=w, ctx=ast.Load()), args=[n.args[0]], keywords=[])
                    yield 3, "wrap arg in %s()" % w, new

        # ---- slices ----------------------------------------------------
        if isinstance(node, ast.Slice):
            if (node.lower is None) != (node.upper is None) and node.step is None:
                new, n = _clone(func, i)
                n.lower, n.upper = n.upper, n.lower
                yield 1, "slice [:n]<->[n:]", new
            if node.step is not None:
                new, n = _clone(func, i)
                n.step = None
                yield 2, "drop slice step", new
            elif node.lower is None and node.upper is None:
                new, n = _clone(func, i)
                n.step = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1))
                yield 2, "add slice step -1", new
            if node.lower is not None and node.upper is not None:
                new, n = _clone(func, i)
                n.lower, n.upper = n.upper, n.lower
                yield 2, "swap slice bounds", new
            # sign of a slice bound: xs[n:] <-> xs[-n:], xs[:n] <-> xs[:-n], xs[:n] <-> xs[-n:]
            if node.step is None:
                for f in ("lower", "upper"):
                    bound = getattr(node, f)
                    if bound is None:
                        continue
                    for also_swap in (False, True):
                        if also_swap and (node.lower is not None and node.upper is not None):
                            continue
                        new, n = _clone(func, i)
                        b = getattr(n, f)
                        if isinstance(b, ast.UnaryOp) and isinstance(b.op, ast.USub):
                            b = b.operand
                        else:
                            b = ast.UnaryOp(op=ast.USub(), operand=b)
                        setattr(n, f, b)
                        if also_swap:
                            n.lower, n.upper = n.upper, n.lower
                        yield 2, "slice bound sign%s" % (" + side" if also_swap else ""), new

        # ---- off-by-one on expressions ---------------------------------
        for field, idx, child, where in _expr_slots(node):
            if isinstance(child, ast.Constant) and not _is_int_const(child):
                continue
            if _is_int_const(child):
                continue  # plain constants are handled above
            if where in ("return", "assign") and max_rank < 3:
                continue
            rank = 1 if where in ("slice", "range", "index") else (2 if where in ("cmp", "arith") else 3)
            for delta in (1, -1):
                new, n = _clone(func, i)
                target = getattr(n, field) if idx is None else getattr(n, field)[idx]
                _replace_child(n, field, idx, _plus(target, delta))
                yield rank, "%s %+d" % (where, delta), new

        # ---- range(n) <-> range(1, n + 1) ------------------------------
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if len(node.args) == 1:
                new, n = _clone(func, i)
                n.args = [ast.Constant(value=1), _plus(n.args[0], 1)]
                yield 2, "range(n)->range(1,n+1)", new
                new, n = _clone(func, i)
                n.args = [ast.Constant(value=1), n.args[0]]
                yield 2, "range(n)->range(1,n)", new
            elif len(node.args) == 2:
                new, n = _clone(func, i)
                n.args = [n.args[1]]
                yield 2, "range(a,n)->range(n)", new

        # ---- control flow ----------------------------------------------
        if isinstance(node, (ast.For, ast.While)) and node.body and isinstance(node.body[-1], ast.Return):
            # "return" accidentally indented into the loop body
            new, n = _clone(func, i)
            ret = n.body.pop()
            if not n.body:
                n.body.append(ast.Pass())
            for parent in _nodes(new):
                for fname_ in ("body", "orelse", "finalbody"):
                    seq = getattr(parent, fname_, None)
                    if isinstance(seq, list) and n in seq:
                        seq.insert(seq.index(n) + 1, ret)
                        break
            yield 2, "move return out of loop", new
        if isinstance(node, (ast.Break, ast.Continue)):
            new, n = _clone(func, i)
            repl = ast.Continue() if isinstance(node, ast.Break) else ast.Break()
            for parent in _nodes(new):
                for fname_ in ("body", "orelse"):
                    seq = getattr(parent, fname_, None)
                    if isinstance(seq, list):
                        for k, item in enumerate(seq):
                            if item is n:
                                seq[k] = repl
            yield 2, "break<->continue", new
        if isinstance(node, ast.If) and node.orelse and max_rank >= 3:
            pass  # covered by "negate condition"

        # ---- variable mix-ups --------------------------------------------
        if max_rank >= 3 and isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                and node.id in value_names:
            for other in value_names:
                if other != node.id:
                    new, n = _clone(func, i)
                    n.id = other
                    yield 3, "name %s->%s" % (node.id, other), new

    # ---- the same inverse swap applied to the whole function --------------
    # e.g. max(lo, min(x, hi)) written as min(lo, max(x, hi))
    if max_rank >= 2:
        present = {n.func.id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)} | \
                  {n.func.attr for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for pair in sorted(tuple(sorted(p)) for p in _INVERSE_NAME_PAIRS):
            a, b = pair
            if a in present and b in present:
                new = copy.deepcopy(func)
                for n in _nodes(new):
                    if isinstance(n, ast.Call):
                        if isinstance(n.func, ast.Name) and n.func.id in pair:
                            n.func.id = b if n.func.id == a else a
                        elif isinstance(n.func, ast.Attribute) and n.func.attr in pair:
                            n.func.attr = b if n.func.attr == a else a
                yield 2, "swap all %s<->%s" % (a, b), new
        if sum(1 for n in nodes if isinstance(n, ast.Compare)) >= 2:
            new = copy.deepcopy(func)
            changed = False
            for n in _nodes(new):
                if isinstance(n, ast.Compare):
                    for j, op in enumerate(n.ops):
                        inv = {ast.Lt: ast.Gt, ast.Gt: ast.Lt, ast.LtE: ast.GtE, ast.GtE: ast.LtE}.get(type(op))
                        if inv:
                            n.ops[j] = inv()
                            changed = True
            if changed:
                yield 2, "invert all comparisons", new

    # ---- statement deletion (long shot) ----------------------------------
    if max_rank >= 3:
        for i, node in enumerate(nodes):
            for fname_ in ("body", "orelse"):
                seq = getattr(node, fname_, None)
                if not isinstance(seq, list) or len(seq) < 2:
                    continue
                for k, stmt in enumerate(seq):
                    if isinstance(stmt, (ast.Return, ast.FunctionDef, ast.ClassDef)):
                        continue
                    new, n = _clone(func, i)
                    getattr(n, fname_)[k] = ast.Pass()
                    yield 3, "delete statement", new


def render(func: ast.AST, indent: int = 0) -> str:
    """Source text of a (mutated) function node, re-indented."""
    ast.fix_missing_locations(func)
    text = ast.unparse(func)
    if indent:
        pad = " " * indent
        text = "\n".join((pad + line) if line.strip() else line for line in text.splitlines())
    return text + "\n"
