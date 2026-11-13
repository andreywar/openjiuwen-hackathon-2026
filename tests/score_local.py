#!/usr/bin/env python3
"""Development-only scorer (NOT part of the Docker image).

Copies a task directory to a temporary place, runs the harness on the copy
exactly as the organisers do (``python run_agent.py --tasks-dir <copy>``) and
then compares the produced answers with the ``expected`` / ``check`` fields of
``task.json``. The harness itself never reads those fields; only this script
does.

    python tests/score_local.py task-dir
    python tests/score_local.py tests/synthetic_tasks --keep
    python tests/score_local.py <processed dir> --no-run --trace trace.jsonl [--original task-dir]

What is checked, and reported separately:

* correctness - only the required file ``<task_dir>/answer.txt`` counts (a copy
  in ``workspace/`` does not). For coding tasks the tests are really executed
  with the same runner and the same requirements as in the harness (tests must
  run, not all skipped, none failing), on a scratch copy; with ``--original``
  (or when the scorer made the copy itself) the test files must be unchanged;
* routing - solved for free / sent to the LLM tier / rejected as infeasible.
  ``expect: "llm"`` in a synthetic task means "expected to be handed to the LLM
  tier"; such a task is listed as NOT solved, it is merely not a surprise;
* spend - taken from the trace. Without a trace (``--no-run`` and no
  ``--trace``) the spend is unknown and is shown as ``n/a``, never as zero.

A task may carry ``"feasible": false`` (synthetic infeasible tasks): for those
the right behaviour is to spend no tokens; an answer does not matter.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agent_harness.sandbox import copy_workspace, remove_scratch  # noqa: E402
from agent_harness.solvers.coding_solver import run_tests  # noqa: E402


def read_answer(task_dir: str):
    p = os.path.join(task_dir, "answer.txt")  # the required location; workspace/answer.txt does not count
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _read(path: str):
    try:
        with open(path, "rb") as f:
            return f.read().replace(b"\r\n", b"\n")
    except OSError:
        return None


def check(task_dir: str, meta: dict, original_dir=None):
    """Returns (verdict, detail). verdict: correct | wrong | no_answer."""
    ans = read_answer(task_dir)
    kind = meta.get("check")
    expected = str(meta.get("expected", ""))
    ws = os.path.join(task_dir, "workspace")
    if not os.path.isdir(ws):
        ws = task_dir
    if ans is None:
        return "no_answer", ""
    a = ans.strip()
    if kind == "unittest":
        if a != "DONE":
            return "wrong", "answer=%r" % a
        tests = sorted(f for f in os.listdir(ws) if (f.startswith("test_") or f.endswith("_test.py"))
                       and f.endswith(".py"))
        if not tests:
            return "wrong", "no test files"
        if original_dir:
            ows = os.path.join(original_dir, "workspace")
            ows = ows if os.path.isdir(ows) else original_dir
            for t in tests:
                if _read(os.path.join(ws, t)) != _read(os.path.join(ows, t)):
                    return "wrong", "test file %s was modified" % t
        scratch = copy_workspace(ws)
        if scratch is None:
            return "wrong", "cannot copy the workspace"
        try:
            r = run_tests(scratch, tests, 60)
        finally:
            remove_scratch(scratch)
        if not r.passed:
            why = r.load_error or ("failing: " + ", ".join(sorted(r.failing)))
            return "wrong", "tests not green: " + why[:120]
        return "correct", "tests pass (%d run, %d executed)" % (r.ran, r.executed)
    if kind == "exact_int":
        try:
            return ("correct" if int(a) == int(expected) else "wrong"), "answer=%r expected=%r" % (a, expected)
        except ValueError:
            return "wrong", "answer=%r expected=%r" % (a, expected)
    return ("correct" if a == expected else "wrong"), "answer=%r expected=%r" % (a, expected)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir")
    ap.add_argument("--keep", action="store_true", help="keep the temporary copy")
    ap.add_argument("--quiet", action="store_true", help="hide harness output")
    ap.add_argument("--only-failures", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="do not run the harness, only check existing answers")
    ap.add_argument("--trace", default=None, help="trace file of the run (needed with --no-run to know the spend)")
    ap.add_argument("--original", default=None, help="pristine copy of the tasks (to prove tests were not edited)")
    args = ap.parse_args()

    src = os.path.abspath(args.tasks_dir)
    if args.no_run:
        # Only check a directory the harness has already processed (e.g. by `docker run`).
        tmp, dst, trace = None, src, args.trace or ""
        original = os.path.abspath(args.original) if args.original else None
        proc = subprocess.CompletedProcess([], 0)
        elapsed = 0.0
    else:
        tmp = tempfile.mkdtemp(prefix="score_")
        dst = os.path.join(tmp, "tasks")
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("answer.txt", "__pycache__", ".DS_Store"))
        trace = os.path.join(tmp, "trace.jsonl")
        original = src
        t0 = time.time()
        proc = subprocess.run([sys.executable, os.path.join(ROOT, "run_agent.py"), "--tasks-dir", dst,
                               "--trace", trace], capture_output=args.quiet, text=True)
        elapsed = time.time() - t0

    have_trace = bool(trace) and os.path.isfile(trace)
    spent, status, planned, calls = {}, {}, set(), 0
    if have_trace:
        with open(trace, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") in ("llm_call", "llm_uncertain_spend"):
                    cost = float(ev.get("cost") or ev.get("booked") or 0.0)
                    spent[ev["task"]] = spent.get(ev["task"], 0.0) + cost
                    calls += ev.get("event") == "llm_call"
                elif ev.get("event") == "outcome":
                    status[ev["task"]] = (ev.get("status"), ev.get("tier"), ev.get("reason") or "")
                elif ev.get("event") == "planned":
                    planned.add(ev["task"])

    rows = []
    for name in sorted(os.listdir(dst)):
        d = os.path.join(dst, name)
        tj = os.path.join(d, "task.json")
        if not os.path.isfile(tj):
            continue
        with open(tj, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tid = meta.get("task_id", name)
        feasible = meta.get("feasible", True)
        expect = meta.get("expect", "free" if feasible else "free_reject")
        verdict, detail = check(d, meta, os.path.join(original, name) if original else None)
        st, tier, reason = status.get(tid, ("?", "", ""))
        tok = spent.get(tid, 0.0) if have_trace else None
        if feasible:
            if verdict == "correct":
                cat = "OK_SPEND_UNKNOWN" if tok is None else ("OK_FREE" if tok == 0 else "OK_LLM")
                bad = False
            elif verdict == "wrong":
                cat, bad = "WRONG_ANSWER", True
            elif st == "infeasible":
                cat, bad = "FALSE_REJECT", True
            elif tid in planned:
                # not solved; "expected" only says that the free tier was not supposed to manage it
                cat, bad = ("UNSOLVED_ROUTED_TO_LLM", expect != "llm")
            else:
                cat, bad = "UNSOLVED", True
        else:
            if tid in planned:
                cat, bad = "INFEASIBLE_TO_LLM", expect != "llm_reject"
            elif st == "infeasible":
                cat, bad = "REJECT_FREE", False
            elif verdict != "no_answer":
                cat, bad = "INFEASIBLE_ANSWERED_FREE", False
            else:
                cat, bad = "INFEASIBLE_NO_ANSWER", False
        rows.append(dict(name=name, type=meta.get("task_type"), reward=float(meta.get("reward", 0)),
                         feasible=feasible, expect=expect, verdict=verdict, detail=detail, tokens=tok, cat=cat,
                         bad=bad, note=meta.get("note", ""), reason=reason, tier=tier))

    for r in rows:
        if args.only_failures and not r["bad"]:
            continue
        tok = "n/a" if r["tokens"] is None else "%.1f" % r["tokens"]
        print("%-26s %-24s tok=%-6s %-30s %s%s" % (
            r["name"], r["cat"], tok, (r["note"] or "")[:30],
            (r["detail"] or r["reason"])[:90], "   <-- CHECK" if r["bad"] else ""))
    print("-" * 110)
    cats = {}
    for r in rows:
        key = (r["type"], r["cat"])
        cats[key] = cats.get(key, 0) + 1
    for key in sorted(cats):
        print("  %-14s %-26s %d" % (key[0], key[1], cats[key]))
    feas = [r for r in rows if r["feasible"]]
    infeas = [r for r in rows if not r["feasible"]]
    got = sum(r["reward"] for r in feas if r["verdict"] == "correct")
    possible = sum(r["reward"] for r in feas)
    count = lambda rs, *cs: sum(1 for r in rs if r["cat"] in cs)  # noqa: E731
    print("CORRECTNESS feasible: %d/%d correct, reward %g/%g | wrong answers: %d | false rejects: %d | "
          "unsolved: %d (of them routed to the LLM tier: %d)" % (
              sum(1 for r in feas if r["verdict"] == "correct"), len(feas), got, possible,
              count(feas, "WRONG_ANSWER"), count(feas, "FALSE_REJECT"),
              count(feas, "UNSOLVED", "UNSOLVED_ROUTED_TO_LLM"), count(feas, "UNSOLVED_ROUTED_TO_LLM")))
    print("ROUTING     infeasible: %d tasks | rejected free: %d | sent to the LLM tier: %d | answered free: %d | "
          "no answer: %d" % (len(infeas), count(infeas, "REJECT_FREE"), count(infeas, "INFEASIBLE_TO_LLM"),
                             count(infeas, "INFEASIBLE_ANSWERED_FREE"), count(infeas, "INFEASIBLE_NO_ANSWER")))
    if have_trace:
        print("SPEND       total tokens: %.1f in %d LLM call(s) | on infeasible tasks: %.1f" % (
            sum(r["tokens"] for r in rows), calls, sum(r["tokens"] for r in infeas)))
    else:
        print("SPEND       unknown: no trace was given (use --trace); answers are NOT assumed to be free")
    print("harness exit code %d | %.1fs | cases to check: %d" % (proc.returncode, elapsed,
                                                                 sum(1 for r in rows if r["bad"])))
    if tmp and args.keep:
        print("copy kept at", tmp)
    elif tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if proc.returncode == 0 and not any(r["bad"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
