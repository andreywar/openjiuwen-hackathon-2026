"""LLM safety net for statements the zero-token solvers did not understand.

A short "write code, we run it" loop: the model gets the problem statement and
a compact description of the task files and answers with a small Python
program. The harness executes the program on a scratch copy of the workspace;
a single printed line is taken as the final answer, anything else is sent back
for one more step. The model can also declare the task INFEASIBLE.

Arithmetic and data access are therefore done by Python, not by the model,
which keeps a non-reasoning model accurate and the token cost low
(typically one call of 150-400 weighted tokens).
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import List, Optional, Tuple

from ..budget import estimate_tokens
from ..llm import BudgetExceeded, LLMError
from .. import wsfiles
from ..sandbox import copy_workspace, remove_scratch, run_python
from .base import Outcome, problem_text

SYSTEM_PROMPT = (
    "You solve a task by writing Python 3 code (standard library only). Your code is executed inside the task "
    "directory and its output is returned to you.\n"
    "Reply with exactly one of:\n"
    "1. one ```python code block that prints ONLY the final answer on a single line "
    "(or prints what you need to inspect first);\n"
    "2. ANSWER: <final answer>\n"
    "3. INFEASIBLE - when the task cannot be solved: a required file, column, value or entity is missing, "
    "a division by zero occurs, or the statement is contradictory or incomplete.\n"
    "No explanations. Only the listed files exist; there is no network. Use integer arithmetic where the task says "
    "'rounded down' (floor division)."
)

MAX_STEPS = 3


def _preview(path: str, max_lines: int, max_chars: int = 600) -> str:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)[:max_chars]
    except OSError:
        return "<unreadable>"


def describe_workspace(workspace: str) -> str:
    """Compact, bounded description of the task files for the prompt."""
    # wsfiles hides task.json / answer.txt and everything outside the workspace.
    entries: List[Tuple[str, str]] = wsfiles.list_files(workspace)
    if not entries:
        return "FILES: (none)"
    out = ["FILES (%d): %s" % (len(entries), ", ".join(e[0] for e in entries[:60]))
           + (" ..." if len(entries) > 60 else "")]
    shown_kinds = set()
    budget = 5
    for rel, path in entries:
        if budget <= 0:
            break
        ext = os.path.splitext(rel)[1].lower()
        kind = (os.path.dirname(rel), re.match(r"[A-Za-z]*", os.path.basename(rel)).group(0), ext)
        if kind in shown_kinds or ext not in (".csv", ".md", ".txt", ".json", ".tsv"):
            continue
        shown_kinds.add(kind)
        budget -= 1
        out.append("--- %s (first lines)\n%s" % (rel, _preview(path, 6)))
    return "\n".join(out)


def build_messages(task) -> List[dict]:
    user = "TASK: %s\n\n%s" % (problem_text(task.instruction)[:3000], describe_workspace(task.workspace))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def estimate_cost(task) -> float:
    msgs = build_messages(task)
    tokens_in = sum(estimate_tokens(m["content"]) for m in msgs)
    return (tokens_in * 0.1 + 160) * 1.5


def clean_answer(text: str) -> str:
    a = text.strip().strip("`").strip()
    if len(a) >= 2 and a[0] == a[-1] and a[0] in "'\"":
        a = a[1:-1].strip()
    if re.fullmatch(r"[^.\s]+(?: [^.\s]+){0,5}\.", a):
        a = a[:-1]
    return a


def _extract_code(reply: str) -> Optional[str]:
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", reply, re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*)\Z", reply, re.S | re.I)
    return m.group(1) if m else None


async def solve(task, llm, settings, trace=None) -> Outcome:
    scratch = await asyncio.to_thread(copy_workspace, task.workspace)
    if scratch is None:
        return Outcome.unknown("cannot create a scratch copy of the workspace")
    try:
        return await _loop(task, llm, settings, scratch, trace)
    finally:
        remove_scratch(scratch)


async def _loop(task, llm, settings, scratch: str, trace) -> Outcome:
    messages = build_messages(task)
    reason = "no answer produced"
    for step in range(1, MAX_STEPS + 1):
        strong = step == MAX_STEPS
        model = settings.model_strong if strong else settings.model_fast
        # every step must fit into what is left of the per-task cap
        max_tokens = llm.affordable_max_tokens(task.task_id, messages, 700 if strong else 320, minimum=120)
        if max_tokens <= 0:
            return Outcome.unknown("budget: step %d does not fit into the token cap" % step)
        try:
            reply = await llm.chat(task.task_id, messages, model=model, max_tokens=max_tokens,
                                   purpose="fallback_step_%d" % step)
        except BudgetExceeded as e:
            return Outcome.unknown("budget: %s" % e)
        except LLMError as e:
            return Outcome.unknown("llm unavailable: %s" % str(e)[:120])
        text = reply.text.strip()

        code = _extract_code(text)
        if code is None:
            m = re.search(r"ANSWER\s*:\s*(.+)", text, re.I)
            if m:
                ans = clean_answer(m.group(1).splitlines()[0])
                if ans.upper() == "INFEASIBLE":
                    return Outcome.infeasible("model judged the task unsolvable", tier="llm")
                if ans:
                    return Outcome.solved(ans, "LLM answer at step %d" % step, tier="llm")
            if re.search(r"\bINFEASIBLE\b", text, re.I) and len(text) < 200:
                return Outcome.infeasible("model judged the task unsolvable", tier="llm")
            if text and "\n" not in text and len(text) <= 60:
                return Outcome.solved(clean_answer(text), "LLM short answer at step %d" % step, tier="llm")
            feedback = "Reply with one python code block, or 'ANSWER: <value>', or INFEASIBLE."
        else:
            res = await asyncio.to_thread(run_python, ["-c", code], scratch, settings.test_timeout_s, None,
                                          llm.deadline)
            out = res.stdout.strip()
            lines = [ln for ln in out.splitlines() if ln.strip()]
            if res.ok and len(lines) == 1 and len(lines[0]) <= 200:
                ans = clean_answer(lines[0])
                if ans.upper() == "INFEASIBLE":
                    return Outcome.infeasible("generated program reported the task unsolvable", tier="llm")
                return Outcome.solved(ans, "program output at step %d" % step, tier="llm")
            err_tail = "\n".join(res.stderr.strip().splitlines()[-4:])[:500]
            if not res.ok and "ZeroDivisionError" in err_tail and task.task_type == "math":
                return Outcome.infeasible("division by zero in the described computation", tier="llm")
            if res.timed_out:
                err_tail = "the program timed out"
            feedback = "OUTPUT:\n%s\n---\n%s%s" % (
                out[:1500] or "(empty)",
                ("ERROR:\n%s\n" % err_tail) if err_tail else "",
                "Now give 'ANSWER: <value>', INFEASIBLE, or another python code block that prints only the answer.")
        reason = "step %d did not produce an answer" % step
        if trace:
            trace.event(task.task_id, "fallback_step", step=step, feedback=feedback[:500])
        messages = messages + [{"role": "assistant", "content": text[:2000]},
                               {"role": "user", "content": feedback}]
    return Outcome.unknown(reason)
