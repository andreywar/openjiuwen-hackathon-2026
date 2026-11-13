"""Planner / scheduler of the harness.

Phase 1 - free work. Every task is first handed to its zero-token solver.
Answers are written to disk the moment they are known. Tasks proven
infeasible are dropped without any LLM call.

Phase 2 - paid work. Whatever is left (statements the free solvers did not
understand, code that mutation repair could not fix) gets a cost estimate and
is processed in order of reward per estimated token, with a bounded number of
parallel LLM sessions, a global soft budget, a per-task cap and a wall-clock
deadline.

Nothing here is allowed to raise: a failure is always confined to one task.
"""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import answers, sandbox
from .budget import Ledger
from .config import Settings
from .llm import LLMClient, preload_openjiuwen
from .solvers import coding_solver, data_solver, llm_fallback, math_solver, search_solver
from .solvers.base import INFEASIBLE, SOLVED, UNKNOWN, Outcome
from .tasks import Task, discover_tasks
from .trace import Trace

log = logging.getLogger("harness")


@dataclass
class TaskState:
    task: Task
    status: str = "pending"      # solved | infeasible | unsolved | skipped
    tier: str = ""               # deterministic | mutation | llm
    answer: Optional[str] = None
    reason: str = ""
    estimate: float = 0.0        # planned weighted tokens
    spent: float = 0.0           # actual weighted tokens
    written: bool = False
    needs_llm: bool = False
    ctx: object = None           # coding context kept between the phases
    seconds: float = 0.0


@dataclass
class RunResult:
    states: List[TaskState] = field(default_factory=list)
    ledger: Optional[Ledger] = None
    elapsed: float = 0.0
    llm_used: bool = False
    transports: Dict[str, int] = field(default_factory=dict)


def _free_solve(task: Task, settings: Settings, coding_time: float = 120.0, run_deadline: Optional[float] = None):
    """Returns (Outcome, coding context or None)."""
    if not (task.instruction or "").strip():
        return Outcome.infeasible("empty instruction"), None
    if task.task_type == "math":
        return math_solver.solve(task.instruction), None
    if task.task_type == "data_analysis":
        return data_solver.solve(task.instruction, task.workspace), None
    if task.task_type == "search":
        return search_solver.solve(task.instruction, task.workspace), None
    if task.task_type == "coding":
        return coding_solver.solve_free(task, settings, coding_time, run_deadline)
    return Outcome.unknown("unknown task type"), None


def _record(state: TaskState, outcome: Outcome, trace: Trace) -> None:
    """Apply an outcome: write the answer at once, remember the verdict."""
    task = state.task
    state.reason = outcome.reason
    state.tier = outcome.tier
    if outcome.status == SOLVED and outcome.answer is not None and str(outcome.answer).strip() != "":
        state.answer = str(outcome.answer).strip()
        state.written = answers.write_answer(task, state.answer)
        state.status = "solved" if state.written else "unsolved"
        if not state.written:
            state.reason = "answer computed but <task_dir>/answer.txt could not be written (%s)" % (
                answers.describe_access(task.task_dir))
        state.needs_llm = False
    elif outcome.status == INFEASIBLE:
        state.status = "infeasible"
        state.needs_llm = False
    else:
        state.status = "unsolved"
    trace.event(task.task_id, "outcome", status=state.status, tier=state.tier, answer=state.answer,
                reason=state.reason, type=task.task_type, reward=task.reward, written=state.written)
    log.info("[%s] %-10s tier=%-13s answer=%r  %s", task.task_id, state.status, state.tier or "-",
             state.answer, (state.reason or "")[:160])


def _estimate(state: TaskState) -> float:
    try:
        if state.task.task_type == "coding" and state.ctx is not None:
            return coding_solver.estimate_llm_cost(state.ctx)
        return llm_fallback.estimate_cost(state.task)
    except Exception:  # noqa: BLE001
        return 400.0


def _release(st: TaskState) -> None:
    """Drop the scratch copies of a task; the coding context itself is kept for the LLM tier."""
    ctx = st.ctx
    if ctx is not None and hasattr(ctx, "cleanup"):
        try:
            ctx.cleanup()
        except Exception:  # noqa: BLE001
            pass


def phase_free(states: List[TaskState], settings: Settings, trace: Trace, deadline: float,
               free_deadline: Optional[float] = None) -> None:
    """Zero-token work. ``free_deadline`` bounds the mutation search of the coding
    tasks so that it cannot eat the time of the paid tier; the cheap parts
    (parsing, one baseline test run) still happen for every task."""
    free_deadline = deadline if free_deadline is None else min(deadline, free_deadline)
    # Instant solvers first, then coding (runs tests); inside a group by reward.
    order = sorted(states, key=lambda s: (s.task.task_type == "coding", -s.task.reward, s.task.task_id))
    coding_left = sum(1 for s in order if s.task.task_type == "coding")
    for st in order:
        if time.monotonic() > deadline:
            st.status, st.reason = "skipped", "deadline reached before the task was started"
            continue
        t0 = time.monotonic()
        try:
            coding_time = 0.0
            if st.task.task_type == "coding":
                # a fair share of what is left of the free phase, never more than the per-task limit
                share = (free_deadline - t0) / max(1, coding_left) * 2.0
                coding_time = max(0.0, min(settings.mutation_time_s, share))
                coding_left -= 1
            outcome, ctx = _free_solve(st.task, settings, coding_time, free_deadline)
            st.ctx = ctx
            _record(st, outcome, trace)
            if outcome.status == UNKNOWN:
                st.needs_llm = st.task.task_type != "coding" or ctx is not None and ctx.baseline is not None
        except Exception as e:  # noqa: BLE001
            st.status, st.reason = "unsolved", "internal error: %s" % e
            st.needs_llm = st.task.task_type != "coding"
            log.error("[%s] free solver crashed: %s", st.task.task_id, traceback.format_exc(limit=3))
        finally:
            _release(st)  # scratch copies are freed as soon as the task leaves the phase
        st.seconds += time.monotonic() - t0


async def _paid_one(st: TaskState, llm: LLMClient, settings: Settings, trace: Trace, sem: asyncio.Semaphore,
                    deadline: float) -> None:
    async with sem:
        task = st.task
        llm.waiting_tasks = max(0, llm.waiting_tasks - 1)  # this task now gets its first chance
        if time.monotonic() > deadline - 30:
            st.reason = "deadline reached"
            return
        if not llm.available:
            st.reason = "LLM unavailable: " + (llm.dead_reason or "back-off would pass the deadline")
            return
        if llm.ledger.remaining() < min(st.estimate, 150.0):
            st.reason = "token budget exhausted (soft limit)"
            trace.event(task.task_id, "skipped", reason=st.reason, estimate=round(st.estimate, 1))
            return
        t0 = time.monotonic()
        try:
            if task.task_type == "coding" and st.ctx is not None:
                outcome = await coding_solver.solve_llm(task, st.ctx, llm, settings, trace)
            else:
                outcome = await llm_fallback.solve(task, llm, settings, trace)
            _record(st, outcome, trace)
        except Exception as e:  # noqa: BLE001
            st.reason = "internal error in LLM tier: %s" % e
            log.error("[%s] LLM tier crashed: %s", task.task_id, traceback.format_exc(limit=3))
        finally:
            _release(st)
            st.spent = llm.ledger.task_spent(task.task_id)
            st.seconds += time.monotonic() - t0


async def phase_paid(queue: List[TaskState], llm: LLMClient, settings: Settings, trace: Trace,
                     deadline: float) -> None:
    sem = asyncio.Semaphore(settings.concurrency)
    jobs = [asyncio.ensure_future(_paid_one(st, llm, settings, trace, sem, deadline)) for st in queue]
    timeout = max(1.0, deadline - time.monotonic())
    done, pending = await asyncio.wait(jobs, timeout=timeout)
    for p in pending:
        p.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def run(tasks_dir: str, settings: Optional[Settings] = None, trace_path: Optional[str] = None) -> RunResult:
    t_start = time.monotonic()
    settings = settings or Settings()
    deadline = t_start + settings.deadline_s
    trace = Trace(trace_path)
    result = RunResult()

    tasks = discover_tasks(tasks_dir)
    log.info("tasks found: %d in %s", len(tasks), tasks_dir)
    states = [TaskState(t) for t in tasks]
    result.states = states
    budget_total, budget_source, budget_problem = settings.budget_info(len(tasks))
    ledger = Ledger(budget_total, settings.budget_soft_fraction)
    result.ledger = ledger
    paid_allowed = settings.llm_configured and budget_total > 0
    log.info("budget: %g weighted tokens, source: %s (soft limit %g)", budget_total, budget_source, ledger.soft_limit)
    if budget_problem:
        log.error("%s; the zero-token solvers still run", budget_problem)
    elif budget_total <= 0:
        log.warning("the budget is 0: no paid LLM calls will be made; the zero-token solvers still run")
    isolation = sandbox.isolation_mode()
    log.info("child process isolation: %s", isolation)
    trace.event("-", "run_start", tasks=len(tasks), budget=ledger.total, soft_limit=ledger.soft_limit,
                budget_source=budget_source, budget_problem=budget_problem, paid_allowed=paid_allowed,
                task_cap=settings.task_token_cap, task_cap_explicit=settings.task_cap_explicit,
                llm_configured=settings.llm_configured, mutation_repair=settings.enable_mutation_repair,
                isolation=isolation, model_fast=settings.model_fast, model_strong=settings.model_strong)
    if not tasks:
        trace.close()
        return result

    # The answers must be created in the task directories. Say so loudly, once,
    # if the permissions of the mounted directory make that impossible.
    locked = [t.task_id for t in tasks if not answers.writable(t.task_dir)]
    if locked:
        log.error("%d of %d task directories are NOT writable for this process (%s); answers for them cannot be "
                  "delivered: %s%s", len(locked), len(tasks), answers.describe_access(tasks[0].task_dir),
                  ", ".join(locked[:5]), " ..." if len(locked) > 5 else "")
        trace.event("-", "not_writable", tasks=locked, access=answers.describe_access(tasks[0].task_dir))

    # ---- phase 1: zero tokens --------------------------------------------
    free_deadline = None
    if paid_allowed:  # leave room for the paid tier
        free_deadline = t_start + settings.deadline_s * max(0.1, min(1.0, settings.free_phase_fraction))
    phase_free(states, settings, trace, deadline, free_deadline)
    free_solved = sum(1 for s in states if s.status == "solved")
    log.info("phase 1 done in %.1fs: solved=%d infeasible=%d need_llm=%d", time.monotonic() - t_start, free_solved,
             sum(1 for s in states if s.status == "infeasible"), sum(1 for s in states if s.needs_llm))

    # ---- phase 2: LLM ---------------------------------------------------
    queue = [s for s in states if s.needs_llm and s.status == "unsolved"]
    if queue:
        for st in queue:
            st.estimate = _estimate(st)
        # Most reward per estimated token first. This is a greedy ordering, not an
        # optimal knapsack: it ignores the chance of success and the time left.
        queue.sort(key=lambda s: (-(s.task.reward / max(s.estimate, 1.0)), s.task.task_id))
        for st in queue:
            trace.event(st.task.task_id, "planned", estimate=round(st.estimate, 1), reward=st.task.reward,
                        priority=round(st.task.reward / max(st.estimate, 1.0), 5))
        if not settings.llm_configured:
            for st in queue:
                st.reason = (st.reason + " | " if st.reason else "") + "LLM not configured"
            log.warning("%d task(s) need the LLM but LLM_BASE_URL / LLM_API_KEY are not set", len(queue))
        elif not paid_allowed:
            why = budget_problem or "the budget is 0"
            for st in queue:
                st.reason = (st.reason + " | " if st.reason else "") + "no paid calls: " + why
                trace.event(st.task.task_id, "skipped", reason="no paid calls: " + why)
            log.warning("%d task(s) need the LLM but paid calls are not allowed: %s", len(queue), why)
        else:
            llm = LLMClient(settings, ledger, trace, deadline)
            llm.waiting_tasks = len(queue)
            if not preload_openjiuwen():
                llm._oj_ok = False  # plain HTTP transport only
            result.llm_used = True
            try:
                asyncio.run(phase_paid(queue, llm, settings, trace, deadline))
            except Exception:  # noqa: BLE001
                log.error("LLM phase aborted: %s", traceback.format_exc(limit=3))
            result.transports = dict(llm.transport_counts)

    for st in states:
        ctx = st.ctx
        if ctx is not None and hasattr(ctx, "cleanup"):
            try:
                ctx.cleanup()
            except Exception:  # noqa: BLE001
                pass
        st.spent = ledger.task_spent(st.task.task_id)

    result.elapsed = time.monotonic() - t_start
    _report(result, trace)
    trace.close()
    return result


def _report(result: RunResult, trace: Trace) -> None:
    states = result.states
    ledger = result.ledger
    log.info("%-22s %-14s %6s %-10s %-13s %8s %8s %6s  %s", "task", "type", "reward", "status", "tier", "est",
             "spent", "sec", "answer")
    for st in sorted(states, key=lambda s: s.task.task_id):
        log.info("%-22s %-14s %6g %-10s %-13s %8.0f %8.1f %6.1f  %s", st.task.task_id, st.task.task_type,
                 st.task.reward, st.status, st.tier or "-", st.estimate, st.spent, st.seconds,
                 "" if st.answer is None else st.answer[:40])
    solved = [s for s in states if s.status == "solved"]
    summary = {
        "tasks": len(states),
        "answered": len(solved),
        "answered_reward": sum(s.task.reward for s in solved),
        "total_reward": sum(s.task.reward for s in states),
        "infeasible": sum(1 for s in states if s.status == "infeasible"),
        "unsolved": sum(1 for s in states if s.status in ("unsolved", "skipped", "pending")),
        "by_tier": {t: sum(1 for s in solved if s.tier == t) for t in ("deterministic", "mutation", "llm")},
        "tokens": ledger.summary() if ledger else {},
        "not_written": [s.task.task_id for s in states if s.answer is not None and not s.written],
        "transports": result.transports,
        "elapsed_s": round(result.elapsed, 1),
    }
    trace.event("-", "run_end", **summary)
    log.info("SUMMARY answered=%d/%d reward=%g/%g infeasible=%d unsolved=%d tiers=%s", summary["answered"],
             summary["tasks"], summary["answered_reward"], summary["total_reward"], summary["infeasible"],
             summary["unsolved"], summary["by_tier"])
    if ledger:
        t = summary["tokens"]
        log.info("TOKENS spent=%.1f of budget %.0f (soft limit %.0f): calls=%d input=%d output=%d reasoning=%d",
                 t["spent"], t["budget_total"], t["soft_limit"], t["calls"], t["input_tokens"], t["output_tokens"],
                 t["reasoning_tokens"])
        if t.get("uncertain_or_estimated_calls"):
            log.info("TOKENS note: %d call record(s) are estimates or uncertain spend (see the trace)",
                     t["uncertain_or_estimated_calls"])
    if summary["not_written"]:
        log.error("NOT WRITTEN: %d computed answer(s) could not be stored: %s", len(summary["not_written"]),
                  ", ".join(summary["not_written"][:10]))
    log.info("elapsed %.1fs", result.elapsed)
