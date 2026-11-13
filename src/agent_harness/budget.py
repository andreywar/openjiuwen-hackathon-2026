"""Token ledger implementing the competition accounting.

cost = output_tokens + reasoning_tokens (if reported separately) + input_tokens / 10

The numbers come from the ``usage`` block of every proxy response. Cached
input is charged in full (the organisers ignore provider caches).

Whether reasoning tokens are already part of ``completion_tokens`` is decided
by the caller from the structure of the usage block (``llm._reasoning_semantics``),
never by comparing magnitudes; an unknown layout is counted on top. A response
without usage is charged by estimate, a request whose answer was lost after it
was sent is charged with its reserved estimate ("uncertain spend").

The ledger is this process' own book-keeping. It is not the proxy's hard
limit and cannot see what the proxy really billed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List

INPUT_WEIGHT = 0.1
CHARS_PER_TOKEN = 3.2  # conservative estimate for English text / code


def estimate_tokens(text: str) -> int:
    """Cheap, slightly pessimistic token estimate used for planning only."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


def weighted_cost(input_tokens: int, output_tokens: int, reasoning_tokens: int = 0,
                  reasoning_separate: bool = False) -> float:
    """Competition cost. ``reasoning_separate`` says that ``reasoning_tokens`` are
    not contained in ``output_tokens`` (or that this is unknown) and must be added."""
    out = output_tokens + (reasoning_tokens if reasoning_separate else 0)
    return out + input_tokens * INPUT_WEIGHT


@dataclass
class CallRecord:
    task_id: str
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost: float
    transport: str
    note: str = ""   # "uncertain spend ...", "... not reported: estimated"


@dataclass
class Ledger:
    total: float
    soft_fraction: float = 0.85
    spent: float = 0.0
    reserved: float = 0.0
    calls: List[CallRecord] = field(default_factory=list)
    per_task: Dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def soft_limit(self) -> float:
        return self.total * self.soft_fraction

    def remaining(self) -> float:
        with self._lock:
            return self.soft_limit - self.spent - self.reserved

    def task_spent(self, task_id: str) -> float:
        with self._lock:
            return self.per_task.get(task_id, 0.0)

    def try_reserve(self, amount: float) -> bool:
        """Reserve budget for a call that is about to be made."""
        with self._lock:
            if self.spent + self.reserved + amount > self.soft_limit:
                return False
            self.reserved += amount
            return True

    def release(self, amount: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - amount)

    def charge(self, rec: CallRecord) -> None:
        with self._lock:
            self.spent += rec.cost
            self.per_task[rec.task_id] = self.per_task.get(rec.task_id, 0.0) + rec.cost
            self.calls.append(rec)

    def settle(self, reserved_amount: float, rec: CallRecord) -> None:
        """End one reserved attempt: drop its reservation and book its cost in one step.

        The real cost may be larger than the reservation (the estimate is not an
        upper bound); it is booked as it is, and later admissions see it.
        """
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved_amount)
            self.spent += rec.cost
            self.per_task[rec.task_id] = self.per_task.get(rec.task_id, 0.0) + rec.cost
            self.calls.append(rec)

    def summary(self) -> dict:
        with self._lock:
            by_model: Dict[str, dict] = {}
            for c in self.calls:
                m = by_model.setdefault(c.model, {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
                m["calls"] += 1
                m["input"] += c.input_tokens
                m["output"] += c.output_tokens
                m["cost"] = round(m["cost"] + c.cost, 1)
            return {
                "budget_total": self.total,
                "soft_limit": round(self.soft_limit, 1),
                "spent": round(self.spent, 1),
                "calls": len(self.calls),
                "input_tokens": sum(c.input_tokens for c in self.calls),
                "output_tokens": sum(c.output_tokens for c in self.calls),
                "reasoning_tokens": sum(c.reasoning_tokens for c in self.calls),
                "uncertain_or_estimated_calls": sum(1 for c in self.calls if c.note),
                "by_model": by_model,
            }
