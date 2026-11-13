"""Run-time settings of the harness.

Everything is read from environment variables so that the image needs no
rebuild to change behaviour. The API key is only ever read here and handed to
the LLM client; it is never logged, written to disk or passed to child
processes that execute task code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

MODEL_FAST = "qwen3.8-27b-noreason"
MODEL_FAST_ALT = "qwen3.6-35b-a3b-noreason"
# Both defaults are non-reasoning models: reasoning tokens are billed 1:1 and
# are therefore an explicit opt-in (HARNESS_MODEL_STRONG=qwen3.8-27b).
MODEL_STRONG = "qwen3.8-27b-noreason"
ALLOWED_MODELS = (
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ[name]))
    except Exception:
        return default


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


def _env_model(name: str, default: str) -> str:
    v = (os.environ.get(name) or "").strip()
    return v if v in ALLOWED_MODELS else default


@dataclass
class Settings:
    # --- LLM proxy -------------------------------------------------------
    base_url: str = field(default_factory=lambda: (os.environ.get("LLM_BASE_URL") or "").strip())
    api_key: str = field(default_factory=lambda: (os.environ.get("LLM_API_KEY") or "").strip(), repr=False)
    model_fast: str = field(default_factory=lambda: _env_model("HARNESS_MODEL_FAST", MODEL_FAST))
    model_strong: str = field(default_factory=lambda: _env_model("HARNESS_MODEL_STRONG", MODEL_STRONG))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("HARNESS_LLM_TIMEOUT", 120.0))
    # Number of simultaneous LLM sessions (rules allow at most 5).
    concurrency: int = field(default_factory=lambda: max(1, min(5, _env_int("HARNESS_CONCURRENCY", 4))))

    # --- token budget ----------------------------------------------------
    # Total budget in "output-token units" (1 output = 1, 10 input = 1).
    # The launch command of the rules passes no budget, and the real limit is
    # enforced by the proxy, not by us. TOKEN_BUDGET sets our own ceiling
    # explicitly; without it the ceiling is HARNESS_BUDGET_PER_TASK x tasks - a
    # frugality assumption of this harness, not a confirmed limit of the proxy.
    budget_total: float = field(default_factory=lambda: _env_float("TOKEN_BUDGET", 0.0))
    budget_per_task: float = field(default_factory=lambda: _env_float("HARNESS_BUDGET_PER_TASK", 1000.0))
    # We stop starting new LLM work once this share of the budget is spent.
    budget_soft_fraction: float = field(default_factory=lambda: _env_float("HARNESS_BUDGET_SOFT", 0.85))
    # Hard cap of what a single task may consume (all attempts together).
    task_token_cap: float = field(default_factory=lambda: _env_float("HARNESS_TASK_CAP", 1800.0))

    # --- time ------------------------------------------------------------
    # Whole run is limited to 1 hour by the organisers; we stop at 50 minutes.
    deadline_s: float = field(default_factory=lambda: _env_float("HARNESS_DEADLINE_S", 50 * 60.0))
    test_timeout_s: float = field(default_factory=lambda: _env_float("HARNESS_TEST_TIMEOUT", 20.0))
    # Zero-token mutation search: at most this long per task, and all free work
    # together at most this share of the run, so that it cannot crowd out the LLM tier.
    mutation_time_s: float = field(default_factory=lambda: _env_float("HARNESS_MUTATION_TIME", 90.0))
    free_phase_fraction: float = field(default_factory=lambda: _env_float("HARNESS_FREE_FRACTION", 0.5))

    # --- feature switches ------------------------------------------------
    enable_llm: bool = field(default_factory=lambda: _env_flag("HARNESS_LLM", True))
    # Classical generate-and-validate program repair (0 tokens) before the LLM.
    enable_mutation_repair: bool = field(default_factory=lambda: _env_flag("HARNESS_MUTATION_REPAIR", True))
    workers: int = field(default_factory=lambda: max(1, min(8, _env_int("HARNESS_WORKERS", 6))))

    def resolve_budget(self, n_tasks: int) -> float:
        if self.budget_total and self.budget_total > 0:
            return self.budget_total
        return max(1.0, self.budget_per_task * max(1, n_tasks))

    @property
    def llm_configured(self) -> bool:
        return bool(self.enable_llm and self.base_url and self.api_key)
