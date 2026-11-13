"""Task discovery and loading.

Only the four documented fields of ``task.json`` are used: ``task_id``,
``task_type``, ``instruction`` and ``reward``. The public examples also carry
grading fields; the harness deliberately never looks at them - they are dropped
right after parsing and are not stored anywhere (see ``_ALLOWED_FIELDS``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

_ALLOWED_FIELDS = ("task_id", "task_type", "instruction", "reward")
KNOWN_TYPES = ("math", "data_analysis", "search", "coding")


@dataclass
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    task_dir: str
    workspace: str

    def __repr__(self) -> str:  # short form for logs
        return "Task(%s, %s, reward=%g)" % (self.task_id, self.task_type, self.reward)


def _infer_type(instruction: str, workspace: str) -> str:
    """Guess the task type when ``task_type`` is absent or unknown."""
    try:
        names = os.listdir(workspace)
    except OSError:
        names = []
    low = (instruction or "").lower()
    if any(n.startswith("test_") and n.endswith(".py") for n in names) or "tests pass" in low:
        return "coding"
    if any(n.lower().endswith(".csv") for n in names) or ".csv" in low:
        return "data_analysis"
    if "kb" in names or "knowledge base" in low:
        return "search"
    if "boxes" in low or "warehouse" in low:
        return "math"
    return "unknown"


def _normalise_type(raw, instruction: str, workspace: str) -> str:
    t = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "data": "data_analysis", "analysis": "data_analysis", "dataanalysis": "data_analysis",
        "code": "coding", "arithmetic": "math", "retrieval": "search",
    }
    t = aliases.get(t, t)
    if t in KNOWN_TYPES:
        return t
    return _infer_type(instruction, workspace)


def load_task(task_dir: str) -> Optional[Task]:
    path = os.path.join(task_dir, "task.json")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = json.load(f)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    # Keep only the documented fields; everything else is discarded here.
    data = {k: raw.get(k) for k in _ALLOWED_FIELDS}
    del raw

    instruction = data["instruction"] if isinstance(data["instruction"], str) else ""
    ws = os.path.join(task_dir, "workspace")
    workspace = ws if os.path.isdir(ws) else task_dir
    try:
        reward = float(data["reward"])
    except Exception:
        reward = 1.0
    task_id = str(data["task_id"] or os.path.basename(os.path.normpath(task_dir)))
    return Task(
        task_id=task_id,
        task_type=_normalise_type(data["task_type"], instruction, workspace),
        instruction=instruction,
        reward=reward,
        task_dir=task_dir,
        workspace=workspace,
    )


def discover_tasks(tasks_dir: str, max_depth: int = 3) -> List[Task]:
    """Find every directory with a ``task.json`` (normally ``<tasks_dir>/<id>/``)."""
    found: List[Task] = []
    root = os.path.abspath(tasks_dir)
    if not os.path.isdir(root):
        return found

    def walk(d: str, depth: int) -> None:
        if os.path.isfile(os.path.join(d, "task.json")):
            t = load_task(d)
            if t is not None:
                found.append(t)
            return  # do not descend into a task directory
        if depth >= max_depth:
            return
        try:
            entries = sorted(os.listdir(d))
        except OSError:
            return
        for e in entries:
            if e.startswith(".") or e == "__MACOSX":
                continue
            p = os.path.join(d, e)
            if os.path.isdir(p):
                walk(p, depth + 1)

    walk(root, 0)
    return found
