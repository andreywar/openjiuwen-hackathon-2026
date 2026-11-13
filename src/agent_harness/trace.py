"""Trajectory log: one JSON line per event (solver decisions, LLM calls, writes).

The file lives in the temporary directory (the image file system is read-only)
and is also what the ``trajectories/`` folder of the repository was produced
from during development.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional


class Trace:
    def __init__(self, path: Optional[str]):
        self.path = path
        self._lock = threading.Lock()
        self._t0 = time.time()
        self._fh = None
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                self._fh = open(path, "a", encoding="utf-8")
            except Exception:
                self._fh = None

    def event(self, task_id: str, kind: str, **data) -> None:
        if self._fh is None:
            return
        rec = {"t": round(time.time() - self._t0, 3), "task": task_id, "event": kind}
        rec.update(data)
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
            with self._lock:
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass
