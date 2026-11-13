#!/usr/bin/env python3
"""Entry point of the harness.

    python /app/run_agent.py --tasks-dir /data/tasks

Environment: LLM_BASE_URL, LLM_API_KEY (proxy of the organisers). Optional
tuning variables are listed in README.md. The process always exits with
code 0: a problem with one task (or with the LLM) must never take the whole
run down, and answers are written as soon as they are known.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import threading
import traceback

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Budget-aware agent harness (openJiuwen Challenge 2026)")
    ap.add_argument("--tasks-dir", "--tasks_dir", dest="tasks_dir", default=os.environ.get("TASKS") or "/data/tasks")
    ap.add_argument("--trace", default=None, help="path of the JSONL trajectory log (default: temp dir)")
    args, _unknown = ap.parse_known_args(argv)
    return args


def main(argv=None) -> int:
    # Only the harness' own logger is verbose; third-party libraries stay at WARNING.
    log = logging.getLogger("harness")
    try:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    except Exception:
        pass
    try:
        args = _parse_args(sys.argv[1:] if argv is None else argv)
        tasks_dir = os.path.abspath(args.tasks_dir)
        trace_path = os.path.abspath(args.trace) if args.trace else None

        # The image file system is read-only; openJiuwen writes ./logs on import,
        # so work from a private directory under the temp dir.
        try:
            scratch = tempfile.mkdtemp(prefix="harness_")
            os.chdir(scratch)
        except Exception:
            scratch = tempfile.gettempdir()
            try:
                os.chdir(scratch)
            except Exception:
                pass
        if trace_path is None:
            trace_path = os.path.join(scratch, "trace.jsonl")

        sys.path.insert(0, os.path.join(APP_DIR, "src"))
        from agent_harness.config import Settings
        from agent_harness.pipeline import run

        settings = Settings()
        # Last-resort guard: leave before the organisers' 1 hour limit kills us.
        watchdog = threading.Timer(settings.deadline_s + 360.0, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()

        log.info("harness start: tasks_dir=%s llm_configured=%s trace=%s", tasks_dir, settings.llm_configured,
                 trace_path)
        run(tasks_dir, settings, trace_path)
    except SystemExit:
        pass
    except BaseException:  # noqa: BLE001 - the run must never crash
        try:
            traceback.print_exc()
        except Exception:
            pass
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    code = main()
    # os._exit: do not wait for stray threads or library shutdown hooks.
    os._exit(code)
