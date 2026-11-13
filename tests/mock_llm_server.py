#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock of the organisers' LLM proxy (development only).

It is NOT part of the Docker image. It lets the LLM code paths of the harness be
exercised offline, without spending team tokens:

    python tests/mock_llm_server.py --port 8765 [--log requests.jsonl] [--budget 5000]

    LLM_BASE_URL=http://127.0.0.1:8765/v1 LLM_API_KEY=mock python run_agent.py --tasks-dir <copy>

Behaviour
---------
* ``GET /v1/models`` lists the four competition models.
* ``POST /v1/chat/completions`` answers with a canned reply chosen by simple
  rules on the last user message (see ``scripted_reply``). The rules imitate a
  reasonably competent model for the prompt formats used by the harness, which
  is enough to test parsing, patch application, retries and token accounting.
* ``usage`` is reported with a crude 1 token ~ 4 chars estimate.
* With ``--budget N`` the server starts answering HTTP 429 once the weighted
  spend (input/10 + output) exceeds N - this imitates the proxy cutting the
  team off when the token budget is exhausted.
* ``--fail-first K`` makes the first K completion requests fail with HTTP 500
  (to test retry / fallback transport).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = [
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
]

STATE = {
    "spent": 0.0,
    "budget": None,
    "requests": 0,
    "fail_first": 0,
    "log": None,
    "replies": [],  # optional queue of forced replies (--reply)
    "max_parallel": 0,
    "active": 0,
}
LOCK = threading.Lock()

# The real fixes for the buggy library functions seen in the public examples.
KNOWN_FIXES = {
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
    "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
    "minimum": "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
    "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
    "average": "def average(xs):\n    return sum(xs) / len(xs)\n",
    "first_n": "def first_n(xs, n):\n    return xs[:n]\n",
    "is_palindrome": "def is_palindrome(s):\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n",
    "fib": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
}


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def scripted_reply(messages: list) -> str:
    """Pick a canned answer from the prompt contents."""
    user = ""
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            user = c if isinstance(c, str) else json.dumps(c)
    full = "\n".join(
        (m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")))
        for m in messages
    )

    if "MOCK_INFEASIBLE" in full:
        return "INFEASIBLE"
    if "MOCK_GARBAGE" in full:
        return "I am not sure what to do here, sorry."

    # Coding repair prompt: return fixed versions of the functions named as failing.
    if "mylib" in full and ("FAILING" in full or "failing" in full):
        names = re.findall(r"\b([a-z_][a-z0-9_]*)\b", user)
        out = []
        seen = set()
        for n in names:
            if n in KNOWN_FIXES and n not in seen:
                seen.add(n)
                out.append(KNOWN_FIXES[n])
        if out:
            return "```python\n" + "\n\n".join(out) + "```"
        return "INFEASIBLE"

    # Generic fallback loop: the harness shows tool output and expects ANSWER.
    m = re.search(r"OUTPUT:\s*\n(.*?)\n(?:---|$)", user, re.S)
    if m and m.group(1).strip():
        return "ANSWER: " + m.group(1).strip().splitlines()[-1].strip()

    # Math word problems -> emit a tiny program.
    if "warehouse" in full.lower() and "boxes" in full.lower():
        nums = re.findall(r"starts with (\d+)", full)
        start = nums[0] if nums else "0"
        return "```python\nprint(%s)\n```" % start
    return "```python\nprint(42)\n```"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default logging
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
        elif self.path.rstrip("/").endswith("/_stats"):
            with LOCK:
                self._send(200, {k: STATE[k] for k in ("spent", "requests", "max_parallel")})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send(400, {"error": {"message": "bad json"}})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return

        with LOCK:
            STATE["requests"] += 1
            STATE["active"] += 1
            STATE["max_parallel"] = max(STATE["max_parallel"], STATE["active"])
            n = STATE["requests"]
            fail = n <= STATE["fail_first"]
            exhausted = STATE["budget"] is not None and STATE["spent"] >= STATE["budget"]
            forced = STATE["replies"].pop(0) if STATE["replies"] else None
        try:
            if STATE["log"]:
                with LOCK, open(STATE["log"], "a", encoding="utf-8") as f:
                    f.write(json.dumps({"n": n, "path": self.path, "body": req,
                                        "auth": bool(self.headers.get("Authorization"))},
                                       ensure_ascii=False) + "\n")
            if fail:
                self._send(500, {"error": {"message": "mock transient failure"}})
                return
            if exhausted:
                self._send(429, {"error": {"message": "token budget exhausted"}})
                return

            messages = req.get("messages") or []
            model = req.get("model") or ""
            if model not in MODELS:
                self._send(404, {"error": {"message": "unknown model %r" % model}})
                return
            reply = forced if forced is not None else scripted_reply(messages)
            max_tokens = req.get("max_tokens") or req.get("max_completion_tokens")
            finish = "stop"
            if max_tokens and approx_tokens(reply) > int(max_tokens):
                reply = reply[: int(max_tokens) * 4]
                finish = "length"
            prompt_tokens = sum(approx_tokens(json.dumps(m.get("content", ""))) + 4 for m in messages)
            completion_tokens = approx_tokens(reply)
            reasoning = 0
            message = {"role": "assistant", "content": reply}
            if not model.endswith("-noreason"):
                reasoning = 40
                completion_tokens += reasoning
                message["reasoning_content"] = "mock reasoning " * 10
            with LOCK:
                STATE["spent"] += prompt_tokens / 10.0 + completion_tokens
            self._send(200, {
                "id": "mock-%d" % n,
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "completion_tokens_details": {"reasoning_tokens": reasoning},
                },
            })
        finally:
            with LOCK:
                STATE["active"] -= 1


def serve(port: int = 8765, budget=None, log=None, fail_first: int = 0, replies=None, host: str = "127.0.0.1"):
    STATE.update(spent=0.0, budget=budget, requests=0, fail_first=fail_first, log=log,
                 replies=list(replies or []), max_parallel=0, active=0)
    server = ThreadingHTTPServer((host, port), Handler)
    return server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--fail-first", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 inside a container")
    args = ap.parse_args()
    server = serve(args.port, args.budget, args.log, args.fail_first, host=args.host)
    print("mock LLM proxy on http://%s:%d/v1" % (args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    sys.stdout.flush()


if __name__ == "__main__":
    main()
