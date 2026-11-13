#!/usr/bin/env python3
"""Fault-injection check of the LLM client with the REAL installed openJiuwen (development only).

The unit regressions use the plain HTTP transport. This script repeats the key
scenarios through ``openjiuwen.core.foundation.llm`` as installed in the image,
against a local scripted server (no internet, no real key):

    docker run --rm --network none --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
        --cap-drop ALL --security-opt no-new-privileges -v "$PWD/tests:/t:ro" \
        --workdir /tmp --entrypoint python <image> /t/docker_llm_faults.py

Prints one JSON line per scenario and exits non-zero if an expectation fails.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/app/src" if os.path.isdir("/app/src") else os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)
os.chdir(os.environ.get("TMPDIR") or "/tmp" if os.name == "posix" else os.getcwd())

from test_regressions import FastClient, FaultServer  # noqa: E402

from agent_harness.budget import Ledger  # noqa: E402
from agent_harness.config import Settings  # noqa: E402
from agent_harness.llm import LLMUnavailable, preload_openjiuwen  # noqa: E402

FAILED = []


def emit(name: str, ok: bool, **data) -> None:
    print("FAULT_CHECK " + json.dumps(dict(scenario=name, ok=ok, **data)), flush=True)
    if not ok:
        FAILED.append(name)


def make(script, **kw):
    srv = FaultServer(script)
    ledger = Ledger(10000)
    client = FastClient(Settings(base_url=srv.url, api_key="local-test", llm_timeout_s=10, enable_llm=True, **kw),
                        ledger)
    return srv, ledger, client


async def call(client, task="t"):
    return await client.chat(task, [{"role": "user", "content": "hello"}], model=client.s.model_fast, max_tokens=40)


async def main() -> None:
    oj = preload_openjiuwen()
    emit("openjiuwen_import", oj)

    for code in (429, 500):
        srv, led, c = make([{"status": code}, {"status": 200}])
        r = await call(c)
        emit("%d_then_success" % code, r.transport == "openjiuwen" and srv.requests == 2 and not c.dead,
             transport=r.transport, http_requests=srv.requests, cost=r.cost)
        srv.close()

    srv, led, c = make([{"status": 500, "headers": {"Retry-After": "1"}}, {"status": 200}])
    t0 = time.monotonic()
    r = await call(c)
    waited = time.monotonic() - t0
    emit("retry_after_honoured", waited >= 0.9 and srv.requests == 2, waited=round(waited, 2))
    srv.close()

    srv, led, c = make([{"status": 503}] * 12 + [{"status": 200}])
    failures = 0
    for _ in range(3):
        try:
            await call(c)
        except LLMUnavailable:
            failures += 1
    after = srv.requests
    r = await call(c)
    emit("persistent_errors_bounded_then_recovery",
         failures == 3 and after == 3 * c.MAX_ATTEMPTS and not c.dead and r.text == "ANSWER: 42"
         and led.reserved == 0.0,
         logical_failures=failures, http_requests_during_failures=after, dead=c.dead,
         transports=c.transport_counts, note="HTTP errors are retried on the same transport, not duplicated")
    srv.close()

    srv, led, c = make([{"status": 401}, {"status": 200}])
    try:
        await call(c)
        stopped = False
    except LLMUnavailable:
        stopped = True
    emit("auth_failure_stops_without_retry", stopped and c.dead and srv.requests == 1, http_requests=srv.requests)
    srv.close()

    srv, led, c = make([{"status": 200, "usage": None}])
    r = await call(c)
    emit("missing_usage_is_charged", r.cost > 0 and led.spent > 0 and "estimated" in r.usage_note,
         cost=r.cost, note=r.usage_note, transport=r.transport)
    srv.close()

    nested = {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200,
              "completion_tokens_details": {"reasoning_tokens": 20}}
    srv, led, c = make([{"status": 200, "usage": nested}])
    r = await call(c)
    emit("openjiuwen_usage_mapping", r.input_tokens == 100 and r.output_tokens == 100 and r.cost in (110.0, 130.0),
         input=r.input_tokens, output=r.output_tokens, reasoning=r.reasoning_tokens, cost=r.cost,
         usage_note=r.usage_note,
         note="110 = recognised as included; 130 = counted on top because openJiuwen hides the usage layout")
    srv.close()

    # openJiuwen cannot be imported / is broken on the client side -> plain HTTP is used instead.
    srv, led, c = make([{"status": 200}])
    c._oj_ok = False
    r = await call(c)
    emit("plain_http_fallback_transport", r.transport == "http" and r.text == "ANSWER: 42", transport=r.transport)
    srv.close()

    srv, led, c = make([{"sleep": 3}, {"sleep": 3}, {"status": 200}])

    async def short(transport, messages, model, max_tokens, timeout):
        if transport == "openjiuwen":
            return await asyncio.wait_for(c._call_openjiuwen(messages, model, max_tokens, 1.0), 1.5)
        return await asyncio.wait_for(asyncio.to_thread(c._call_http_sync, messages, model, max_tokens, 1.0), 1.5)

    c._send = short
    try:
        await call(c)
        timed = False
    except LLMUnavailable:
        timed = True
    notes = [x.note for x in led.calls]
    emit("timeout_after_send_books_uncertain_spend", timed and len(notes) == 2 and led.spent > 0
         and srv.requests == 2, http_requests=srv.requests, booked=led.spent)
    srv.close()


asyncio.run(main())
print("FAULT_CHECK_SUMMARY failed=%s" % (FAILED or "none"), flush=True)
sys.exit(1 if FAILED else 0)
