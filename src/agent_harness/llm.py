"""LLM access.

Primary transport: the openJiuwen model layer
(``openjiuwen.core.foundation.llm.init_model`` -> ``Model.invoke``), verified
against openjiuwen==0.1.19:

    model = init_model("OpenAI", model_name, api_key, api_base,
                       temperature=0.0, timeout=..., verify_ssl=True)
    msg = await model.invoke(messages, model=..., max_tokens=..., temperature=0.0)
    msg.content, msg.finish_reason, msg.reasoning_content
    msg.usage_metadata.input_tokens / .output_tokens / .reasoning_tokens

Second transport: a plain standard-library HTTP call to the same
OpenAI-compatible ``/chat/completions`` endpoint. It is used only when the
openJiuwen *client side* is broken (import problem, unexpected response shape)
or cannot connect at all. An HTTP error status returned by the proxy is not a
transport problem: the request is then retried on the same transport, it is
not sent a second time over the other one.

Error policy (see ``classify``):

* 401 / 403 - the key is rejected: LLM work stops at once (no retries, they
  would only burn the limited number of authorisation attempts);
* 400 / 404 / 422 ... - this request is wrong: it is not repeated, other
  requests go on;
* 429 / 5xx / network errors - temporary: bounded retries with exponential
  back-off and jitter, ``Retry-After`` is honoured, the delay is shared by all
  parallel sessions and never runs past the deadline of the run. After a
  logical call has failed completely the client cools down and then recovers;
  it is not switched off for the rest of the run. A 429 is a rate limit unless
  it says unambiguously that the total budget is exhausted and carries nothing
  that points to a temporary limit (no Retry-After, no "rate limit" / "per
  minute" wording); only such a message (or 402) seen twice in a row stops the
  LLM work;
* time-out or a lost answer after the request was sent - the proxy may have
  generated (and billed) the completion: the reserved estimate is booked as
  *uncertain* spend and the request is repeated at most once.

Every attempt, including a retry, is admitted separately against the per-task
cap and the global soft limit and settles its own reservation; a retry that no
longer fits the budget is not sent.

TLS certificates are always verified against the system / certifi CA bundle;
there is no fallback to an unverified connection.

Both transports talk exclusively to ``LLM_BASE_URL``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from .budget import CallRecord, Ledger, estimate_tokens, weighted_cost
from .config import Settings

log = logging.getLogger("harness.llm")


class LLMError(Exception):
    """Base class for LLM related problems (never fatal for the run)."""


class BudgetExceeded(LLMError):
    """The call would exceed the global soft limit or the per-task cap."""


class LLMUnavailable(LLMError):
    """The proxy did not answer (down, budget exhausted, time is over...)."""


class ProxyHTTPError(Exception):
    """An HTTP error status of the proxy, with the (short) body it sent."""

    def __init__(self, status_code: int, headers, body: str):
        super().__init__("HTTP Error %d: %s" % (status_code, body[:300]))
        self.status_code = status_code
        self.headers = headers


@dataclass
class Reply:
    text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost: float
    model: str
    transport: str
    # None: reasoning tokens are part of output_tokens (OpenAI semantics, or none reported);
    # True: reported separately; "unknown": cannot be told -> counted on top (conservative).
    reasoning_separate: object = None
    usage_note: str = ""


@dataclass
class Failure:
    kind: str                 # auth | budget | request | rate | server | network | timeout | client
    status: Optional[int]
    retry_after: Optional[float]
    text: str
    maybe_paid: bool          # the proxy may have produced (and billed) a completion


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
# An unambiguous statement that the *total* allowance is used up. Rate limits
# ("limit exceeded", "tokens per minute quota") are deliberately not matched.
_BUDGET_WORDS = re.compile(
    r"insufficient[_ ](?:quota|balance|funds|credits?|budget)"
    r"|budget (?:is |has been |was )?(?:exhausted|exceeded|spent|used up|depleted)"
    r"|(?:exhausted|exceeded|spent|used up) (?:the |your |its |total |token )*budget"
    r"|budget limit (?:reached|exceeded)"
    r"|out of (?:tokens|budget|credits?)"
    r"|(?:exceeded|exhausted) your (?:current )?quota"
    r"|no (?:remaining |more )?(?:budget|credits?)(?: left)?\b", re.I)
# Signs of a temporary limit. If any is present the refusal is treated as
# temporary even when budget-like words appear next to it.
_RATE_WORDS = re.compile(
    r"rate.?limit|per[ -](?:second|minute|hour|day)|\b[rt]pm\b|\b[rt]pd\b|retry|try again"
    r"|slow down|temporar|throttl|overload|concurren|please wait", re.I)


def clean_text(text: Optional[str]) -> str:
    """Drop inline reasoning blocks some servers leave in ``content``."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # unterminated opening tag variant
        text = text.split("</think>", 1)[1]
    return text.strip()


def _chain(exc: BaseException):
    seen = []
    cur: Optional[BaseException] = exc
    while cur is not None and cur not in seen and len(seen) < 8:
        seen.append(cur)
        yield cur
        cur = cur.__cause__ or (None if cur.__suppress_context__ else cur.__context__)


def _retry_after(headers) -> Optional[float]:
    try:
        v = headers.get("Retry-After") if headers is not None else None
        if v is None:
            return None
        return max(0.0, float(str(v).strip()))
    except Exception:
        return None  # HTTP-date form or garbage: fall back to our own back-off


def classify(exc: BaseException) -> Failure:
    """Turn an exception of either transport into a retry decision."""
    status: Optional[int] = None
    retry_after: Optional[float] = None
    names, texts = [], []
    for e in _chain(exc):
        names.append(type(e).__name__)
        texts.append(str(e)[:400])
        if status is None:
            for attr in ("status_code", "code", "status"):
                v = getattr(e, attr, None)
                if isinstance(v, int) and 100 <= v <= 599:
                    status = v
                    break
        if retry_after is None:
            retry_after = _retry_after(getattr(e, "headers", None))
            if retry_after is None:
                retry_after = _retry_after(getattr(getattr(e, "response", None), "headers", None))
    text = "%s: %s" % (names[0], " | ".join(t for t in texts if t)[:300])
    blob = " ".join(names + texts)
    if status is None:
        m = re.search(r"(?:Error code|HTTP Error|status(?: code)?)[:= ]+(\d{3})\b", blob)
        if m:
            status = int(m.group(1))
    if status is not None:
        # Exhaustion of the total budget must be stated unambiguously: 402, or an
        # explicit budget message with nothing that points to a temporary limit
        # (no Retry-After, no rate-limit wording). Everything else on 429 is a
        # rate limit and is retried; the contract of the proxy is not guessed.
        # Only what the proxy said counts: class names such as "RateLimitError" (every
        # 429 of the OpenAI client) and the standard reason phrase carry no information.
        said = re.sub(r"\b\w+(?:Error|Exception)\b|too many requests", " ", " ".join(texts), flags=re.I)
        if status == 402 or (status in (403, 429) and retry_after is None and _BUDGET_WORDS.search(said)
                             and not _RATE_WORDS.search(said)):
            return Failure("budget", status, retry_after, text, False)
        if status in (401, 403):
            return Failure("auth", status, retry_after, text, False)
        if status in (408, 409, 425, 429):
            return Failure("rate", status, retry_after, text, False)
        if status >= 500:
            return Failure("server", status, retry_after, text, status in (502, 504))
        return Failure("request", status, retry_after, text, False)
    low = blob.lower()
    if any(isinstance(e, (asyncio.TimeoutError, TimeoutError)) for e in _chain(exc)) or "timeout" in low \
            or "timed out" in low:
        return Failure("timeout", None, None, text, True)
    if any(isinstance(e, (urllib.error.URLError, ConnectionError, OSError)) for e in _chain(exc)) \
            or re.search(r"connect|name resolution|getaddrinfo|unreachable|refused|reset by peer|ssl|certificate", low):
        # nothing was accepted by the proxy, except when the connection died mid-answer
        return Failure("network", None, None, text, bool(re.search(r"reset|incomplete|closed|eof", low)))
    return Failure("client", None, None, text, True)


class LLMClient:
    MAX_ATTEMPTS = 4          # per logical call, temporary errors only
    BACKOFF_BASE_S = 2.0
    BACKOFF_CAP_S = 60.0
    RETRY_AFTER_CAP_S = 90.0
    COOLDOWN_CAP_S = 240.0

    def __init__(self, settings: Settings, ledger: Ledger, trace=None, deadline: Optional[float] = None):
        self.s = settings
        self.ledger = ledger
        self.trace = trace
        self.deadline = deadline  # time.monotonic() value
        self.dead = False         # permanent stop: rejected key or confirmed budget exhaustion
        self.dead_reason = ""
        self._consecutive_failures = 0
        self._budget_signals = 0
        self._gate_until = 0.0    # shared back-off: no session sends before this moment
        self._oj_models = {}
        self._oj_ok: Optional[bool] = None  # None = not tried yet
        self._oj_failures = 0
        self.transport_counts = {"openjiuwen": 0, "http": 0}
        self.retries = 0
        # Tasks of the paid queue that have not had their first attempt yet
        # (maintained by the pipeline); they keep their share of the free budget.
        self.waiting_tasks = 0
        self._ssl_context: Optional[ssl.SSLContext] = None

    # ------------------------------------------------------------------ utils
    @property
    def available(self) -> bool:
        if not self.s.llm_configured or self.dead:
            return False
        return self._time_left() > max(0.0, self._gate_until - time.monotonic()) + 25

    def _time_left(self) -> float:
        if self.deadline is None:
            return 1e9
        return self.deadline - time.monotonic()

    def _chat_url(self) -> str:
        return self.s.base_url.rstrip("/") + "/chat/completions"

    def _stop(self, reason: str) -> None:
        if not self.dead:
            self.dead, self.dead_reason = True, reason
            log.error("LLM work stopped for this run: %s", reason)

    def _hold(self, seconds: float) -> None:
        self._gate_until = max(self._gate_until, time.monotonic() + max(0.0, seconds))

    async def _wait_gate(self) -> None:
        waited = False
        while True:
            wait = self._gate_until - time.monotonic()
            if wait <= 0:
                break
            if self._time_left() < wait + 25:
                raise LLMUnavailable("the deadline would pass during the back-off")
            waited = True
            await asyncio.sleep(min(wait, 2.0))
        if waited:
            await asyncio.sleep(random.uniform(0.0, 1.0))  # do not release all sessions in the same instant

    def estimate_call(self, messages: List[dict], max_tokens: int) -> float:
        prompt_text = "".join(str(m.get("content", "")) for m in messages)
        return (estimate_tokens(prompt_text) + 8 * len(messages)) * 0.1 + max_tokens

    def task_cap(self, task_id: str) -> float:
        """What this task may have consumed in total after its next call.

        An explicit HARNESS_TASK_CAP is a hard cap. Otherwise the configured
        value (1800) is the floor, and the task may go on beyond it as long as
        it stays within its equal share of the budget that is free right now,
        the other shares belonging to the queued tasks that are still waiting
        for their first attempt. So one hard task cannot take what the others
        need for a first try, but the last promising task is not stopped at
        1800 while thousands of tokens are free. The global soft limit and the
        three logical attempts per task stay in force either way.
        """
        if self.s.task_cap_explicit:
            return self.s.task_token_cap
        spent = self.ledger.task_spent(task_id)
        share = spent + max(0.0, self.ledger.remaining()) / (max(0, self.waiting_tasks) + 1)
        return max(self.s.task_token_cap, share)

    def admission(self, task_id: str, messages: List[dict], wanted: int, minimum: int = 64) -> dict:
        """Can a call with up to ``wanted`` output tokens be made now, and if not - why exactly."""
        fixed = self.estimate_call(messages, 0)
        spent = self.ledger.task_spent(task_id)
        cap = self.task_cap(task_id)
        task_room = cap - spent - fixed
        global_free = self.ledger.remaining()
        global_room = global_free - fixed
        allowed = int(min(wanted, task_room, global_room))
        limit = ""
        if allowed < minimum:
            allowed = 0
            limit = "global budget" if global_room < task_room else "per-task cap"
        return {"allowed": allowed, "limit": limit, "wanted": int(wanted), "minimum": int(minimum),
                "input_cost": round(fixed, 1), "task_spent": round(spent, 1), "task_cap": round(cap, 1),
                "task_cap_explicit": bool(self.s.task_cap_explicit), "global_free": round(global_free, 1),
                "reserved": round(self.ledger.reserved, 1), "waiting_tasks": self.waiting_tasks}

    def affordable_max_tokens(self, task_id: str, messages: List[dict], wanted: int, minimum: int = 64) -> int:
        """Largest ``max_tokens`` <= ``wanted`` that fits the per-task cap and the global soft limit (0 = none)."""
        return self.admission(task_id, messages, wanted, minimum)["allowed"]

    # ----------------------------------------------------------- openJiuwen
    def _get_oj_model(self, model: str):
        m = self._oj_models.get(model)
        if m is None:
            from openjiuwen.core.foundation.llm import init_model  # heavy import, done lazily

            _quiet_openjiuwen_logs()
            m = init_model("OpenAI", model, self.s.api_key, self.s.base_url, temperature=0.0,
                           timeout=self.s.llm_timeout_s, verify_ssl=True)
            self._oj_models[model] = m
        return m

    async def _call_openjiuwen(self, messages: List[dict], model: str, max_tokens: int, timeout: float) -> Reply:
        m = self._get_oj_model(model)
        msg = await m.invoke(messages, model=model, max_tokens=max_tokens, temperature=0.0, timeout=timeout)
        usage = getattr(msg, "usage_metadata", None)
        content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
        i = int(getattr(usage, "input_tokens", 0) or 0)
        o = int(getattr(usage, "output_tokens", 0) or 0)
        r = int(getattr(usage, "reasoning_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0)
        return Reply(clean_text(content), str(getattr(msg, "finish_reason", "") or ""), i, o, r, 0.0, model,
                     "openjiuwen", reasoning_separate=_reasoning_semantics(i, o, r, total, None))

    # ------------------------------------------------------------ plain HTTP
    def _ssl(self) -> ssl.SSLContext:
        if self._ssl_context is None:
            ctx = ssl.create_default_context()  # verifies the certificate and the host name
            try:
                import certifi

                ctx.load_verify_locations(certifi.where())
            except Exception:
                pass  # the system CA store is used
            self._ssl_context = ctx
        return self._ssl_context

    def _call_http_sync(self, messages: List[dict], model: str, max_tokens: int, timeout: float) -> Reply:
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url(),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.s.api_key},
        )
        ctx = self._ssl() if self._chat_url().lower().startswith("https") else None
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read(2000).decode("utf-8", "replace")
            except Exception:
                detail = ""
            raise ProxyHTTPError(e.code, e.headers, detail or str(e.reason)) from None
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        i = int(usage.get("prompt_tokens") or 0)
        o = int(usage.get("completion_tokens") or 0)
        details = usage.get("completion_tokens_details") or {}
        nested = details.get("reasoning_tokens")
        r = int(nested or usage.get("reasoning_tokens") or 0)
        total = int(usage.get("total_tokens") or 0)
        return Reply(clean_text(message.get("content")), str(choice.get("finish_reason") or ""), i, o, r, 0.0,
                     model, "http", reasoning_separate=_reasoning_semantics(i, o, r, total, nested is not None))

    async def _send(self, transport: str, messages: List[dict], model: str, max_tokens: int, timeout: float) -> Reply:
        if transport == "openjiuwen":
            return await asyncio.wait_for(self._call_openjiuwen(messages, model, max_tokens, timeout), timeout + 15)
        return await asyncio.wait_for(
            asyncio.to_thread(self._call_http_sync, messages, model, max_tokens, timeout), timeout + 15)

    # ------------------------------------------------------------------ chat
    async def chat(self, task_id: str, messages: List[dict], *, model: str, max_tokens: int,
                   purpose: str = "") -> Reply:
        """One chat completion. Raises ``BudgetExceeded`` or ``LLMUnavailable``."""
        if not self.s.llm_configured:
            raise LLMUnavailable("LLM proxy is not configured")
        if self.dead:
            raise LLMUnavailable("LLM stopped: " + self.dead_reason)
        if self._time_left() < 20:
            raise LLMUnavailable("deadline reached")

        prompt_text = "".join(str(m.get("content", "")) for m in messages)
        est_in = estimate_tokens(prompt_text) + 8 * len(messages)
        est = est_in * 0.1 + max_tokens

        # Every attempt that may be billed has its own life cycle:
        #   admit (per-task cap + atomic reservation in the ledger)
        #   -> send
        #   -> settle: the reservation is dropped and exactly one cost is booked -
        #      the reported usage, or the reserved estimate when the answer was lost
        #      after sending ("uncertain spend"); an attempt the proxy certainly did
        #      not serve just releases its reservation.
        # A retry is a new attempt and must be admitted again, whatever the transport.
        reply: Optional[Reply] = None
        errors: List[str] = []
        attempts = 0
        timeouts = 0
        use_http_next = False
        denied = ""          # why an attempt was not admitted
        holding = False      # a reservation of ``est`` is currently held
        try:
            while attempts < self.MAX_ATTEMPTS:
                await self._wait_gate()
                if self.dead:
                    break
                left = self._time_left()
                if left < 20:
                    errors.append("deadline reached")
                    break
                if self.ledger.task_spent(task_id) + est > self.task_cap(task_id):
                    denied = "per-task cap"
                    break
                if not self.ledger.try_reserve(est):
                    denied = "global soft limit"
                    break
                holding = True
                timeout = max(10.0, min(self.s.llm_timeout_s, left - 10))
                transport = "http" if (self._oj_ok is False or use_http_next) else "openjiuwen"
                use_http_next = False
                attempts += 1
                try:
                    reply = await self._send(transport, messages, model, max_tokens, timeout)
                    if transport == "openjiuwen":
                        self._oj_ok = True
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    f = classify(e)
                    errors.append("%s -> %s [%s]" % (transport, f.text, f.kind))
                    holding = False
                    if f.maybe_paid:
                        self._settle_uncertain(task_id, model, purpose, est, transport, f)
                    else:
                        self.ledger.release(est)
                    if f.kind == "auth":
                        self._stop("the proxy rejected the key (HTTP %s)" % f.status)
                        break
                    if f.kind == "request":
                        break  # this request is wrong; repeating it cannot help
                    if f.kind == "budget":
                        self._budget_signals += 1
                        if self._budget_signals >= 2:
                            self._stop("the proxy reports the token budget as exhausted")
                            break
                        self._hold(max(30.0, f.retry_after or 0.0))
                        continue
                    if f.kind == "client":
                        # openJiuwen itself failed (not the proxy): use the plain client next
                        if transport == "openjiuwen":
                            self._oj_failures += 1
                            use_http_next = True
                            if self._oj_failures >= 2 and self._oj_ok is not True:
                                self._oj_ok = False
                                log.warning("openJiuwen transport disabled, using plain HTTP")
                            continue
                        break
                    if f.kind == "timeout":
                        timeouts += 1
                        if timeouts >= 2:
                            break  # do not keep paying for answers that never arrive
                    if f.kind == "network" and transport == "openjiuwen":
                        use_http_next = True  # nothing was billed; let the other client try to connect
                    delay = min(self.BACKOFF_CAP_S, self.BACKOFF_BASE_S * (2 ** (attempts - 1)))
                    delay *= random.uniform(0.5, 1.0)
                    if f.retry_after is not None:
                        delay = max(delay, min(self.RETRY_AFTER_CAP_S, f.retry_after))
                    if attempts >= self.MAX_ATTEMPTS or self._time_left() < delay + 30:
                        break
                    self.retries += 1
                    self._hold(delay)
            if reply is not None:
                self._finalise_usage(reply, est_in)
                holding = False
                self.ledger.settle(est, CallRecord(task_id, model, purpose, reply.input_tokens, reply.output_tokens,
                                                   reply.reasoning_tokens, reply.cost, reply.transport,
                                                   reply.usage_note))
        except asyncio.CancelledError:
            if holding:  # cancelled with a request on the wire: it may still be served and billed
                holding = False
                self._settle_uncertain(task_id, model, purpose, est, "cancelled",
                                       Failure("timeout", None, None, "cancelled while waiting for the answer", True))
            raise
        finally:
            if holding:
                self.ledger.release(est)

        if reply is None:
            if errors:
                self._consecutive_failures += 1
                if not self.dead and not denied:
                    # cool down and recover later instead of switching the LLM off for good
                    self._hold(min(self.COOLDOWN_CAP_S, 15.0 * (2 ** (self._consecutive_failures - 1))))
            if self.trace and (errors or attempts):
                self.trace.event(task_id, "llm_error", purpose=purpose, model=model, errors=errors,
                                 attempts=attempts, stopped=self.dead, denied=denied)
            if denied:
                # first attempt: plain refusal; later: the retry itself is not affordable any more
                raise BudgetExceeded(denied if not attempts else "%s: no budget left to retry after %s" % (
                    denied, errors[-1][:120] if errors else "a failed attempt"))
            raise LLMUnavailable("; ".join(errors) or "no response")

        self._consecutive_failures = 0
        self._budget_signals = 0
        self.transport_counts[reply.transport] = self.transport_counts.get(reply.transport, 0) + 1
        if self.trace:
            self.trace.event(
                task_id, "llm_call", purpose=purpose, model=model, transport=reply.transport,
                input_tokens=reply.input_tokens, output_tokens=reply.output_tokens,
                reasoning_tokens=reply.reasoning_tokens, cost=round(reply.cost, 1),
                estimated_cost=round(est, 1), finish_reason=reply.finish_reason, attempts=attempts,
                usage_note=reply.usage_note, messages=messages, reply=reply.text,
            )
        return reply

    # --------------------------------------------------------------- accounting
    def _finalise_usage(self, reply: Reply, est_in: int) -> None:
        """Fill in missing usage conservatively and compute the weighted cost."""
        notes = []
        if reply.input_tokens <= 0:
            reply.input_tokens = est_in
            notes.append("input tokens not reported: estimated")
        if reply.output_tokens <= 0:
            reply.output_tokens = estimate_tokens(reply.text) if reply.text else 1
            notes.append("output tokens not reported: estimated")
        separate = reply.reasoning_separate
        if reply.reasoning_tokens > 0 and separate == "unknown":
            notes.append("reasoning token semantics unknown: counted on top of output")
        reply.cost = weighted_cost(reply.input_tokens, reply.output_tokens, reply.reasoning_tokens,
                                   reasoning_separate=bool(separate))
        reply.usage_note = "; ".join(notes)

    def _settle_uncertain(self, task_id: str, model: str, purpose: str, est: float, transport: str,
                          f: Failure) -> None:
        """End an attempt that may have been served and billed although no answer arrived:
        its reservation becomes booked spend (once)."""
        self.ledger.settle(est, CallRecord(
            task_id, model, purpose, 0, 0, 0, est, transport,
            "uncertain spend: no answer received (%s), reserved estimate booked" % f.kind))
        if self.trace:
            self.trace.event(task_id, "llm_uncertain_spend", purpose=purpose, model=model, booked=round(est, 1),
                             reason=f.text[:200])


def _reasoning_semantics(i: int, o: int, r: int, total: int, nested: Optional[bool]):
    """Are ``r`` reasoning tokens part of ``o``? None = yes / not applicable, True = separate, "unknown"."""
    if r <= 0:
        return None
    if nested:  # OpenAI ``completion_tokens_details``: a breakdown of completion_tokens
        return None
    if total > 0:
        if total == i + o:
            return None
        if total == i + o + r:
            return True
    return "unknown"


_QUIET_DONE = False


def _quiet_openjiuwen_logs() -> None:
    """openJiuwen logs every prompt and response at INFO level; keep the console readable."""
    global _QUIET_DONE
    if _QUIET_DONE:
        return
    _QUIET_DONE = True
    try:
        import openjiuwen.core.common.logging as oj_logging

        # Its loggers are created lazily; force them into existence at ERROR level.
        for lazy in list(getattr(oj_logging, "_lazy_loggers", [])):
            try:
                lazy.set_level(logging.ERROR)
            except Exception:
                pass
    except Exception:
        pass
    for name, lg in list(logging.root.manager.loggerDict.items()):
        if not isinstance(lg, logging.Logger) or name.startswith("harness"):
            continue
        if any(type(h).__name__ == "SafeRotatingFileHandler" for h in lg.handlers):
            lg.setLevel(logging.ERROR)
            lg.propagate = False


def preload_openjiuwen() -> bool:
    """Import openJiuwen up-front (it takes a few seconds). Returns success."""
    try:
        import openjiuwen.core.foundation.llm  # noqa: F401

        _quiet_openjiuwen_logs()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("openJiuwen import failed (%s: %s); plain HTTP transport will be used",
                    type(e).__name__, str(e)[:200])
        return False
