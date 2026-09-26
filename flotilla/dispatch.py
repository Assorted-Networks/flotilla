"""Send chat requests to nodes with load balancing and failover."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import httpx

from flotilla import llm
from flotilla.config import ClusterConfig
from flotilla.llm import BackendError, Usage
from flotilla.registry import Lease, NoCandidate, Placement, QueueTimeout, Registry
from flotilla.util import ThinkStreamFilter, estimate_tokens, strip_think

log = logging.getLogger("flotilla.dispatch")


@dataclass
class CallSpec:
    candidates: list[str]
    body: dict[str, Any]                      # OpenAI request body without "model"
    placement: Placement = field(default_factory=Placement)
    timeout: float | None = None
    # Called with (node name, model) each time a node is picked for this call.
    on_assign: Callable[[str, str], None] | None = None


@dataclass
class Attempt:
    node: str
    model: str
    error: str
    kind: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {"node": self.node, "model": self.model, "error": self.error, "kind": self.kind, "seconds": round(self.seconds, 2)}


@dataclass
class CallResult:
    content: str
    reasoning: str
    finish_reason: str | None
    usage: Usage
    node: str
    model: str
    latency: float
    attempts: list[Attempt] = field(default_factory=list)


class DispatchError(Exception):
    def __init__(self, message: str, attempts: list[Attempt] | None = None, status: int = 502, mid_stream: bool = False):
        super().__init__(message)
        self.message = message
        self.attempts = attempts or []
        self.status = status
        self.mid_stream = mid_stream


def _prompt_chars(body: dict) -> int:
    total = 0
    for m in body.get("messages") or []:
        c = m.get("content")
        total += len(c) if isinstance(c, str) else len(str(c or ""))
    return total


def _thinking_rejected(exc: BackendError, body: dict) -> bool:
    """A 400 caused by reasoning_effort (model or server does not support it)."""
    if exc.status not in (400, 422) or "reasoning_effort" not in body:
        return False
    msg = exc.message.lower()
    return "think" in msg or "reasoning" in msg


def _is_node_fault(exc: BackendError) -> bool:
    return exc.status is None or exc.status >= 500 or exc.status in (401, 403, 404, 408, 429)


def _chunk_has_output(chunk: dict) -> bool:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") or delta.get("content") or delta.get("tool_calls"):
            return True
        if delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking"):
            return True
    return False


class Dispatcher:
    def __init__(self, registry: Registry, client: httpx.AsyncClient, cfg: ClusterConfig):
        self.registry = registry
        self.client = client
        self.cfg = cfg
        self.on_call: Callable[[str, str, bool, float, Usage | None], None] | None = None
        # (node, model) pairs whose server rejected `reasoning_effort`; the
        # field is left out for them from then on.
        self.no_effort: set[tuple[str, str]] = set()

    def _body_for(self, lease: Lease, body: dict) -> dict:
        out = {**body, "model": lease.model}
        if "reasoning_effort" in out and (lease.node.name, lease.canonical) in self.no_effort:
            out.pop("reasoning_effort")
        return out

    def _drop_effort(self, lease: Lease) -> None:
        log.info("%s on %s does not accept reasoning_effort; leaving it out", lease.model, lease.node.name)
        self.no_effort.add((lease.node.name, lease.canonical))

    def _timeout(self, spec: CallSpec) -> httpx.Timeout:
        return llm.make_timeout(spec.timeout or self.cfg.request_timeout, self.cfg.connect_timeout)

    def _record(self, node: str, model: str, ok: bool, seconds: float, usage: Usage | None) -> None:
        if self.on_call:
            try:
                self.on_call(node, model, ok, seconds, usage)
            except Exception:  # metrics must never break a request
                log.exception("metrics hook failed")

    async def _acquire(self, spec: CallSpec, exclude: set[str], attempts: list[Attempt]) -> Lease:
        try:
            lease = await self.registry.acquire(spec.candidates, spec.placement, exclude)
            if spec.on_assign:
                try:
                    spec.on_assign(lease.node.name, lease.model)
                except Exception:  # noqa: BLE001 - observers must not break calls
                    log.exception("on_assign hook failed")
            return lease
        except (NoCandidate, QueueTimeout) as exc:
            msg = str(exc)
            if attempts:
                msg += " | earlier attempts: " + "; ".join(f"{a.node}/{a.model}: {a.error}" for a in attempts)
            raise DispatchError(msg, attempts, status=503) from exc

    def _handle_failure(self, lease: Lease, exc: BackendError, t0: float, attempts: list[Attempt], exclude: set[str]) -> None:
        seconds = time.monotonic() - t0
        self.registry.release(lease, ok=False if _is_node_fault(exc) else None, error_kind=exc.kind, error=exc.message)
        attempts.append(Attempt(lease.node.name, lease.model, exc.message, exc.kind, seconds))
        self._record(lease.node.name, lease.model, False, seconds, None)
        log.warning("call to %s on %s failed (%s): %s", lease.model, lease.node.name, exc.kind, exc.message)
        if exc.kind != "not_found":
            # A 404 already removed the model from that node; the node may
            # still serve a fallback model.
            exclude.add(lease.node.name)

    # -- non-streaming --------------------------------------------------------

    async def complete_raw(self, spec: CallSpec) -> tuple[dict, Lease, float, list[Attempt]]:
        """Run a non-streaming request with failover. Returns the raw response."""
        attempts: list[Attempt] = []
        exclude: set[str] = set()
        body = dict(spec.body)
        body.pop("stream", None)
        body.pop("stream_options", None)
        failures = 0
        while True:
            lease = await self._acquire(spec, exclude, attempts)
            t0 = time.monotonic()
            try:
                data = await llm.post_json(
                    self.client, llm.chat_url(lease.node.url), self._body_for(lease, body),
                    lease.node.api_key, self._timeout(spec),
                )
            except BackendError as exc:
                if _thinking_rejected(exc, self._body_for(lease, body)):
                    self.registry.release(lease, ok=None)
                    self._drop_effort(lease)
                    continue
                self._handle_failure(lease, exc, t0, attempts, exclude)
                failures += 1
                if not exc.retryable:
                    raise DispatchError(exc.message, attempts, status=exc.status or 502) from exc
                if failures > self.cfg.max_retries:
                    raise DispatchError(f"gave up after {failures} failed attempts: {exc.message}", attempts) from exc
                continue
            except BaseException:
                # Cancelled (client went away) or an unexpected error: free the slot.
                self.registry.release(lease, ok=None)
                raise
            seconds = time.monotonic() - t0
            usage = Usage.from_openai(data.get("usage")) if isinstance(data, dict) else None
            self.registry.release(lease, ok=True, completion_tokens=usage.completion_tokens if usage else 0)
            self._record(lease.node.name, lease.model, True, seconds, usage)
            return data, lease, seconds, attempts

    async def complete(self, spec: CallSpec) -> CallResult:
        data, lease, seconds, attempts = await self.complete_raw(spec)
        comp = llm.parse_completion(data, _prompt_chars(spec.body))
        return CallResult(
            content=comp.content,
            reasoning=comp.reasoning,
            finish_reason=comp.finish_reason,
            usage=comp.usage,
            node=lease.node.name,
            model=lease.model,
            latency=seconds,
            attempts=attempts,
        )

    # -- streaming ------------------------------------------------------------

    async def stream_raw(self, spec: CallSpec) -> AsyncIterator[tuple[str, Any]]:
        """Stream a request with failover until the first output token.

        Yields ("start", lease) once a node is producing output, then
        ("chunk", dict) for every chunk, then ("end", (lease, seconds, attempts)).
        Chunks received before the first real output are held back so a
        failure at that point can be retried on another node invisibly.
        """
        attempts: list[Attempt] = []
        exclude: set[str] = set()
        body = dict(spec.body)
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})
        failures = 0
        while True:
            lease = await self._acquire(spec, exclude, attempts)
            t0 = time.monotonic()
            started = False
            held: list[dict] = []
            completion_tokens = 0
            finished = False
            try:
                async for chunk in llm.stream_chat(
                    self.client, lease.node.url, self._body_for(lease, body), lease.node.api_key, self._timeout(spec)
                ):
                    usage = Usage.from_openai(chunk.get("usage"))
                    if usage:
                        completion_tokens = usage.completion_tokens
                    if not started:
                        held.append(chunk)
                        if not _chunk_has_output(chunk):
                            continue
                        started = True
                        yield "start", lease
                        for h in held:
                            yield "chunk", h
                        held = []
                        continue
                    yield "chunk", chunk
                if not started:
                    started = True
                    yield "start", lease
                    for h in held:
                        yield "chunk", h
                finished = True
            except BackendError as exc:
                if started:
                    self._handle_failure(lease, exc, t0, attempts, exclude)
                    raise DispatchError(f"{lease.node.name} failed mid-stream: {exc.message}", attempts, mid_stream=True) from exc
                if _thinking_rejected(exc, self._body_for(lease, body)):
                    self.registry.release(lease, ok=None)
                    self._drop_effort(lease)
                    continue
                self._handle_failure(lease, exc, t0, attempts, exclude)
                failures += 1
                if not exc.retryable:
                    raise DispatchError(exc.message, attempts, status=exc.status or 502) from exc
                if failures > self.cfg.max_retries:
                    raise DispatchError(f"gave up after {failures} failed attempts: {exc.message}", attempts) from exc
                continue
            finally:
                if not finished and not lease.released:
                    # Cancelled or the consumer stopped reading.
                    self.registry.release(lease, ok=None)
            seconds = time.monotonic() - t0
            self.registry.release(lease, ok=True, completion_tokens=completion_tokens)
            self._record(lease.node.name, lease.model, True, seconds, Usage(completion_tokens=completion_tokens))
            yield "end", (lease, seconds, attempts)
            return

    async def stream(self, spec: CallSpec) -> AsyncIterator[tuple[str, Any]]:
        """Stream a request as ("delta", (content, reasoning)) events and a
        final ("done", CallResult). Inline <think> blocks go to reasoning."""
        think = ThinkStreamFilter()
        content: list[str] = []
        reasoning: list[str] = []
        usage: Usage | None = None
        finish: str | None = None
        inner = self.stream_raw(spec)
        try:
            async for kind, payload in inner:
                if kind == "chunk":
                    c, r, fin, u = llm.parse_chunk(payload)
                    if u and u.total_tokens:
                        usage = u
                    if fin:
                        finish = fin
                    c2, r2 = think.feed(c) if c else ("", "")
                    r_all = r + r2
                    if c2 or r_all:
                        content.append(c2)
                        reasoning.append(r_all)
                        yield "delta", (c2, r_all)
                elif kind == "end":
                    tail_c, tail_r = think.flush()
                    if tail_c or tail_r:
                        content.append(tail_c)
                        reasoning.append(tail_r)
                        yield "delta", (tail_c, tail_r)
                    lease, seconds, attempts = payload
                    text = "".join(content)
                    final_text, extra_reasoning = strip_think(text)
                    reasoning_text = "".join(reasoning)
                    if extra_reasoning:
                        reasoning_text += "\n" + extra_reasoning
                    if usage is None:
                        usage = Usage(
                            prompt_tokens=_prompt_chars(spec.body) // 4,
                            completion_tokens=estimate_tokens(text) + estimate_tokens(reasoning_text),
                            estimated=True,
                        )
                    yield "done", CallResult(
                        content=final_text,
                        reasoning=reasoning_text.strip(),
                        finish_reason=finish,
                        usage=usage,
                        node=lease.node.name,
                        model=lease.model,
                        latency=seconds,
                        attempts=attempts,
                    )
        finally:
            await inner.aclose()

    # -- embeddings -----------------------------------------------------------

    async def embeddings(self, spec: CallSpec) -> dict:
        attempts: list[Attempt] = []
        exclude: set[str] = set()
        failures = 0
        while True:
            lease = await self._acquire(spec, exclude, attempts)
            t0 = time.monotonic()
            try:
                data = await llm.post_json(
                    self.client, llm.api_base(lease.node.url) + "/embeddings",
                    {**spec.body, "model": lease.model}, lease.node.api_key, self._timeout(spec),
                )
            except BackendError as exc:
                self._handle_failure(lease, exc, t0, attempts, exclude)
                failures += 1
                if not exc.retryable or failures > self.cfg.max_retries:
                    raise DispatchError(exc.message, attempts, status=exc.status or 502) from exc
                continue
            except BaseException:
                self.registry.release(lease, ok=None)
                raise
            self.registry.release(lease, ok=True)
            self._record(lease.node.name, lease.model, True, time.monotonic() - t0, None)
            return data
