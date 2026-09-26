"""Execution context shared by every step of one task."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from flotilla.config import FlotillaConfig, MemberRef, Params, resolve_member
from flotilla.dispatch import CallSpec, DispatchError, Dispatcher
from flotilla.llm import Usage
from flotilla.registry import Placement
from flotilla.tasks import StepRecord, TaskRecord
from flotilla.util import new_id, now, prepend_system, truncate

# Receives streamed content of the final step (the answer the user sees).
Sink = Callable[[str], Awaitable[None]]


class StepFailed(Exception):
    def __init__(self, message: str, step: StepRecord | None = None):
        super().__init__(message)
        self.step = step


class TaskTimeout(Exception):
    pass


@dataclass
class StepOutput:
    content: str
    reasoning: str
    model: str | None
    node: str | None
    usage: Usage
    seconds: float
    step_id: str


class RunContext:
    def __init__(
        self,
        cfg: FlotillaConfig,
        dispatcher: Dispatcher,
        task: TaskRecord,
        deadline: float,
        depth: int = 0,
        parent_step: str | None = None,
        final_overrides: dict[str, Any] | None = None,
    ):
        self.cfg = cfg
        self.dispatcher = dispatcher
        self.task = task
        self.deadline = deadline               # time.monotonic() based
        self.depth = depth
        self.parent_step = parent_step
        # Client-supplied settings (e.g. max_tokens) applied to the step that
        # produces the final answer.
        self.final_overrides = final_overrides or {}

    def child(self, parent_step: str) -> "RunContext":
        return RunContext(
            self.cfg, self.dispatcher, self.task, self.deadline,
            depth=self.depth + 1, parent_step=parent_step, final_overrides=self.final_overrides,
        )

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check_deadline(self) -> None:
        if self.remaining() <= 0:
            raise TaskTimeout("task exceeded its time limit")

    def note(self, text: str) -> None:
        """A free-form progress line for traces and live progress views."""
        self.task.emit("task.note", text=text, parent=self.parent_step)

    # -- calling members ------------------------------------------------------

    def _body(self, params: Params, messages: list[dict], final: bool, response_format: dict | None,
              max_tokens: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": prepend_system(messages, params.system)}
        for key in ("temperature", "top_p", "max_tokens", "seed", "stop", "reasoning_effort"):
            value = getattr(params, key)
            if value is not None:
                body[key] = value
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format:
            body["response_format"] = response_format
        if final:
            for key, value in self.final_overrides.items():
                if value is not None:
                    body[key] = value
        body.update(params.extra or {})
        return body

    async def call(
        self,
        ref: MemberRef,
        messages: list[dict],
        *,
        role: str,
        label: str | None = None,
        final: bool = False,
        sink: Sink | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
    ) -> StepOutput:
        """Run one member (a model call or a nested team) and record the step."""
        from flotilla.engine.runner import run_team  # circular import at module load

        self.check_deadline()
        member = resolve_member(self.cfg, ref)
        step = self.task.add_step(StepRecord(
            id=new_id("step_"),
            parent=self.parent_step,
            role=role,
            label=label or role,
            member=member.label,
            kind="team" if member.team else "llm",
            candidates=list(member.candidates),
            final=final,
        ))
        self.task.emit("step.started", step=step.as_dict(full=False))
        t0 = time.monotonic()

        if member.team:
            if self.depth + 1 > self.cfg.server.max_depth:
                return self._fail(step, t0, f"nested teams deeper than max_depth={self.cfg.server.max_depth}")
            usage_before = Usage(self.task.usage.prompt_tokens, self.task.usage.completion_tokens)
            try:
                content = await run_team(member.team, messages, self.child(step.id), sink=sink if final else None)
            except StepFailed as exc:
                return self._fail(step, t0, f"team '{member.team}' failed: {exc}")
            used = Usage(
                self.task.usage.prompt_tokens - usage_before.prompt_tokens,
                self.task.usage.completion_tokens - usage_before.completion_tokens,
            )
            return self._ok(step, t0, content, "", f"team:{member.team}", None, used, count_usage=False)

        params = member.params
        timeout = min(params.timeout or self.cfg.cluster.request_timeout, max(1.0, self.remaining()))
        body = self._body(params, messages, final, response_format, max_tokens)
        def assigned(node: str, model: str) -> None:
            step.node, step.model, step.assigned_at = node, model, now()
            self.task.emit("step.assigned", step=step.id, label=step.label, node=node, model=model)

        spec = CallSpec(
            candidates=member.candidates,
            body=body,
            placement=Placement(node=params.node, labels=dict(params.labels), prefer_labels=dict(params.prefer_labels)),
            timeout=timeout,
            on_assign=assigned,
        )
        try:
            if sink is not None and final:
                result = await self._stream_into(spec, step, sink)
            else:
                result = await self.dispatcher.complete(spec)
                step.attempts = [a.as_dict() for a in result.attempts]
                if not result.content.strip() and body.get("reasoning_effort") != "none":
                    # Thinking models sometimes spend the whole budget
                    # reasoning and return no answer; ask again without it.
                    step.note = "empty answer; retried with reasoning_effort=none"
                    self.task.emit("step.retry", step=step.id, reason=step.note)
                    retry = await self.dispatcher.complete(CallSpec(
                        candidates=spec.candidates, body={**body, "reasoning_effort": "none"},
                        placement=spec.placement, timeout=spec.timeout, on_assign=spec.on_assign,
                    ))
                    retry.usage.add(result.usage)
                    retry.reasoning = result.reasoning or retry.reasoning
                    result = retry
        except DispatchError as exc:
            step.attempts = [a.as_dict() for a in exc.attempts]
            return self._fail(step, t0, exc.message)
        except TaskTimeout:
            raise
        if not result.content.strip():
            step.model, step.node = result.model, result.node
            return self._fail(step, t0, "the model returned an empty answer")
        return self._ok(step, t0, result.content, result.reasoning, result.model, result.node, result.usage)

    async def _stream_into(self, spec: CallSpec, step: StepRecord, sink: Sink):
        """Stream the final answer to the client while recording the step."""
        result = None
        got_content = False
        gen = self.dispatcher.stream(spec)
        try:
            async for kind, payload in gen:
                if kind == "delta":
                    content, reasoning = payload
                    if reasoning:
                        self.task.emit("step.delta", step=step.id, reasoning=reasoning)
                    if content:
                        if not got_content and not content.strip():
                            continue  # drop leading whitespace-only deltas
                        got_content = True
                        await sink(content)
                elif kind == "done":
                    result = payload
        finally:
            await gen.aclose()
        assert result is not None
        step.attempts = [a.as_dict() for a in result.attempts]
        if not result.content.strip() and spec.body.get("reasoning_effort") != "none":
            step.note = "empty answer; retried with reasoning_effort=none"
            self.task.emit("step.retry", step=step.id, reason=step.note)
            retry = await self.dispatcher.complete(CallSpec(
                candidates=spec.candidates, body={**spec.body, "reasoning_effort": "none"},
                placement=spec.placement, timeout=spec.timeout, on_assign=spec.on_assign,
            ))
            retry.usage.add(result.usage)
            retry.reasoning = result.reasoning or retry.reasoning
            if retry.content.strip():
                await sink(retry.content)
            result = retry
        return result

    def _ok(self, step: StepRecord, t0: float, content: str, reasoning: str, model: str | None,
            node: str | None, usage: Usage, count_usage: bool = True) -> StepOutput:
        step.status = "ok"
        step.ended_at = step.started_at + (time.monotonic() - t0)
        step.output = content
        step.reasoning = reasoning or None
        step.model = model
        step.node = node
        step.usage = usage.as_dict()
        if count_usage:
            self.task.usage.add(usage)
        self.task.emit("step.completed", step=step.as_dict(full=True))
        return StepOutput(content, reasoning, model, node, usage, time.monotonic() - t0, step.id)

    def _fail(self, step: StepRecord, t0: float, message: str) -> StepOutput:
        step.status = "error"
        step.ended_at = step.started_at + (time.monotonic() - t0)
        step.error = message
        self.task.emit("step.failed", step=step.as_dict(full=False))
        raise StepFailed(f"{step.label}: {truncate(message, 600)}", step)

    async def gather(self, calls: list[Awaitable[StepOutput]]) -> list[StepOutput | StepFailed]:
        """Run calls in parallel; failed steps come back as StepFailed values."""
        results = await asyncio.gather(*calls, return_exceptions=True)
        out: list[StepOutput | StepFailed] = []
        for r in results:
            if isinstance(r, (StepOutput, StepFailed)):
                out.append(r)
            elif isinstance(r, BaseException):
                raise r
        return out
