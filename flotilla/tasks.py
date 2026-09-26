"""Task records: what each run did, step by step, with live event fan-out."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from flotilla.llm import Usage
from flotilla.util import new_id, now, truncate

log = logging.getLogger("flotilla.tasks")

# Events that are only useful live (token deltas); they are not kept for replay.
EPHEMERAL = {"task.delta", "step.delta"}
MAX_EVENTS = 5000


@dataclass
class StepRecord:
    id: str
    parent: str | None
    role: str
    label: str
    member: str
    kind: str                    # "llm" or "team"
    candidates: list[str] = field(default_factory=list)
    status: str = "running"
    model: str | None = None
    node: str | None = None
    started_at: float = field(default_factory=now)
    assigned_at: float | None = None       # when a node slot was obtained
    ended_at: float | None = None
    usage: dict[str, Any] | None = None
    output: str | None = None
    reasoning: str | None = None
    error: str | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    final: bool = False
    note: str | None = None

    @property
    def seconds(self) -> float | None:
        return round(self.ended_at - self.started_at, 2) if self.ended_at else None

    def as_dict(self, full: bool = True) -> dict[str, Any]:
        d = {
            "id": self.id,
            "parent": self.parent,
            "role": self.role,
            "label": self.label,
            "member": self.member,
            "kind": self.kind,
            "candidates": self.candidates,
            "status": self.status,
            "model": self.model,
            "node": self.node,
            "started_at": self.started_at,
            "assigned_at": self.assigned_at,
            "ended_at": self.ended_at,
            "seconds": self.seconds,
            "usage": self.usage,
            "error": self.error,
            "attempts": self.attempts,
            "final": self.final,
            "note": self.note,
        }
        if full:
            d["output"] = self.output
            d["reasoning"] = self.reasoning
        else:
            d["output_preview"] = truncate(self.output, 280)
        return d


class TaskRecord:
    def __init__(self, target: str, kind: str, messages: list[dict], source: str):
        self.id = new_id("task_")
        self.target = target          # team name or model name
        self.kind = kind              # "team" or "model"
        self.source = source          # "openai", "api", "ui"
        self.messages = messages
        self.status = "running"
        self.created_at = now()
        self.finished_at: float | None = None
        self.output: str | None = None
        self.error: str | None = None
        self.usage = Usage()
        self.steps: list[StepRecord] = []
        self.events: list[dict[str, Any]] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._listeners: list = []
        self.done = asyncio.Event()

    # -- events -------------------------------------------------------------

    def emit(self, etype: str, **data: Any) -> dict[str, Any]:
        event = {"type": etype, "task": self.id, "ts": now(), **data}
        if etype not in EPHEMERAL and len(self.events) < MAX_EVENTS:
            self.events.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        for fn in list(self._listeners):
            try:
                fn(event)
            except Exception:  # noqa: BLE001 - a listener must not break the run
                log.exception("task listener failed")
        return event

    def add_listener(self, fn) -> None:
        """Call `fn(event)` synchronously for every event (keeps ordering)."""
        self._listeners.append(fn)

    def remove_listener(self, fn) -> None:
        with contextlib.suppress(ValueError):
            self._listeners.remove(fn)

    def subscribe(self, replay: bool = True) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10000)
        if replay:
            for ev in self.events:
                q.put_nowait(ev)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    # -- steps --------------------------------------------------------------

    def add_step(self, step: StepRecord) -> StepRecord:
        self.steps.append(step)
        return step

    def finish(self, status: str, output: str | None = None, error: str | None = None) -> None:
        self.status = status
        self.output = output
        self.error = error
        self.finished_at = now()
        self.emit(
            "task.completed" if status == "ok" else "task.failed",
            status=status,
            output=output,
            error=error,
            usage=self.usage.as_dict(),
            seconds=self.seconds,
        )
        self.done.set()

    @property
    def seconds(self) -> float | None:
        end = self.finished_at or now()
        return round(end - self.created_at, 2)

    def nodes_used(self) -> list[str]:
        return sorted({s.node for s in self.steps if s.node})

    def models_used(self) -> list[str]:
        return sorted({s.model for s in self.steps if s.model and s.kind == "llm"})

    def input_preview(self) -> str:
        for m in reversed(self.messages):
            if m.get("role") == "user":
                c = m.get("content")
                return truncate(c if isinstance(c, str) else json.dumps(c)[:400], 200)
        return ""

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target": self.target,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "seconds": self.seconds,
            "input_preview": self.input_preview(),
            "output_preview": truncate(self.output, 200),
            "error": self.error,
            "usage": self.usage.as_dict(),
            "steps": len(self.steps),
            "nodes": self.nodes_used(),
            "models": self.models_used(),
        }

    def as_dict(self) -> dict[str, Any]:
        d = self.summary()
        d.update({
            "messages": self.messages,
            "output": self.output,
            "step_details": [s.as_dict() for s in self.steps],
        })
        return d


class TaskStore:
    """Keeps the most recent tasks in memory; optionally appends finished
    tasks to a JSON-lines file so traces survive restarts."""

    def __init__(self, limit: int = 200, data_dir: str | None = None):
        self.limit = limit
        self.tasks: "OrderedDict[str, TaskRecord | dict]" = OrderedDict()
        self.path = os.path.join(data_dir, "tasks.jsonl") if data_dir else None
        if self.path:
            self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                lines = fh.readlines()[-self.limit:]
            for line in lines:
                try:
                    d = json.loads(line)
                    self.tasks[d["id"]] = d
                except (ValueError, KeyError):
                    continue
            log.info("loaded %d past tasks from %s", len(self.tasks), self.path)
        except OSError as exc:
            log.warning("could not read %s: %s", self.path, exc)

    def create(self, target: str, kind: str, messages: list[dict], source: str) -> TaskRecord:
        task = TaskRecord(target, kind, messages, source)
        self.tasks[task.id] = task
        while len(self.tasks) > self.limit:
            self.tasks.popitem(last=False)
        return task

    def get(self, task_id: str) -> "TaskRecord | dict | None":
        return self.tasks.get(task_id)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        out = []
        for t in reversed(self.tasks.values()):
            out.append(t.summary() if isinstance(t, TaskRecord) else {k: v for k, v in t.items() if k not in ("messages", "output", "step_details")})
            if len(out) >= limit:
                break
        return out

    def persist(self, task: TaskRecord) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(task.as_dict(), ensure_ascii=False) + "\n")
            self._compact_if_needed()
        except OSError as exc:
            log.warning("could not write %s: %s", self.path, exc)

    def _compact_if_needed(self) -> None:
        """Keep the file from growing without bound (rewrite at 5x the limit)."""
        assert self.path
        try:
            if os.path.getsize(self.path) < 1_000_000:
                return
            with open(self.path, encoding="utf-8") as fh:
                lines = fh.readlines()
            if len(lines) <= self.limit * 5:
                return
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(lines[-self.limit:])
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning("could not compact %s: %s", self.path, exc)
