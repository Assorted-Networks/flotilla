"""Entry points for running a team as a task."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from flotilla.config import FlotillaConfig
from flotilla.dispatch import Dispatcher
from flotilla.engine.context import RunContext, Sink, StepFailed, TaskTimeout
from flotilla.engine.strategies import STRATEGIES
from flotilla.tasks import TaskRecord

log = logging.getLogger("flotilla.engine")


async def run_team(name: str, messages: list[dict], ctx: RunContext, sink: Sink | None = None) -> str:
    team = ctx.cfg.teams.get(name)
    if team is None:
        raise StepFailed(f"unknown team '{name}'")
    ctx.task.emit("team.started", team=name, strategy=team.strategy, parent=ctx.parent_step, depth=ctx.depth)
    content = await STRATEGIES[team.strategy](team, messages, ctx, sink)
    ctx.task.emit("team.completed", team=name, parent=ctx.parent_step, depth=ctx.depth)
    return content


async def execute_team_task(
    task: TaskRecord,
    cfg: FlotillaConfig,
    dispatcher: Dispatcher,
    team_name: str,
    messages: list[dict],
    sink: Sink | None = None,
    final_overrides: dict[str, Any] | None = None,
) -> str:
    """Run a team to completion, recording everything on `task`."""
    team = cfg.teams[team_name]
    limit = team.timeout or cfg.server.task_timeout
    ctx = RunContext(cfg, dispatcher, task, time.monotonic() + limit, final_overrides=final_overrides)
    task.emit("task.started", target=team_name, kind="team", strategy=team.strategy)
    try:
        content = await asyncio.wait_for(run_team(team_name, messages, ctx, sink), timeout=limit)
    except StepFailed as exc:
        task.finish("error", error=str(exc))
        raise
    except (TaskTimeout, asyncio.TimeoutError) as exc:
        message = str(exc) or f"task exceeded its time limit ({limit:g}s)"
        task.finish("error", error=message)
        raise TaskTimeout(message) from exc
    except asyncio.CancelledError:
        task.finish("cancelled", error="cancelled by the client")
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
        log.exception("task %s crashed", task.id)
        task.finish("error", error=f"internal error: {exc}")
        raise
    task.finish("ok", output=content)
    return content
