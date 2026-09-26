"""The coordinator: node registry, team engine, OpenAI-compatible API and dashboard."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

from flotilla import __version__, llm
from flotilla import openai_format as oai
from flotilla.config import ConfigError, FlotillaConfig, MemberConfig, load_config, resolve_member, team_member_refs, team_models
from flotilla.dispatch import CallSpec, DispatchError, Dispatcher
from flotilla.engine.context import RunContext, StepFailed, TaskTimeout
from flotilla.engine.runner import execute_team_task
from flotilla.llm import BackendError, Usage
from flotilla.registry import NodeModel, Registry
from flotilla.settings import CoordinatorSettings, ssl_context
from flotilla.tasks import StepRecord, TaskRecord, TaskStore
from flotilla.util import bearer_token, canonical_model, consteq, new_id, now, truncate

log = logging.getLogger("flotilla.coordinator")

STATIC_DIR = Path(__file__).parent / "static"
MAX_BODY = 32 * 1024 * 1024
KEEPALIVE_SECONDS = 15.0


class HTTPProblem(Exception):
    def __init__(self, status: int, message: str, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class Metrics:
    """A few Prometheus counters, kept in memory."""

    def __init__(self) -> None:
        self.requests: dict[tuple[str, str, str], int] = {}
        self.calls: dict[tuple[str, str, str], int] = {}
        self.call_seconds: dict[tuple[str, str], float] = {}
        self.tokens: dict[tuple[str, str, str], int] = {}

    def request(self, kind: str, target: str, status: str) -> None:
        key = (kind, target, status)
        self.requests[key] = self.requests.get(key, 0) + 1

    def call(self, node: str, model: str, ok: bool, seconds: float, usage: Usage | None) -> None:
        key = (node, model, "ok" if ok else "error")
        self.calls[key] = self.calls.get(key, 0) + 1
        self.call_seconds[(node, model)] = self.call_seconds.get((node, model), 0.0) + seconds
        if usage:
            for kind, n in (("prompt", usage.prompt_tokens), ("completion", usage.completion_tokens)):
                tkey = (node, model, kind)
                self.tokens[tkey] = self.tokens.get(tkey, 0) + n

    def render(self, registry: Registry) -> str:
        def esc(v: str) -> str:
            return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

        lines = [
            "# HELP flotilla_node_up 1 if the node can take work.",
            "# TYPE flotilla_node_up gauge",
        ]
        for n in registry.snapshot():
            lines.append(f'flotilla_node_up{{node="{esc(n["name"])}"}} {1 if n["status"] == "online" else 0}')
        lines += ["# HELP flotilla_node_in_flight Requests running on the node.", "# TYPE flotilla_node_in_flight gauge"]
        for n in registry.snapshot():
            lines.append(f'flotilla_node_in_flight{{node="{esc(n["name"])}"}} {n["in_flight"]}')
        lines += ["# HELP flotilla_node_capacity Parallel request slots on the node.", "# TYPE flotilla_node_capacity gauge"]
        for n in registry.snapshot():
            lines.append(f'flotilla_node_capacity{{node="{esc(n["name"])}"}} {n["capacity"]}')
        lines += ["# HELP flotilla_requests_total Client requests.", "# TYPE flotilla_requests_total counter"]
        for (kind, target, status), v in sorted(self.requests.items()):
            lines.append(f'flotilla_requests_total{{kind="{esc(kind)}",target="{esc(target)}",status="{status}"}} {v}')
        lines += ["# HELP flotilla_model_calls_total Model calls made on nodes.", "# TYPE flotilla_model_calls_total counter"]
        for (node, model, status), v in sorted(self.calls.items()):
            lines.append(f'flotilla_model_calls_total{{node="{esc(node)}",model="{esc(model)}",status="{status}"}} {v}')
        lines += ["# HELP flotilla_model_call_seconds_total Time spent in model calls.", "# TYPE flotilla_model_call_seconds_total counter"]
        for (node, model), v in sorted(self.call_seconds.items()):
            lines.append(f'flotilla_model_call_seconds_total{{node="{esc(node)}",model="{esc(model)}"}} {v:.3f}')
        lines += ["# HELP flotilla_tokens_total Tokens processed.", "# TYPE flotilla_tokens_total counter"]
        for (node, model, kind), v in sorted(self.tokens.items()):
            lines.append(f'flotilla_tokens_total{{node="{esc(node)}",model="{esc(model)}",type="{kind}"}} {v}')
        return "\n".join(lines) + "\n"


class Coordinator:
    def __init__(self, settings: CoordinatorSettings, cfg: FlotillaConfig):
        self.settings = settings
        self.cfg = cfg
        self.registry = Registry(cfg.cluster, agent_token=settings.cluster_token)
        self.tasks = TaskStore(cfg.server.trace_limit, settings.data_dir)
        self.metrics = Metrics()
        self.started_at = time.time()
        self.client: httpx.AsyncClient | None = None
        self.dispatcher: Dispatcher | None = None
        self._background: set[asyncio.Task] = set()
        self._loops: list[asyncio.Task] = []

    # -- lifecycle ------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        self.client = httpx.AsyncClient(
            verify=ssl_context(self.settings.ca_file),
            trust_env=self.settings.trust_env_proxy,
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=64),
        )
        self.dispatcher = Dispatcher(self.registry, self.client, self.cfg.cluster)
        self.dispatcher.on_call = self.metrics.call
        if not self.settings.cluster_token:
            log.error("FLOTILLA_CLUSTER_TOKEN is not set: agents cannot join until it is")
        if not self.settings.api_keys:
            log.warning("FLOTILLA_API_KEYS is not set: the API and dashboard accept requests without a key")
        log.info(
            "coordinator %s ready: %d teams, %d static nodes", __version__, len(self.cfg.teams), len(self.cfg.cluster.static_nodes)
        )
        self._loops = [asyncio.create_task(self._probe_loop()), asyncio.create_task(self._prune_loop())]
        try:
            yield
        finally:
            for t in self._loops + list(self._background):
                t.cancel()
            await asyncio.gather(*self._loops, *self._background, return_exceptions=True)
            await self.client.aclose()

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    async def _probe_loop(self) -> None:
        while True:
            try:
                await self.probe_once()
            except Exception:  # noqa: BLE001 - keep probing
                log.exception("probe loop error")
            await asyncio.sleep(self.cfg.cluster.probe_interval)

    async def _prune_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                self.registry.prune()
            except Exception:  # noqa: BLE001
                log.exception("prune failed")

    async def probe_once(self) -> None:
        """Refresh static nodes and re-check agents the coordinator cannot reach."""
        jobs = []
        for node in list(self.registry.nodes.values()):
            if node.static:
                jobs.append(self._probe_static(node))
            elif node.reachable is not True and self.registry.status(node) != "offline":
                jobs.append(self._check_reachable(node))
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    async def _probe_static(self, node) -> None:
        assert self.client
        try:
            if node.kind == "ollama":
                tags = await llm.get_json(self.client, f"{node.url}/api/tags", node.api_key)
                models = {}
                for m in tags.get("models") or []:
                    if m.get("remote_host") or str(m.get("name", "")).endswith("cloud"):
                        continue
                    d = m.get("details") or {}
                    nm = NodeModel(m.get("name") or m.get("model"), m.get("size"), d.get("family"),
                                   d.get("parameter_size"), d.get("quantization_level"))
                    models[canonical_model(nm.name)] = nm
                loaded = None
                with contextlib.suppress(BackendError):
                    ps = await llm.get_json(self.client, f"{node.url}/api/ps", node.api_key)
                    loaded = {canonical_model(m.get("name") or m.get("model")) for m in ps.get("models") or []}
            elif node.kind == "agent":
                info = await llm.get_json(self.client, f"{node.url}/info", node.api_key)
                models = {canonical_model(m["name"]): NodeModel(m["name"], m.get("size"), m.get("family"),
                          m.get("parameter_size"), m.get("quantization")) for m in info.get("models") or []}
                loaded = {canonical_model(m) for m in info.get("loaded_models") or []}
            else:
                data = await llm.get_json(self.client, llm.api_base(node.url) + "/models", node.api_key)
                models = {canonical_model(m["id"]): NodeModel(m["id"]) for m in data.get("data") or [] if m.get("id")}
                loaded = set(models)
            configured = {s.name: s for s in self.cfg.cluster.static_nodes}.get(node.name)
            if configured and configured.models:
                models = {canonical_model(m): models.get(canonical_model(m)) or NodeModel(m) for m in configured.models}
            self.registry.update_static(node, models, loaded, ok=True)
        except (BackendError, KeyError, TypeError, AttributeError) as exc:
            self.registry.update_static(node, None, None, ok=False, error=str(exc))

    async def _check_reachable(self, node) -> None:
        assert self.client
        try:
            resp = await self.client.get(f"{node.url}/health", timeout=5.0)
            ok = resp.status_code < 500
            self.registry.set_reachable(node, ok, None if ok else f"HTTP {resp.status_code}")
        except httpx.HTTPError as exc:
            self.registry.set_reachable(node, False, f"{type(exc).__name__}: {exc}")

    def reload(self) -> FlotillaConfig:
        cfg = load_config(self.settings.config_path)
        self.cfg = cfg
        self.registry.reconfigure(cfg.cluster)
        if self.dispatcher:
            self.dispatcher.cfg = cfg.cluster
        self.tasks.limit = cfg.server.trace_limit
        log.info("configuration reloaded: %d teams, %d members", len(cfg.teams), len(cfg.members))
        return cfg

    # -- request helpers --------------------------------------------------------

    def check_client(self, request: Request) -> None:
        if not self.settings.api_keys:
            return
        token = bearer_token(request.headers)
        if not token or not any(consteq(token, k) for k in self.settings.api_keys):
            raise HTTPProblem(401, "missing or invalid API key", "invalid_api_key")

    def check_agent(self, request: Request) -> None:
        if not self.settings.cluster_token:
            raise HTTPProblem(503, "the coordinator has no FLOTILLA_CLUSTER_TOKEN configured", "no_cluster_token")
        if not consteq(bearer_token(request.headers), self.settings.cluster_token):
            raise HTTPProblem(401, "invalid cluster token", "invalid_cluster_token")

    async def read_json(self, request: Request) -> dict:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY:
            raise HTTPProblem(413, "request body too large")
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise HTTPProblem(413, "request body too large")
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise HTTPProblem(400, "request body is not valid JSON")
        if not isinstance(data, dict):
            raise HTTPProblem(400, "request body must be a JSON object")
        return data

    def team_for(self, model: str | None) -> str | None:
        if not model:
            return None
        prefix = self.cfg.server.team_prefix
        if prefix and model.startswith(prefix) and model[len(prefix):] in self.cfg.teams:
            return model[len(prefix):]
        if model in self.cfg.teams and not self.registry.has_model(model):
            return model
        return None

    @staticmethod
    def _validate_messages(messages: Any) -> list[dict]:
        if not isinstance(messages, list) or not messages:
            raise HTTPProblem(400, "`messages` must be a non-empty list")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m:
                raise HTTPProblem(400, "each message needs a `role`")
        return messages

    # -- progress rendering (for the reasoning field) ---------------------------

    def progress_line(self, ev: dict) -> str | None:
        t = ev.get("type")
        if t == "team.started" and ev.get("depth", 0) > 0:
            return f"team {ev['team']} ({ev['strategy']}) started"
        if t == "step.started" and ev["step"]["kind"] == "team":
            s = ev["step"]
            return f"{s['label']} -> {s['member']}"
        if t == "step.assigned":
            return f"{ev['label']} -> {ev['model']} on {ev['node']}"
        if t == "step.completed":
            s = ev["step"]
            if s["kind"] == "team":
                return f"{s['label']} done in {s['seconds']}s"
            tokens = (s.get("usage") or {}).get("completion_tokens")
            where = f"{s['model']} on {s['node']}"
            return f"{s['label']} done: {where}, {s['seconds']}s" + (f", {tokens} tokens" if tokens else "")
        if t == "step.failed":
            s = ev["step"]
            return f"{s['label']} failed: {truncate(s.get('error'), 300)}"
        if t == "step.retry":
            return f"retrying: {ev.get('reason')}"
        if t == "task.note":
            return ev.get("text")
        return None

    # -- OpenAI-compatible endpoints ----------------------------------------------

    async def v1_models(self, request: Request) -> Response:
        self.check_client(request)
        created = int(self.started_at)
        data = []
        for name, team in self.cfg.teams.items():
            data.append({
                "id": f"{self.cfg.server.team_prefix}{name}",
                "object": "model",
                "created": created,
                "owned_by": "flotilla",
                "description": team.description or f"{team.strategy} team",
            })
        if self.cfg.server.expose_models:
            for canon, entry in sorted(self.registry.models_index().items()):
                data.append({
                    "id": entry["name"],
                    "object": "model",
                    "created": created,
                    "owned_by": ",".join(entry["nodes"]),
                })
        return JSONResponse({"object": "list", "data": data})

    async def v1_chat(self, request: Request) -> Response:
        self.check_client(request)
        body = await self.read_json(request)
        model = body.get("model")
        if not isinstance(model, str):
            raise HTTPProblem(400, "`model` must be a string")
        messages = self._validate_messages(body.get("messages"))
        team = self.team_for(model)
        if team:
            return await self.team_chat(body, team, messages)
        if self.cfg.server.expose_models and model and self.registry.has_model(model):
            return await self.model_chat(body, model, messages)
        available = [f"{self.cfg.server.team_prefix}{t}" for t in self.cfg.teams]
        raise HTTPProblem(404, f"model '{model}' not found. Teams: {', '.join(available) or 'none'}", "model_not_found")

    @staticmethod
    def _wants_usage(body: dict) -> bool:
        options = body.get("stream_options")
        return isinstance(options, dict) and bool(options.get("include_usage"))

    async def team_chat(self, body: dict, team: str, messages: list[dict]) -> Response:
        assert self.dispatcher
        stream = bool(body.get("stream"))
        include_usage = self._wants_usage(body)
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        overrides = {"max_tokens": max_tokens} if isinstance(max_tokens, int) and max_tokens > 0 else {}
        model_id = body.get("model")
        task = self.tasks.create(team, "team", messages, source="openai")
        cid = f"chatcmpl-{task.id}"
        created = int(time.time())
        progress = self.cfg.server.progress == "reasoning"
        rfield = self.cfg.server.progress_field
        headers = {"x-flotilla-task-id": task.id}

        if not stream:
            try:
                content = await execute_team_task(task, self.cfg, self.dispatcher, team, messages, final_overrides=overrides)
            except (StepFailed, TaskTimeout, DispatchError) as exc:
                self.metrics.request("team", team, "error")
                self.tasks.persist(task)
                status = 504 if isinstance(exc, TaskTimeout) else 502
                return oai.error(status, f"team '{team}' failed: {exc}", headers=headers)
            self.metrics.request("team", team, "ok")
            self.tasks.persist(task)
            log_lines = [ln for ev in task.events if (ln := self.progress_line(ev))] if progress else []
            return JSONResponse(
                oai.completion(cid, model_id, content, task.usage.as_dict(),
                               reasoning="\n".join(log_lines) or None, reasoning_field=rfield,
                               extra={"flotilla": {"task_id": task.id, "steps": len(task.steps), "nodes": task.nodes_used()}}),
                headers=headers,
            )

        queue: asyncio.Queue = asyncio.Queue()

        async def sink(text: str) -> None:
            task.emit("task.delta", content=text)
            queue.put_nowait(("content", text))

        def on_event(ev: dict) -> None:
            # Runs synchronously inside task.emit, so progress lines and
            # answer tokens reach the client in the order they happened.
            if ev["type"] == "step.delta" and ev.get("reasoning"):
                queue.put_nowait(("reasoning", ev["reasoning"]))
                return
            line = self.progress_line(ev)
            if line:
                queue.put_nowait(("reasoning", line + "\n"))

        if progress:
            task.add_listener(on_event)

        async def run() -> None:
            try:
                await execute_team_task(task, self.cfg, self.dispatcher, team, messages, sink=sink, final_overrides=overrides)
                self.metrics.request("team", team, "ok")
                queue.put_nowait(("end", None))
            except (StepFailed, TaskTimeout, DispatchError) as exc:
                self.metrics.request("team", team, "error")
                queue.put_nowait(("error", f"team '{team}' failed: {exc}"))
            except asyncio.CancelledError:
                self.metrics.request("team", team, "cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                queue.put_nowait(("error", f"internal error: {exc}"))
            finally:
                task.remove_listener(on_event)
                self.tasks.persist(task)

        runner = self._spawn(run())

        async def generate() -> AsyncIterator[bytes]:
            try:
                yield oai.sse(oai.chunk(cid, model_id, created, role="assistant", content=""))
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield oai.SSE_PING
                        continue
                    if kind == "content":
                        yield oai.sse(oai.chunk(cid, model_id, created, content=payload))
                    elif kind == "reasoning":
                        yield oai.sse(oai.chunk(cid, model_id, created, reasoning=payload, reasoning_field=rfield))
                    elif kind == "error":
                        yield oai.sse(oai.error_body(payload))
                        yield oai.SSE_DONE
                        return
                    elif kind == "end":
                        yield oai.sse(oai.chunk(cid, model_id, created, finish_reason="stop"))
                        if include_usage:
                            yield oai.sse(oai.chunk(cid, model_id, created, usage=task.usage.as_dict(), include_choice=False))
                        yield oai.SSE_DONE
                        return
            finally:
                if not runner.done():
                    runner.cancel()

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"})

    async def model_chat(self, body: dict, model: str, messages: list[dict]) -> Response:
        """Direct model call: load-balanced passthrough to a node."""
        assert self.dispatcher
        stream = bool(body.get("stream"))
        req = {k: v for k, v in body.items() if k != "model"}
        spec = CallSpec(candidates=[model], body=req)
        task = self.tasks.create(model, "model", messages, source="openai")
        step = task.add_step(StepRecord(id=new_id("step_"), parent=None, role="model", label="model", member=model,
                                        kind="llm", candidates=[model], final=True))
        task.emit("task.started", target=model, kind="model")
        task.emit("step.started", step=step.as_dict(full=False))
        headers = {"x-flotilla-task-id": task.id}

        def fail(exc: DispatchError) -> None:
            step.status, step.error, step.ended_at = "error", exc.message, now()
            step.attempts = [a.as_dict() for a in exc.attempts]
            task.emit("step.failed", step=step.as_dict(full=False))
            task.finish("error", error=exc.message)
            self.tasks.persist(task)
            self.metrics.request("model", model, "error")

        def succeed(node: str, actual: str, content: str, usage: Usage) -> None:
            step.status, step.node, step.model, step.ended_at = "ok", node, actual, now()
            step.output, step.usage = content, usage.as_dict()
            task.usage.add(usage)
            task.emit("step.completed", step=step.as_dict(full=True))
            task.finish("ok", output=content)
            self.tasks.persist(task)
            self.metrics.request("model", model, "ok")

        if not stream:
            try:
                data, lease, _seconds, _attempts = await self.dispatcher.complete_raw(spec)
            except DispatchError as exc:
                fail(exc)
                return oai.error(exc.status if exc.status >= 400 else 502, exc.message, headers=headers)
            if isinstance(data, dict):
                data["model"] = model
            try:
                comp = llm.parse_completion(data)
                succeed(lease.node.name, lease.model, comp.content, comp.usage)
            except BackendError:
                # Unusual response shape: pass it on untouched, record what we can.
                succeed(lease.node.name, lease.model, "", Usage())
            return JSONResponse(data, headers=headers)

        agen = self.dispatcher.stream_raw(spec)
        try:
            kind, lease = await agen.__anext__()
        except DispatchError as exc:
            await agen.aclose()
            fail(exc)
            return oai.error(exc.status if exc.status >= 400 else 502, exc.message, headers=headers)
        except StopAsyncIteration:
            fail(DispatchError("empty stream"))
            return oai.error(502, "empty stream", headers=headers)

        # The dispatcher asks every node for usage; pass the usage-only chunk
        # (empty `choices`) on only to clients that asked for it, as OpenAI does.
        include_usage = self._wants_usage(body)

        async def generate() -> AsyncIterator[bytes]:
            parts: list[str] = []
            usage = Usage()
            try:
                async for kind, payload in agen:
                    if kind == "chunk":
                        payload["model"] = model
                        c, _r, _f, u = llm.parse_chunk(payload)
                        if c:
                            parts.append(c)
                        if u:
                            usage = u
                        if not include_usage and not payload.get("choices") and "usage" in payload:
                            continue
                        yield oai.sse(payload)
                    elif kind == "end":
                        yield oai.SSE_DONE
                        succeed(lease.node.name, lease.model, "".join(parts), usage)
            except DispatchError as exc:
                fail(exc)
                yield oai.sse(oai.error_body(exc.message))
                yield oai.SSE_DONE
            finally:
                await agen.aclose()
                if task.status == "running":
                    task.finish("cancelled", error="client disconnected")

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={**headers, "cache-control": "no-cache", "x-accel-buffering": "no"})

    async def v1_embeddings(self, request: Request) -> Response:
        self.check_client(request)
        assert self.dispatcher
        body = await self.read_json(request)
        model = body.get("model")
        if not model or not self.registry.has_model(model):
            raise HTTPProblem(404, f"embedding model '{model}' is not available on any node", "model_not_found")
        try:
            data = await self.dispatcher.embeddings(CallSpec(candidates=[model], body={k: v for k, v in body.items() if k != "model"}))
        except DispatchError as exc:
            return oai.error(exc.status if exc.status >= 400 else 502, exc.message)
        if isinstance(data, dict):
            data["model"] = model
        return JSONResponse(data)

    # -- native API -----------------------------------------------------------------

    async def api_cluster(self, request: Request) -> Response:
        self.check_client(request)
        nodes = self.registry.snapshot()
        online = [n for n in nodes if n["status"] == "online"]
        return JSONResponse({
            "cluster": self.settings.cluster_name,
            "version": __version__,
            "uptime_seconds": round(time.time() - self.started_at),
            "auth_required": bool(self.settings.api_keys),
            "nodes": nodes,
            "models": self.registry.models_index(),
            "totals": {
                "nodes": len(nodes),
                "online": len(online),
                "slots": sum(n["capacity"] for n in online),
                "in_flight": sum(n["in_flight"] for n in nodes),
            },
        })

    def team_summary(self, name: str) -> dict[str, Any]:
        team = self.cfg.teams[name]
        index = self.registry.models_index()
        roles = []
        missing: set[str] = set()
        for role, ref in team_member_refs(team):
            res = resolve_member(self.cfg, ref)
            if res.team:
                available = res.team in self.cfg.teams
                roles.append({"role": role, "member": res.label, "team": res.team, "available": available})
                continue
            present = [m for m in res.candidates if canonical_model(m) in index]
            if not present:
                missing.update(res.candidates)
            roles.append({
                "role": role,
                "member": res.label,
                "models": res.candidates,
                "available_models": present,
                "available": bool(present),
            })
        return {
            "name": name,
            "id": f"{self.cfg.server.team_prefix}{name}",
            "strategy": team.strategy,
            "description": team.description,
            "roles": roles,
            "models": sorted(team_models(self.cfg, name)),
            "missing_models": sorted(missing),
            "ready": all(r["available"] for r in roles),
        }

    async def api_teams(self, request: Request) -> Response:
        self.check_client(request)
        return JSONResponse({"teams": [self.team_summary(n) for n in self.cfg.teams]})

    async def api_create_task(self, request: Request) -> Response:
        self.check_client(request)
        assert self.dispatcher
        body = await self.read_json(request)
        messages = body.get("messages")
        if not messages and body.get("prompt"):
            messages = [{"role": "user", "content": str(body["prompt"])}]
            if body.get("system"):
                messages.insert(0, {"role": "system", "content": str(body["system"])})
        messages = self._validate_messages(messages)
        team = body.get("team")
        model = body.get("model")
        for key, value in (("team", team), ("model", model)):
            if value is not None and not isinstance(value, str):
                raise HTTPProblem(400, f"`{key}` must be a string")
        if team and team.startswith(self.cfg.server.team_prefix):
            team = team[len(self.cfg.server.team_prefix):]
        if team and team not in self.cfg.teams:
            raise HTTPProblem(404, f"unknown team '{team}'", "team_not_found")
        if not team and not model:
            raise HTTPProblem(400, "give either `team` or `model`")
        wait = body.get("wait", True)
        target = team or model
        task = self.tasks.create(target, "team" if team else "model", messages, source=str(body.get("source") or "api"))

        async def sink(text: str) -> None:
            task.emit("task.delta", content=text)

        async def run() -> None:
            try:
                if team:
                    await execute_team_task(task, self.cfg, self.dispatcher, team, messages, sink=sink)
                else:
                    await self._run_model_task(task, model, messages, sink)
                self.metrics.request("api", target, "ok")
            except (StepFailed, TaskTimeout, DispatchError):
                self.metrics.request("api", target, "error")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded on the task
                log.exception("task %s failed", task.id)
                if task.status == "running":
                    task.finish("error", error=f"internal error: {exc}")
            finally:
                self.tasks.persist(task)

        job = self._spawn(run())
        if not wait:
            return JSONResponse({"id": task.id, "status": task.status, "events": f"/api/tasks/{task.id}/events"}, status_code=202)
        # shield: if the client goes away the task still finishes and is kept.
        await asyncio.shield(job)
        return JSONResponse(task.as_dict())

    async def _run_model_task(self, task: TaskRecord, model: str, messages: list[dict], sink) -> None:
        assert self.dispatcher
        ctx = RunContext(self.cfg, self.dispatcher, task, time.monotonic() + self.cfg.server.task_timeout)
        task.emit("task.started", target=model, kind="model")
        try:
            out = await ctx.call(MemberConfig(model=model), messages, role="model", final=True, sink=sink)
        except (StepFailed, TaskTimeout) as exc:
            task.finish("error", error=str(exc) or "task exceeded its time limit")
            raise
        task.finish("ok", output=out.content)

    async def api_list_tasks(self, request: Request) -> Response:
        self.check_client(request)
        try:
            limit = int(request.query_params.get("limit", "50"))
        except ValueError:
            raise HTTPProblem(400, "`limit` must be a whole number")
        return JSONResponse({"tasks": self.tasks.recent(max(1, limit))})

    async def api_get_task(self, request: Request) -> Response:
        self.check_client(request)
        task = self.tasks.get(request.path_params["task_id"])
        if task is None:
            raise HTTPProblem(404, "task not found")
        return JSONResponse(task.as_dict() if isinstance(task, TaskRecord) else task)

    async def api_task_events(self, request: Request) -> Response:
        self.check_client(request)
        task = self.tasks.get(request.path_params["task_id"])
        if task is None:
            raise HTTPProblem(404, "task not found")

        async def generate() -> AsyncIterator[bytes]:
            if not isinstance(task, TaskRecord):
                yield oai.sse({"type": "task.snapshot", "task": task})
                return
            q = task.subscribe(replay=True)
            try:
                if task.done.is_set() and q.empty():
                    return
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield oai.SSE_PING
                        continue
                    yield oai.sse(ev)
                    if ev["type"] in ("task.completed", "task.failed"):
                        return
            finally:
                task.unsubscribe(q)

        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    async def api_pull(self, request: Request) -> Response:
        """Ask agents to download a model (Ollama nodes only)."""
        self.check_client(request)
        assert self.client
        body = await self.read_json(request)
        model = str(body.get("model") or "").strip()
        if not model:
            raise HTTPProblem(400, "give `model`")
        nodes = body.get("nodes") or []
        if isinstance(nodes, str):
            nodes = [nodes]
        if not isinstance(nodes, list):
            raise HTTPProblem(400, "`nodes` must be a list of node names")
        wanted = {str(n) for n in nodes}
        results = {name: {"ok": False, "error": "unknown node"} for name in wanted if name not in self.registry.nodes}
        for node in list(self.registry.nodes.values()):
            if wanted and node.name not in wanted:
                continue
            if node.static:
                results[node.name] = {"ok": False, "error": "static node: pull it on that server directly"}
                continue
            if self.registry.status(node) == "offline":
                results[node.name] = {"ok": False, "error": "node is offline"}
                continue
            try:
                data = await llm.post_json(self.client, f"{node.url}/admin/pull", {"model": model},
                                           self.settings.cluster_token, llm.make_timeout(30))
                results[node.name] = {"ok": True, **(data if isinstance(data, dict) else {})}
            except BackendError as exc:
                results[node.name] = {"ok": False, "error": exc.message}
        return JSONResponse({"model": model, "nodes": results})

    async def api_reload(self, request: Request) -> Response:
        self.check_client(request)
        try:
            cfg = self.reload()
        except ConfigError as exc:
            return JSONResponse({"ok": False, "problems": exc.problems}, status_code=400)
        return JSONResponse({"ok": True, "teams": list(cfg.teams), "members": list(cfg.members)})

    # -- agent endpoints ----------------------------------------------------------------

    async def agent_heartbeat(self, request: Request) -> Response:
        self.check_agent(request)
        hb = await self.read_json(request)
        client_host = request.client.host if request.client else None
        try:
            node = self.registry.upsert_agent(hb, client_host)
        except ValueError as exc:
            raise HTTPProblem(400, str(exc))
        if node.reachable is not True:
            self._spawn(self._check_reachable(node))
        return JSONResponse({
            "ok": True,
            "node": node.name,
            "url": node.url,
            "reachable": node.reachable,
            "cluster": self.settings.cluster_name,
            "coordinator_version": __version__,
        })

    async def agent_deregister(self, request: Request) -> Response:
        self.check_agent(request)
        body = await self.read_json(request)
        return JSONResponse({"ok": self.registry.deregister(str(body.get("name") or ""))})

    # -- misc ---------------------------------------------------------------------------

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "version": __version__, "nodes_online": len(self.registry.online_nodes())})

    async def metrics_endpoint(self, request: Request) -> Response:
        self.check_client(request)
        return PlainTextResponse(self.metrics.render(self.registry), media_type="text/plain; version=0.0.4")

    async def dashboard(self, request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


def build_app(settings: CoordinatorSettings | None = None, cfg: FlotillaConfig | None = None) -> Starlette:
    settings = settings or CoordinatorSettings.from_env()
    cfg = cfg or load_config(settings.config_path)
    coord = Coordinator(settings, cfg)

    def wrap(handler):
        async def endpoint(request: Request) -> Response:
            try:
                return await handler(request)
            except HTTPProblem as exc:
                return oai.error(exc.status, exc.message, code=exc.code)
        return endpoint

    routes = [
        Route("/", wrap(coord.dashboard)),
        Route("/health", wrap(coord.health)),
        Route("/metrics", wrap(coord.metrics_endpoint)),
        Route("/v1/models", wrap(coord.v1_models)),
        Route("/v1/chat/completions", wrap(coord.v1_chat), methods=["POST"]),
        Route("/v1/embeddings", wrap(coord.v1_embeddings), methods=["POST"]),
        Route("/api/cluster", wrap(coord.api_cluster)),
        Route("/api/teams", wrap(coord.api_teams)),
        Route("/api/tasks", wrap(coord.api_create_task), methods=["POST"]),
        Route("/api/tasks", wrap(coord.api_list_tasks), methods=["GET"]),
        Route("/api/tasks/{task_id}", wrap(coord.api_get_task)),
        Route("/api/tasks/{task_id}/events", wrap(coord.api_task_events)),
        Route("/api/pull", wrap(coord.api_pull), methods=["POST"]),
        Route("/api/config/reload", wrap(coord.api_reload), methods=["POST"]),
        Route("/api/agents/heartbeat", wrap(coord.agent_heartbeat), methods=["POST"]),
        Route("/api/agents/deregister", wrap(coord.agent_deregister), methods=["POST"]),
    ]
    app = Starlette(routes=routes, lifespan=coord.lifespan)
    app.state.coordinator = coord
    return app


def serve(settings: CoordinatorSettings | None = None) -> None:
    import uvicorn

    settings = settings or CoordinatorSettings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if settings.log_level != "debug":
        logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        cfg = load_config(settings.config_path)
    except ConfigError as exc:
        for p in exc.problems:
            log.error("config: %s", p)
        raise SystemExit(2)
    if settings.data_dir:
        os.makedirs(settings.data_dir, exist_ok=True)
    app = build_app(settings, cfg)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level if settings.log_level in ("critical", "error", "warning", "info", "debug") else "info",
        access_log=settings.log_level == "debug",
        ssl_certfile=settings.tls_cert,
        ssl_keyfile=settings.tls_key,
        timeout_keep_alive=30,
    )
