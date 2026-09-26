"""The node agent: runs next to an inference server on each machine.

It registers the machine with the coordinator (heartbeats with the model list,
load and hardware), and exposes an authenticated OpenAI-compatible proxy so
the inference server itself never has to be reachable from the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, AsyncIterator

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from flotilla import __version__, llm
from flotilla import openai_format as oai
from flotilla.llm import BackendError
from flotilla.settings import AgentSettings, ssl_context
from flotilla.util import bearer_token, canonical_model, consteq, new_id

log = logging.getLogger("flotilla.agent")

MAX_BODY = 32 * 1024 * 1024


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cpus": os.cpu_count()}
    with contextlib.suppress(OSError, ValueError, IndexError):
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    info["memory_gb"] = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    if shutil.which("nvidia-smi"):
        with contextlib.suppress(OSError, subprocess.SubprocessError, ValueError):
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout
            gpus = []
            for line in out.strip().splitlines():
                name, mem = [x.strip() for x in line.split(",")[:2]]
                gpus.append({"name": name, "memory_gb": round(float(mem) / 1024, 1)})
            if gpus:
                info["gpus"] = gpus
    return info


class Backend:
    """Talks to the local inference server (Ollama or any OpenAI-compatible one)."""

    def __init__(self, settings: AgentSettings, client: httpx.AsyncClient):
        self.s = settings
        self.client = client
        self.kind = settings.backend if settings.backend in ("ollama", "openai") else "openai"

    @property
    def v1(self) -> str:
        return llm.api_base(self.s.backend_url)

    async def list_models(self) -> list[dict[str, Any]]:
        if self.kind == "ollama":
            data = await llm.get_json(self.client, f"{self.s.backend_url}/api/tags", None)
            out = []
            for m in data.get("models") or []:
                name = m.get("name") or m.get("model")
                if not name:
                    continue
                remote = bool(m.get("remote_host") or m.get("remote_model")) or name.endswith("cloud")
                if remote and not self.s.include_remote_models:
                    continue
                d = m.get("details") or {}
                out.append({
                    "name": name,
                    "size": m.get("size"),
                    "family": d.get("family"),
                    "parameter_size": d.get("parameter_size"),
                    "quantization": d.get("quantization_level"),
                })
        else:
            data = await llm.get_json(self.client, f"{self.v1}/models", self.s.backend_api_key)
            out = [{"name": m["id"]} for m in data.get("data") or [] if m.get("id")]
        if self.s.static_models:
            known = {canonical_model(m["name"]): m for m in out}
            out = [known.get(canonical_model(n), {"name": n}) for n in self.s.static_models]
        return out

    async def loaded_models(self, models: list[dict[str, Any]]) -> list[str]:
        if self.kind != "ollama":
            return [m["name"] for m in models]
        with contextlib.suppress(BackendError):
            data = await llm.get_json(self.client, f"{self.s.backend_url}/api/ps", None)
            return [m.get("name") or m.get("model") for m in data.get("models") or []]
        return []

    async def version(self) -> str | None:
        if self.kind != "ollama":
            return None
        with contextlib.suppress(BackendError):
            data = await llm.get_json(self.client, f"{self.s.backend_url}/api/version", None)
            return data.get("version")
        return None


class Agent:
    def __init__(self, settings: AgentSettings):
        self.s = settings
        self.instance_id = new_id()
        self.client: httpx.AsyncClient | None = None
        self.backend: Backend | None = None
        self.slots = asyncio.Semaphore(settings.max_concurrency)
        self.in_flight = 0
        self.models: list[dict[str, Any]] = []
        self.loaded: list[str] = []
        self.backend_ok = False
        self.backend_error: str | None = None
        self.backend_version: str | None = None
        self.hardware = hardware_info()
        self.pulls: dict[str, dict[str, Any]] = {}
        self._pull_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        self._coord_state: str | None = None
        self._warned_unreachable = False

    # -- lifecycle ------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        self.client = httpx.AsyncClient(verify=ssl_context(self.s.ca_file), trust_env=self.s.trust_env_proxy)
        self.backend = Backend(self.s, self.client)
        log.info(
            "agent %s for node '%s': backend %s at %s, %d slots",
            __version__, self.s.node_name, self.backend.kind, self.s.backend_url, self.s.max_concurrency,
        )
        loops = [asyncio.create_task(self._heartbeat_loop())]
        if self.s.pull_models:
            loops.append(asyncio.create_task(self._initial_pulls()))
        try:
            yield
        finally:
            for t in loops + list(self._tasks):
                t.cancel()
            await asyncio.gather(*loops, *self._tasks, return_exceptions=True)
            await self._deregister()
            await self.client.aclose()

    def _spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)
        return t

    async def refresh_backend(self) -> None:
        assert self.backend
        try:
            models = await self.backend.list_models()
            self.loaded = await self.backend.loaded_models(models)
            if self.backend_version is None:
                self.backend_version = await self.backend.version()
            if not self.backend_ok:
                log.info("backend reachable with %d models: %s", len(models), ", ".join(m["name"] for m in models) or "none yet")
            self.models = models
            self.backend_ok = True
            self.backend_error = None
        except (BackendError, KeyError, TypeError, AttributeError) as exc:
            message = getattr(exc, "message", None) or f"unexpected reply: {type(exc).__name__}: {exc}"
            if self.backend_ok or self.backend_error is None:
                log.warning("backend %s unavailable: %s", self.s.backend_url, message)
            self.backend_ok = False
            self.backend_error = message

    def heartbeat_payload(self) -> dict[str, Any]:
        return {
            "name": self.s.node_name,
            "instance_id": self.instance_id,
            "url": self.s.advertise_url,
            "port": self.s.advertise_port or self.s.port,
            "backend": self.backend.kind if self.backend else self.s.backend,
            "backend_ok": self.backend_ok,
            "backend_error": self.backend_error,
            "backend_version": self.backend_version,
            "version": __version__,
            "models": self.models,
            "loaded_models": self.loaded,
            "max_concurrency": self.s.max_concurrency,
            "in_flight": self.in_flight,
            "labels": self.s.labels,
            "hardware": self.hardware,
        }

    async def _heartbeat_loop(self) -> None:
        assert self.client
        while True:
            try:
                await self.refresh_backend()
                if self.s.coordinator_url:
                    await self._heartbeat_once()
            except Exception:  # noqa: BLE001 - a bad reply must not stop the heartbeats
                log.exception("heartbeat cycle failed")
            await asyncio.sleep(self.s.heartbeat_interval)

    async def _heartbeat_once(self) -> None:
        assert self.client
        url = f"{self.s.coordinator_url}/api/agents/heartbeat"
        try:
            resp = await self.client.post(
                url, json=self.heartbeat_payload(),
                headers={"authorization": f"Bearer {self.s.cluster_token}"}, timeout=10.0,
            )
        except httpx.HTTPError as exc:
            self._coord(f"unreachable ({type(exc).__name__})", logging.WARNING,
                        f"cannot reach coordinator at {self.s.coordinator_url}: {exc}")
            return
        if resp.status_code == 200:
            try:
                data = resp.json()
            except ValueError:
                data = {}
            self._coord("ok", logging.INFO,
                        f"registered with coordinator '{data.get('cluster')}' as '{data.get('node')}'; "
                        f"coordinator will call this node at {data.get('url')}")
            if data.get("reachable") is False and not self._warned_unreachable:
                self._warned_unreachable = True
                log.warning(
                    "the coordinator cannot reach this agent at %s. Publish port %d on this machine and/or set "
                    "FLOTILLA_ADVERTISE_URL to an address the coordinator can reach", data.get("url"),
                    self.s.advertise_port or self.s.port,
                )
            elif data.get("reachable"):
                self._warned_unreachable = False
        elif resp.status_code == 401:
            self._coord("rejected", logging.ERROR, "coordinator rejected the cluster token (FLOTILLA_CLUSTER_TOKEN differs)")
        else:
            self._coord(f"http {resp.status_code}", logging.WARNING,
                        f"heartbeat failed: HTTP {resp.status_code} {resp.text[:200]}")

    def _coord(self, state: str, level: int, message: str) -> None:
        """Log coordinator connection changes once, not on every heartbeat."""
        if state != self._coord_state:
            log.log(level, message)
            self._coord_state = state

    async def _deregister(self) -> None:
        if not (self.s.coordinator_url and self.client):
            return
        with contextlib.suppress(httpx.HTTPError):
            await self.client.post(
                f"{self.s.coordinator_url}/api/agents/deregister", json={"name": self.s.node_name},
                headers={"authorization": f"Bearer {self.s.cluster_token}"}, timeout=3.0,
            )

    # -- model downloads (Ollama) -------------------------------------------------

    async def _initial_pulls(self) -> None:
        # Wait for the backend to come up first.
        for _ in range(60):
            if self.backend_ok:
                break
            await asyncio.sleep(2)
        present = {canonical_model(m["name"]) for m in self.models}
        for model in self.s.pull_models:
            if canonical_model(model) in present:
                log.info("model already present: %s", model)
                continue
            await self.pull(model)
        await self.refresh_backend()

    def start_pull(self, model: str) -> dict[str, Any]:
        state = self.pulls.get(model)
        if state and state["status"] in ("queued", "downloading"):
            return {"status": "already running", **state}
        self.pulls[model] = {"status": "queued", "completed": 0, "total": 0, "error": None, "started_at": time.time()}
        self._spawn(self.pull(model))
        return {"status": "started"}

    async def pull(self, model: str) -> None:
        assert self.client
        if self.s.backend != "ollama":
            self.pulls[model] = {"status": "error", "error": "pulling is only supported for Ollama backends"}
            return
        state = self.pulls.setdefault(model, {"status": "queued", "completed": 0, "total": 0, "error": None, "started_at": time.time()})
        async with self._pull_lock:  # one download at a time
            state["status"] = "downloading"
            log.info("pulling %s ...", model)
            last_pct = -10
            try:
                async with self.client.stream(
                    "POST", f"{self.s.backend_url}/api/pull", json={"model": model, "stream": True},
                    timeout=httpx.Timeout(connect=10, read=600, write=60, pool=60),
                ) as resp:
                    if resp.status_code >= 400:
                        raise BackendError((await resp.aread()).decode("utf-8", "replace")[:300], resp.status_code)
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        with contextlib.suppress(ValueError):
                            ev = json.loads(line)
                            if ev.get("error"):
                                raise BackendError(str(ev["error"]))
                            if ev.get("total"):
                                state["total"] = ev["total"]
                                state["completed"] = ev.get("completed") or 0
                                pct = int(100 * state["completed"] / max(1, state["total"]))
                                if pct >= last_pct + 10:
                                    last_pct = pct
                                    log.info("pulling %s: %d%%", model, pct)
                            if ev.get("status") == "success":
                                state["status"] = "done"
                if state["status"] != "done":
                    state["status"] = "done"
                log.info("pulled %s", model)
            except (BackendError, httpx.HTTPError) as exc:
                state["status"] = "error"
                state["error"] = getattr(exc, "message", None) or str(exc)
                log.error("pull of %s failed: %s", model, state["error"])
        await self.refresh_backend()

    # -- HTTP handlers --------------------------------------------------------------

    def _auth(self, request: Request) -> Response | None:
        if not consteq(bearer_token(request.headers), self.s.cluster_token):
            return oai.error(401, "invalid cluster token")
        return None

    async def _body(self, request: Request) -> tuple[bytes, dict] | Response:
        raw = await request.body()
        if len(raw) > MAX_BODY:
            return oai.error(413, "request body too large")
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            return oai.error(400, "request body is not valid JSON")
        if not isinstance(data, dict):
            return oai.error(400, "request body must be a JSON object")
        return raw, data

    async def _acquire_slot(self) -> bool:
        try:
            await asyncio.wait_for(self.slots.acquire(), timeout=self.s.queue_timeout)
        except asyncio.TimeoutError:
            return False
        self.in_flight += 1
        return True

    def _release_slot(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.slots.release()

    def _backend_headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.s.backend_api_key:
            h["authorization"] = f"Bearer {self.s.backend_api_key}"
        return h

    async def chat(self, request: Request) -> Response:
        return await self._proxy(request, "/chat/completions")

    async def embeddings(self, request: Request) -> Response:
        return await self._proxy(request, "/embeddings")

    async def _proxy(self, request: Request, path: str) -> Response:
        denied = self._auth(request)
        if denied:
            return denied
        parsed = await self._body(request)
        if isinstance(parsed, Response):
            return parsed
        raw, data = parsed
        assert self.client and self.backend
        if not await self._acquire_slot():
            return oai.error(503, f"node '{self.s.node_name}' is busy")
        url = self.backend.v1 + path
        timeout = httpx.Timeout(connect=10, read=self.s.request_timeout, write=60, pool=self.s.request_timeout)
        if not data.get("stream"):
            try:
                resp = await self.client.post(url, content=raw, headers=self._backend_headers(), timeout=timeout)
                return Response(resp.content, status_code=resp.status_code,
                                media_type=resp.headers.get("content-type", "application/json"))
            except httpx.HTTPError as exc:
                return oai.error(502, f"backend error on node '{self.s.node_name}': {type(exc).__name__}: {exc}")
            finally:
                self._release_slot()

        req = self.client.build_request("POST", url, content=raw, headers=self._backend_headers(), timeout=timeout)
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.HTTPError as exc:
            self._release_slot()
            return oai.error(502, f"backend error on node '{self.s.node_name}': {type(exc).__name__}: {exc}")
        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            self._release_slot()
            return Response(body, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"))

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for piece in resp.aiter_raw():
                    yield piece
            except httpx.HTTPError as exc:
                yield oai.sse(oai.error_body(f"backend stream broke on node '{self.s.node_name}': {exc}"))
            finally:
                await resp.aclose()
                self._release_slot()

        return StreamingResponse(relay(), status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type", "text/event-stream"),
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    async def v1_models(self, request: Request) -> Response:
        denied = self._auth(request)
        if denied:
            return denied
        return JSONResponse({"object": "list", "data": [
            {"id": m["name"], "object": "model", "created": 0, "owned_by": self.s.node_name} for m in self.models
        ]})

    async def info(self, request: Request) -> Response:
        denied = self._auth(request)
        if denied:
            return denied
        return JSONResponse({**self.heartbeat_payload(), "pulls": self.pulls})

    async def health(self, request: Request) -> Response:
        return JSONResponse({"status": "ok", "node": self.s.node_name, "backend_ok": self.backend_ok, "version": __version__})

    async def admin_pull(self, request: Request) -> Response:
        denied = self._auth(request)
        if denied:
            return denied
        if request.method == "GET":
            return JSONResponse({"pulls": self.pulls})
        parsed = await self._body(request)
        if isinstance(parsed, Response):
            return parsed
        model = str(parsed[1].get("model") or "").strip()
        if not model:
            return oai.error(400, "give `model`")
        if self.s.backend != "ollama":
            return oai.error(400, "this node's backend cannot download models on request")
        if canonical_model(model) in {canonical_model(m["name"]) for m in self.models}:
            return JSONResponse({"status": "present"})
        return JSONResponse(self.start_pull(model), status_code=202)


def build_app(settings: AgentSettings | None = None) -> Starlette:
    settings = settings or AgentSettings.from_env()
    agent = Agent(settings)
    app = Starlette(
        routes=[
            Route("/health", agent.health),
            Route("/info", agent.info),
            Route("/v1/models", agent.v1_models),
            Route("/v1/chat/completions", agent.chat, methods=["POST"]),
            Route("/v1/embeddings", agent.embeddings, methods=["POST"]),
            Route("/admin/pull", agent.admin_pull, methods=["GET", "POST"]),
        ],
        lifespan=agent.lifespan,
    )
    app.state.agent = agent
    return app


def serve(settings: AgentSettings | None = None) -> None:
    import uvicorn

    settings = settings or AgentSettings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if settings.log_level != "debug":
        logging.getLogger("httpx").setLevel(logging.WARNING)
    if not settings.cluster_token:
        log.error("FLOTILLA_CLUSTER_TOKEN is required (the same value as on the coordinator)")
        raise SystemExit(2)
    if not settings.coordinator_url:
        log.warning("FLOTILLA_COORDINATOR_URL is not set: this agent will not register anywhere")
    uvicorn.run(
        build_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level if settings.log_level in ("critical", "error", "warning", "info", "debug") else "info",
        access_log=settings.log_level == "debug",
        ssl_certfile=settings.tls_cert,
        ssl_keyfile=settings.tls_key,
        timeout_keep_alive=30,
    )
