"""Start a real multi-process cluster on localhost for end-to-end tests.

Every "machine" is a fake inference server plus an agent process; the
coordinator is its own process. They only talk over HTTP, as they would
across real machines.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

TEST_CLUSTER_CONFIG = """
cluster:
  heartbeat_timeout: 2.5
  failure_cooldown: 3
  probe_interval: 0.5
  request_timeout: 30
  queue_timeout: 20
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Cluster:
    def __init__(self, nodes: dict[str, list[str]], api_key: str | None = None, token: str = "cluster-secret",
                 extra_config: str = ""):
        self.nodes = nodes
        self.api_key = api_key
        self.token = token
        self.dir = Path(tempfile.mkdtemp(prefix="flotilla-test-"))
        self.procs: dict[str, subprocess.Popen] = {}
        self._logs: list = []
        self.ports: dict[str, int] = {}
        default = (ROOT / "config" / "flotilla.yaml").read_text()
        # Keep the default teams and members, override cluster timings.
        teams_part = default[default.index("defaults:"):]
        self.config_path = self.dir / "flotilla.yaml"
        self.config_path.write_text(TEST_CLUSTER_CONFIG + extra_config + "\n" + teams_part)
        self.coord_port = free_port()
        self.url = f"http://127.0.0.1:{self.coord_port}"

    # -- processes ------------------------------------------------------------

    def _spawn(self, name: str, args: list[str], env: dict[str, str]) -> subprocess.Popen:
        log = open(self.dir / f"{name}.log", "w")
        self._logs.append(log)
        full_env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1", **env}
        for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            full_env.pop(k, None)
        proc = subprocess.Popen([sys.executable, "-m", "flotilla", *args], env=full_env, stdout=log,
                                stderr=subprocess.STDOUT, cwd=str(ROOT))
        self.procs[name] = proc
        return proc

    def start_backend(self, node: str, models: list[str]) -> None:
        port = free_port()
        self.ports[f"backend:{node}"] = port
        self._spawn(f"backend-{node}", ["fake-backend", "--name", f"fake-{node}", "--models", ",".join(models),
                                        "--port", str(port), "--latency", "0.02"], {})

    def start_agent(self, node: str, token: str | None = None, advertise: str | None = "auto", **env) -> None:
        port = self.ports.get(f"agent:{node}") or free_port()
        self.ports[f"agent:{node}"] = port
        if advertise == "auto":
            advertise = f"http://127.0.0.1:{port}"
        self._spawn(f"agent-{node}", ["agent"], {
            "FLOTILLA_COORDINATOR_URL": self.url,
            "FLOTILLA_CLUSTER_TOKEN": token or self.token,
            "FLOTILLA_NODE_NAME": node,
            "FLOTILLA_PORT": str(port),
            "FLOTILLA_BACKEND_URL": f"http://127.0.0.1:{self.ports[f'backend:{node}']}",
            "FLOTILLA_ADVERTISE_URL": advertise or "",
            "FLOTILLA_HEARTBEAT_INTERVAL": "0.5",
            "FLOTILLA_MAX_CONCURRENCY": "3",
            **env,
        })

    def start(self) -> "Cluster":
        env = {
            "FLOTILLA_CONFIG": str(self.config_path),
            "FLOTILLA_PORT": str(self.coord_port),
            "FLOTILLA_CLUSTER_TOKEN": self.token,
            "FLOTILLA_DATA_DIR": str(self.dir / "data"),
        }
        if self.api_key:
            env["FLOTILLA_API_KEYS"] = self.api_key
        self._spawn("coordinator", ["coordinator"], env)
        for node, models in self.nodes.items():
            self.start_backend(node, models)
        self.wait_http(f"{self.url}/health")
        for node in self.nodes:
            self.wait_http(f"http://127.0.0.1:{self.ports[f'backend:{node}']}/health")
            self.start_agent(node)
        self.wait_for(lambda: self.online() == set(self.nodes), 20, "all nodes online")
        return self

    def stop(self) -> None:
        for proc in self.procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs.values():
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for log in self._logs:
            log.close()

    def kill(self, name: str) -> None:
        proc = self.procs[name]
        proc.kill()
        proc.wait(timeout=5)

    def logs(self) -> str:
        out = []
        for f in sorted(self.dir.glob("*.log")):
            out.append(f"===== {f.name}\n{f.read_text()[-4000:]}")
        return "\n".join(out)

    # -- helpers ----------------------------------------------------------------

    def headers(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def client(self, **kw) -> httpx.Client:
        return httpx.Client(base_url=self.url, headers=self.headers(), timeout=60, trust_env=False, **kw)

    @staticmethod
    def wait_http(url: str, timeout: float = 20) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if httpx.get(url, timeout=1, trust_env=False).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise TimeoutError(f"{url} did not come up")

    @staticmethod
    def wait_for(pred, timeout: float, what: str) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if pred():
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise TimeoutError(f"timed out waiting for: {what}")

    def cluster_state(self) -> dict:
        with self.client() as c:
            return c.get("/api/cluster").json()

    def online(self) -> set[str]:
        return {n["name"] for n in self.cluster_state()["nodes"] if n["status"] == "online"}

    def node(self, name: str) -> dict:
        return next(n for n in self.cluster_state()["nodes"] if n["name"] == name)

    def backend_control(self, node: str, **body) -> dict:
        port = self.ports[f"backend:{node}"]
        return httpx.post(f"http://127.0.0.1:{port}/_fake/control", json=body, trust_env=False).json()

    def backend_stats(self, node: str) -> dict:
        port = self.ports[f"backend:{node}"]
        return httpx.get(f"http://127.0.0.1:{port}/_fake/stats", trust_env=False).json()


def parse_sse(text: str) -> list:
    events = []
    for block in text.split("\n\n"):
        data = [ln[5:].lstrip() for ln in block.splitlines() if ln.startswith("data:")]
        if not data:
            continue
        payload = "\n".join(data)
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events
