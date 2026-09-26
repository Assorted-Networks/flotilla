"""Cluster membership and slot scheduling.

Nodes are either agents (which heartbeat to the coordinator) or static
OpenAI-compatible servers listed in the config (which the coordinator probes).
The registry tracks how many requests the coordinator has in flight on every
node and hands out slots: least-loaded first, preferring nodes that already
have the model in memory and nodes that have been fast.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from flotilla.config import ClusterConfig, StaticNodeConfig
from flotilla.util import canonical_model

log = logging.getLogger("flotilla.registry")


class NoCandidate(Exception):
    """No online node can serve any of the requested models."""


class QueueTimeout(Exception):
    """Nodes exist for the model but none freed a slot in time."""


@dataclass
class Placement:
    node: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    prefer_labels: dict[str, str] = field(default_factory=dict)

    def allows(self, node: "Node") -> bool:
        if self.node and node.name != self.node:
            return False
        return all(node.labels.get(k) == v for k, v in self.labels.items())

    def preference(self, node: "Node") -> int:
        return sum(1 for k, v in self.prefer_labels.items() if node.labels.get(k) == v)

    def describe(self) -> str:
        parts = []
        if self.node:
            parts.append(f"node={self.node}")
        parts += [f"{k}={v}" for k, v in self.labels.items()]
        return ", ".join(parts)


@dataclass
class NodeModel:
    name: str
    size: int | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization": self.quantization,
        }


@dataclass
class Node:
    name: str
    url: str
    kind: str                      # "agent", "ollama" or "openai"
    api_key: str | None = None
    static: bool = False
    capacity: int = 2
    models: dict[str, NodeModel] = field(default_factory=dict)
    loaded: set[str] = field(default_factory=set)
    labels: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    backend: str | None = None
    version: str | None = None
    instance_id: str | None = None
    backend_ok: bool = True
    # Whether the coordinator could open a connection to `url` (agents only;
    # None until checked). A node whose heartbeats arrive but whose URL is
    # unreachable usually needs FLOTILLA_ADVERTISE_URL.
    reachable: bool | None = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = 0.0
    in_flight: int = 0
    reported_in_flight: int = 0
    fail_streak: int = 0
    cooldown_until: float = 0.0
    last_error: str | None = None
    spt: float | None = None       # EWMA seconds per completion token
    requests: int = 0
    failures: int = 0
    completion_tokens: int = 0


@dataclass
class Lease:
    node: Node
    model: str                     # exact model name on that node
    canonical: str
    started: float = field(default_factory=time.monotonic)
    released: bool = False


def _models_from_list(items: Iterable[Any]) -> dict[str, NodeModel]:
    out: dict[str, NodeModel] = {}
    for item in items or []:
        if isinstance(item, str):
            m = NodeModel(name=item)
        elif isinstance(item, dict) and item.get("name"):
            m = NodeModel(
                name=str(item["name"]),
                size=item.get("size"),
                family=item.get("family"),
                parameter_size=item.get("parameter_size"),
                quantization=item.get("quantization"),
            )
        else:
            continue
        out.setdefault(canonical_model(m.name), m)
    return out


class Registry:
    def __init__(self, cfg: ClusterConfig, agent_token: str | None = None):
        self.cfg = cfg
        # Agents authenticate the coordinator with the shared cluster token.
        self.agent_token = agent_token
        self.nodes: dict[str, Node] = {}
        self._waiters: set[asyncio.Future] = set()
        for s in cfg.static_nodes:
            self.add_static(s)

    # -- membership ---------------------------------------------------------

    def add_static(self, s: StaticNodeConfig) -> Node:
        node = Node(
            name=s.name,
            url=s.url.rstrip("/"),
            kind=s.kind,
            api_key=s.api_key,
            static=True,
            capacity=s.max_concurrency,
            models=_models_from_list(s.models),
            labels=dict(s.labels),
            backend=s.kind,
            backend_ok=False,       # until the first successful probe
        )
        self.nodes[node.name] = node
        return node

    def reconfigure(self, cfg: ClusterConfig) -> None:
        """Apply a reloaded cluster config: replace static nodes, keep agents."""
        self.cfg = cfg
        wanted = {s.name: s for s in cfg.static_nodes}
        for name in [n for n, node in self.nodes.items() if node.static and n not in wanted]:
            del self.nodes[name]
        for name, s in wanted.items():
            old = self.nodes.get(name)
            if old is None or not old.static or old.url != s.url.rstrip("/") or old.kind != s.kind:
                self.add_static(s)
            else:
                old.capacity, old.labels, old.api_key = s.max_concurrency, dict(s.labels), s.api_key
        self._wake()

    def upsert_agent(self, hb: dict[str, Any], client_host: str | None) -> Node:
        name = str(hb.get("name") or "").strip()
        if not name:
            raise ValueError("heartbeat without a node name")
        url = (hb.get("url") or "").rstrip("/")
        if not url:
            if not client_host:
                raise ValueError("heartbeat without url and unknown client address")
            host = f"[{client_host}]" if ":" in client_host else client_host
            url = f"http://{host}:{int(hb.get('port') or 8801)}"
        node = self.nodes.get(name)
        if node is not None and node.static:
            raise ValueError(f"node name '{name}' is already used by a static node in the config")
        before = None
        if node is None:
            node = Node(name=name, url=url, kind="agent")
            self.nodes[name] = node
            log.info("node joined: %s at %s", name, url)
        else:
            before = self.status(node)
            if node.instance_id and hb.get("instance_id") and node.instance_id != hb.get("instance_id"):
                if time.time() - node.last_seen < self.cfg.heartbeat_timeout and node.url != url:
                    log.warning("two agents are using the node name '%s' (%s and %s)", name, node.url, url)
                else:
                    log.info("node restarted: %s", name)
            if node.url != url:
                log.info("node %s moved: %s -> %s", name, node.url, url)
                node.reachable = None
        node.url = url
        node.api_key = self.agent_token
        node.instance_id = hb.get("instance_id")
        node.capacity = max(1, int(hb.get("max_concurrency") or 2))
        node.models = _models_from_list(hb.get("models") or [])
        node.loaded = {canonical_model(m) for m in hb.get("loaded_models") or []}
        node.labels = {str(k): str(v) for k, v in (hb.get("labels") or {}).items()}
        node.hardware = hb.get("hardware") or {}
        node.backend = hb.get("backend")
        node.version = hb.get("version")
        node.backend_ok = bool(hb.get("backend_ok", True))
        if not node.backend_ok:
            node.last_error = f"backend unavailable: {hb.get('backend_error') or 'unknown error'}"
        node.reported_in_flight = int(hb.get("in_flight") or 0)
        node.last_seen = time.time()
        after = self.status(node)
        if before is not None and before != after:
            if after == "offline":
                log.warning("node %s is offline: %s", name, node.last_error)
            elif before == "offline":
                log.info("node back online: %s", name)
        self._wake()
        return node

    def deregister(self, name: str) -> bool:
        node = self.nodes.get(name)
        if node is None or node.static:
            return False
        node.last_seen = 0.0
        log.info("node left: %s", name)
        self._wake()
        return True

    def set_reachable(self, node: Node, ok: bool, error: str | None = None) -> None:
        if node.reachable is not ok:
            if ok:
                log.info("node %s is reachable at %s", node.name, node.url)
            else:
                log.warning(
                    "node %s sends heartbeats but %s is unreachable from the coordinator (%s); "
                    "set FLOTILLA_ADVERTISE_URL on that agent", node.name, node.url, error,
                )
        node.reachable = ok
        if not ok:
            node.last_error = f"unreachable at {node.url}: {error}"
        self._wake()

    def update_static(self, node: Node, models: dict[str, NodeModel] | None, loaded: set[str] | None, ok: bool, error: str | None = None) -> None:
        was_ok = node.backend_ok
        node.backend_ok = ok
        if ok:
            node.last_seen = time.time()
            if models is not None:
                node.models = models
            if loaded is not None:
                node.loaded = loaded
            node.last_error = None
            if not was_ok:
                log.info("static node online: %s", node.name)
        else:
            node.last_error = error
            if was_ok:
                log.warning("static node unreachable: %s (%s)", node.name, error)
        self._wake()

    def prune(self) -> list[str]:
        """Forget agents that have been silent for longer than node_expiry."""
        cutoff = time.time() - self.cfg.node_expiry
        gone = [n for n, node in self.nodes.items() if not node.static and node.last_seen < cutoff and node.in_flight == 0]
        for name in gone:
            left = self.nodes[name].last_seen == 0.0
            del self.nodes[name]
            if left:
                log.info("node removed after leaving: %s", name)
            else:
                log.info("node forgotten after %ss offline: %s", int(self.cfg.node_expiry), name)
        return gone

    # -- status ---------------------------------------------------------------

    def status(self, node: Node, now: float | None = None) -> str:
        now = time.time() if now is None else now
        if node.static:
            alive = node.backend_ok
        else:
            alive = node.backend_ok and (now - node.last_seen) <= self.cfg.heartbeat_timeout
        if not alive:
            return "offline"
        if node.reachable is False:
            return "unreachable"
        if node.cooldown_until > now:
            return "cooldown"
        return "online"

    def schedulable(self, node: Node, now: float | None = None) -> bool:
        return self.status(node, now) == "online"

    def online_nodes(self) -> list[Node]:
        now = time.time()
        return [n for n in self.nodes.values() if self.status(n, now) in ("online", "cooldown")]

    def models_index(self) -> dict[str, dict[str, Any]]:
        """canonical name -> {name, nodes, loaded_on} for nodes that are up."""
        index: dict[str, dict[str, Any]] = {}
        for node in self.online_nodes():
            for canon, m in node.models.items():
                entry = index.setdefault(canon, {"name": m.name, "nodes": [], "loaded_on": [], "details": m.as_dict()})
                entry["nodes"].append(node.name)
                if canon in node.loaded:
                    entry["loaded_on"].append(node.name)
        return index

    def has_model(self, name: str) -> bool:
        canon = canonical_model(name)
        return any(canon in n.models for n in self.online_nodes())

    def snapshot(self) -> list[dict[str, Any]]:
        now = time.time()
        out = []
        for node in sorted(self.nodes.values(), key=lambda n: n.name):
            out.append({
                "name": node.name,
                "url": node.url,
                "kind": node.kind,
                "static": node.static,
                "status": self.status(node, now),
                "backend": node.backend,
                "backend_ok": node.backend_ok,
                "reachable": node.reachable,
                "version": node.version,
                "capacity": node.capacity,
                "in_flight": node.in_flight,
                "reported_in_flight": node.reported_in_flight,
                "models": [m.as_dict() for m in node.models.values()],
                "loaded_models": sorted(node.models[c].name for c in node.loaded if c in node.models),
                "labels": node.labels,
                "hardware": node.hardware,
                "last_seen": node.last_seen or None,
                "seconds_since_seen": round(now - node.last_seen, 1) if node.last_seen else None,
                "first_seen": node.first_seen,
                "requests": node.requests,
                "failures": node.failures,
                "completion_tokens": node.completion_tokens,
                "seconds_per_token": round(node.spt, 4) if node.spt else None,
                "last_error": node.last_error,
            })
        return out

    # -- scheduling -----------------------------------------------------------

    def _wake(self) -> None:
        for fut in list(self._waiters):
            if not fut.done():
                fut.set_result(None)
        self._waiters.clear()

    def _cost(self, node: Node, canon: str, placement: Placement) -> tuple:
        load = node.in_flight / max(1, node.capacity)
        cold = 0.0 if canon in node.loaded else 0.5
        speed = min(node.spt, 1.0) if node.spt else 0.1
        return (-placement.preference(node), load + cold + 2.0 * speed, random.random())

    def _pick(self, candidates: list[str], placement: Placement, exclude: set[str]) -> tuple[Lease | None, bool]:
        """Returns (lease, any_capable)."""
        now = time.time()
        any_capable = False
        for cand in candidates:
            canon = canonical_model(cand)
            capable = [
                n for n in self.nodes.values()
                if canon in n.models and n.name not in exclude and placement.allows(n) and self.schedulable(n, now)
            ]
            if not capable:
                continue
            any_capable = True
            free = [n for n in capable if n.in_flight < n.capacity]
            if free:
                best = min(free, key=lambda n: self._cost(n, canon, placement))
                best.in_flight += 1
                return Lease(node=best, model=best.models[canon].name, canonical=canon), True
            if not self.cfg.fallback_on_busy:
                # The preferred model exists but is busy: wait for it rather
                # than quietly downgrading to a fallback model.
                return None, True
        return None, any_capable

    def explain(self, candidates: list[str], placement: Placement, exclude: set[str]) -> str:
        wanted = ", ".join(candidates) or "(no model configured)"
        where = f" matching {placement.describe()}" if placement.describe() else ""
        online = self.online_nodes()
        if not online:
            return f"no nodes are online (wanted: {wanted})"
        detail = "; ".join(
            f"{n.name}: {', '.join(sorted(m.name for m in n.models.values())) or 'no models'}" for n in online
        )
        tried = f" (already failed on: {', '.join(sorted(exclude))})" if exclude else ""
        return f"no online node{where} serves any of [{wanted}]{tried}. Online nodes: {detail}"

    async def acquire(
        self,
        candidates: list[str],
        placement: Placement | None = None,
        exclude: set[str] | None = None,
        timeout: float | None = None,
    ) -> Lease:
        placement = placement or Placement()
        exclude = exclude or set()
        timeout = self.cfg.queue_timeout if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            lease, any_capable = self._pick(candidates, placement, exclude)
            if lease is not None:
                return lease
            if not any_capable:
                raise NoCandidate(self.explain(candidates, placement, exclude))
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise QueueTimeout(
                    f"all nodes serving [{', '.join(candidates)}] stayed busy for {int(timeout)}s"
                )
            fut: asyncio.Future = loop.create_future()
            self._waiters.add(fut)
            try:
                # Wake on any release/heartbeat, and re-check at least every
                # few seconds so expiring cooldowns are noticed.
                await asyncio.wait_for(fut, timeout=min(remaining, 2.0))
            except asyncio.TimeoutError:
                pass
            finally:
                self._waiters.discard(fut)

    def release(
        self,
        lease: Lease,
        ok: bool | None,
        completion_tokens: int = 0,
        error_kind: str | None = None,
        error: str | None = None,
    ) -> None:
        """Return a slot. ok=None means cancelled (no health signal)."""
        if lease.released:
            return
        lease.released = True
        node = lease.node
        node.in_flight = max(0, node.in_flight - 1)
        duration = time.monotonic() - lease.started
        if ok:
            node.requests += 1
            node.fail_streak = 0
            node.loaded.add(lease.canonical)
            node.completion_tokens += completion_tokens
            if completion_tokens >= 8 and duration > 0:
                sample = duration / completion_tokens
                node.spt = sample if node.spt is None else 0.7 * node.spt + 0.3 * sample
        elif ok is False:
            node.failures += 1
            node.fail_streak += 1
            node.last_error = error
            # A node that refuses connections is probably down: stop sending
            # it work until the next heartbeat or the cooldown ends.
            if error_kind == "connection" or node.fail_streak >= 2:
                node.cooldown_until = time.time() + self.cfg.failure_cooldown
                log.warning("node %s in cooldown for %ss after: %s", node.name, int(self.cfg.failure_cooldown), error)
            if error_kind == "not_found":
                node.models.pop(lease.canonical, None)
                node.loaded.discard(lease.canonical)
        self._wake()

    def in_flight_total(self) -> int:
        return sum(n.in_flight for n in self.nodes.values())
