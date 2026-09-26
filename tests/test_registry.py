import asyncio
import time
import unittest

from flotilla.config import ClusterConfig, StaticNodeConfig
from flotilla.registry import NoCandidate, Placement, QueueTimeout, Registry


def hb(name, models, capacity=2, loaded=(), labels=None, backend_ok=True):
    return {
        "name": name,
        "url": f"http://{name}:8801",
        "instance_id": f"id-{name}",
        "models": [{"name": m} for m in models],
        "loaded_models": list(loaded),
        "max_concurrency": capacity,
        "labels": labels or {},
        "backend_ok": backend_ok,
    }


class RegistryTests(unittest.IsolatedAsyncioTestCase):
    def make(self, **kw):
        reg = Registry(ClusterConfig(**kw), agent_token="tok")
        return reg

    async def test_least_loaded_spreads_work(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["m"]), None)
        reg.upsert_agent(hb("b", ["m"]), None)
        l1 = await reg.acquire(["m"])
        l2 = await reg.acquire(["m"])
        self.assertNotEqual(l1.node.name, l2.node.name)
        self.assertEqual(l1.node.api_key, "tok")

    async def test_waits_for_free_slot(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["m"], capacity=1), None)
        first = await reg.acquire(["m"])
        waiter = asyncio.create_task(reg.acquire(["m"], timeout=5))
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        reg.release(first, ok=True)
        second = await asyncio.wait_for(waiter, 1)
        self.assertEqual(second.node.name, "a")

    async def test_queue_timeout(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["m"], capacity=1), None)
        await reg.acquire(["m"])
        with self.assertRaises(QueueTimeout):
            await reg.acquire(["m"], timeout=0.1)

    async def test_no_candidate(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["m"]), None)
        with self.assertRaises(NoCandidate) as ctx:
            await reg.acquire(["other"])
        self.assertIn("a: m", str(ctx.exception))

    async def test_fallback_model_when_first_missing(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["small"]), None)
        lease = await reg.acquire(["big", "small"])
        self.assertEqual(lease.model, "small")

    async def test_prefers_first_model_when_busy_unless_configured(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["big"], capacity=1), None)
        reg.upsert_agent(hb("b", ["small"]), None)
        await reg.acquire(["big"])
        with self.assertRaises(QueueTimeout):
            await reg.acquire(["big", "small"], timeout=0.1)
        reg2 = self.make(fallback_on_busy=True)
        reg2.upsert_agent(hb("a", ["big"], capacity=1), None)
        reg2.upsert_agent(hb("b", ["small"]), None)
        await reg2.acquire(["big"])
        lease = await reg2.acquire(["big", "small"], timeout=0.1)
        self.assertEqual(lease.model, "small")

    async def test_placement(self):
        reg = self.make()
        reg.upsert_agent(hb("cpu", ["m"]), None)
        reg.upsert_agent(hb("gpu", ["m"], labels={"gpu": "yes"}), None)
        for _ in range(3):
            lease = await reg.acquire(["m"], Placement(labels={"gpu": "yes"}))
            self.assertEqual(lease.node.name, "gpu")
            reg.release(lease, ok=True)
        lease = await reg.acquire(["m"], Placement(node="cpu"))
        self.assertEqual(lease.node.name, "cpu")
        reg.release(lease, ok=True)
        lease = await reg.acquire(["m"], Placement(prefer_labels={"gpu": "yes"}))
        self.assertEqual(lease.node.name, "gpu")

    async def test_prefers_loaded_model(self):
        reg = self.make()
        reg.upsert_agent(hb("cold", ["m"]), None)
        reg.upsert_agent(hb("hot", ["m"], loaded=["m"]), None)
        for _ in range(5):
            lease = await reg.acquire(["m"])
            self.assertEqual(lease.node.name, "hot")
            reg.release(lease, ok=True)

    async def test_ollama_name_matching(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["llama3.2:latest"]), None)
        lease = await reg.acquire(["llama3.2"])
        self.assertEqual(lease.model, "llama3.2:latest")

    async def test_cooldown_after_connection_failure(self):
        reg = self.make(failure_cooldown=60)
        reg.upsert_agent(hb("a", ["m"]), None)
        reg.upsert_agent(hb("b", ["m"]), None)
        lease = await reg.acquire(["m"])
        bad = lease.node.name
        reg.release(lease, ok=False, error_kind="connection", error="refused")
        self.assertEqual(reg.status(reg.nodes[bad]), "cooldown")
        for _ in range(3):
            lease = await reg.acquire(["m"])
            self.assertNotEqual(lease.node.name, bad)
            reg.release(lease, ok=True)

    async def test_not_found_removes_model(self):
        reg = self.make()
        reg.upsert_agent(hb("a", ["m", "n"]), None)
        lease = await reg.acquire(["m"])
        reg.release(lease, ok=False, error_kind="not_found", error="no such model")
        self.assertNotIn("m", reg.nodes["a"].models)
        with self.assertRaises(NoCandidate):
            await reg.acquire(["m"])

    async def test_offline_after_heartbeat_timeout_and_back(self):
        reg = self.make(heartbeat_timeout=1)
        node = reg.upsert_agent(hb("a", ["m"]), None)
        node.last_seen = time.time() - 5
        self.assertEqual(reg.status(node), "offline")
        with self.assertRaises(NoCandidate):
            await reg.acquire(["m"])
        reg.upsert_agent(hb("a", ["m"]), None)
        self.assertEqual(reg.status(node), "online")

    async def test_backend_down_means_offline(self):
        reg = self.make()
        node = reg.upsert_agent(hb("a", ["m"], backend_ok=False), None)
        self.assertEqual(reg.status(node), "offline")

    async def test_unreachable_agent_not_scheduled(self):
        reg = self.make()
        node = reg.upsert_agent(hb("a", ["m"]), None)
        reg.set_reachable(node, False, "refused")
        self.assertEqual(reg.status(node), "unreachable")
        with self.assertRaises(NoCandidate):
            await reg.acquire(["m"])

    async def test_url_inferred_from_client_address(self):
        reg = self.make()
        data = hb("a", ["m"])
        data["url"] = None
        data["port"] = 9000
        node = reg.upsert_agent(data, "10.1.2.3")
        self.assertEqual(node.url, "http://10.1.2.3:9000")

    async def test_static_nodes(self):
        reg = self.make(static_nodes=[StaticNodeConfig(name="s", url="http://s:11434/", models=["m"])])
        node = reg.nodes["s"]
        self.assertEqual(reg.status(node), "offline")          # until probed
        reg.update_static(node, None, None, ok=True)
        lease = await reg.acquire(["m"])
        self.assertEqual(lease.node.url, "http://s:11434")
        with self.assertRaises(ValueError):
            reg.upsert_agent(hb("s", ["m"]), None)             # name taken by a static node

    async def test_static_agent_uses_cluster_token(self):
        nodes = [
            StaticNodeConfig(name="agent", url="http://a:8801", kind="agent"),
            StaticNodeConfig(name="agent-key", url="http://b:8801", kind="agent", api_key="own"),
            StaticNodeConfig(name="plain", url="http://c:8000", kind="openai"),
        ]
        reg = self.make(static_nodes=nodes)
        keys = {n: reg.nodes[n].api_key for n in ("agent", "agent-key", "plain")}
        self.assertEqual(keys, {"agent": "tok", "agent-key": "own", "plain": None})
        reg.reconfigure(ClusterConfig(static_nodes=nodes))    # an unchanged node keeps its key
        self.assertEqual(reg.nodes["agent"].api_key, "tok")

    async def test_speed_breaks_ties(self):
        reg = self.make()
        reg.upsert_agent(hb("slow", ["m"]), None)
        reg.upsert_agent(hb("fast", ["m"]), None)
        reg.nodes["slow"].spt = 0.3
        reg.nodes["fast"].spt = 0.02
        lease = await reg.acquire(["m"])
        self.assertEqual(lease.node.name, "fast")


if __name__ == "__main__":
    unittest.main()
