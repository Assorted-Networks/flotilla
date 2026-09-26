"""End-to-end tests: coordinator, agents and fake backends as separate processes."""

import json
import threading
import time
import unittest

import httpx

from tests.cluster import Cluster, parse_sse

NODES = {
    # Every small model lives on at least two machines, so any one can fail.
    "node-a": ["qwen3.5:4b", "granite4.2:3b", "ministral-3:3b", "nomic-embed-text"],
    "node-b": ["qwen3.5:4b", "granite4.2:3b", "qwen3.5:9b"],
    "node-c": ["ministral-3:3b", "granite4.2:3b", "qwen3.5:4b"],
}

TEAMS = ["moa", "council", "planner", "critique", "vote", "pipeline", "auto"]


class LogOnFailure:
    """Print the process logs when a test fails."""

    def run(self, result=None):
        before = len(result.failures) + len(result.errors) if result else 0
        res = super().run(result)
        if result is not None and len(result.failures) + len(result.errors) > before and getattr(self, "cluster", None):
            print(self.cluster.logs())
        return res


class ClusterTests(LogOnFailure, unittest.TestCase):
    cluster: Cluster

    @classmethod
    def setUpClass(cls):
        cls.cluster = Cluster(NODES, api_key="client-key")
        try:
            cls.cluster.start()
        except Exception:
            print(cls.cluster.logs())
            cls.cluster.stop()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.cluster.stop()

    def chat(self, model, content="How do heat pumps work?", **extra):
        with self.cluster.client() as c:
            return c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": content}], **extra})

    def test_nodes_registered_with_models(self):
        state = self.cluster.cluster_state()
        self.assertEqual(state["totals"]["online"], 3)
        self.assertEqual(state["totals"]["slots"], 9)
        self.assertIn("qwen3.5:9b", state["models"])
        self.assertEqual(sorted(state["models"]["granite4.2:3b"]["nodes"]), ["node-a", "node-b", "node-c"])
        for n in state["nodes"]:
            self.assertTrue(n["reachable"], n)
            self.assertEqual(n["kind"], "agent")

    def test_models_endpoint_lists_teams_and_models(self):
        with self.cluster.client() as c:
            ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        for team in TEAMS:
            self.assertIn(f"team/{team}", ids)
        self.assertIn("qwen3.5:9b", ids)

    def test_every_team_non_streaming(self):
        for team in TEAMS:
            with self.subTest(team=team):
                r = self.chat(f"team/{team}")
                self.assertEqual(r.status_code, 200, r.text)
                data = r.json()
                self.assertTrue(data["choices"][0]["message"]["content"].strip())
                self.assertGreater(data["usage"]["total_tokens"], 0)
                self.assertTrue(data["flotilla"]["nodes"])
                self.assertIn("x-flotilla-task-id", r.headers)

    def test_work_spreads_across_machines(self):
        r = self.chat("team/moa")
        task_id = r.headers["x-flotilla-task-id"]
        with self.cluster.client() as c:
            task = c.get(f"/api/tasks/{task_id}").json()
        proposer_nodes = {s["node"] for s in task["step_details"] if s["role"] == "proposer"}
        self.assertGreaterEqual(len(proposer_nodes), 2, task["step_details"])
        agg = next(s for s in task["step_details"] if s["role"] == "aggregator")
        self.assertEqual((agg["model"], agg["node"]), ("qwen3.5:9b", "node-b"))

    def test_streaming_team(self):
        with self.cluster.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "team/planner", "stream": True, "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Plan a home network upgrade"}],
            }) as r:
                self.assertEqual(r.status_code, 200)
                self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
                body = r.read().decode()
        events = parse_sse(body)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [e for e in events if isinstance(e, dict)]
        content = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks if ch.get("choices"))
        progress = "".join(ch["choices"][0]["delta"].get("reasoning_content") or "" for ch in chunks if ch.get("choices"))
        self.assertTrue(content.startswith("Synthesis by"), content)
        self.assertIn("plan: s1", progress)
        self.assertIn("synthesizer done", progress)
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertGreater(chunks[-1]["usage"]["total_tokens"], 0)
        finishes = [ch["choices"][0]["finish_reason"] for ch in chunks if ch.get("choices")]
        self.assertEqual(finishes.count("stop"), 1)

    def test_direct_model_calls(self):
        r = self.chat("granite4.2:3b")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["model"], "granite4.2:3b")
        with self.cluster.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "qwen3.5:4b", "stream": True, "messages": [{"role": "user", "content": "hi"}],
            }) as resp:
                events = parse_sse(resp.read().decode())
        self.assertEqual(events[-1], "[DONE]")
        text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if isinstance(e, dict) and e.get("choices"))
        self.assertTrue(text.startswith("Answer from qwen3.5:4b@fake-"), text)
        self.assertTrue(all(e["model"] == "qwen3.5:4b" for e in events if isinstance(e, dict)))

    def test_embeddings(self):
        with self.cluster.client() as c:
            r = c.post("/v1/embeddings", json={"model": "nomic-embed-text", "input": ["a", "b"]})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["data"]), 2)

    def test_unknown_model(self):
        r = self.chat("no-such-model")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["error"]["code"], "model_not_found")

    def test_bad_requests(self):
        with self.cluster.client() as c:
            self.assertEqual(c.post("/v1/chat/completions", content=b"{not json").status_code, 400)
            self.assertEqual(c.post("/v1/chat/completions", json={"model": "team/moa", "messages": []}).status_code, 400)

    def test_api_key_required(self):
        with httpx.Client(base_url=self.cluster.url, trust_env=False) as c:
            self.assertEqual(c.get("/v1/models").status_code, 401)
            self.assertEqual(c.get("/v1/models", headers={"authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(c.get("/v1/models", headers={"x-api-key": "client-key"}).status_code, 200)
            self.assertEqual(c.get("/health").status_code, 200)            # health stays open
            self.assertEqual(c.get("/").status_code, 200)                  # the page itself loads
            # Agents cannot be impersonated with a client key.
            r = c.post("/api/agents/heartbeat", json={"name": "evil", "url": "http://evil"},
                       headers={"authorization": "Bearer client-key"})
            self.assertEqual(r.status_code, 401)

    def test_agent_rejects_calls_without_cluster_token(self):
        port = self.cluster.ports["agent:node-a"]
        r = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", trust_env=False,
                       json={"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(r.status_code, 401)

    def test_native_task_api_and_events(self):
        with self.cluster.client() as c:
            r = c.post("/api/tasks", json={"team": "council", "prompt": "Is RAID a backup?", "wait": False})
            self.assertEqual(r.status_code, 202)
            task_id = r.json()["id"]
            with c.stream("GET", f"/api/tasks/{task_id}/events") as resp:
                events = parse_sse(resp.read().decode())
            types = [e["type"] for e in events]
            self.assertEqual(types[0], "task.started")
            self.assertEqual(types[-1], "task.completed")
            self.assertIn("step.completed", types)
            self.assertIn("task.delta", types)
            task = c.get(f"/api/tasks/{task_id}").json()
            self.assertEqual(task["status"], "ok")
            self.assertEqual(len(task["step_details"]), 7)
            recent = c.get("/api/tasks").json()["tasks"]
            self.assertIn(task_id, [t["id"] for t in recent])
            # Replaying a finished task's events still works.
            with c.stream("GET", f"/api/tasks/{task_id}/events") as resp:
                replay = parse_sse(resp.read().decode())
            self.assertEqual(replay[-1]["type"], "task.completed")

    def test_native_task_single_model(self):
        with self.cluster.client() as c:
            task = c.post("/api/tasks", json={"model": "ministral-3:3b", "prompt": "hello"}).json()
        self.assertEqual(task["status"], "ok")
        self.assertIn("ministral-3:3b", task["output"])

    def test_teams_endpoint(self):
        with self.cluster.client() as c:
            teams = {t["name"]: t for t in c.get("/api/teams").json()["teams"]}
        self.assertTrue(teams["moa"]["ready"])
        self.assertEqual(teams["moa"]["missing_models"], [])
        writer = next(r for r in teams["moa"]["roles"] if r["role"] == "aggregator")
        self.assertEqual(writer["available_models"], ["qwen3.5:9b", "qwen3.5:4b"])

    def test_metrics(self):
        self.chat("team/vote")
        with self.cluster.client() as c:
            text = c.get("/metrics").text
        self.assertIn('flotilla_node_up{node="node-a"} 1', text)
        self.assertIn("flotilla_model_calls_total", text)
        self.assertIn('flotilla_requests_total{kind="team",target="vote",status="ok"}', text)

    def test_parallel_requests_respect_node_slots(self):
        results = []

        def call():
            results.append(self.chat("team/moa").status_code)

        threads = [threading.Thread(target=call) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, [200] * 6)
        for node in NODES:
            self.assertLessEqual(self.cluster.backend_stats(node)["max_in_flight"], 3)
        self.assertEqual(self.cluster.cluster_state()["totals"]["in_flight"], 0)

    def test_pull_through_coordinator(self):
        with self.cluster.client() as c:
            r = c.post("/api/pull", json={"model": "smollm2:1.7b", "nodes": ["node-c"]})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["nodes"]["node-c"]["ok"])
        self.cluster.wait_for(
            lambda: "smollm2:1.7b" in [m["name"] for m in self.cluster.node("node-c")["models"]], 10, "pulled model",
        )

    def test_client_disconnect_cancels_team(self):
        with self.cluster.client() as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": "team/critique", "stream": True,
                "messages": [{"role": "user", "content": "FAKE_SLOW please"}],
            }) as r:
                task_id = r.headers["x-flotilla-task-id"]
                for _line in r.iter_lines():
                    break  # read the first chunk, then hang up
        self.cluster.wait_for(lambda: self.task_status(task_id) == "cancelled", 10, "task cancelled")
        self.cluster.wait_for(lambda: self.cluster.cluster_state()["totals"]["in_flight"] == 0, 10, "slots released")

    def task_status(self, task_id):
        with self.cluster.client() as c:
            return c.get(f"/api/tasks/{task_id}").json()["status"]

    def test_config_reload(self):
        text = self.cluster.config_path.read_text()
        self.cluster.config_path.write_text(text + "\n  duo:\n    strategy: vote\n    voters: [qwen, granite]\n")
        try:
            with self.cluster.client() as c:
                r = c.post("/api/config/reload")
                self.assertTrue(r.json()["ok"], r.text)
                self.assertIn("team/duo", [m["id"] for m in c.get("/v1/models").json()["data"]])
            self.assertEqual(self.chat("team/duo").status_code, 200)
            self.cluster.config_path.write_text(text + "\n  broken: {strategy: mixture}\n")
            with self.cluster.client() as c:
                r = c.post("/api/config/reload")
            self.assertEqual(r.status_code, 400)
            self.assertFalse(r.json()["ok"])
        finally:
            self.cluster.config_path.write_text(text)
            with self.cluster.client() as c:
                c.post("/api/config/reload")


class FailureTests(LogOnFailure, unittest.TestCase):
    """Machines failing while the cluster is in use."""

    def setUp(self):
        self.cluster = Cluster(NODES)
        try:
            self.cluster.start()
        except Exception:
            print(self.cluster.logs())
            self.cluster.stop()
            raise

    def tearDown(self):
        self.cluster.stop()

    def chat(self, model, content="How do heat pumps work?"):
        with self.cluster.client() as c:
            return c.post("/v1/chat/completions", json={"model": model, "messages": [{"role": "user", "content": content}]})

    def task(self, task_id):
        with self.cluster.client() as c:
            return c.get(f"/api/tasks/{task_id}").json()

    def test_backend_errors_fail_over(self):
        # node-b's inference server starts returning 500s; its agent still heartbeats.
        self.cluster.backend_control("node-b", fail=True)
        for _ in range(3):
            r = self.chat("team/moa")
            self.assertEqual(r.status_code, 200, r.text)
            steps = self.task(r.headers["x-flotilla-task-id"])["step_details"]
            self.assertTrue(all(s["node"] != "node-b" for s in steps if s["status"] == "ok"), steps)
        # The writer model only exists on node-b, so the aggregator used its fallback.
        agg = next(s for s in steps if s["role"] == "aggregator")
        self.assertEqual(agg["model"], "qwen3.5:4b")
        self.assertEqual(self.cluster.node("node-b")["status"], "cooldown")

    def test_machine_disappears(self):
        self.cluster.kill("agent-node-a")
        self.cluster.kill("backend-node-a")
        # Immediately after the crash the coordinator still thinks node-a is up;
        # calls to it fail with connection errors and move to other nodes.
        for _ in range(3):
            r = self.chat("team/council")
            self.assertEqual(r.status_code, 200, r.text)
        self.cluster.wait_for(lambda: self.cluster.node("node-a")["status"] == "offline", 10, "node-a offline")
        r = self.chat("team/planner")
        self.assertEqual(r.status_code, 200, r.text)
        # The machine comes back and rejoins on its own.
        self.cluster.start_backend("node-a", NODES["node-a"])
        self.cluster.wait_http(f"http://127.0.0.1:{self.cluster.ports['backend:node-a']}/health")
        self.cluster.start_agent("node-a")
        self.cluster.wait_for(lambda: "node-a" in self.cluster.online(), 15, "node-a back online")

    def test_everything_down_gives_clear_error(self):
        for node in NODES:
            self.cluster.kill(f"agent-{node}")
        self.cluster.wait_for(lambda: not self.cluster.online(), 10, "all offline")
        r = self.chat("team/moa")
        self.assertEqual(r.status_code, 502)
        self.assertIn("no nodes are online", r.json()["error"]["message"])

    def test_wrong_token_never_joins(self):
        self.cluster.start_backend("intruder", ["qwen3.5:4b"])
        self.cluster.wait_http(f"http://127.0.0.1:{self.cluster.ports['backend:intruder']}/health")
        self.cluster.start_agent("intruder", token="wrong-token")
        time.sleep(2)
        names = {n["name"] for n in self.cluster.cluster_state()["nodes"]}
        self.assertNotIn("intruder", names)
        log = (self.cluster.dir / "agent-intruder.log").read_text()
        self.assertIn("rejected the cluster token", log)

    def test_unreachable_agent_is_flagged(self):
        self.cluster.start_backend("hidden", ["qwen3.5:4b"])
        self.cluster.wait_http(f"http://127.0.0.1:{self.cluster.ports['backend:hidden']}/health")
        from tests.cluster import free_port

        self.cluster.start_agent("hidden", advertise=f"http://127.0.0.1:{free_port()}")
        self.cluster.wait_for(lambda: any(n["name"] == "hidden" and n["status"] == "unreachable"
                                          for n in self.cluster.cluster_state()["nodes"]), 10, "flagged unreachable")
        self.assertEqual(self.chat("team/moa").status_code, 200)
        self.cluster.wait_for(lambda: "cannot reach this agent" in (self.cluster.dir / "agent-hidden.log").read_text(),
                              10, "agent warned")


if __name__ == "__main__":
    unittest.main()
