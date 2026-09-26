"""Strategy tests with in-process fake nodes (no sockets, no subprocesses)."""

import textwrap
import unittest

import httpx

from flotilla import fake_backend
from flotilla.config import load_config
from flotilla.dispatch import Dispatcher
from flotilla.engine.context import StepFailed, TaskTimeout
from flotilla.engine.runner import execute_team_task
from flotilla.registry import Registry
from flotilla.tasks import TaskRecord

CONFIG = """
cluster: {max_retries: 2, failure_cooldown: 30}
defaults: {temperature: 0.5}
members:
  qa: {model: small-a, description: first small model}
  qb: {model: small-b, description: second small model}
  qc: {model: small-c, description: third small model}
  big: {model: [big-x, small-a], temperature: 0.2}
  only-on-c: {model: rare}
teams:
  moa: {strategy: mixture, proposers: [qa, qb, qc], aggregator: big}
  moa2: {strategy: mixture, proposers: [qa, qb], aggregator: big, layers: 2}
  strict: {strategy: mixture, proposers: [qa, qb, only-on-c], aggregator: big, min_success: 3}
  tolerant: {strategy: mixture, proposers: [qa, qb, only-on-c], aggregator: big}
  council: {strategy: council, members: [qa, qb, qc], chairman: big}
  plan: {strategy: plan, planner: qa, workers: [qa, qb, qc], synthesizer: big}
  badplan:
    strategy: plan
    planner: qa
    workers: [qb, qc]
    synthesizer: big
    prompts: {plan: "Say something about: {{task}}"}
  critique: {strategy: critique, writer: big, critics: [qb, qc], max_rounds: 2}
  critique1: {strategy: critique, writer: big, critics: [qb], max_rounds: 1}
  vote: {strategy: vote, voters: [qa, qb, qc], judge: big}
  majority: {strategy: vote, voters: [qa, qb], samples: 2}
  pipeline:
    strategy: pipeline
    stages:
      - {member: qa, instruction: Draft it.}
      - {member: qb, instruction: Fix it.}
      - {member: qc}
  single: {strategy: single, member: qa}
  route:
    strategy: route
    router: qa
    default: quick
    routes:
      - {name: quick, description: small talk, target: qb}
      - {name: build, description: multi-part work, target: "team:plan"}
  slow: {strategy: single, member: qa, timeout: 0.5}
  d1: {strategy: single, member: "team:d2"}
  d2: {strategy: single, member: "team:d3"}
  d3: {strategy: single, member: qa}
"""

NODES = {
    "node-a": ["small-a", "small-b", "big-x"],
    "node-b": ["small-a", "small-b", "small-c"],
    "node-c": ["small-c", "rare"],
}


class HostRouter(httpx.AsyncBaseTransport):
    """Routes requests to in-process ASGI apps by host name."""

    def __init__(self, apps):
        self.transports = {host: httpx.ASGITransport(app=app) for host, app in apps.items()}

    async def handle_async_request(self, request):
        transport = self.transports.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"no route to {request.url.host}", request=request)
        return await transport.handle_async_request(request)


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = load_config(text=textwrap.dedent(CONFIG))
        self.apps = {name: fake_backend.build_app(name, models, latency=0.0) for name, models in NODES.items()}
        self.fakes = {name: app.state.fake for name, app in self.apps.items()}
        self.client = httpx.AsyncClient(transport=HostRouter(self.apps))
        self.registry = Registry(self.cfg.cluster, agent_token="tok")
        for name, models in NODES.items():
            self.registry.upsert_agent({"name": name, "url": f"http://{name}", "models": [{"name": m} for m in models],
                                        "max_concurrency": 4}, None)
        self.dispatcher = Dispatcher(self.registry, self.client, self.cfg.cluster)

    async def asyncTearDown(self):
        await self.client.aclose()

    async def run_team(self, team, prompt="How do heat pumps work?", sink=None, overrides=None):
        messages = [{"role": "user", "content": prompt}]
        task = TaskRecord(team, "team", messages, "test")
        out = await execute_team_task(task, self.cfg, self.dispatcher, team, messages, sink=sink, final_overrides=overrides)
        return out, task

    def steps(self, task, role=None):
        return [s for s in task.steps if role is None or s.role == role]

    async def test_mixture(self):
        out, task = await self.run_team("moa")
        self.assertTrue(out.startswith("Combined answer by big-x@node-a from 3 answers"), out)
        self.assertEqual(len(self.steps(task, "proposer")), 3)
        self.assertEqual(task.status, "ok")
        self.assertGreaterEqual(len(task.nodes_used()), 2)
        self.assertGreater(task.usage.total_tokens, 0)

    async def test_mixture_layers(self):
        out, task = await self.run_team("moa2")
        self.assertEqual(len(self.steps(task, "proposer")), 4)
        self.assertIn("(layer 2)", self.steps(task, "proposer")[-1].label)

    async def test_failover_to_other_node(self):
        self.fakes["node-a"].failing = True
        out, task = await self.run_team("moa")
        self.assertEqual(task.status, "ok")
        for s in self.steps(task, "proposer"):
            self.assertIn(s.node, ("node-b", "node-c"))
        failed_attempts = [a for s in task.steps for a in s.attempts if a["node"] == "node-a"]
        self.assertTrue(failed_attempts, "expected at least one recorded failed attempt on node-a")
        # big-x only lives on node-a, so the aggregator falls back to small-a
        agg = self.steps(task, "aggregator")[0]
        self.assertEqual(agg.model, "small-a")

    async def test_partial_failure_tolerated(self):
        self.fakes["node-c"].failing = True
        out, task = await self.run_team("tolerant")
        self.assertEqual(task.status, "ok")
        self.assertEqual([s.status for s in self.steps(task, "proposer")].count("error"), 1)
        with self.assertRaises(StepFailed):
            await self.run_team("strict")

    async def test_council(self):
        out, task = await self.run_team("council")
        self.assertTrue(out.startswith("Council verdict"), out)
        self.assertEqual(len(self.steps(task, "reviewer")), 3)
        notes = [e["text"] for e in task.events if e["type"] == "task.note"]
        self.assertTrue(any(n.startswith("peer ranking: C") for n in notes), notes)

    async def test_plan_with_dependencies(self):
        out, task = await self.run_team("plan")
        self.assertTrue(out.startswith("Synthesis by big-x@node-a of 3 subtasks"), out)
        workers = {s.label.split(":")[0]: s for s in self.steps(task, "worker")}
        self.assertEqual(set(workers), {"s1", "s2", "s3"})
        self.assertGreaterEqual(workers["s3"].started_at, workers["s1"].ended_at)
        self.assertEqual({workers[k].member for k in workers}, {"qa", "qb", "qc"})

    async def test_plan_fallback_when_planner_rambles(self):
        out, task = await self.run_team("badplan")
        self.assertEqual(task.status, "ok")
        self.assertEqual(len(self.steps(task, "planner")), 2)       # first try + retry
        self.assertEqual(len(self.steps(task, "worker")), 1)        # one catch-all subtask
        notes = " ".join(e["text"] for e in task.events if e["type"] == "task.note")
        self.assertIn("one subtask", notes)

    async def test_critique_until_approved(self):
        out, task = await self.run_team("critique")
        self.assertIn("(revised)", out)
        labels = [s.label for s in task.steps]
        self.assertEqual(labels[0], "draft")
        self.assertIn("revision 1", labels)
        self.assertNotIn("revision 2", labels)       # critics approved in round 2

    async def test_critique_last_round_streams(self):
        chunks = []

        async def sink(text):
            chunks.append(text)

        out, task = await self.run_team("critique1", sink=sink)
        self.assertEqual("".join(chunks).strip(), out)
        self.assertTrue(self.steps(task, "writer")[-1].final)

    async def test_vote_with_judge(self):
        out, task = await self.run_team("vote")
        self.assertIn("small-b", out)       # the fake judge always picks candidate B
        self.assertEqual(len(self.steps(task, "judge")), 1)

    async def test_vote_majority(self):
        out, task = await self.run_team("majority", prompt="2+2?")
        self.assertEqual(len(self.steps(task, "voter")), 4)
        notes = " ".join(e["text"] for e in task.events if e["type"] == "task.note")
        self.assertIn("majority answer", notes)

    async def test_pipeline(self):
        out, task = await self.run_team("pipeline")
        stages = self.steps(task, "stage")
        self.assertEqual([s.member for s in stages], ["qa", "qb", "qc"])
        self.assertIn("small-c", out)
        # later stages get the previous version appended to the user's turn,
        # on whichever machines the scheduler picked
        bodies = [b for fake in self.fakes.values() for b in fake.last_bodies]
        later = [b for b in bodies if "current version from the previous step" in b["messages"][-1]["content"]]
        self.assertEqual(len(later), 2, bodies)
        for body in later:
            self.assertEqual([m["role"] for m in body["messages"]], ["user"])
            self.assertTrue(body["messages"][0]["content"].startswith("How do heat pumps work?"))

    async def test_route_to_nested_team(self):
        out, task = await self.run_team("route", prompt="Please build a migration plan")
        self.assertTrue(out.startswith("Synthesis"), out)
        team_step = next(s for s in task.steps if s.kind == "team")
        children = [s for s in task.steps if s.parent == team_step.id]
        self.assertGreaterEqual(len(children), 5)
        out2, _ = await self.run_team("route", prompt="hello there")
        self.assertIn("small-b", out2)       # quick route -> qb

    async def test_streaming_sink_gets_final_answer(self):
        chunks = []

        async def sink(text):
            chunks.append(text)

        out, task = await self.run_team("moa", sink=sink)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks).strip(), out)

    async def test_empty_answer_is_retried_without_thinking(self):
        out, task = await self.run_team("single", prompt="FAKE_EMPTY please")
        self.assertTrue(out.startswith("Answer from"), out)
        step = task.steps[0]
        self.assertIn("retried", step.note or "")
        last = [b for f in self.fakes.values() for b in f.last_bodies][-1]
        self.assertEqual(last.get("reasoning_effort"), "none")

    async def test_reasoning_effort_dropped_for_servers_that_reject_it(self):
        self.cfg.defaults.reasoning_effort = "none"
        for fake in self.fakes.values():
            fake.reject_effort = True
        out, task = await self.run_team("single")
        self.assertEqual(task.status, "ok")
        before = sum(f.requests for f in self.fakes.values())
        await self.run_team("single")
        after = sum(f.requests for f in self.fakes.values())
        self.assertEqual(after - before, 1, "the rejected field should not be sent again to the same server")

    async def test_think_blocks_removed(self):
        out, task = await self.run_team("single", prompt="FAKE_THINK")
        self.assertNotIn("<think>", out)
        self.assertIn("private reasoning", task.steps[0].reasoning)

    async def test_final_overrides_only_on_final_step(self):
        await self.run_team("moa", overrides={"max_tokens": 77})
        bodies = self.fakes["node-a"].last_bodies + self.fakes["node-b"].last_bodies + self.fakes["node-c"].last_bodies
        with_override = [b for b in bodies if b.get("max_tokens") == 77]
        self.assertEqual(len(with_override), 1)
        self.assertIn("Write the single best response", with_override[0]["messages"][0]["content"])

    async def test_depth_limit(self):
        self.cfg.server.max_depth = 1
        with self.assertRaises(StepFailed) as ctx:
            await self.run_team("d1")
        self.assertIn("max_depth", str(ctx.exception))

    async def test_task_timeout(self):
        with self.assertRaises(TaskTimeout):
            await self.run_team("slow", prompt="FAKE_SLOW")

    async def test_missing_model_error_is_clear(self):
        for node in list(self.registry.nodes.values()):
            node.models.pop("small-a", None)
            node.models.pop("big-x", None)
        with self.assertRaises(StepFailed) as ctx:
            await self.run_team("single")
        self.assertIn("serves any of [small-a]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
