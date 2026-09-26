"""A fake inference server for tests and dry runs (no models, no GPU).

It answers like Ollama (native /api/* endpoints plus /v1) or a plain
OpenAI-compatible server, and recognises Flotilla's own prompts (planner,
reviewer, critic, judge, router) so every strategy can be exercised end to end.

Magic words in the last user message: FAKE_FAIL (HTTP 500), FAKE_SLOW (delay),
FAKE_EMPTY (answer only in the reasoning field unless thinking is off),
FAKE_THINK (answer wrapped with an inline <think> block).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import Counter
from typing import Any, AsyncIterator

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from flotilla import openai_format as oai
from flotilla.util import content_to_text, last_user_text

FILLER = (
    "Small models can cover for each other: one notices what another missed, and a final "
    "pass keeps the parts that hold up."
)


class FakeBackend:
    def __init__(self, name: str, models: list[str], latency: float = 0.05, mode: str = "ollama",
                 token_delay: float = 0.005):
        self.name = name
        self.models = list(models)
        self.latency = latency
        self.mode = mode
        self.token_delay = token_delay
        self.loaded: set[str] = set()
        self.by_model: Counter = Counter()
        self.requests = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.last_bodies: list[dict] = []
        self.failing = False          # toggled with POST /_fake/control {"fail": true}
        self.reject_effort = False    # behave like a server that rejects reasoning_effort

    # -- answer logic ---------------------------------------------------------------

    @staticmethod
    def _section_names(text: str, header: str) -> list[str]:
        idx = text.find(header)
        if idx == -1:
            return []
        names = []
        for line in text[idx + len(header):].splitlines()[1:]:
            m = re.match(r"-\s+([^:]+):", line.strip())
            if m:
                names.append(m.group(1).strip())
            elif names and not line.strip():
                break
        return names

    def answer(self, model: str, body: dict) -> tuple[str, str]:
        msgs = body.get("messages") or []
        everything = "\n".join(content_to_text(m.get("content")) for m in msgs)
        user = last_user_text(msgs)
        tag = f"{model}@{self.name}"
        if "Write the single best response" in everything:
            n = len(re.findall(r"\[Answer \d+\]", everything))
            return f"Combined answer by {tag} from {n} answers. {FILLER}", ""
        if "You chair a council" in everything:
            return f"Council verdict by {tag}. {FILLER}", ""
        if "final writer for a team of AI workers" in everything:
            n = len(re.findall(r"\[Subtask ", everything))
            return f"Synthesis by {tag} of {n} subtasks. {FILLER}", ""
        if '"subtasks"' in everything:
            workers = self._section_names(everything, "Available workers:") or ["worker"]
            plan = {"subtasks": [
                {"id": "s1", "title": "Gather the facts", "instructions": "List the key facts.", "worker": workers[0], "depends_on": []},
                {"id": "s2", "title": "Consider trade-offs", "instructions": "Explain the trade-offs.", "worker": workers[1 % len(workers)], "depends_on": []},
                {"id": "s3", "title": "Draft recommendations", "instructions": "Recommend next steps using s1.", "worker": workers[2 % len(workers)], "depends_on": ["s1"]},
            ]}
            return "```json\n" + json.dumps(plan) + "\n```", ""
        if "FINAL RANKING" in everything:
            labels = sorted(set(re.findall(r"\[Response ([A-Z])\]", everything)))
            return f"Review by {tag}: every answer is plausible.\nFINAL RANKING: {', '.join(reversed(labels))}", ""
        if "reply with exactly:" in everything:
            if "(revised)" in everything:
                return "APPROVED", ""
            return f"- The draft needs one concrete example ({tag}).\n- Tighten the opening.", ""
        if "BEST: <label>" in everything:
            labels = sorted(set(re.findall(r"\[Candidate ([A-Z])\]", everything)))
            pick = "B" if "B" in labels else (labels[0] if labels else "A")
            return f"Candidate {pick} is the most complete.\nBEST: {pick}", ""
        if '{"route"' in everything:
            routes = self._section_names(everything, "Routes:")
            request = everything.split("Request:", 1)[-1].lower()
            chosen = next((r for r in routes if re.search(rf"\b{re.escape(r.lower())}\b", request)), routes[0] if routes else "default")
            return json.dumps({"route": chosen}), ""
        if "Revise the draft" in everything:
            return f"Final text by {tag} (revised). {FILLER}", ""
        if "Your subtask:" in everything:
            title = re.search(r"Your subtask: (.*)", everything)
            return f"Result for '{title.group(1).strip() if title else '?'}' by {tag}. {FILLER}", ""
        if "FAKE_EMPTY" in user and body.get("reasoning_effort") != "none":
            return "", "I thought about it for too long and ran out of tokens."
        answer = f"Answer from {tag} to: {user[:80]}. {FILLER}"
        if "FAKE_THINK" in user:
            answer = f"<think>private reasoning of {tag}</think>{answer}"
        rf = body.get("response_format") or {}
        if rf.get("type") in ("json_object", "json_schema"):
            return json.dumps({"answer": answer}), ""
        return answer, ""

    # -- HTTP -------------------------------------------------------------------------

    async def chat(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model", "")
        if model not in self.models and f"{model}:latest" not in self.models:
            return JSONResponse({"error": {"message": f"model '{model}' not found", "type": "not_found_error"}}, status_code=404)
        self.requests += 1
        self.by_model[model] += 1
        self.last_bodies = (self.last_bodies + [body])[-50:]
        user = last_user_text(body.get("messages") or [])
        if self.reject_effort and "reasoning_effort" in body:
            return JSONResponse({"error": {"message": "reasoning_effort: Input should be 'low', 'medium' or 'high'",
                                           "type": "BadRequestError"}}, status_code=400)
        if self.failing or "FAKE_FAIL" in user:
            return JSONResponse({"error": {"message": "fake failure", "type": "server_error"}}, status_code=500)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.latency + (1.5 if "FAKE_SLOW" in user else 0))
            content, reasoning = self.answer(model, body)
        finally:
            self.in_flight -= 1
        self.loaded.add(model)
        prompt_chars = sum(len(content_to_text(m.get("content"))) for m in body.get("messages") or [])
        usage = {"prompt_tokens": prompt_chars // 4, "completion_tokens": len(content) // 4 + 1,
                 "total_tokens": prompt_chars // 4 + len(content) // 4 + 1}
        cid = f"chatcmpl-fake{self.requests}"
        if not body.get("stream"):
            message: dict[str, Any] = {"role": "assistant", "content": content}
            if reasoning:
                message["reasoning"] = reasoning
            return JSONResponse({
                "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage,
            })
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def gen() -> AsyncIterator[bytes]:
            created = int(time.time())
            yield oai.sse(oai.chunk(cid, model, created, role="assistant", content=""))
            if reasoning:
                yield oai.sse(oai.chunk(cid, model, created, reasoning=reasoning, reasoning_field="reasoning"))
            words = re.split(r"(\s+)", content)
            for i in range(0, len(words), 4):
                await asyncio.sleep(self.token_delay)
                yield oai.sse(oai.chunk(cid, model, created, content="".join(words[i:i + 4])))
            yield oai.sse(oai.chunk(cid, model, created, finish_reason="stop"))
            if include_usage:
                yield oai.sse(oai.chunk(cid, model, created, usage=usage, include_choice=False))
            yield oai.SSE_DONE

        return StreamingResponse(gen(), media_type="text/event-stream")

    async def embeddings(self, request: Request) -> Response:
        body = await request.json()
        inputs = body.get("input")
        inputs = inputs if isinstance(inputs, list) else [inputs]
        data = []
        for i, text in enumerate(inputs):
            digest = hashlib.sha256(str(text).encode()).digest()
            data.append({"object": "embedding", "index": i, "embedding": [b / 255 for b in digest[:8]]})
        return JSONResponse({"object": "list", "data": data, "model": body.get("model"),
                             "usage": {"prompt_tokens": 1, "total_tokens": 1}})

    async def v1_models(self, request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": m, "object": "model", "owned_by": self.name} for m in self.models]})

    async def tags(self, request: Request) -> Response:
        return JSONResponse({"models": [{
            "name": m, "model": m, "size": 1_000_000_000, "digest": "0" * 64, "modified_at": "2026-01-01T00:00:00Z",
            "details": {"format": "gguf", "family": "fake", "families": ["fake"], "parameter_size": "1B", "quantization_level": "Q4_K_M"},
        } for m in self.models]})

    async def ps(self, request: Request) -> Response:
        return JSONResponse({"models": [{"name": m, "model": m, "size": 1, "size_vram": 1} for m in sorted(self.loaded)]})

    async def version(self, request: Request) -> Response:
        return JSONResponse({"version": "0.0.0-fake"})

    async def pull(self, request: Request) -> Response:
        body = await request.json()
        model = body.get("model") or body.get("name")

        async def gen() -> AsyncIterator[bytes]:
            yield b'{"status":"pulling manifest"}\n'
            for done in (25, 50, 75, 100):
                await asyncio.sleep(0.02)
                yield json.dumps({"status": "downloading", "total": 100, "completed": done}).encode() + b"\n"
            if model not in self.models:
                self.models.append(model)
            yield b'{"status":"success"}\n'

        if body.get("stream") is False:
            async for _ in gen():
                pass
            return JSONResponse({"status": "success"})
        return StreamingResponse(gen(), media_type="application/x-ndjson")

    async def control(self, request: Request) -> Response:
        body = await request.json()
        if "fail" in body:
            self.failing = bool(body["fail"])
        if "latency" in body:
            self.latency = float(body["latency"])
        if "reject_effort" in body:
            self.reject_effort = bool(body["reject_effort"])
        return JSONResponse({"failing": self.failing, "latency": self.latency, "reject_effort": self.reject_effort})

    async def stats(self, request: Request) -> Response:
        return JSONResponse({"name": self.name, "requests": self.requests, "by_model": dict(self.by_model),
                             "in_flight": self.in_flight, "max_in_flight": self.max_in_flight,
                             "last_bodies": self.last_bodies[-5:]})


def build_app(name: str, models: list[str], latency: float = 0.05, mode: str = "ollama") -> Starlette:
    fake = FakeBackend(name, models, latency, mode)
    routes = [
        Route("/v1/chat/completions", fake.chat, methods=["POST"]),
        Route("/v1/embeddings", fake.embeddings, methods=["POST"]),
        Route("/v1/models", fake.v1_models),
        Route("/_fake/stats", fake.stats),
        Route("/_fake/control", fake.control, methods=["POST"]),
        Route("/health", lambda r: JSONResponse({"status": "ok"})),
    ]
    if mode == "ollama":
        routes += [
            Route("/api/tags", fake.tags),
            Route("/api/ps", fake.ps),
            Route("/api/version", fake.version),
            Route("/api/pull", fake.pull, methods=["POST"]),
        ]
    app = Starlette(routes=routes)
    app.state.fake = fake
    return app


def serve(name: str, models: list[str], host: str, port: int, latency: float, mode: str) -> None:
    import uvicorn

    uvicorn.run(build_app(name, models, latency, mode), host=host, port=port, log_level="warning")
