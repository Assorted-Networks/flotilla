"""Minimal client for OpenAI-compatible chat endpoints (Ollama, vLLM, llama.cpp, agents)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from flotilla.util import content_to_text, estimate_tokens, strip_think

# Status codes worth retrying on a different node. 400/413/422 are problems
# with the request itself and would fail everywhere.
_NOT_RETRYABLE = {400, 413, 422}


class BackendError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable

    @property
    def kind(self) -> str:
        if self.status is None:
            return "connection"
        if self.status == 404:
            return "not_found"
        if self.status in (401, 403):
            return "auth"
        if self.status in (429, 503):
            return "busy"
        return f"http_{self.status}"


def error_from_response(status: int, body: bytes | str) -> BackendError:
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    message = text.strip()[:500] or f"HTTP {status}"
    try:
        data = json.loads(text)
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            message = str(err.get("message") or err)
        elif isinstance(err, str):
            message = err
        elif isinstance(data, dict) and data.get("detail"):
            message = str(data["detail"])
    except (ValueError, AttributeError):
        pass
    return BackendError(message, status=status, retryable=status not in _NOT_RETRYABLE)


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def chat_url(base_url: str) -> str:
    return api_base(base_url) + "/chat/completions"


def api_base(base_url: str) -> str:
    """`http://host:11434` or `http://host:11434/v1` -> `http://host:11434/v1`."""
    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def make_timeout(total: float, connect: float = 5.0) -> httpx.Timeout:
    # `read` bounds the gap between bytes: a CPU node can take a while to
    # process a long prompt before the first token arrives.
    return httpx.Timeout(connect=connect, read=total, write=60.0, pool=total)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.estimated = self.estimated or other.estimated

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_openai(cls, data: dict | None) -> "Usage | None":
        if not isinstance(data, dict):
            return None
        return cls(
            prompt_tokens=int(data.get("prompt_tokens") or 0),
            completion_tokens=int(data.get("completion_tokens") or 0),
        )


@dataclass
class Completion:
    content: str
    reasoning: str = ""
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: dict | None = None


def _reasoning_of(obj: dict) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = obj.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def parse_completion(data: dict, prompt_chars: int = 0) -> Completion:
    try:
        choice = data["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise BackendError(f"malformed completion: {str(data)[:200]}", retryable=True) from exc
    msg = choice.get("message") or {}
    content, inline_reasoning = strip_think(content_to_text(msg.get("content")))
    reasoning = _reasoning_of(msg) or inline_reasoning
    usage = Usage.from_openai(data.get("usage"))
    if usage is None or usage.total_tokens == 0:
        usage = Usage(
            prompt_tokens=prompt_chars // 4,
            completion_tokens=estimate_tokens(content) + estimate_tokens(reasoning),
            estimated=True,
        )
    return Completion(
        content=content,
        reasoning=reasoning,
        finish_reason=choice.get("finish_reason"),
        usage=usage,
        raw=data,
    )


def parse_chunk(chunk: dict) -> tuple[str, str, str | None, Usage | None]:
    """(content_delta, reasoning_delta, finish_reason, usage) from a stream chunk."""
    usage = Usage.from_openai(chunk.get("usage"))
    choices = chunk.get("choices") or []
    if not choices:
        return "", "", None, usage
    choice = choices[0] or {}
    delta = choice.get("delta") or {}
    content = delta.get("content")
    content = content_to_text(content) if content else ""
    return content, _reasoning_of(delta), choice.get("finish_reason"), usage


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    body: dict,
    api_key: str | None,
    timeout: httpx.Timeout,
) -> dict:
    try:
        resp = await client.post(url, json=body, headers=_headers(api_key), timeout=timeout)
    except httpx.TimeoutException as exc:
        raise BackendError(f"timeout calling {url}: {type(exc).__name__}") from exc
    except httpx.HTTPError as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise error_from_response(resp.status_code, resp.content)
    try:
        return resp.json()
    except ValueError as exc:
        raise BackendError(f"non-JSON response from {url}") from exc


async def get_json(
    client: httpx.AsyncClient, url: str, api_key: str | None, timeout: float = 10.0
) -> Any:
    try:
        resp = await client.get(url, headers=_headers(api_key), timeout=timeout)
    except httpx.HTTPError as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise error_from_response(resp.status_code, resp.content)
    try:
        return resp.json()
    except ValueError as exc:
        raise BackendError(f"non-JSON response from {url}") from exc


async def iter_sse(response: httpx.Response) -> AsyncIterator[str]:
    """Yield the data payload of each server-sent event."""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        yield "\n".join(data_lines)


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    body: dict,
    api_key: str | None,
    timeout: httpx.Timeout,
) -> AsyncIterator[dict]:
    """POST a streaming chat request and yield each parsed chunk.

    Raises BackendError before the first chunk for HTTP errors, and mid-stream
    if the server sends an error event or the connection drops.
    """
    url = chat_url(base_url)
    body = {**body, "stream": True}
    try:
        async with client.stream("POST", url, json=body, headers=_headers(api_key), timeout=timeout) as resp:
            if resp.status_code >= 400:
                raise error_from_response(resp.status_code, await resp.aread())
            async for data in iter_sse(resp):
                if data.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise BackendError(f"stream error: {msg}", status=500)
                yield chunk
    except httpx.TimeoutException as exc:
        raise BackendError(f"timeout streaming from {url}: {type(exc).__name__}") from exc
    except httpx.HTTPError as exc:
        raise BackendError(f"stream from {url} failed: {exc}") from exc
