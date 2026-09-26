"""Builders for OpenAI-style responses, stream chunks and errors."""

from __future__ import annotations

import json
import time
from typing import Any

from starlette.responses import JSONResponse


def sse(obj: Any) -> bytes:
    if isinstance(obj, str):
        return f"data: {obj}\n\n".encode()
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


SSE_DONE = b"data: [DONE]\n\n"
SSE_PING = b": ping\n\n"


def chunk(
    cid: str,
    model: str,
    created: int,
    content: str | None = None,
    role: str | None = None,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
    finish_reason: str | None = None,
    usage: dict | None = None,
    include_choice: bool = True,
) -> dict:
    out: dict[str, Any] = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
    if include_choice:
        delta: dict[str, Any] = {}
        if role:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        if reasoning:
            delta[reasoning_field] = reasoning
        out["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    else:
        out["choices"] = []
    if usage is not None:
        out["usage"] = usage
    return out


def completion(
    cid: str,
    model: str,
    content: str,
    usage: dict,
    reasoning: str | None = None,
    reasoning_field: str = "reasoning_content",
    extra: dict | None = None,
) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message[reasoning_field] = reasoning
    out = {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": usage,
    }
    if extra:
        out.update(extra)
    return out


def error_body(message: str, etype: str = "server_error", code: str | None = None) -> dict:
    return {"error": {"message": message, "type": etype, "code": code, "param": None}}


def error(status: int, message: str, etype: str | None = None, code: str | None = None,
          headers: dict | None = None) -> JSONResponse:
    if etype is None:
        etype = {
            400: "invalid_request_error",
            401: "authentication_error",
            403: "permission_error",
            404: "invalid_request_error",
            413: "invalid_request_error",
            429: "rate_limit_error",
        }.get(status, "server_error")
    return JSONResponse(error_body(message, etype, code), status_code=status, headers=headers)
