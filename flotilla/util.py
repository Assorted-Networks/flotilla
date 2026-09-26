"""Small helpers shared by the coordinator, the agent and the engine."""

from __future__ import annotations

import hmac
import json
import re
import secrets
import time
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# ids, time, auth
# ---------------------------------------------------------------------------


def new_id(prefix: str = "") -> str:
    token = secrets.token_hex(6)
    return f"{prefix}{token}" if prefix else token


def now() -> float:
    return time.time()


def consteq(a: str | None, b: str | None) -> bool:
    """Constant-time string comparison that tolerates None."""
    if a is None or b is None:
        return False
    return hmac.compare_digest(a.encode(), b.encode())


def bearer_token(headers: Any) -> str | None:
    """Extract a token from `Authorization: Bearer x` or `x-api-key: x`."""
    auth = headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    key = headers.get("x-api-key")
    return key.strip() if key else None


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate (~4 characters per token) for backends without usage."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def content_to_text(content: Any) -> str:
    """OpenAI message content may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") in ("image_url", "input_image"):
                    parts.append("[image]")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def last_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return content_to_text(msg.get("content"))
    return ""


def render_conversation(messages: list[dict], limit: int = 12000) -> str:
    """Render a chat as plain text for prompts that need a single task string.

    A single user message is returned as-is. Longer conversations keep the most
    recent turns within `limit` characters and label the latest request.
    """
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not turns:
        return ""
    if len(turns) == 1 and turns[0].get("role") == "user":
        return truncate(content_to_text(turns[0].get("content")), limit)
    latest = content_to_text(turns[-1].get("content")) if turns[-1].get("role") == "user" else ""
    history = turns[:-1] if latest else turns
    lines: list[str] = []
    budget = max(0, limit - len(latest) - 200)
    for msg in reversed(history):
        who = "User" if msg.get("role") == "user" else "Assistant"
        line = f"{who}: {content_to_text(msg.get('content'))}"
        if len(line) > budget:
            break
        lines.append(line)
        budget -= len(line)
    lines.reverse()
    out = "Conversation so far:\n" + "\n\n".join(lines) if lines else ""
    if latest:
        out = (out + "\n\n" if out else "") + "Latest request:\n" + latest
    return out


def system_text(messages: list[dict]) -> str:
    return "\n\n".join(
        content_to_text(m.get("content")) for m in messages if m.get("role") == "system"
    ).strip()


def without_system(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]


def prepend_system(messages: list[dict], text: str | None) -> list[dict]:
    """Put `text` in front of any existing system prompt (one merged message)."""
    if not text or not text.strip():
        return messages
    existing = system_text(messages)
    merged = text.strip() + ("\n\n" + existing if existing else "")
    return [{"role": "system", "content": merged}] + without_system(messages)


def with_system(messages: list[dict], *parts: str | None) -> list[dict]:
    """Return messages with one merged system message at the front.

    Many small models only honour a single, leading system message, so the
    caller's own system prompt and any role instructions are merged.
    """
    pieces = [system_text(messages)] + [p for p in parts if p]
    merged = "\n\n".join(p.strip() for p in pieces if p and p.strip())
    rest = without_system(messages)
    return ([{"role": "system", "content": merged}] if merged else []) + rest


# ---------------------------------------------------------------------------
# reasoning ("thinking") handling
# ---------------------------------------------------------------------------

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_THINK_BLOCK = re.compile(r"<(think|thinking)>(.*?)</\1>", re.S | re.I)


def strip_think(text: str | None) -> tuple[str, str]:
    """Split inline <think> blocks out of model output.

    Returns (answer, reasoning). Handles an unclosed opening tag (the model ran
    out of tokens while thinking) and a stray closing tag (the chat template
    opened the block inside the prompt).
    """
    if not text:
        return "", ""
    reasoning: list[str] = []

    def _take(match: re.Match) -> str:
        reasoning.append(match.group(2))
        return ""

    out = _THINK_BLOCK.sub(_take, text)
    low = out.lower()
    for tag in _OPEN_TAGS:
        idx = low.find(tag)
        if idx != -1:
            reasoning.append(out[idx + len(tag):])
            out = out[:idx]
            low = out.lower()
    for tag in _CLOSE_TAGS:
        idx = low.find(tag)
        if idx != -1:
            reasoning.append(out[:idx])
            out = out[idx + len(tag):]
            low = out.lower()
    joined = "\n".join(r.strip() for r in reasoning if r and r.strip())
    return out.strip(), joined


def _partial_suffix(buf_lower: str, tags: Iterable[str]) -> int:
    """Length of the longest suffix of buf that is a proper prefix of a tag."""
    best = 0
    for tag in tags:
        for k in range(min(len(tag) - 1, len(buf_lower)), 0, -1):
            if tag.startswith(buf_lower[-k:]):
                best = max(best, k)
                break
    return best


def _find_first(buf_lower: str, tags: Iterable[str]) -> tuple[int, int]:
    best_idx, best_len = -1, 0
    for tag in tags:
        idx = buf_lower.find(tag)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best_len = idx, len(tag)
    return best_idx, best_len


class ThinkStreamFilter:
    """Incrementally separates <think>...</think> text from streamed content."""

    def __init__(self) -> None:
        self.buf = ""
        self.in_think = False

    def feed(self, text: str) -> tuple[str, str]:
        self.buf += text or ""
        content: list[str] = []
        reasoning: list[str] = []
        while self.buf:
            low = self.buf.lower()
            tags = _CLOSE_TAGS if self.in_think else _OPEN_TAGS
            idx, taglen = _find_first(low, tags)
            target = reasoning if self.in_think else content
            if idx == -1:
                keep = _partial_suffix(low, tags)
                cut = len(self.buf) - keep
                target.append(self.buf[:cut])
                self.buf = self.buf[cut:]
                break
            target.append(self.buf[:idx])
            self.buf = self.buf[idx + taglen:]
            self.in_think = not self.in_think
        return "".join(content), "".join(reasoning)

    def flush(self) -> tuple[str, str]:
        rest, self.buf = self.buf, ""
        return ("", rest) if self.in_think else (rest, "")


# ---------------------------------------------------------------------------
# parsing structured replies from small models
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str | None) -> Any | None:
    """Best-effort JSON extraction from an LLM reply.

    Accepts plain JSON, JSON inside code fences, or JSON embedded in prose.
    Returns the first object/array that parses, or None.
    """
    if not text:
        return None
    text, _ = strip_think(text)
    candidates = [text.strip()]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (ValueError, TypeError):
            pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _end = decoder.raw_decode(text, i)
                if isinstance(value, (dict, list)):
                    return value
            except ValueError:
                continue
    return None


def parse_ranking(text: str, labels: list[str]) -> list[str]:
    """Parse `FINAL RANKING: C, A, B` (or a numbered list after it).

    Unknown labels are ignored, duplicates dropped, and labels the reviewer
    forgot are appended in their original order so every label gets a rank.
    """
    valid = {lab.upper() for lab in labels}
    order: list[str] = []
    body, _ = strip_think(text)
    marker = re.search(r"final\s+ranking\s*[:\-]?", body, re.I)
    section = body[marker.end():marker.end() + 600] if marker else ""
    if section:
        named = re.findall(r"\b(?:response|answer|candidate)\s+([A-Za-z])\b", section, re.I)
        if named:
            tokens = [t.upper() for t in named]
        else:
            # Plain lists such as "C, A, B" or "1. C 2. A 3. B": only look at the
            # first non-empty lines so prose after the ranking is ignored.
            lines = [ln for ln in section.splitlines() if ln.strip()][: max(1, len(labels))]
            tokens = re.findall(r"\b([A-Z])\b", "\n".join(lines))
        for tok in tokens:
            if tok in valid and tok not in order:
                order.append(tok)
    for lab in labels:
        if lab.upper() not in order:
            order.append(lab.upper())
    return order


def parse_choice(text: str, labels: list[str], key: str = "BEST") -> str | None:
    """Parse `BEST: B` style verdicts. Returns the label or None."""
    body, _ = strip_think(text)
    valid = {lab.upper() for lab in labels}
    pattern = rf"{key}\s*[:\-]\s*[\s\*\(\[\"']*(?:response|answer|candidate)?\s*([A-Za-z])\b"
    found = None
    for match in re.finditer(pattern, body, re.I):
        lab = match.group(1).upper()
        if lab in valid:
            found = lab  # the last verdict wins if the model restates it
    return found


def is_approval(text: str, token: str = "APPROVED") -> bool:
    """True when a critic's reply is an approval rather than a list of issues."""
    body, _ = strip_think(text)
    body = body.strip()
    if not body:
        return False
    first = re.sub(r"[^a-z]", "", body.splitlines()[0].lower())
    tok = re.sub(r"[^a-z]", "", token.lower())
    if first == tok:
        return True
    # Short replies such as "Approved - looks good." also count.
    return len(body) < 60 and body.lower().lstrip("*# ").startswith(token.lower())


def normalize_answer(text: str) -> str:
    """Normalization used for majority voting on short answers."""
    body, _ = strip_think(text)
    body = body.strip().lower()
    body = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", body)
    body = re.sub(r"[\s\W_]+", " ", body)
    return body.strip()


# ---------------------------------------------------------------------------
# models, labels, misc
# ---------------------------------------------------------------------------


def canonical_model(name: str) -> str:
    """Canonical form for matching model names across nodes.

    Ollama treats `llama3.2` and `llama3.2:latest` as the same model, and names
    are case-insensitive there; other backends are matched case-insensitively
    as well, but requests always use the node's exact spelling.
    """
    name = (name or "").strip().lower()
    if name.endswith(":latest"):
        name = name[: -len(":latest")]
    return name


def parse_labels(spec: str | None) -> dict[str, str]:
    """Parse `gpu=nvidia,vram=12` into a dict."""
    out: dict[str, str] = {}
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition("=")
        out[key.strip()] = value.strip() or "true"
    return out


def parse_list(spec: str | None) -> list[str]:
    return [s.strip() for s in (spec or "").replace(";", ",").split(",") if s.strip()]


def letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(min(n, 26))]


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
