"""Custom httpx transport that reroutes OpenAI chat-completion traffic to
vLLM's `/v1/chunked_chat/completions` endpoint.

Plug an instance of `ChunkedChatHTTPClient` into `langchain_openai.ChatOpenAI`
via its `http_async_client` kwarg. From then on every chat-completion request
gets:

- its path rewritten from `/chat/completions` to `/chunked_chat/completions`
- its body augmented with `chunks` and `anchor_indices`

Chunk layout:
- chunks[0] = concatenated system message content (the static system
  prompt) and is marked as an anchor.
- chunks[1] = serialized non-system messages (user / assistant /
  tool turns). Not an anchor.

Only the system prompt is anchored — the project direction is "static
chunks are the not-changing system prompts".
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_CHAT_SUFFIX = "/chat/completions"
_CHUNKED_SUFFIX = "/chunked_chat/completions"
_DEBUG = os.getenv("CHUNKED_CHAT_DEBUG", "").strip() not in ("", "0", "false", "False")


def _debug(msg: str) -> None:
    if _DEBUG:
        print(f"[chunked_chat] {msg}", file=sys.stderr, flush=True)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "".join(parts)
    return str(content)


def _render_non_system(msg: dict) -> str:
    role = msg.get("role", "user")
    text = _content_to_text(msg.get("content"))
    lines = [f"<|{role}|>"]
    if text:
        lines.append(text)
    if msg.get("tool_calls"):
        lines.append("<|tool_calls|>" + json.dumps(msg["tool_calls"], ensure_ascii=False))
    if msg.get("tool_call_id"):
        lines.append(f"<|tool_call_id|>{msg['tool_call_id']}")
    return "\n".join(lines)


def split_messages_into_chunks(messages: list[dict]) -> tuple[str, str]:
    """Return (anchor_chunk, dynamic_chunk) for an OpenAI messages array."""
    system_parts: list[str] = []
    other_parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "system":
            text = _content_to_text(msg.get("content"))
            if text:
                system_parts.append(text)
        else:
            other_parts.append(_render_non_system(msg))
    return "\n\n".join(system_parts), "\n\n".join(other_parts)


class ChunkedChatHTTPClient(httpx.AsyncClient):
    """AsyncClient that rewrites chat-completion requests to chunked_chat."""

    def __init__(self, *args: Any, agent_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._agent_id = agent_id

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        original_url = str(request.url)
        if request.url.path.endswith(_CHAT_SUFFIX):
            request = self._rewrite(request)
            _debug(f"rewrote {original_url} -> {request.url}")
        else:
            _debug(f"pass-through {request.method} {request.url}")
        response = await super().send(request, **kwargs)
        if response.status_code >= 400:
            msg = f"{request.method} {request.url} -> {response.status_code}"
            logger.warning("chunked_chat: %s", msg)
            _debug(msg)
        return response

    def _rewrite(self, request: httpx.Request) -> httpx.Request:
        try:
            body = json.loads(request.content) if request.content else {}
        except Exception as exc:
            logger.warning("chunked_chat: body parse failed (%s); passing through", exc)
            return request

        anchor, dynamic = split_messages_into_chunks(body.get("messages") or [])
        chunks: list[str] = []
        anchor_indices: list[int] = []
        if anchor:
            chunks.append(anchor)
            anchor_indices.append(0)
        if dynamic:
            chunks.append(dynamic)
        if not chunks:
            chunks = [""]

        body["chunks"] = chunks
        body["anchor_indices"] = anchor_indices
        if self._agent_id is not None and "agent_id" not in body:
            body["agent_id"] = self._agent_id
        # Drop messages so the serving layer reconstructs them from chunks,
        # keeping per-chunk token spans aligned with the engine's tokens.
        body["messages"] = []

        new_path = request.url.path[: -len(_CHAT_SUFFIX)] + _CHUNKED_SUFFIX
        new_url = request.url.copy_with(path=new_path)
        new_content = json.dumps(body).encode("utf-8")
        new_headers = httpx.Headers(request.headers)
        new_headers["content-length"] = str(len(new_content))

        return httpx.Request(
            method=request.method,
            url=new_url,
            headers=new_headers,
            content=new_content,
            extensions=dict(request.extensions),
        )


def build_chunked_chat_http_client(
    *, timeout: float = 300.0, agent_id: str | None = None
) -> ChunkedChatHTTPClient:
    return ChunkedChatHTTPClient(
        timeout=httpx.Timeout(connect=10.0, read=timeout, write=timeout, pool=10.0),
        agent_id=agent_id,
    )
