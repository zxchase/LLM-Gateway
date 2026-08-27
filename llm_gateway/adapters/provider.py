"""Provider protocol and adapter implementations.

Every adapter translates its upstream response into the single unified shape
``(str content, Usage)`` so the rest of the gateway is provider-agnostic.

Supported upstream formats:

* ``openai_chat``       — OpenAI ``/chat/completions``
* ``openai_responses``  — OpenAI ``/v1/responses``
* ``anthropic``         — Anthropic ``/v1/messages``
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Protocol

import httpx

from ..config import ModelConfig
from ..errors import GatewayError
from ..models import Message, Usage

ANTHROPIC_DEFAULT_MAX_TOKENS = 4096


class Provider(Protocol):
    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]: ...

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]: ...


def _json_instruction(response_schema: dict[str, Any]) -> str:
    return (
        "只返回一个合法 JSON 对象，必须严格符合下列 JSON Schema，"
        "不要返回 Markdown 或额外文字："
        f"{json.dumps(response_schema, ensure_ascii=False)}"
    )


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Separate system messages from the conversation.

    Anthropic and the Responses API take ``system``/``instructions`` as a
    top-level field instead of a message with role ``system``.
    """
    system_parts = [message.content for message in messages if message.role == "system"]
    rest = [message for message in messages if message.role != "system"]
    return "\n\n".join(system_parts), rest


def parse_chat_completion(data: dict[str, Any]) -> tuple[str, Usage]:
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    raw = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(raw.get("prompt_tokens", 0) or 0),
        output_tokens=int(raw.get("completion_tokens", 0) or 0),
    )
    return content, usage


def parse_responses(data: dict[str, Any]) -> tuple[str, Usage]:
    parts: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for block in item.get("content", []):
            if block.get("type") in ("output_text", "text"):
                parts.append(block.get("text", "") or "")
            elif block.get("type") == "refusal":
                parts.append(block.get("refusal", "") or "")
    raw = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
    )
    return "".join(parts), usage


def parse_anthropic(data: dict[str, Any]) -> tuple[str, Usage]:
    parts: list[str] = []
    for block in data.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", "") or "")
    raw = data.get("usage") or {}
    usage = Usage(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
    )
    return "".join(parts), usage


def _load_payload(line: str) -> dict[str, Any] | None:
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def _parse_chat_delta(line: str) -> str:
    data = _load_payload(line)
    if not data:
        return ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content", "") or ""


def _parse_responses_delta(line: str) -> str:
    data = _load_payload(line)
    if not data or data.get("type") != "response.output_text.delta":
        return ""
    return data.get("delta", "") or ""


def _parse_anthropic_delta(line: str) -> str:
    data = _load_payload(line)
    if not data or data.get("type") != "content_block_delta":
        return ""
    delta = data.get("delta") or {}
    if delta.get("type") != "text_delta":
        return ""
    return delta.get("text", "") or ""


class OpenAICompatibleProvider:
    """OpenAI ``/chat/completions`` adapter with Bearer auth."""

    def create_client(self, config: ModelConfig) -> httpx.AsyncClient:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        return httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_seconds=60.0, connect=10.0),
        )

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        body: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
        }
        if response_schema is not None:
            body = self._apply_structured_output(config, body, response_schema)

        async with self.create_client(config) as client:
            response = await client.post(
                "/chat/completions",
                json=body,
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        return parse_chat_completion(response.json())

    def _apply_structured_output(
        self,
        config: ModelConfig,
        body: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if config.structured_output_mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_response",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            body["messages"] = [
                {"role": "system", "content": _json_instruction(response_schema)},
                *body["messages"],
            ]
        return body

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        body: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [message.model_dump() for message in messages],
            "stream": True,
        }
        async with self.create_client(config) as client:
            async with client.stream(
                "POST",
                "/chat/completions",
                json=body,
                timeout=timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = _parse_chat_delta(line)
                    if delta:
                        yield delta


class OpenAIResponsesProvider(OpenAICompatibleProvider):
    """OpenAI ``/v1/responses`` adapter."""

    def _base_body(self, config: ModelConfig, messages: list[Message]) -> dict[str, Any]:
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "model": config.provider_model,
            "input": [message.model_dump() for message in rest],
        }
        if system:
            body["instructions"] = system
        return body

    def _apply_structured_output(
        self,
        config: ModelConfig,
        body: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        instruction = _json_instruction(response_schema)
        existing = body.get("instructions")
        body["instructions"] = f"{existing}\n\n{instruction}" if existing else instruction
        return body

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        body = self._base_body(config, messages)
        body["stream"] = False
        if response_schema is not None:
            body = self._apply_structured_output(config, body, response_schema)

        async with self.create_client(config) as client:
            response = await client.post(
                "/v1/responses",
                json=body,
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        return parse_responses(response.json())

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        body = self._base_body(config, messages)
        body["stream"] = True
        async with self.create_client(config) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json=body,
                timeout=timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = _parse_responses_delta(line)
                    if delta:
                        yield delta


class AnthropicProvider:
    """Native Anthropic ``/v1/messages`` adapter."""

    def create_client(self, config: ModelConfig) -> httpx.AsyncClient:
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise GatewayError("gateway_misconfigured", "Gateway 模型凭据未配置", 503)
        return httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout=httpx.Timeout(timeout_seconds=60.0, connect=10.0),
        )

    def _base_body(
        self,
        config: ModelConfig,
        messages: list[Message],
        response_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        system, rest = _split_system(messages)
        if response_schema is not None:
            instruction = _json_instruction(response_schema)
            system = f"{system}\n\n{instruction}" if system else instruction
        body: dict[str, Any] = {
            "model": config.provider_model,
            "max_tokens": ANTHROPIC_DEFAULT_MAX_TOKENS,
            "messages": [{"role": message.role, "content": message.content} for message in rest],
        }
        if system:
            body["system"] = system
        return body

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        body = self._base_body(config, messages, response_schema)
        body["stream"] = False
        async with self.create_client(config) as client:
            response = await client.post(
                "/v1/messages",
                json=body,
                timeout=timeout_seconds,
            )
        response.raise_for_status()
        return parse_anthropic(response.json())

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        body = self._base_body(config, messages, None)
        body["stream"] = True
        async with self.create_client(config) as client:
            async with client.stream(
                "POST",
                "/v1/messages",
                json=body,
                timeout=timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    delta = _parse_anthropic_delta(line)
                    if delta:
                        yield delta


class ProviderDispatcher:
    """Routes a request to the right adapter via ``ModelConfig.provider_type``."""

    def __init__(self, providers: dict[str, Provider] | None = None) -> None:
        self._providers = providers or {
            "openai_chat": OpenAICompatibleProvider(),
            "openai_responses": OpenAIResponsesProvider(),
            "anthropic": AnthropicProvider(),
        }

    def _resolve(self, config: ModelConfig) -> Provider:
        provider = self._providers.get(config.provider_type)
        if provider is None:
            raise GatewayError(
                "unsupported_provider",
                f"不支持的供应商类型: {config.provider_type}",
                500,
            )
        return provider

    async def complete(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
        response_schema: dict[str, Any] | None,
    ) -> tuple[str, Usage]:
        return await self._resolve(config).complete(config, messages, timeout_seconds, response_schema)

    async def stream(
        self,
        config: ModelConfig,
        messages: list[Message],
        timeout_seconds: float,
    ) -> AsyncIterator[str]:
        async for delta in self._resolve(config).stream(config, messages, timeout_seconds):
            yield delta