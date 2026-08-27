"""Orchestration: fallback, structured-output validation, and audit."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from string import Template
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

import httpx
from jsonschema import ValidationError as JsonSchemaError
from jsonschema import validate

from ..adapters.provider import Provider, ProviderDispatcher
from ..config import (
    CALL_TRACES,
    MODEL_CONFIGS,
    PRICE_PER_MILLION,
    PROMPT_TEMPLATES,
    ModelConfig,
)
from ..errors import GatewayError
from ..models import CallTrace, LLMRequest, LLMResponse, Message, PromptSelection, Usage

logger = logging.getLogger("llm_gateway")

DEFAULT_PROVIDER: Provider = ProviderDispatcher()


def render_prompt(selection: PromptSelection) -> Message:
    template = PROMPT_TEMPLATES.get((selection.name, selection.version))
    if template is None:
        raise GatewayError("unknown_prompt_template", "Prompt 模板不存在", 400)
    try:
        content = Template(template.system_template).substitute(selection.variables)
    except KeyError as exc:
        raise GatewayError("missing_prompt_variable", f"缺少 Prompt 变量: {exc.args[0]}", 400) from exc
    return Message(role="system", content=content)


def build_messages(request: LLMRequest) -> list[Message]:
    if request.prompt is None:
        return request.messages
    return [render_prompt(request.prompt), *request.messages]


def validate_model(model: str, response_schema: dict[str, Any] | None) -> ModelConfig:
    config = MODEL_CONFIGS.get(model)
    if config is None:
        raise GatewayError("unknown_model", "模型不在 Gateway 允许列表中", 400)
    if response_schema is not None and not config.supports_structured_output:
        raise GatewayError("structured_output_unsupported", "模型不支持 Structured Output", 400)
    return config


def calculate_cost(model: str, usage: Usage) -> float:
    price = PRICE_PER_MILLION[model]
    return (usage.input_tokens * price["input"] + usage.output_tokens * price["output"]) / 1_000_000


def record_trace(
    request_id: str,
    requested_model: str,
    actual_model: str | None,
    prompt: PromptSelection | None,
    usage: Usage,
    latency_ms: int,
    attempts: int,
    status: Literal["success", "failed"],
    error_code: str | None = None,
) -> None:
    trace = CallTrace(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc),
        requested_model=requested_model,
        actual_model=actual_model,
        prompt_name=prompt.name if prompt else None,
        prompt_version=prompt.version if prompt else None,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=calculate_cost(actual_model, usage) if actual_model else 0,
        latency_ms=latency_ms,
        attempts=attempts,
        status=status,
        error_code=error_code,
    )
    CALL_TRACES.append(trace)
    logger.info("llm_call_trace=%s", trace.model_dump_json())


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


# 单个模型的最大尝试次数（含首次调用）
MAX_ATTEMPTS_PER_MODEL = 3
# 指数退避基数：第 n 次重试前等待 base * 2^n 秒
RETRY_BACKOFF_BASE_SECONDS = 0.1


def backoff_delay(retry_number: int) -> float:
    """指数退避间隔：0.1s、0.2s、0.4s ..."""
    return RETRY_BACKOFF_BASE_SECONDS * (2 ** retry_number)


async def call_with_fallback(
    request: LLMRequest,
    provider: Provider | None = None,
) -> LLMResponse:
    """非流式调用：每个模型最多 MAX_ATTEMPTS_PER_MODEL 次尝试，指数退避后降级备用模型。

    幂等降级保证：
    * request_id 在重试与降级全程保持不变，审计按最终结果只记录一次；
    * messages 只构建一次，重试/降级复用同一输入，不会重复渲染 Prompt；
    * 降级链经 dict.fromkeys 去重，同一模型不会被降级两次。
    """
    provider = provider or DEFAULT_PROVIDER
    requested_model = request.model
    request_id = str(uuid4())
    started = time.perf_counter()
    attempts = 0
    last_error: Exception | None = None
    messages = build_messages(request)

    for model_name in dict.fromkeys([requested_model, "general-backup"]):
        try:
            config = validate_model(model_name, request.response_schema)
        except GatewayError as exc:
            if model_name == requested_model:
                raise exc
            last_error = exc
            continue

        for retry_number in range(MAX_ATTEMPTS_PER_MODEL):
            attempts += 1
            try:
                content, usage = await provider.complete(
                    config,
                    messages,
                    request.timeout_seconds,
                    request.response_schema,
                )
                parsed: dict[str, Any] | list[Any] | None = None
                if request.response_schema is not None:
                    try:
                        parsed = json.loads(content)
                        validate(instance=parsed, schema=request.response_schema)
                    except json.JSONDecodeError as exc:
                        raise GatewayError("invalid_json", "模型没有返回合法 JSON") from exc
                    except JsonSchemaError as exc:
                        raise GatewayError("schema_validation_failed", "模型结果不符合 response_schema") from exc

                response = LLMResponse(
                    request_id=request_id,
                    model=model_name,
                    content=content,
                    parsed=parsed,
                    usage=usage,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    attempts=attempts,
                )
                record_trace(
                    request_id,
                    requested_model,
                    model_name,
                    request.prompt,
                    usage,
                    response.latency_ms,
                    attempts,
                    "success",
                )
                return response
            except GatewayError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if is_retryable(exc) and retry_number < MAX_ATTEMPTS_PER_MODEL - 1:
                    await asyncio.sleep(backoff_delay(retry_number))
                    continue
                break

    latency_ms = int((time.perf_counter() - started) * 1000)
    error_code = "model_unavailable"
    record_trace(
        request_id,
        requested_model,
        None,
        request.prompt,
        Usage(input_tokens=0, output_tokens=0),
        latency_ms,
        attempts,
        "failed",
        error_code,
    )
    raise GatewayError(error_code, "主模型和备用模型均不可用") from last_error


def encode_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def stream_with_fallback(
    request: LLMRequest,
    provider: Provider | None = None,
) -> AsyncIterator[str]:
    """流式调用：每个模型最多 MAX_ATTEMPTS_PER_MODEL 次尝试，指数退避后降级备用模型。

    幂等降级保证：
    * 已向客户端下发过内容（emitted）后不再重试/降级，避免内容重复下发；
    * request_id 全程不变，审计只在最终成功/失败时记录一次；
    * messages 只构建一次，重试/降级复用同一输入。
    """
    provider = provider or DEFAULT_PROVIDER
    messages = build_messages(request)
    started = time.perf_counter()
    attempts = 0
    emitted = False
    last_error: Exception | None = None
    request_id = str(uuid4())

    for model_name in dict.fromkeys([request.model, "general-backup"]):
        try:
            config = validate_model(model_name, None)
        except GatewayError as exc:
            if model_name == request.model:
                raise exc
            last_error = exc
            continue

        stop_all = False
        for retry_number in range(MAX_ATTEMPTS_PER_MODEL):
            attempts += 1
            try:
                async for delta in provider.stream(config, messages, request.timeout_seconds):
                    emitted = True
                    yield encode_sse({"type": "content.delta", "delta": delta})
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if emitted or not is_retryable(exc):
                    # 已下发部分内容或错误不可重试：降级不再幂等安全，直接终止
                    stop_all = True
                    break
                if retry_number < MAX_ATTEMPTS_PER_MODEL - 1:
                    await asyncio.sleep(backoff_delay(retry_number))
                    continue
                break

            record_trace(
                request_id,
                request.model,
                model_name,
                request.prompt,
                Usage(input_tokens=0, output_tokens=0),
                int((time.perf_counter() - started) * 1000),
                attempts,
                "success",
            )
            yield encode_sse({"type": "response.completed", "model": model_name})
            return
        if stop_all:
            break

    logger.exception("upstream stream failed", exc_info=last_error)
    record_trace(
        request_id,
        request.model,
        None,
        request.prompt,
        Usage(input_tokens=0, output_tokens=0),
        int((time.perf_counter() - started) * 1000),
        attempts,
        "failed",
        "upstream_stream_failed",
    )
    yield encode_sse({"type": "response.failed", "error": "upstream_stream_failed"})

