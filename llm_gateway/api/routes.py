"""HTTP entry points."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..adapters.provider import Provider
from ..config import CALL_TRACES
from ..errors import GatewayError
from ..models import CallTrace, LLMRequest, LLMResponse
from ..services.gateway import (
    build_messages,
    call_with_fallback,
    stream_with_fallback,
    validate_model,
)


def build_router(provider: Provider | None = None) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/llm", response_model=LLMResponse)
    async def create_llm_response(request: LLMRequest) -> LLMResponse:
        if request.stream:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "use_stream_endpoint",
                    "message": "流式请求请使用 /v1/llm/stream",
                },
            )
        try:
            return await call_with_fallback(request, provider=provider)
        except GatewayError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc

    @router.post("/v1/llm/stream")
    async def create_stream(request: LLMRequest) -> StreamingResponse:
        if request.response_schema is not None:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "unsupported_combination",
                    "message": "流式输出不支持 response_schema",
                },
            )
        try:
            validate_model(request.model, None)
            build_messages(request)
        except GatewayError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"code": exc.code, "message": exc.message},
            ) from exc
        return StreamingResponse(
            stream_with_fallback(request, provider=provider),
            media_type="text/event-stream",
        )

    @router.get("/v1/traces", response_model=list[CallTrace])
    async def list_traces() -> list[CallTrace]:
        return CALL_TRACES

    return router

