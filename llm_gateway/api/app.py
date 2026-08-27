"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..adapters.provider import Provider
from ..errors import GatewayError
from .routes import build_router


def create_app(provider: Provider | None = None) -> FastAPI:
    app = FastAPI(title="Agent LLM Gateway", version="0.0.1")
    app.state.provider = provider
    app.include_router(build_router(provider=provider))

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail()})

    return app


app = create_app()

