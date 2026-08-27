"""Development entry point for the LLM Gateway service."""

from __future__ import annotations

from llm_gateway.api.app import app

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "gateway:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

