"""Unified internal error type used across the gateway."""

from __future__ import annotations


class GatewayError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 502) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)

    def detail(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

