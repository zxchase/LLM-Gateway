"""按模型名分桶的滑动窗口限流器。

每个模型拥有独立的滑动窗口桶（窗口内最多 ``max_requests`` 次请求），
模型之间互不影响：耗尽一个模型的桶不会阻塞其他模型的请求。
超限的请求由调用方以 429 ``rate_limited`` 拒绝，不重试、不降级。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

from ..errors import GatewayError

DEFAULT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "60"))
DEFAULT_WINDOW_SECONDS = float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


class ModelRateLimiter:
    """模型级滑动窗口限流器。

    ``try_acquire`` 在窗口内有余量时记录时间戳并放行，否则拒绝；
    过期时间戳在每次访问时惰性清理，无需后台任务。
    """

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def try_acquire(self, model: str) -> bool:
        """尝试占用模型的一个请求名额，成功返回 True。"""
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.setdefault(model, deque())
            threshold = now - self.window_seconds
            while bucket and bucket[0] <= threshold:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        """清空所有模型的桶（限流参数保持不变）。"""
        with self._lock:
            self._buckets.clear()

    def configure(self, max_requests: int, window_seconds: float) -> None:
        """调整限流参数并清空所有桶。"""
        with self._lock:
            self.max_requests = max_requests
            self.window_seconds = window_seconds
            self._buckets.clear()


model_rate_limiter = ModelRateLimiter(DEFAULT_MAX_REQUESTS, DEFAULT_WINDOW_SECONDS)


def acquire_model_or_raise(model: str) -> None:
    """为请求的模型获取限流名额，超限时抛出 429 GatewayError。"""
    if not model_rate_limiter.try_acquire(model):
        raise GatewayError(
            "rate_limited",
            f"模型 {model} 已达限流阈值"
            f"（{model_rate_limiter.max_requests} 次/{model_rate_limiter.window_seconds:.0f}s），请稍后重试",
            429,
        )
