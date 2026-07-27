"""
限流中间件 — 滑动窗口 + Redis
"""

import time
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..redis_client import get_redis


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """全局限流中间件"""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        # 跳过内部路径
        if request.url.path in ("/health", "/favicon.ico"):
            return await call_next(request)

        # 获取客户端标识（IP + UserAgent）
        client_id = self._get_client_id(request)
        key = f"rate_limit:{client_id}"

        redis = get_redis()
        now = int(time.time() * 1000)
        window_ms = self.window_seconds * 1000

        # 移除窗口外的记录
        await redis.zremrangebyscore(key, 0, now - window_ms)
        # 当前窗口请求数
        current = await redis.zcard(key)

        if current >= self.max_requests:
            retry_after = self.window_seconds
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "请求过于频繁，请稍后再试",
                    "retry_after": retry_after,
                },
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Retry-After": str(retry_after),
                },
            )

        # 添加当前请求
        request_id = str(uuid.uuid4())
        await redis.zadd(key, {request_id: now})
        await redis.expire(key, self.window_seconds)

        remaining = self.max_requests - current - 1

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Window"] = f"{self.window_seconds}s"

        return response

    def _get_client_id(self, request: Request) -> str:
        """获取客户端唯一标识"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        ua = request.headers.get("User-Agent", "")[:100]
        return f"{ip}:{hash(ua) & 0xffffffff}"
