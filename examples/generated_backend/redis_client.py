"""
Redis 客户端 — 连接池 + 缓存工具
"""

import json
from typing import Any, Optional

import redis.asyncio as redis

from .config import settings


_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """获取 Redis 客户端单例"""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.REDIS_POOL_SIZE,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    return _redis_client


# ===== 缓存装饰器 =====

async def cache_get(key: str) -> Optional[Any]:
    """获取缓存"""
    r = get_redis()
    data = await r.get(key)
    if data:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return None


async def cache_set(key: str, value: Any, ttl: int = None) -> None:
    """设置缓存"""
    r = get_redis()
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    elif not isinstance(value, str):
        value = str(value)
    if ttl:
        await r.setex(key, ttl, value)
    else:
        await r.set(key, value)


async def cache_delete(key: str) -> None:
    """删除缓存"""
    r = get_redis()
    await r.delete(key)


async def cache_invalidate_pattern(pattern: str) -> int:
    """按模式批量删除缓存，返回删除数量"""
    r = get_redis()
    count = 0
    async for key in r.scan_iter(match=pattern):
        await r.delete(key)
        count += 1
    return count


# ===== 限流器 =====

class RateLimiter:
    """滑动窗口限流（基于 Redis）"""

    def __init__(self, key_prefix: str = "rl"):
        self.key_prefix = key_prefix

    async def is_allowed(self, identifier: str, max_requests: int, window_seconds: int) -> bool:
        """检查是否允许请求"""
        r = get_redis()
        key = f"{self.key_prefix}:{identifier}"
        now = int(__import__("time").time() * 1000)

        # 移除窗口外的记录
        await r.zremrangebyscore(key, 0, now - window_seconds * 1000)
        # 当前窗口请求数
        current = await r.zcard(key)

        if current >= max_requests:
            return False

        # 添加当前请求
        await r.zadd(key, {f"{now}-{id}": now})
        # 设置过期
        await r.expire(key, window_seconds)
        return True

    async def get_remaining(self, identifier: str, max_requests: int, window_seconds: int) -> int:
        """获取剩余请求数"""
        r = get_redis()
        key = f"{self.key_prefix}:{identifier}"
        now = int(__import__("time").time() * 1000)
        await r.zremrangebyscore(key, 0, now - window_seconds * 1000)
        current = await r.zcard(key)
        return max(0, max_requests - current)
