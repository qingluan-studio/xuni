"""
配置管理 — 支持环境变量覆盖 + 多环境
"""

import os
from typing import List
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置"""

    # 基础配置
    ENV: str = os.getenv("ENV", "development")
    DEBUG: bool = ENV != "production"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-please!!")

    # 数据库
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://user:pass@localhost:5432/xenith_blog"
    )
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_ECHO: bool = DEBUG

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_POOL_SIZE: int = int(os.getenv("REDIS_POOL_SIZE", "50"))

    # JWT
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_TTL", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_TTL", "7"))

    # 限流
    RATE_LIMIT_MAX: int = int(os.getenv("RATE_LIMIT_MAX", "100"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

    # 分页
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # CORS
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    ).split(",")

    # 上传
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # 日志
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if DEBUG else "WARNING")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")

    # 搜索
    ELASTICSEARCH_URL: str = os.getenv("ES_URL", "http://localhost:9200")

    # 缓存
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL", "300"))

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()
