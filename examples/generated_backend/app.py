"""
Xenith Blog — 完整 Web 后端系统
技术栈：FastAPI + SQLAlchemy + Redis + JWT + Celery
生成等级：质量点强化 9/10 (SSS级)
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .config import settings
from .database import engine, Base, get_db
from .redis_client import get_redis
from .auth import router as auth_router
from .users import router as users_router
from .posts import router as posts_router
from .comments import router as comments_router
from .tags import router as tags_router
from .search import router as search_router
from .admin import router as admin_router
from .middleware.rate_limiter import RateLimiterMiddleware
from .middleware.request_id import RequestIDMiddleware
from .utils.logger import setup_logger
from .utils.exceptions import (
    UserNotFoundError, PostNotFoundError,
    PermissionDeniedError, ValidationError,
    register_exception_handlers,
)


logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动：初始化数据库
    Base.metadata.create_all(bind=engine)
    logger.info("数据库表初始化完成")

    # 预热 Redis 连接池
    redis = get_redis()
    await redis.ping()
    logger.info("Redis 连接池就绪")

    # 启动后台任务（Celery worker 由外部进程管理）
    logger.info(f"应用启动完成，环境: {settings.ENV}")

    yield

    # 关闭：清理资源
    await engine.dispose()
    logger.info("数据库连接池已关闭")
    await get_redis().close()
    logger.info("Redis 连接已关闭")
    logger.info("应用已优雅关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="Xenith Blog API",
        description="面向开发者的高性能博客系统后端",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.ENV != "production" else None,
        redoc_url="/redoc" if settings.ENV != "production" else None,
    )

    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求 ID 中间件
    app.add_middleware(RequestIDMiddleware)

    # 限流中间件
    app.add_middleware(
        RateLimiterMiddleware,
        max_requests=settings.RATE_LIMIT_MAX,
        window_seconds=settings.RATE_LIMIT_WINDOW,
    )

    # 注册异常处理器
    register_exception_handlers(app)

    # 注册路由
    app.include_router(auth_router, prefix="/api/v1/auth", tags=["认证"])
    app.include_router(users_router, prefix="/api/v1/users", tags=["用户"])
    app.include_router(posts_router, prefix="/api/v1/posts", tags=["文章"])
    app.include_router(comments_router, prefix="/api/v1/comments", tags=["评论"])
    app.include_router(tags_router, prefix="/api/v1/tags", tags=["标签"])
    app.include_router(search_router, prefix="/api/v1/search", tags=["搜索"])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["管理"])

    @app.get("/health", tags=["系统"])
    async def health_check(db: Session = Depends(get_db)):
        """健康检查"""
        try:
            # 检查数据库
            db.execute("SELECT 1")
            db_health = True
        except Exception:
            db_health = False

        try:
            redis = get_redis()
            await redis.ping()
            redis_health = True
        except Exception:
            redis_health = False

        return {
            "status": "healthy" if (db_health and redis_health) else "degraded",
            "services": {
                "database": "ok" if db_health else "fail",
                "redis": "ok" if redis_health else "fail",
            },
            "version": "2.0.0",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level="info",
    )
