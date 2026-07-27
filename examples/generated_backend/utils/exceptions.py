"""
自定义异常 + 全局异常处理器
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError


class AppError(Exception):
    """应用基础异常"""
    def __init__(self, message: str, status_code: int = 400, code: str = "app_error"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(message)


class ValidationError(AppError):
    """参数验证错误"""
    def __init__(self, message: str):
        super().__init__(message, status_code=400, code="validation_error")


class AuthenticationError(AppError):
    """认证错误"""
    def __init__(self, message: str = "认证失败"):
        super().__init__(message, status_code=401, code="authentication_error")


class UserNotFoundError(AppError):
    """用户不存在"""
    def __init__(self, message: str = "用户不存在"):
        super().__init__(message, status_code=404, code="user_not_found")


class PostNotFoundError(AppError):
    """文章不存在"""
    def __init__(self, message: str = "文章不存在"):
        super().__init__(message, status_code=404, code="post_not_found")


class PermissionDeniedError(AppError):
    """权限不足"""
    def __init__(self, message: str = "权限不足"):
        super().__init__(message, status_code=403, code="permission_denied")


class RateLimitError(AppError):
    """限流错误"""
    def __init__(self, message: str = "请求过于频繁"):
        super().__init__(message, status_code=429, code="rate_limit_exceeded")


def register_exception_handlers(app: FastAPI):
    """注册全局异常处理器"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.code,
                "message": exc.message,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "http_error",
                "message": exc.detail,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = []
        for err in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
                "type": err["type"],
            })
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_failed",
                "message": "参数验证失败",
                "details": errors,
            },
        )

    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(request: Request, exc: IntegrityError):
        return JSONResponse(
            status_code=409,
            content={
                "error": "conflict",
                "message": "数据冲突，请检查唯一约束",
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def sql_error_handler(request: Request, exc: SQLAlchemyError):
        return JSONResponse(
            status_code=500,
            content={
                "error": "database_error",
                "message": "数据库操作失败",
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "服务器内部错误",
            },
        )
