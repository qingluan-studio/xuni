"""
日志工具
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from ..config import settings


_LOGGERS: dict[str, logging.Logger] = {}


def setup_logger(name: str) -> logging.Logger:
    """配置并返回 logger"""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_get_formatter())
    logger.addHandler(console_handler)

    # 文件输出（生产环境）
    if settings.ENV == "production" and settings.LOG_FILE:
        os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(_get_formatter())
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def _get_formatter() -> logging.Formatter:
    """获取日志格式化器"""
    fmt = (
        "%(asctime)s | %(levelname)-7s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
