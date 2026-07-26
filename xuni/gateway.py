"""
XuniGateway —— 虚拟 API 网关

提供统一的 HTTP 风格请求接口，把"凭证认证 + 模型路由 + 能量计费"串起来。

    APIRequest  —— 一次调用请求（端点 + 凭证 + 提示词 + 参数）
    APIResponse —— 一次调用响应（success + data + error）
    APIEndpoint —— 可用端点枚举
    APIError     —— 网关层异常
    XuniGateway  —— 网关本体，handle_request(req) -> resp

设计为最小可用：任何有效凭证可读，模型调用与系统统计需要相应凭证类型。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional

from .model import ModelInput


class APIEndpoint(Enum):
    MODELS_LIST = auto()
    MODELS_INFO = auto()
    MODELS_PREDICT = auto()
    SYSTEM_STATISTICS = auto()


@dataclass
class APIRequest:
    """虚拟 API 请求。"""
    endpoint: APIEndpoint
    token_id: str
    model_id: Optional[str] = None
    prompt: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """虚拟 API 响应。"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0


class APIError(Exception):
    """网关层异常。"""


# 凭证类型需要的枚举名（延迟导入避免循环依赖）
_MODEL_TYPES = {"MODEL_TOKEN", "PREMIUM_TOKEN", "API_KEY"}
_PREMIUM_TYPES = {"PREMIUM_TOKEN", "API_KEY"}


class XuniGateway:
    """
    虚拟 API 网关。

    Args:
        credential_manager: XuniCredential 实例，负责凭证验证与消耗
        model_registry: XuniModelRegistry 实例，负责模型查找与调用
    """

    def __init__(self, credential_manager=None, model_registry=None):
        self.credential_manager = credential_manager
        self.model_registry = model_registry
        self._request_counter = 0

    # ------------------------------------------------------------------ #
    def handle_request(self, req: APIRequest) -> APIResponse:
        """处理一次 API 请求。"""
        start = time.time()
        self._request_counter += 1

        # 1) 凭证认证
        token = None
        if self.credential_manager is not None:
            token = self.credential_manager.validate(req.token_id)
        if token is None:
            return APIResponse(False, error="无效或已过期的凭证",
                               latency_ms=(time.time() - start) * 1000)

        token_type_name = token.token_type.name

        try:
            if req.endpoint == APIEndpoint.MODELS_LIST:
                if self.model_registry is None:
                    return APIResponse(False, error="未配置模型注册表")
                data = {"models": self.model_registry.list_models()}

            elif req.endpoint == APIEndpoint.MODELS_INFO:
                if not req.model_id:
                    return APIResponse(False, error="缺少 model_id")
                model = self.model_registry.get_model(req.model_id) if self.model_registry else None
                if model is None:
                    return APIResponse(False, error=f"模型不存在: {req.model_id}")
                data = model.get_info()

            elif req.endpoint == APIEndpoint.MODELS_PREDICT:
                if token_type_name not in _MODEL_TYPES:
                    return APIResponse(False, error="需要 MODEL_TOKEN 或更高凭证才能调用模型")
                if not req.model_id:
                    return APIResponse(False, error="缺少 model_id")
                model = self.model_registry.get_model(req.model_id) if self.model_registry else None
                if model is None:
                    return APIResponse(False, error=f"模型不存在: {req.model_id}")
                # 消耗凭证
                if hasattr(self.credential_manager, "consume_token"):
                    self.credential_manager.consume_token(req.token_id, cost=1)
                output = model.predict(ModelInput(
                    prompt=req.prompt,
                    parameters=req.parameters or {},
                ))
                data = {
                    "model_id": model.model_id,
                    "text": output.text,
                    "json": output.json,
                    "classification": output.classification,
                    "prediction": output.prediction,
                    "latency_ms": output.latency_ms,
                    "energy_consumed": output.energy_consumed,
                }

            elif req.endpoint == APIEndpoint.SYSTEM_STATISTICS:
                if token_type_name not in _PREMIUM_TYPES:
                    return APIResponse(False, error="需要 PREMIUM_TOKEN 或 API_KEY 才能查看系统统计")
                data = {
                    "total_requests": self._request_counter,
                    "models": self.model_registry.statistics() if self.model_registry else {},
                    "credentials": (
                        self.credential_manager.statistics()
                        if self.credential_manager is not None else {}
                    ),
                }

            else:
                return APIResponse(False, error=f"未知端点: {req.endpoint}")

            # 记录执行
            if self.credential_manager is not None and hasattr(self.credential_manager, "record_execution"):
                self.credential_manager.record_execution(req.token_id, success=True, action=req.endpoint.name)

            return APIResponse(
                True,
                data=data,
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            if self.credential_manager is not None and hasattr(self.credential_manager, "record_execution"):
                self.credential_manager.record_execution(req.token_id, success=False,
                                                        action=req.endpoint.name, details={"error": str(e)})
            return APIResponse(False, error=f"网关处理失败: {e}",
                               latency_ms=(time.time() - start) * 1000)

    def statistics(self) -> Dict[str, Any]:
        return {"total_requests": self._request_counter}
