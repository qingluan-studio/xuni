"""
XuniCredential —— 虚拟凭证系统

核心理念：采样点产生的虚拟电场能量可以兑换为"虚拟凭证"，
凭证可以：
- 作为虚拟 API 的认证令牌
- 兑换虚拟模型的调用次数
- 存储、转移、消耗
- 过期、刷新、升级

凭证类型：
- ACCESS_TOKEN: 通用访问令牌
- MODEL_TOKEN: 模型调用凭证
- PREMIUM_TOKEN: 高级功能凭证
- API_KEY: 虚拟 API 密钥

能量转换率：场能量 → 凭证强度
"""

import hashlib
import time
import uuid
import base64
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List
import numpy as np


class CredentialType(Enum):
    ACCESS_TOKEN = auto()
    MODEL_TOKEN = auto()
    PREMIUM_TOKEN = auto()
    API_KEY = auto()


class TokenStatus(Enum):
    ACTIVE = auto()
    EXPIRED = auto()
    REVOKED = auto()
    CONSUMED = auto()


@dataclass
class XuniToken:
    """虚拟凭证令牌"""
    token_id: str
    token_type: CredentialType
    status: TokenStatus = TokenStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    consumed_at: float = 0.0
    energy_value: float = 0.0
    remaining_calls: int = 0
    scope: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def is_valid(self) -> bool:
        return (
            self.status == TokenStatus.ACTIVE
            and (self.expires_at == 0.0 or time.time() < self.expires_at)
            and (self.remaining_calls == 0 or self.remaining_calls > 0)
        )

    def consume(self, cost: int = 1) -> bool:
        if not self.is_valid():
            return False
        if self.remaining_calls > 0:
            self.remaining_calls -= cost
            if self.remaining_calls <= 0:
                self.status = TokenStatus.CONSUMED
                self.consumed_at = time.time()
        return True

    def to_jwt(self) -> str:
        """生成 JWT 格式令牌"""
        header = {
            "alg": "XUNI",
            "typ": "JWT",
            "type": self.token_type.name.lower(),
        }
        payload = {
            "tid": self.token_id,
            "energy": round(self.energy_value, 4),
            "exp": int(self.expires_at) if self.expires_at > 0 else 0,
            "scope": self.scope,
        }
        
        def _b64url_encode(data: dict) -> str:
            import json
            return base64.urlsafe_b64encode(
                json.dumps(data, separators=(",", ":")).encode()
            ).decode().rstrip("=")
        
        signature = hashlib.sha256(
            f"{self.token_id}{self.created_at}".encode()
        ).hexdigest()[:32]
        
        return ".".join([
            _b64url_encode(header),
            _b64url_encode(payload),
            signature,
        ])

    @classmethod
    def from_jwt(cls, token_str: str) -> Optional["XuniToken"]:
        """从 JWT 解析令牌"""
        try:
            parts = token_str.split(".")
            if len(parts) != 3:
                return None
            
            import json
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
            
            return cls(
                token_id=payload.get("tid", ""),
                token_type=CredentialType[payload.get("type", "ACCESS_TOKEN").upper()],
                status=TokenStatus.ACTIVE,
                energy_value=payload.get("energy", 0.0),
                expires_at=payload.get("exp", 0.0),
                scope=payload.get("scope", []),
            )
        except Exception:
            return None


class XuniCredential:
    """
    虚拟凭证管理器。
    
    核心能力：
    1. 场能量 → 凭证兑换
    2. 凭证发放、验证、消耗
    3. 凭证存储与检索
    4. 凭证升级与降级
    5. 批量生成凭证
    6. 凭证执行验证（验证凭证是否真能执行）
    """

    def __init__(self, energy_conversion_rate: float = 100.0):
        self.conversion_rate = energy_conversion_rate
        self.tokens: Dict[str, XuniToken] = {}
        self._token_counter = 0
        self._execution_records: Dict[str, List[Dict[str, Any]]] = {}

    def _generate_token_id(self) -> str:
        """生成24位唯一令牌 ID（字母+数字）"""
        self._token_counter += 1
        raw = hashlib.md5(
            f"{uuid.uuid4()}{self._token_counter}{time.time()}".encode()
        ).hexdigest()[:24]
        
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        rng = np.random.default_rng(int(hashlib.md5(raw.encode()).hexdigest(), 16) % 1000000)
        result = list(raw)
        for i in range(len(result)):
            if rng.random() < 0.3:
                result[i] = rng.choice(list(chars))
        return "".join(result)

    def mint(
        self,
        field_energy: float,
        token_type: CredentialType = CredentialType.ACCESS_TOKEN,
        duration_hours: float = 24.0,
        max_calls: int = 0,
        scope: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> XuniToken:
        """
        用虚拟场能量铸造凭证。
        
        Args:
            field_energy: 虚拟电场能量
            token_type: 凭证类型
            duration_hours: 有效期（小时），0 表示永不过期
            max_calls: 最大调用次数，0 表示无限
            scope: 权限范围
            metadata: 附加元数据
        
        Returns:
            XuniToken 凭证对象
        """
        energy_value = field_energy * self.conversion_rate
        
        token = XuniToken(
            token_id=self._generate_token_id(),
            token_type=token_type,
            status=TokenStatus.ACTIVE,
            created_at=time.time(),
            expires_at=time.time() + duration_hours * 3600 if duration_hours > 0 else 0.0,
            energy_value=energy_value,
            remaining_calls=max_calls,
            scope=scope or [],
            metadata=metadata or {},
        )
        
        self.tokens[token.token_id] = token
        return token

    def mint_batch(
        self,
        field_energy: float,
        count: int = 10,
        token_type: CredentialType = CredentialType.MODEL_TOKEN,
        duration_hours: float = 24.0,
        max_calls: int = 100,
    ) -> list[XuniToken]:
        """批量铸造凭证"""
        tokens = []
        for _ in range(count):
            tokens.append(self.mint(
                field_energy=field_energy / count,
                token_type=token_type,
                duration_hours=duration_hours,
                max_calls=max_calls,
            ))
        return tokens

    def validate(self, token_id: str) -> Optional[XuniToken]:
        """验证凭证是否有效"""
        token = self.tokens.get(token_id)
        if token and token.is_valid():
            return token
        return None

    def consume_token(self, token_id: str, cost: int = 1) -> bool:
        """消耗凭证"""
        token = self.tokens.get(token_id)
        if token and token.consume(cost):
            return True
        return False

    def revoke(self, token_id: str) -> bool:
        """吊销凭证"""
        token = self.tokens.get(token_id)
        if token:
            token.status = TokenStatus.REVOKED
            return True
        return False

    def refresh(self, token_id: str, additional_energy: float = 0.0) -> Optional[XuniToken]:
        """刷新凭证（延长有效期、增加能量）"""
        token = self.tokens.get(token_id)
        if not token:
            return None
        
        token.created_at = time.time()
        token.expires_at = time.time() + 24 * 3600
        if additional_energy > 0:
            token.energy_value += additional_energy * self.conversion_rate
        
        if token.status in (TokenStatus.EXPIRED, TokenStatus.CONSUMED):
            token.status = TokenStatus.ACTIVE
            if token.remaining_calls <= 0:
                token.remaining_calls = int(token.energy_value // 10)
        
        return token

    def upgrade(self, token_id: str) -> Optional[XuniToken]:
        """升级凭证（ACCESS → MODEL → PREMIUM）"""
        token = self.tokens.get(token_id)
        if not token:
            return None
        
        upgrade_map = {
            CredentialType.ACCESS_TOKEN: CredentialType.MODEL_TOKEN,
            CredentialType.MODEL_TOKEN: CredentialType.PREMIUM_TOKEN,
            CredentialType.PREMIUM_TOKEN: CredentialType.PREMIUM_TOKEN,
            CredentialType.API_KEY: CredentialType.API_KEY,
        }
        
        new_type = upgrade_map.get(token.token_type, token.token_type)
        if new_type != token.token_type:
            token.token_type = new_type
            token.energy_value *= 2.0
            token.scope.extend(["upgrade", "enhanced"])
        
        return token

    def get_token_info(self, token_id: str) -> Optional[dict]:
        """获取凭证详情"""
        token = self.tokens.get(token_id)
        if not token:
            return None
        
        return {
            "token_id": token.token_id,
            "type": token.token_type.name,
            "status": token.status.name,
            "created_at": token.created_at,
            "expires_at": token.expires_at,
            "energy_value": token.energy_value,
            "remaining_calls": token.remaining_calls,
            "scope": token.scope,
            "metadata": token.metadata,
            "jwt": token.to_jwt(),
        }

    def list_tokens(self, status_filter: Optional[TokenStatus] = None) -> list[dict]:
        """列出所有凭证"""
        result = []
        for token in self.tokens.values():
            if status_filter and token.status != status_filter:
                continue
            result.append(self.get_token_info(token.token_id))
        return result

    def statistics(self) -> dict:
        """凭证统计"""
        active = sum(1 for t in self.tokens.values() if t.status == TokenStatus.ACTIVE)
        expired = sum(1 for t in self.tokens.values() if t.status == TokenStatus.EXPIRED)
        revoked = sum(1 for t in self.tokens.values() if t.status == TokenStatus.REVOKED)
        consumed = sum(1 for t in self.tokens.values() if t.status == TokenStatus.CONSUMED)
        
        total_energy = sum(t.energy_value for t in self.tokens.values())
        total_calls = sum(t.remaining_calls for t in self.tokens.values() if t.remaining_calls > 0)
        
        return {
            "total_tokens": len(self.tokens),
            "active": active,
            "expired": expired,
            "revoked": revoked,
            "consumed": consumed,
            "total_energy_value": round(total_energy, 2),
            "total_remaining_calls": total_calls,
        }

    def record_execution(self, token_id: str, success: bool, action: str = "", details: Dict[str, Any] = None):
        """记录凭证执行记录"""
        if token_id not in self._execution_records:
            self._execution_records[token_id] = []
        
        record = {
            "timestamp": time.time(),
            "success": success,
            "action": action,
            "details": details or {},
        }
        self._execution_records[token_id].append(record)
        
        if len(self._execution_records[token_id]) > 100:
            self._execution_records[token_id] = self._execution_records[token_id][-100:]

    def verify_execution(self, token_id: str) -> Dict[str, Any]:
        """
        验证凭证是否真能执行。
        
        返回执行历史和成功率，判断凭证是否有效可用。
        """
        token = self.tokens.get(token_id)
        if not token:
            return {"valid": False, "error": "Token not found"}
        
        records = self._execution_records.get(token_id, [])
        
        if not records:
            return {
                "valid": True,
                "token_info": self.get_token_info(token_id),
                "execution_history": "No execution records",
                "success_rate": "N/A",
                "can_execute": token.is_valid(),
            }
        
        success_count = sum(1 for r in records if r["success"])
        success_rate = success_count / len(records)
        
        return {
            "valid": token.is_valid(),
            "token_info": self.get_token_info(token_id),
            "execution_history": len(records),
            "success_rate": round(success_rate * 100, 2),
            "last_execution": records[-1]["timestamp"],
            "last_action": records[-1]["action"],
            "can_execute": token.is_valid() and success_rate > 0,
        }

    def get_execution_history(self, token_id: str) -> List[Dict[str, Any]]:
        """获取凭证执行历史"""
        return self._execution_records.get(token_id, [])
