"""
TokenAttributeSubstances —— 真实 Token 属性物质

把"真实 token 缺失的 7 个属性"做成 7 种虚拟物质。
让虚拟 DownloadToken 融合这些物质，逐步"长"成接近真实 token 的样子。

每种属性物质携带一个真实 token 的属性值，融合时把该值写入 DownloadToken.metadata。
- TokenIdSubstance       → token_id
- TokenTextSubstance     → text（子词文本）
- TokenLogProbSubstance  → logprob（对数概率）
- TokenRankSubstance     → rank（候选排名）
- TokenEntropySubstance  → entropy_bits（自信息量）
- TokenPositionSubstance → position（序列位置）
- TokenEmbeddingSubstance→ embedding（向量）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .multiverse_resources import (
    DownloadToken,
    MultiverseResourceFactory,
    ResourceDimension,
    ResourceRarity,
    VirtualResource,
)


# ============================================================
# 7 种真实属性物质
# ============================================================

@dataclass
class TokenIdSubstance(VirtualResource):
    """token_id 物质——携带一个词表里的整数 ID"""
    token_id: int = 0

    def __post_init__(self):
        if self.name == "":
            self.name = "TokenId物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.RARE
        self.metadata["token_id"] = self.token_id


@dataclass
class TokenTextSubstance(VirtualResource):
    """text 物质——携带子词文本"""
    text: str = ""

    def __post_init__(self):
        if self.name == "":
            self.name = "TokenText物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.RARE
        self.metadata["text"] = self.text
        self.metadata["byte_length"] = len(self.text.encode("utf-8"))
        self.metadata["char_length"] = len(self.text)


@dataclass
class TokenLogProbSubstance(VirtualResource):
    """logprob 物质——携带对数概率"""
    logprob: float = 0.0

    def __post_init__(self):
        if self.name == "":
            self.name = "LogProb物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.EPIC
        self.metadata["logprob"] = self.logprob


@dataclass
class TokenRankSubstance(VirtualResource):
    """rank 物质——携带候选分布中的排名"""
    rank: int = 1

    def __post_init__(self):
        if self.name == "":
            self.name = "Rank物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.EPIC
        self.metadata["rank"] = self.rank


@dataclass
class TokenEntropySubstance(VirtualResource):
    """entropy_bits 物质——携带单 token 自信息量"""
    entropy_bits: float = 0.0

    def __post_init__(self):
        if self.name == "":
            self.name = "Entropy物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.EPIC
        self.metadata["entropy_bits"] = self.entropy_bits


@dataclass
class TokenPositionSubstance(VirtualResource):
    """position 物质——携带序列位置"""
    position: int = 0

    def __post_init__(self):
        if self.name == "":
            self.name = "Position物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.RARE
        self.metadata["position"] = self.position


@dataclass
class TokenEmbeddingSubstance(VirtualResource):
    """embedding 物质——携带高维向量"""
    embedding: Optional[np.ndarray] = None
    embedding_dim: int = 0

    def __post_init__(self):
        if self.name == "":
            self.name = "Embedding物质"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self.rarity = ResourceRarity.LEGENDARY
        if self.embedding is not None:
            self.embedding_dim = int(self.embedding.shape[0])
            self.metadata["embedding_dim"] = self.embedding_dim
            self.metadata["embedding_norm"] = float(np.linalg.norm(self.embedding))


# ============================================================
# Token 属性锻造炉——让虚拟 Token 吸收属性物质
# ============================================================

class TokenAttributeForge:
    """
    属性锻造炉——把 7 种属性物质融合到 DownloadToken 上。

    每融合一种物质，对应的属性值就写入 DownloadToken.metadata。
    融合 7 种之后，虚拟 Token 就具备了真实 token 的所有 7 个核心属性。
    """

    # 7 种属性 → metadata 字段名映射
    ATTRIBUTE_MAP = {
        "TokenId物质":       "token_id",
        "TokenText物质":     "text",
        "LogProb物质":       "logprob",
        "Rank物质":          "rank",
        "Entropy物质":       "entropy_bits",
        "Position物质":      "position",
        "Embedding物质":     "embedding",
    }

    def __init__(self, factory: Optional[MultiverseResourceFactory] = None):
        self.factory = factory or MultiverseResourceFactory()
        self.fusion_log: List[Dict[str, Any]] = []

    def absorb(self, token: DownloadToken, substance: VirtualResource) -> Dict[str, Any]:
        """
        让 token 吸收一个属性物质——把属性值写入 metadata。

        模拟"负负得正"提纯同款守恒律：
        - 每次吸收消耗一点点 quality 作为"认知代价"
        - 但属性值完整进入 token
        """
        substance_name = substance.name
        attr_key = self.ATTRIBUTE_MAP.get(substance_name)
        if attr_key is None:
            # 按子串匹配兜底
            for key, val in self.ATTRIBUTE_MAP.items():
                if key in substance_name or substance_name in key:
                    attr_key = val
                    break

        if attr_key is None:
            return {
                "absorbed": False,
                "reason": f"未知物质: {substance_name}",
            }

        # 提取属性值
        if attr_key == "embedding":
            value = getattr(substance, "embedding", None)
            if value is not None:
                value = np.asarray(value, dtype=np.float32)
        else:
            value = substance.metadata.get(attr_key)
            if value is None:
                value = getattr(substance, attr_key, None)

        old_value = token.metadata.get(attr_key)

        # 写入 token.metadata
        token.metadata[attr_key] = value

        # 守恒代价：每次吸收消耗 0.5 quality（但不低于 1.0）
        quality_cost = 0.5
        old_quality = token.quality
        token.quality = max(1.0, token.quality - quality_cost)

        # 记录融合历史
        token.collision_history.append(
            f"吸收 {substance_name} → {attr_key}={value!r}"
        )

        log_entry = {
            "absorbed": True,
            "substance": substance_name,
            "attribute": attr_key,
            "old_value": old_value,
            "new_value": value,
            "quality_cost": quality_cost,
            "quality_before": old_quality,
            "quality_after": token.quality,
        }
        self.fusion_log.append(log_entry)
        return log_entry

    def absorb_all(
        self,
        token: DownloadToken,
        substances: List[VirtualResource],
    ) -> Dict[str, Any]:
        """依次吸收多个属性物质"""
        results = []
        for s in substances:
            r = self.absorb(token, s)
            results.append(r)
        return {
            "total_absorbed": sum(1 for r in results if r.get("absorbed")),
            "total_attempted": len(substances),
            "results": results,
            "final_metadata_keys": list(token.metadata.keys()),
            "final_quality": token.quality,
        }

    def has_real_attribute(self, token: DownloadToken, attr: str) -> bool:
        """检查 token 是否已经持有某个真实属性"""
        return attr in token.metadata and token.metadata[attr] is not None

    def real_attributes_status(self, token: DownloadToken) -> Dict[str, bool]:
        """返回 7 个真实属性的持有状态"""
        return {
            attr: self.has_real_attribute(token, attr)
            for attr in self.ATTRIBUTE_MAP.values()
        }


# ============================================================
# 工厂方法：从真实 token 数据生成属性物质
# ============================================================

def substance_from_real_token(real_token: Dict[str, Any]) -> List[VirtualResource]:
    """
    从一个真实 token 的属性字典，生成 7 种属性物质。

    Args:
        real_token: 字典，包含 token_id, text, logprob, rank,
                    entropy_bits, position, embedding 等键

    Returns:
        7 种属性物质的列表
    """
    substances: List[VirtualResource] = []

    # 1. token_id
    if "token_id" in real_token:
        substances.append(TokenIdSubstance(token_id=int(real_token["token_id"])))

    # 2. text
    if "text" in real_token:
        substances.append(TokenTextSubstance(text=str(real_token["text"])))

    # 3. logprob
    if "logprob" in real_token:
        substances.append(TokenLogProbSubstance(logprob=float(real_token["logprob"])))

    # 4. rank
    if "rank" in real_token:
        substances.append(TokenRankSubstance(rank=int(real_token["rank"])))

    # 5. entropy_bits
    if "entropy_bits" in real_token:
        substances.append(TokenEntropySubstance(
            entropy_bits=float(real_token["entropy_bits"])
        ))

    # 6. position
    if "position" in real_token:
        substances.append(TokenPositionSubstance(position=int(real_token["position"])))

    # 7. embedding（如果有的话）
    if "embedding" in real_token and real_token["embedding"] is not None:
        emb = np.asarray(real_token["embedding"], dtype=np.float32)
        substances.append(TokenEmbeddingSubstance(embedding=emb, embedding_dim=emb.shape[0]))

    return substances


def synthesize_embedding(token_id: int, dim: int = 12288) -> np.ndarray:
    """
    合成一个伪 embedding——用 token_id 做种子的确定向量。
    这不是真实 embedding，只是占位（结构上跟真实 embedding 等长）。
    """
    rng = np.random.default_rng(token_id)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)  # 单位化


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TokenIdSubstance",
    "TokenTextSubstance",
    "TokenLogProbSubstance",
    "TokenRankSubstance",
    "TokenEntropySubstance",
    "TokenPositionSubstance",
    "TokenEmbeddingSubstance",
    "TokenAttributeForge",
    "substance_from_real_token",
    "synthesize_embedding",
]
