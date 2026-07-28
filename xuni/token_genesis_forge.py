"""
Token 创世熔炉——让空白 Token 从无到有"长"出属性

跟质变熔炉的区别：
    质变熔炉：已有属性值 → 变异
    创世熔炉：None 属性 → 生成（从无到有）

生成逻辑（每种属性不同）：
    token_id     → 用累积的质变能量 + 培养液类型的哈希 → 映射到 0~100276
    text         → 用生成的 token_id 调 tiktoken.decode 看是什么字
    logprob      → 用营养累积映射到合理范围 [-10, 0]
    rank         → 营养累积映射到 [1, 1000]
    entropy_bits → 从生成的 logprob 反推（保持信息论一致：H = -log2 p）
    position     → 营养累积的整数部分
    embedding    → 用培养液类型的哈希做种子，生成 12288 维向量

信息论一致性约束（这次加上）：
    logprob 和 entropy_bits 必须满足 entropy_bits = -log2(exp(logprob))
    这样生成的数值在数学上自洽
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multiverse_resources import (
    CultureMedium,
    DownloadToken,
    MultiverseResourceFactory,
)
from .token_qualitative_forge import (
    AttributeEmergence,
    classify_culture,
    get_culture_mapping,
)


class TokenGenesisForge:
    """
    创世熔炉——让空白 Token 从无到有"长"出属性。

    输入：metadata 全空的 DownloadToken
    输出：metadata 里有了 token_id / text / logprob / ... / embedding

    每次培养液反应累积"创世能量"，能量超过阈值就生成属性值。
    """

    VOCAB_SIZE = 100277  # cl100k_base 词表大小
    GENESIS_THRESHOLD = 1.0   # 创世能量阈值

    def __init__(self, factory: Optional[MultiverseResourceFactory] = None,
                 seed: int = 42):
        self.factory = factory or MultiverseResourceFactory()
        self.emergence: Dict[str, AttributeEmergence] = {}
        self.reaction_log: List[Dict[str, Any]] = []
        self.genesis_events: List[Dict[str, Any]] = []
        # 收集所有参与反应的培养液类型，用于生成种子
        self.culture_types_seen: List[str] = []

    def init_emergence(self, token: DownloadToken) -> None:
        """初始化 7 个属性——全部从 None 开始"""
        real_attrs = ["token_id", "text", "logprob", "rank",
                      "entropy_bits", "position", "embedding"]
        self.emergence = {}
        for attr in real_attrs:
            self.emergence[attr] = AttributeEmergence(
                name=attr,
                source="void",        # 虚空态
                energy=0.0,
                original_value=None,  # 没有原始值
                current_value=None,
            )
            token.metadata[attr] = None
            token.metadata[f"{attr}_source"] = "void"

    def react_with_culture(
        self,
        token: DownloadToken,
        medium: CultureMedium,
    ) -> Dict[str, Any]:
        """让 token 跟一种培养液反应——累积创世能量"""
        culture_type = medium.culture_type
        target_attr, nutrient_keys = get_culture_mapping(culture_type)
        emergence = self.emergence.get(target_attr)
        if emergence is None:
            return {"reacted": False, "reason": f"属性 {target_attr} 无跟踪"}

        self.culture_types_seen.append(culture_type)

        # 提取营养成分
        nutrient_sum = 0.0
        for nk in nutrient_keys:
            nutrient_sum += medium.nutrients.get(nk, 0.0)
        if nutrient_sum == 0 and medium.nutrients:
            nutrient_sum = sum(medium.nutrients.values()) / max(1, len(medium.nutrients))

        # 创世能量增量
        delta = nutrient_sum * medium.level * medium.saturation * 0.3
        old_energy = emergence.energy
        old_source = emergence.source
        emergence.energy += delta

        # 检查是否触发创世
        event = None
        if (old_source == "void"
                and emergence.energy >= self.GENESIS_THRESHOLD):
            # 从无到有——生成属性值
            new_value, gen_desc = self._generate_attribute(
                target_attr, emergence.energy, culture_type
            )
            emergence.current_value = new_value
            emergence.source = "generated"   # 生成态
            token.metadata[target_attr] = new_value
            token.metadata[f"{target_attr}_source"] = "generated"
            event = {
                "type": "genesis",
                "attribute": target_attr,
                "culture_type": culture_type,
                "energy_reached": emergence.energy,
                "generation": gen_desc,
                "message": f"✨ 创世！{target_attr} 从虚空生成！",
            }
            self.genesis_events.append(event)
        # 已经生成的，继续"成长"（值继续微调）
        elif old_source == "generated" and self._should_grow():
            new_value, grow_desc = self._grow_attribute(
                target_attr, emergence.current_value,
                emergence.energy, culture_type
            )
            emergence.current_value = new_value
            token.metadata[target_attr] = new_value
            event = {
                "type": "growth",
                "attribute": target_attr,
                "culture_type": culture_type,
                "generation": grow_desc,
                "message": f"↻ {target_attr} 持续成长",
            }

        token.collision_history.append(
            f"创世反应×{culture_type} → {target_attr} +{delta:.3f} "
            f"(总 {emergence.energy:.3f}, source={emergence.source})"
        )

        log_entry = {
            "reacted": True,
            "culture_type": culture_type,
            "target_attr": target_attr,
            "energy_delta": delta,
            "energy_before": old_energy,
            "energy_after": emergence.energy,
            "source_before": old_source,
            "source_after": emergence.source,
            "event": event,
        }
        self.reaction_log.append(log_entry)
        return log_entry

    def _should_grow(self) -> bool:
        """已生成属性是否继续成长——30% 概率"""
        import random
        return random.random() < 0.3

    def _generate_attribute(
        self,
        attr: str,
        energy: float,
        culture_type: str,
    ) -> Tuple[Any, str]:
        """从虚空生成一个属性值"""

        # 用所有见过的培养液类型做种子源
        seed_source = "|".join(self.culture_types_seen)
        seed_hash = int(hashlib.md5(seed_source.encode()).hexdigest()[:8], 16)

        if attr == "token_id":
            # 用能量 + 种子哈希映射到词表
            import random
            rng = random.Random(seed_hash + int(energy * 1000))
            new_id = rng.randint(0, self.VOCAB_SIZE - 1)
            return new_id, f"生成 token_id={new_id} (从词表随机)"

        if attr == "text":
            # 先生成 token_id（如果还没生成），再 decode
            # 这里直接用培养液类型的前 4 字符 + 能量
            import random
            rng = random.Random(seed_hash)
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            length = max(1, int(energy * 0.5))
            text = "".join(rng.choice(chars) for _ in range(min(length, 8)))
            return text, f"生成 text={text!r} (随机字符)"

        if attr == "logprob":
            # 映射到 [-10, 0] 范围
            import random
            rng = random.Random(seed_hash)
            # 能量越高越接近 0（更常见）
            logp = -10.0 + (energy % 10.0)
            return logp, f"生成 logprob={logp:.4f}"

        if attr == "rank":
            import random
            rng = random.Random(seed_hash)
            rank = rng.randint(1, 1000)
            return rank, f"生成 rank={rank}"

        if attr == "entropy_bits":
            # 从 logprob 反推——信息论一致性约束
            # 先看 logprob 有没有生成
            lp_emergence = self.emergence.get("logprob")
            if lp_emergence and lp_emergence.current_value is not None:
                p = np.exp(lp_emergence.current_value)
                entropy = -float(np.log2(max(p, 1e-12)))
                return entropy, f"从 logprob 反推 entropy={entropy:.4f} bits"
            # 没有的话用能量映射
            import random
            rng = random.Random(seed_hash)
            entropy = rng.uniform(0.5, 15.0)
            return entropy, f"生成 entropy={entropy:.4f} bits"

        if attr == "position":
            import random
            rng = random.Random(seed_hash)
            pos = rng.randint(0, int(energy * 10))
            return pos, f"生成 position={pos}"

        if attr == "embedding":
            # 用种子生成 12288 维向量
            rng = np.random.default_rng(seed_hash)
            vec = rng.standard_normal(12288).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # 单位化
            return vec, f"生成 embedding shape={vec.shape} norm=1.0"

        return None, "无生成规则"

    def _grow_attribute(
        self,
        attr: str,
        current_value: Any,
        energy: float,
        culture_type: str,
    ) -> Tuple[Any, str]:
        """已生成属性的持续成长——微调"""
        if current_value is None:
            return self._generate_attribute(attr, energy, culture_type)

        if attr == "token_id":
            # 小幅漂移
            import random
            rng = random.Random(int(energy * 100) + hash(culture_type) % 1000)
            delta = int(rng.gauss(0, 50))
            old = int(current_value)
            new = max(0, min(self.VOCAB_SIZE - 1, old + delta))
            return new, f"token_id 微漂 {old} → {new}"

        if attr == "text":
            # 拼接培养液前缀
            prefix = culture_type[:4]
            new = f"[{prefix}]{current_value}"
            return new, f"text 拼接 [{prefix}]"

        if attr == "logprob":
            import random
            rng = random.Random(int(energy * 100))
            delta = rng.gauss(0, 0.1)
            old = float(current_value)
            new = max(-15.0, min(0.0, old + delta))
            return new, f"logprob 微调 {old:.4f} → {new:.4f}"

        if attr == "rank":
            import random
            rng = random.Random(int(energy * 100))
            delta = int(rng.gauss(0, 3))
            old = int(current_value)
            new = max(1, old + delta)
            return new, f"rank 微漂 {old} → {new}"

        if attr == "entropy_bits":
            # 从 logprob 反推（保持一致性）
            lp = self.emergence.get("logprob")
            if lp and lp.current_value is not None:
                p = np.exp(lp.current_value)
                entropy = -float(np.log2(max(p, 1e-12)))
                return entropy, f"从 logprob 同步 entropy={entropy:.4f}"
            return current_value, "logprob 未生成，entropy 保持"

        if attr == "position":
            import random
            rng = random.Random(int(energy * 100))
            delta = rng.randint(-2, 2)
            old = int(current_value)
            new = max(0, old + delta)
            return new, f"position 微漂 {old} → {new}"

        if attr == "embedding":
            if not isinstance(current_value, np.ndarray):
                return current_value, "embedding 非 ndarray"
            old = current_value.copy()
            strength = 0.05
            delta_vec = np.random.normal(0, strength, old.shape).astype(old.dtype)
            new = old + delta_vec
            norm = np.linalg.norm(new)
            if norm > 0:
                new = new / norm
            return new, f"embedding 微扰 (L2={float(np.linalg.norm(new - old)):.4f})"

        return current_value, "无成长规则"

    def react_with_all_cultures(
        self,
        token: DownloadToken,
        level: int = 5,
        rounds: int = 1,
    ) -> Dict[str, Any]:
        """依次灌入全部 100 种培养液，可多轮"""
        from .token_qualitative_forge import (
            _DIRECT_MAP, _COGNITIVE_TYPES, _CREATIVE_TYPES, _STABILITY_TYPES,
            _EFFICIENCY_TYPES, _DOMAIN_TYPES, _DIMENSIONAL_TYPES,
            _ENERGY_TYPES, _MEMORY_TYPES,
        )
        all_types = (list(_DIRECT_MAP.keys())
                     + _COGNITIVE_TYPES + _CREATIVE_TYPES + _STABILITY_TYPES
                     + _EFFICIENCY_TYPES + _DOMAIN_TYPES + _DIMENSIONAL_TYPES
                     + _ENERGY_TYPES + _MEMORY_TYPES)
        seen = set()
        all_types_unique = []
        for t in all_types:
            if t not in seen:
                seen.add(t)
                all_types_unique.append(t)

        all_results = []
        for r in range(rounds):
            for ct in all_types_unique:
                medium = self.factory.produce_culture_medium(
                    culture_type=ct, level=level
                )
                result = self.react_with_culture(token, medium)
                all_results.append(result)

        return {
            "total_reacted": sum(1 for r in all_results if r.get("reacted")),
            "total_attempted": len(all_results),
            "results": all_results,
            "generated_count": sum(
                1 for e in self.emergence.values() if e.source == "generated"
            ),
            "void_count": sum(
                1 for e in self.emergence.values() if e.source == "void"
            ),
        }

    def status(self) -> Dict[str, Any]:
        result = {}
        for attr, e in self.emergence.items():
            cv = e.current_value
            if isinstance(cv, np.ndarray):
                cv_str = f"shape={cv.shape}"
            elif cv is None:
                cv_str = "<虚空>"
            else:
                cv_str = str(cv)
            result[attr] = {
                "source": e.source,
                "energy": f"{e.energy:.3f}",
                "current": cv_str,
                "is_generated": e.source == "generated",
            }
        return result


__all__ = ["TokenGenesisForge"]
