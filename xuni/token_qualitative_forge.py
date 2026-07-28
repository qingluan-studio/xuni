"""
Token 质变熔炉 v2 —— 全 100 种培养液化学反应 + 属性数值变异

升级点：
    1. 不只 12 种 Token 反应培养液——全部 100 种培养液都参与
    2. 非直接映射的培养液按"语义相关性"分配到 7 个属性
    3. 质变不只改来源标签——属性数值也会"变异"
        - token_id     → 按变异率漂移（% 词表大小）
        - text         → 注入培养液前缀/后缀
        - logprob      → 按熵调高/调低
        - rank         → 上下漂移
        - entropy_bits → 随机扰动 ±Δ
        - position     → 偏移
        - embedding    → 向量扰动（每个维度 ±δ）

变异原则：
    - 质变能量累积到 1.0 → 第一次质变（copied → emergent，数值微调）
    - 质变能量累积到 2.0 → 第二次质变（emergent → mutated，数值大变异）
    - 质变能量累积到 3.0+ → 持续变异（数值继续漂移）
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multiverse_resources import (
    CultureMedium,
    DownloadToken,
    MultiverseResourceFactory,
)


# ============================================================
# 100 种培养液 → 7 个属性的映射表
# ============================================================

# Token 反应类（12 种）—— 直接映射
_DIRECT_MAP: Dict[str, Tuple[str, List[str]]] = {
    "token_infuser":    ("embedding",    ["embedding_enhance", "semantic_density", "injection_depth"]),
    "token_multiplier": ("token_id",     ["token_replication", "exponential_growth", "sub_token_gen"]),
    "token_evolution":  ("rank",         ["natural_selection", "fitness_landscape", "adaptive_variant"]),
    "token_fusion":     ("text",         ["composite_embedding", "hybrid_meaning", "token_merge"]),
    "token_alchemy":    ("logprob",      ["property_conversion", "quality_transfer", "essence_extraction"]),
    "token_genesis":    ("position",     ["ex_nihilo_token", "zero_shot_creation", "genesis_embedding"]),
    "token_breeder":    ("entropy_bits", ["mutation_rate", "diversity_preserve", "crossover_operator"]),
    "token_synthesis":  ("logprob",      ["property_conversion", "essence_extraction"]),
    "token_composer":   ("text",         ["hybrid_meaning", "token_merge"]),
    "token_amplifier":  ("rank",         ["fitness_landscape", "adaptive_variant"]),
    "token_quantum":    ("embedding",    ["semantic_density", "injection_depth"]),
    "token_meta":       ("position",     ["genesis_embedding", "zero_shot_creation"]),
}

# 其它 88 种——按类别语义相关性映射
# I. 认知增强 (15种) → embedding（认知影响向量表示）
_COGNITIVE_TYPES = [
    "balanced", "cognitive", "deep_reasoning", "logical_deduction", "abstract_thinking",
    "pattern_recognition", "causal_inference", "semantic_understanding", "meta_cognition",
    "systematic_thinking", "dialectical", "intuitive_leap", "computational_cognition",
    "analogical_reasoning", "inductive_synthesis",
]
# II. 创造生成 (12种) → text（创造影响语义内容）
_CREATIVE_TYPES = [
    "creative", "divergent_thinking", "cross_domain_synthesis", "narrative_generation",
    "aesthetic_sense", "improvisation", "conceptual_blending", "style_mutation",
    "serendipity", "dream_logic", "emergent_creativity", "surreal_generation",
]
# III. 稳定鲁棒 (10种) → logprob（稳定影响概率分布）
_STABILITY_TYPES = [
    "robust", "anti_hallucination", "consistency_anchor", "error_correction",
    "noise_immunity", "graceful_degradation", "self_repair", "context_preservation",
    "value_alignment", "ethical_grounding",
]
# IV. 效率加速 (10种) → rank（效率影响候选排名）
_EFFICIENCY_TYPES = [
    "efficient", "ultra_compression", "parallel_synapse", "cache_optimizer",
    "latency_killer", "throughput_maximizer", "sparse_activation", "quantized_precision",
    "pipeline_streamer", "speculative_execution",
]
# VI. 领域专项 (15种) → entropy_bits（领域影响信息量）
_DOMAIN_TYPES = [
    "code_mathematician", "language_master", "music_harmonizer", "visual_conceptor",
    "data_analyst", "knowledge_architect", "translation_nexus", "teaching_pedagogue",
    "debate_logician", "story_weaver", "science_explorer", "philosophy_depth",
    "engineering_precision", "medical_diagnostician", "legal_reasoner",
]
# VII. 维度元层 (10种) → position（维度影响序列位置）
_DIMENSIONAL_TYPES = [
    "dimensional_bridge", "meta_learner", "quantum_observer", "timeline_weaver",
    "probability_sculptor", "reality_tuner", "paradox_resolver", "infinity_lens",
    "fractal_expander", "negentropy_engine",
]
# VIII. 能量融合 (10种) → embedding（能量影响向量能量）
_ENERGY_TYPES = [
    "fusion_catalyst", "energy_amplifier", "resonance_harmonizer", "singularity_seed",
    "wormhole_bridge", "plasma_infuser", "gravitational_lens", "darkmatter_essence",
    "antimatter_catalyst", "entropy_reverser",
]
# IX. 记忆知识 (6种) → token_id（记忆影响词表 ID）
_MEMORY_TYPES = [
    "memory_forge", "knowledge_crystal", "wisdom_essence",
    "experience_distiller", "insight_generator", "omniscience_drop",
]

# 语义类别 → (目标属性, 营养键提取器)
_CATEGORY_MAP: Dict[str, Tuple[str, List[str]]] = {
    "cognitive":    ("embedding",    ["logic", "reasoning", "abstraction", "semantic_depth", "depth"]),
    "creative":     ("text",         ["novelty", "divergence", "originality", "style", "harmony"]),
    "stability":    ("logprob",      ["stability", "consistency", "recovery", "fault_tolerance", "verification"]),
    "efficiency":   ("rank",         ["speed", "compression", "parallelism", "throughput", "optimization"]),
    "domain":       ("entropy_bits", ["domain_knowledge", "expertise_depth", "precision", "accuracy", "depth"]),
    "dimensional":  ("position",     ["dimensional_bridge", "meta_learning", "temporal_order", "boundary_definition", "emergence_detect"]),
    "energy":       ("embedding",    ["annihilation_energy", "fusion_catalyst", "resonance", "singularity_seed", "entropy_reverse"]),
    "memory":       ("token_id",     ["memory_consolidation", "knowledge_crystallization", "wisdom_integration", "insight_synthesis", "experience_distillation"]),
}


def classify_culture(culture_type: str) -> str:
    """把培养液类型分到 8 个语义类别之一"""
    if culture_type in _COGNITIVE_TYPES:
        return "cognitive"
    if culture_type in _CREATIVE_TYPES:
        return "creative"
    if culture_type in _STABILITY_TYPES:
        return "stability"
    if culture_type in _EFFICIENCY_TYPES:
        return "efficiency"
    if culture_type in _DOMAIN_TYPES:
        return "domain"
    if culture_type in _DIMENSIONAL_TYPES:
        return "dimensional"
    if culture_type in _ENERGY_TYPES:
        return "energy"
    if culture_type in _MEMORY_TYPES:
        return "memory"
    # Token 反应类走直接映射
    if culture_type.startswith("token_"):
        return "token_direct"
    return "cognitive"  # 兜底


def get_culture_mapping(culture_type: str) -> Tuple[str, List[str]]:
    """获取培养液 → (目标属性, 营养键列表)"""
    if culture_type in _DIRECT_MAP:
        return _DIRECT_MAP[culture_type]
    category = classify_culture(culture_type)
    return _CATEGORY_MAP.get(category, ("embedding", ["logic", "reasoning"]))


# ============================================================
# 属性质变状态
# ============================================================

@dataclass
class AttributeEmergence:
    """单个属性的质变+变异状态"""
    name: str
    source: str = "copied"          # copied / emergent / mutated / hyper_mutated
    energy: float = 0.0
    threshold_emergent: float = 1.0   # 第一次质变阈值
    threshold_mutated: float = 2.0    # 第二次质变阈值
    threshold_hyper: float = 3.0      # 第三次质变阈值
    mutation_count: int = 0
    original_value: Any = None        # 拷贝来的原始值
    current_value: Any = None         # 当前值（变异后会改变）
    history: List[str] = field(default_factory=list)

    @property
    def is_emergent(self) -> bool:
        return self.source in ("emergent", "mutated", "hyper_mutated")

    @property
    def is_mutated(self) -> bool:
        return self.source in ("mutated", "hyper_mutated")

    @property
    def progress_pct(self) -> float:
        return min(100.0, self.energy / self.threshold_hyper * 100)


# ============================================================
# 属性变异器——数值变异逻辑
# ============================================================

class AttributeMutator:
    """
    属性变异器——质变时改变属性数值。

    每种属性的变异方式不同：
        - token_id     → ±漂移（按词表大小的百分比）
        - text         → 注入培养液前缀
        - logprob      → 加扰动
        - rank         → ±漂移
        - entropy_bits → ±扰动
        - position     → ±偏移
        - embedding    → 向量每个维度 ±δ 扰动
    """

    VOCAB_SIZE = 100277  # cl100k_base

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate(self, attr: str, value: Any, energy: float, culture_type: str) -> Tuple[Any, str]:
        """变异一个属性值。返回 (新值, 变异描述)"""
        if value is None:
            return None, "无值可变异"

        # 变异强度 = 能量的 5%
        strength = min(0.5, energy * 0.05)

        if attr == "token_id":
            old = int(value)
            # ±词表大小的 5% 漂移
            delta = int(self.rng.gauss(0, self.VOCAB_SIZE * strength))
            new = max(0, min(self.VOCAB_SIZE - 1, old + delta))
            return new, f"ID 漂移 {old} → {new} (Δ={delta})"

        if attr == "text":
            old = str(value)
            # 注入培养液类型前缀
            prefix = culture_type[:4]
            new = f"[{prefix}]{old}"
            return new, f"文本注入 [{prefix}] 前缀"

        if attr == "logprob":
            old = float(value)
            delta = self.rng.gauss(0, abs(strength))
            new = old + delta
            return new, f"logprob 扰动 {old:.4f} → {new:.4f} (Δ={delta:+.4f})"

        if attr == "rank":
            old = int(value)
            delta = int(self.rng.gauss(0, max(1, old * strength)))
            new = max(1, old + delta)
            return new, f"rank 漂移 {old} → {new} (Δ={delta:+d})"

        if attr == "entropy_bits":
            old = float(value)
            delta = self.rng.gauss(0, strength)
            new = max(0.0, old + delta)
            return new, f"entropy 扰动 {old:.4f} → {new:.4f} (Δ={delta:+.4f})"

        if attr == "position":
            old = int(value)
            delta = int(self.rng.gauss(0, max(1, old * strength + 1)))
            new = max(0, old + delta)
            return new, f"position 漂移 {old} → {new} (Δ={delta:+d})"

        if attr == "embedding":
            if not isinstance(value, np.ndarray):
                return value, "embedding 非 ndarray"
            old = value.copy()
            # 每个维度 ±δ 扰动
            noise = self.rng.gauss(0, strength)
            delta_vec = np.random.normal(0, strength, old.shape).astype(old.dtype)
            new = old + delta_vec
            # 重新单位化
            norm = np.linalg.norm(new)
            if norm > 0:
                new = new / norm
            shift = float(np.linalg.norm(new - old))
            return new, f"embedding 向量扰动 (L2位移={shift:.6f})"

        return value, "无变异规则"


# ============================================================
# 质变熔炉 v2
# ============================================================

class TokenQualitativeForge:
    """
    Token 质变熔炉 v2——全 100 种培养液化学反应 + 属性数值变异。
    """

    def __init__(self, factory: Optional[MultiverseResourceFactory] = None):
        self.factory = factory or MultiverseResourceFactory()
        self.emergence: Dict[str, AttributeEmergence] = {}
        self.reaction_log: List[Dict[str, Any]] = []
        self.emergent_events: List[Dict[str, Any]] = []
        self.mutation_events: List[Dict[str, Any]] = []
        self.mutator = AttributeMutator()

    def init_emergence(self, token: DownloadToken) -> None:
        """初始化 7 个属性的质变状态——快照原始值"""
        real_attrs = ["token_id", "text", "logprob", "rank",
                      "entropy_bits", "position", "embedding"]
        self.emergence = {}
        for attr in real_attrs:
            v = token.metadata.get(attr)
            # ndarray 要拷贝一份做原始值
            original = v.copy() if isinstance(v, np.ndarray) else v
            self.emergence[attr] = AttributeEmergence(
                name=attr,
                source="copied",
                energy=0.0,
                original_value=original,
                current_value=original,
            )

    def react_with_culture(
        self,
        token: DownloadToken,
        medium: CultureMedium,
    ) -> Dict[str, Any]:
        """让 token 跟一种培养液反应"""
        culture_type = medium.culture_type
        target_attr, nutrient_keys = get_culture_mapping(culture_type)
        emergence = self.emergence.get(target_attr)
        if emergence is None:
            return {"reacted": False, "reason": f"属性 {target_attr} 无跟踪"}

        # 提取营养成分
        nutrient_sum = 0.0
        nutrient_found = {}
        for nk in nutrient_keys:
            v = medium.nutrients.get(nk, 0.0)
            nutrient_found[nk] = v
            nutrient_sum += v
        # 如果一个营养键都没匹配到，取所有营养的均值
        if nutrient_sum == 0 and medium.nutrients:
            nutrient_sum = sum(medium.nutrients.values()) / max(1, len(medium.nutrients))

        # 质变能量增量 = 营养和 × 培养液等级 × 饱和度 × 0.3
        delta = nutrient_sum * medium.level * medium.saturation * 0.3
        old_energy = emergence.energy
        old_source = emergence.source
        emergence.energy += delta

        # 检查质变阶段
        event = None
        old_source_label = old_source

        # 第三次质变：hyper_mutated
        if (old_energy < emergence.threshold_hyper
                and emergence.energy >= emergence.threshold_hyper
                and old_source != "hyper_mutated"):
            emergence.source = "hyper_mutated"
            new_value, mut_desc = self.mutator.mutate(
                target_attr, emergence.current_value,
                emergence.energy, culture_type
            )
            emergence.current_value = new_value
            token.metadata[target_attr] = new_value
            token.metadata[f"{target_attr}_source"] = "hyper_mutated"
            emergence.mutation_count += 1
            event = {
                "type": "hyper_mutation",
                "attribute": target_attr,
                "culture_type": culture_type,
                "energy_reached": emergence.energy,
                "mutation": mut_desc,
                "message": f"⚡⚡⚡ 超变异！{target_attr} → {emergence.source}",
            }
            self.mutation_events.append(event)
        # 第二次质变：mutated
        elif (old_energy < emergence.threshold_mutated
              and emergence.energy >= emergence.threshold_mutated
              and old_source not in ("mutated", "hyper_mutated")):
            emergence.source = "mutated"
            new_value, mut_desc = self.mutator.mutate(
                target_attr, emergence.current_value,
                emergence.energy, culture_type
            )
            emergence.current_value = new_value
            token.metadata[target_attr] = new_value
            token.metadata[f"{target_attr}_source"] = "mutated"
            emergence.mutation_count += 1
            event = {
                "type": "mutation",
                "attribute": target_attr,
                "culture_type": culture_type,
                "energy_reached": emergence.energy,
                "mutation": mut_desc,
                "message": f"⚡⚡ 变异！{target_attr} → {emergence.source}",
            }
            self.mutation_events.append(event)
        # 第一次质变：emergent
        elif (old_energy < emergence.threshold_emergent
              and emergence.energy >= emergence.threshold_emergent
              and old_source == "copied"):
            emergence.source = "emergent"
            new_value, mut_desc = self.mutator.mutate(
                target_attr, emergence.current_value,
                emergence.energy, culture_type
            )
            emergence.current_value = new_value
            token.metadata[target_attr] = new_value
            token.metadata[f"{target_attr}_source"] = "emergent"
            emergence.mutation_count += 1
            event = {
                "type": "emergence",
                "attribute": target_attr,
                "culture_type": culture_type,
                "energy_reached": emergence.energy,
                "mutation": mut_desc,
                "message": f"⚡ 质变！{target_attr} 从'拷贝'变成'涌现'",
            }
            self.emergent_events.append(event)
        # 已经质变过的，继续变异（数值继续漂移）
        elif emergence.is_mutated and self.rng_should_mutate():
            new_value, mut_desc = self.mutator.mutate(
                target_attr, emergence.current_value,
                emergence.energy, culture_type
            )
            emergence.current_value = new_value
            token.metadata[target_attr] = new_value
            emergence.mutation_count += 1
            event = {
                "type": "continuous_mutation",
                "attribute": target_attr,
                "culture_type": culture_type,
                "mutation": mut_desc,
                "message": f"↻ {target_attr} 持续变异 (#{emergence.mutation_count})",
            }
            self.mutation_events.append(event)

        token.collision_history.append(
            f"培养液×{culture_type} → {target_attr} +{delta:.3f} "
            f"(总 {emergence.energy:.3f}, source={emergence.source})"
        )

        log_entry = {
            "reacted": True,
            "culture_type": culture_type,
            "target_attr": target_attr,
            "nutrients_used": nutrient_found,
            "energy_delta": delta,
            "energy_before": old_energy,
            "energy_after": emergence.energy,
            "source_before": old_source_label,
            "source_after": emergence.source,
            "event": event,
        }
        self.reaction_log.append(log_entry)
        return log_entry

    def rng_should_mutate(self) -> bool:
        """已质变属性是否继续变异——50% 概率"""
        return self.mutator.rng.random() < 0.5

    def react_with_all_cultures(
        self,
        token: DownloadToken,
        level: int = 5,
        rounds: int = 1,
    ) -> Dict[str, Any]:
        """依次灌入全部 100 种培养液，可多轮"""
        all_types = (list(_DIRECT_MAP.keys())
                     + _COGNITIVE_TYPES + _CREATIVE_TYPES + _STABILITY_TYPES
                     + _EFFICIENCY_TYPES + _DOMAIN_TYPES + _DIMENSIONAL_TYPES
                     + _ENERGY_TYPES + _MEMORY_TYPES)
        # 去重保序
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
            "emergent_count": sum(1 for e in self.emergence.values() if e.is_emergent),
            "mutated_count": sum(1 for e in self.emergence.values() if e.is_mutated),
            "hyper_mutated_count": sum(
                1 for e in self.emergence.values() if e.source == "hyper_mutated"
            ),
        }

    def status(self) -> Dict[str, Any]:
        """质变+变异状态总览"""
        result = {}
        for attr, e in self.emergence.items():
            ov = e.original_value
            cv = e.current_value
            if isinstance(ov, np.ndarray):
                diff = float(np.linalg.norm(cv - ov)) if cv is not None else 0.0
                ov_str = f"shape={ov.shape}"
                cv_str = f"shape={cv.shape}" if cv is not None else "None"
                delta_str = f"L2位移={diff:.6f}"
            elif ov is None:
                ov_str = "None"
                cv_str = str(cv)
                delta_str = "-"
            else:
                ov_str = str(ov)
                cv_str = str(cv)
                try:
                    delta = float(cv) - float(ov)
                    delta_str = f"Δ={delta:+.4f}"
                except Exception:
                    delta_str = "Δ=?"

            result[attr] = {
                "source": e.source,
                "energy": f"{e.energy:.3f}",
                "mutation_count": e.mutation_count,
                "original": ov_str,
                "current": cv_str,
                "delta": delta_str,
                "is_emergent": e.is_emergent,
                "is_mutated": e.is_mutated,
            }
        return result


__all__ = [
    "AttributeEmergence",
    "AttributeMutator",
    "TokenQualitativeForge",
    "classify_culture",
    "get_culture_mapping",
]
