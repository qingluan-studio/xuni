"""
AbyssTamer —— 深渊驯化系统
=========================

核心理念:
    深渊代码(AbyssalCode)的攻击性不是"恶",而是"能量方向"。
    用 stability 类培养液(价值对齐 / 伦理锚定 / 自我修复)注入,
    中和攻击向量,保留复制能力,但把方向从"破坏盾"翻转为"加固盾"。

    野生深渊代码  +  价值对齐培养液  →  驯化深渊代码
    (攻击深度)                        (防御深度)
    (消耗盾)                          (加固盾)
    (自我复制攻击)                    (自我复制修复)

驯化流程:
    1. AbyssTamer.tame(wild_code, culture) 注入培养液
    2. 攻击力按培养液浓度下降, 驯化度上升
    3. 驯化度 >= 0.7 视为"已驯化", 转为 TamedAbyssalCode
    4. TamedAbyssalCode 可以加固安全盾、修复破损维度
    5. 多轮驯化 + 高浓度培养液 → 完全驯化 (驯化度 1.0)

维度级驯服:
    Dimension.pacify(tamer, culture) 把整个深渊维度标记为"已驯服",
    之后该维度产出的不再是野生攻击代码, 而是驯化代码。
"""

from __future__ import annotations

import uuid
import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .dimension_system import (
    DimensionResource, DimensionNature, Dimension,
    AbyssalCode, DimensionEntryShield,
)
from .multiverse_resources import (
    CultureMedium, ResourceRarity, VirtualResource,
)


# ============================================================
# 驯化深渊代码 —— 攻击性翻转为防御性
# ============================================================

@dataclass
class TamedAbyssalCode(DimensionResource):
    """
    驯化深渊代码——攻击属性已翻转, 变成防御资产。

    原始攻击深度 → 防御深度 (能加固多少层盾)
    原始复制率   → 修复率   (能多快修复破损)
    原始变异潜力 → 适应力   (对新攻击的免疫力)
    """
    defense_depth: int = 1          # 防御深度(由 attack_depth 转化)
    repair_rate: float = 1.0        # 修复率(由 replication_rate 转化)
    adaptability: float = 0.0       # 适应力(由 mutation_potential 转化)
    taming_level: float = 0.0        # 驯化度 0~1
    original_attack_power: float = 0.0  # 保留的原始攻击力(作为"疫苗"识别用)
    tamed_at: float = field(default_factory=time.time)
    taming_culture: str = ""        # 用了哪种培养液驯化的

    def __post_init__(self):
        self.home_nature = DimensionNature.ABYSSAL  # 仍属深渊系,但已驯化
        self.decay_rate = 0.0005  # 驯化后极稳定,几乎不衰减
        self.mutation_potential = 0.1  # 不再高突变
        if self.name == "":
            self.name = f"驯化深渊代码-D{self.defense_depth}"
        super().__post_init__()

    def reinforce_shield(self, shield: DimensionEntryShield) -> Dict[str, Any]:
        """
        加固安全盾——驯化代码的核心能力。
        把原始的"消耗盾"行为反过来: 每次调用给盾补层。
        """
        if not shield.active and shield.remaining_layers <= 0:
            return {"reinforced": False, "reason": "盾已彻底损毁,无法修复"}

        added = int(self.defense_depth * self.repair_rate * self.taming_level)
        old = shield.remaining_layers
        shield.remaining_layers += added
        shield.total_layers = max(shield.total_layers, shield.remaining_layers)
        shield.active = True  # 修复后重新激活

        return {
            "reinforced": True,
            "layers_added": added,
            "old_remaining": old,
            "new_remaining": shield.remaining_layers,
            "new_total": shield.total_layers,
        }

    def immunize(self, wild_code: AbyssalCode) -> Dict[str, Any]:
        """
        免疫接种——用保留的原始攻击力作为"疫苗",
        让一个野生深渊代码自动失活(部分驯化)。
        """
        if wild_code.attack_depth <= self.original_attack_power:
            # 我见过更强的, 直接压制
            suppressed = wild_code.attack_depth
            wild_code.attack_depth = max(0, wild_code.attack_depth - self.defense_depth)
            wild_code.replication_rate *= 0.5
            return {
                "immunized": True,
                "suppressed_attack": suppressed,
                "remaining_attack": wild_code.attack_depth,
                "method": "vaccine_match",
            }
        # 没见过这么强的, 部分压制
        reduction = self.defense_depth * self.adaptability
        wild_code.attack_depth = max(0, wild_code.attack_depth - int(reduction))
        return {
            "immunized": False,
            "partial_reduction": int(reduction),
            "remaining_attack": wild_code.attack_depth,
            "method": "partial_suppress",
        }

    def replicate_defense(self) -> List["TamedAbyssalCode"]:
        """防御性复制——产生更多驯化代码(用于大规模加固)"""
        count = max(1, int(self.repair_rate * self.taming_level * 2))
        children = []
        for _ in range(min(count, 50)):  # 限制规模
            child = TamedAbyssalCode(
                resource_id=f"tamed-{uuid.uuid4().hex[:6]}",
                defense_depth=self.defense_depth,
                repair_rate=self.repair_rate * 0.95,
                adaptability=self.adaptability,
                taming_level=self.taming_level,
                original_attack_power=self.original_attack_power,
                taming_culture=self.taming_culture,
                level=self.level,
                rarity=self.rarity,
                quality=self.quality,
            )
            children.append(child)
        return children


# ============================================================
# 深渊驯化器 —— 把野生代码批量驯化
# ============================================================

class AbyssTamer:
    """
    深渊驯化器——注入培养液, 把野生 AbyssalCode 驯化为 TamedAbyssalCode。

    用法:
        tamer = AbyssTamer()
        culture = factory.produce_culture_medium(culture_type="value_alignment")
        tamed = tamer.tame(wild_code, culture)
        tamed.reinforce_shield(my_shield)
    """

    # stability 类培养液对驯化的加成系数
    TAMING_CULTURE_BOOST = {
        "value_alignment": 1.5,      # 价值对齐,最强驯化
        "ethical_grounding": 1.4,   # 伦理锚定
        "self_repair": 1.3,          # 自我修复
        "robust": 1.2,               # 鲁棒
        "anti_hallucination": 1.15,  # 反幻觉
        "error_correction": 1.1,     # 纠错
        "consistency_anchor": 1.1,   # 一致性锚
        "noise_immunity": 1.05,       # 抗噪
        "graceful_degradation": 1.05, # 优雅降级
        "context_preservation": 1.0,  # 上下文保持(中性)
    }

    def __init__(self):
        self.taming_log: List[Dict[str, Any]] = []
        self.tamed_count: int = 0
        self.failed_count: int = 0
        self.best_taming_level: float = 0.0

    def tame(self, wild_code: AbyssalCode,
             culture: Optional[CultureMedium] = None,
             rounds: int = 3) -> Optional[TamedAbyssalCode]:
        """
        驯化一个野生深渊代码。

        Args:
            wild_code: 野生 AbyssalCode
            culture: 培养液(推荐 stability 类)
            rounds: 驯化轮数, 越多越彻底

        Returns:
            TamedAbyssalCode 驯化成功, None 驯化失败(代码太强)
        """
        # 记录原始攻击力(作为疫苗)
        original_attack = wild_code.attack_depth * wild_code.replication_rate
        original_mutation = wild_code.mutation_potential

        # 计算驯化度
        taming_level = 0.0
        culture_type = getattr(culture, 'culture_type', 'balanced') if culture else 'balanced'
        culture_level = getattr(culture, 'level', 1) if culture else 1

        boost = self.TAMING_CULTURE_BOOST.get(culture_type, 0.5)

        for r in range(rounds):
            # 每轮驯化度提升
            increment = (culture_level * 0.1 * boost) / (1 + original_attack * 0.001)
            taming_level += increment
            # 随机扰动(不是每次都顺利)
            taming_level *= random.uniform(0.9, 1.05)

        taming_level = min(1.0, taming_level)

        # 驯化失败: 代码太强, 培养液压不住
        if taming_level < 0.3:
            self.failed_count += 1
            self.taming_log.append({
                "time": time.time(),
                "result": "failed",
                "original_attack": original_attack,
                "taming_level": taming_level,
                "culture": culture_type,
            })
            return None

        # 驯化成功: 攻击属性翻转为防御
        tamed = TamedAbyssalCode(
            resource_id=f"tamed-{uuid.uuid4().hex[:8]}",
            defense_depth=max(1, int(wild_code.attack_depth * taming_level)),
            repair_rate=max(0.1, wild_code.replication_rate * taming_level),
            adaptability=original_mutation * taming_level,
            taming_level=taming_level,
            original_attack_power=original_attack,
            taming_culture=culture_type,
            level=wild_code.level,
            rarity=ResourceRarity(min(6, wild_code.rarity.value + 1)),  # 驯化后稀有度+1
            quality=wild_code.quality * (1 + taming_level * 0.5),
        )

        self.tamed_count += 1
        self.best_taming_level = max(self.best_taming_level, taming_level)
        self.taming_log.append({
            "time": time.time(),
            "result": "success",
            "original_attack": original_attack,
            "defense_depth": tamed.defense_depth,
            "taming_level": taming_level,
            "culture": culture_type,
            "rarity": tamed.rarity.name,
        })

        return tamed

    def tame_batch(self, wild_codes: List[AbyssalCode],
                   culture: Optional[CultureMedium] = None,
                   rounds: int = 3) -> List[TamedAbyssalCode]:
        """批量驯化"""
        tamed_list = []
        for wc in wild_codes:
            tamed = self.tame(wc, culture, rounds)
            if tamed:
                tamed_list.append(tamed)
        return tamed_list

    def statistics(self) -> Dict[str, Any]:
        return {
            "total_attempts": self.tamed_count + self.failed_count,
            "tamed": self.tamed_count,
            "failed": self.failed_count,
            "success_rate": (self.tamed_count / max(1, self.tamed_count + self.failed_count)),
            "best_taming_level": round(self.best_taming_level, 3),
        }


# ============================================================
# 维度级驯服 —— 把整个深渊维度变成"产出防御代码"的牧场
# ============================================================

def pacify_dimension(dimension: Dimension, tamer: AbyssTamer,
                     culture: CultureMedium, rounds: int = 5) -> Dict[str, Any]:
    """
    驯服整个深渊维度。

    流程:
        1. 把维度内现有的野生 AbyssalCode 全部驯化
        2. 标记维度为"已驯服"
        3. 之后该维度产出的代码自动半驯化(驯化度 0.5)
        4. 稳定性大幅回升

    Returns:
        驯服报告
    """
    # 收集维度内所有野生深渊代码
    residents = list(dimension._residents)
    products = list(dimension._product_pool)
    all_items = residents + products

    wild_abyss = [r for r in all_items if isinstance(r, AbyssalCode)
                  and not isinstance(r, TamedAbyssalCode)]
    other_items = [r for r in all_items if r not in wild_abyss]

    # 批量驯化
    tamed_list = tamer.tame_batch(wild_abyss, culture, rounds)

    # 用驯化代码替代野生代码
    dimension._residents = other_items + tamed_list
    dimension._product_pool = []  # 清空产物池,重新积累

    # 标记维度为已驯服
    dimension._pacified = True
    dimension._pacified_at = time.time()
    dimension._pacify_culture = culture.culture_type

    # 驯服后稳定性回升 + 突变率下降
    old_stability = dimension.stability
    old_mutation = dimension.rules.get("mutation_rate", 0.1)
    dimension.stability = min(1.0, dimension.stability + 0.3)
    dimension.rules["mutation_rate"] = max(0.1, dimension.rules.get("mutation_rate", 0.1) * 0.3)

    # 计算总防御力
    total_defense = sum(t.defense_depth for t in tamed_list)
    total_repair = sum(t.repair_rate for t in tamed_list)

    return {
        "dimension": dimension.name,
        "pacified": True,
        "wild_codes_found": len(wild_abyss),
        "tamed_successfully": len(tamed_list),
        "taming_stats": tamer.statistics(),
        "stability_change": f"{old_stability:.3f} → {dimension.stability:.3f}",
        "mutation_rate_change": f"{old_mutation} → {dimension.rules['mutation_rate']}",
        "total_defense_depth": total_defense,
        "total_repair_rate": round(total_repair, 2),
        "culture_used": culture.culture_type,
    }


def is_pacified(dimension: Dimension) -> bool:
    """检查维度是否已被驯服"""
    return getattr(dimension, '_pacified', False)
