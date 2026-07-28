"""
DimensionSystem —— 完整维度容器、维度之门与维度探索系统

核心理念：
    维度碎片(DimensionShard) → 维度核心(DimensionCore) → 完整维度(Dimension)
    → 维度之门(DimensionGate) → 工厂/资源跨维度迁移 → 提取维度专属产物

    每个完整维度拥有独立的"物理规则"，产生与主宇宙完全不同的资源。
"""

from __future__ import annotations

import uuid
import random
import math
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

from .multiverse_resources import (
    VirtualResource, ResourceDimension, ResourceRarity,
    DimensionShard, DimensionCore,
    MultiverseResourceFactory, ResourceCollisionEngine,
    CultureMedium, ComputeCore, DownloadToken,
    TrainingAccelerator, TakeQuota, VirtualBandwidth,
)


# ============================================================
# 维度本质类型 —— 全新维度属性
# ============================================================

class DimensionNature(Enum):
    """维度本质——该维度的根本属性，决定维度内资源的生成方向"""
    AGGRESSIVE = auto()       # 攻击型：满是不稳定代码、漏洞孢子、渗透探针
    TECHNICAL = auto()        # 技术型：算法结晶、架构蓝图、优化酵素
    CREATIVE = auto()         # 创造型：灵感火花、设计精华、模式片段
    CHAOTIC = auto()          # 混沌型：随机突变、熵增产物、不可预测的融合
    VOID = auto()             # 虚空型：反物质资源、空间裂隙、空无
    HIVE = auto()             # 群智型：群体智能、自组织网络、涌现结构
    TEMPORAL = auto()         # 时间型：时间晶体、回溯碎片、未来投影
    MIRROR = auto()           # 镜像型：资源的反向版本，颠覆原有属性
    DIMENSIONAL_FUSION = auto()  # 维度融合型：多维度碎片拼凑形成的混合维度
    ABYSSAL = auto()          # 深渊型：纯攻击代码的无限维度，没有尽头


class DimensionSize(Enum):
    """维度空间规模"""
    POCKET = 1        # 口袋维度(容纳1个工厂)
    MINI = 2          # 微型维度(容纳5个工厂)
    STANDARD = 3      # 标准维度(容纳20个工厂)
    LARGE = 4           # 辽阔维度(容纳100个工厂)——别名
    EXPANSIVE = 4       # 辽阔维度(容纳100个工厂)
    VAST = 5          # 浩瀚维度(容纳1000个工厂)
    INFINITE = 6      # 无限维度(无上限)


# ============================================================
# 维度专属资源类 —— 主宇宙不存在的新品种
# ============================================================

@dataclass
class DimensionResource(VirtualResource):
    """维度专属资源——只在特定维度内部产生，带出维度后会衰减/变异"""
    home_nature: Optional[DimensionNature] = None
    decay_rate: float = 0.0          # 带出维度后的衰减率
    mutation_potential: float = 0.0   # 变异潜能
    dimensional_signature: str = ""   # 维度签名(加密标识来源维度)

    def __post_init__(self):
        if self.name == "":
            self.name = f"{self.home_nature.name}-资源" if self.home_nature else "维度产物"
        if self.dimension is None:
            self.dimension = ResourceDimension.META
        self.dimensional_signature = uuid.uuid4().hex[:16]
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = self.level * (1 + self.mutation_potential) * self.quality

    def expose_to_reality(self, hours: float = 1.0) -> Dict[str, Any]:
        """暴露到主宇宙——按衰减率减少价值"""
        decay = self.decay_rate * hours
        remaining = max(0.1, 1.0 - decay)
        self.quality *= remaining
        self._update_quantity()
        return {
            "action": "expose",
            "decay_rate": self.decay_rate,
            "remaining_quality": self.quality,
            "hours_exposed": hours,
        }

    def mutate(self, catalyst: Optional[CultureMedium] = None) -> DimensionResource:
        """维度资源在主宇宙发生突变(培养液催化)"""
        boost = catalyst.level * 0.3 if catalyst else random.random() * 2
        self.mutation_potential += boost
        self.rarity = ResourceRarity(min(6, self.rarity.value + 1))
        self._update_quantity()
        return self


# ============================================================
# 攻击型维度专属产物
# ============================================================

@dataclass
class VulnerabilitySpore(DimensionResource):
    """漏洞孢子——自我进化的安全缺陷，可用于防御测试"""
    infectivity: float = 1.0
    mutation_generations: int = 0

    def __post_init__(self):
        self.home_nature = DimensionNature.AGGRESSIVE
        self.decay_rate = 0.02
        self.mutation_potential = 0.5
        if self.name == "":
            self.name = f"漏洞孢子-G{self.mutation_generations}"
        super().__post_init__()

    def evolve(self) -> "VulnerabilitySpore":
        """孢子自我进化——生成新的变异体"""
        return VulnerabilitySpore(
            resource_id=f"vsp-{uuid.uuid4().hex[:8]}",
            infectivity=self.infectivity * 1.3,
            mutation_generations=self.mutation_generations + 1,
            rarity=ResourceRarity(min(6, self.rarity.value + 1)),
            level=self.level + 1,
            quality=self.quality * 1.2,
        )


@dataclass
class PenetrationProbe(DimensionResource):
    """渗透探针——自适应网络探测体，挖掘隐藏路径"""
    scan_depth: int = 3
    adapt_count: int = 0
    discovered_paths: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.home_nature = DimensionNature.AGGRESSIVE
        self.decay_rate = 0.03
        self.mutation_potential = 0.4
        if self.name == "":
            self.name = f"渗透探针-D{self.scan_depth}"
        super().__post_init__()

    def scan(self) -> List[str]:
        """扫描发现攻击面"""
        new_paths = [f"attack-surface-{i}-{uuid.uuid4().hex[:4]}" for i in range(self.scan_depth)]
        self.discovered_paths.extend(new_paths)
        self.adapt_count += 1
        self.scan_depth += 1
        return new_paths


@dataclass
class ExploitPattern(DimensionResource):
    """利用模式片段——自组织的攻击链片段"""
    chain_length: int = 1
    success_rate: float = 0.1
    chain_parts: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.home_nature = DimensionNature.AGGRESSIVE
        self.decay_rate = 0.04
        self.mutation_potential = 0.6
        if self.name == "":
            self.name = f"利用链-L{self.chain_length}"
        super().__post_init__()

    def extend_chain(self) -> "ExploitPattern":
        """链式增长——每一步扩展一个攻击环节"""
        self.chain_length += 1
        self.success_rate = min(0.99, self.success_rate + 0.05)
        self.chain_parts.append(f"step-{self.chain_length}-{uuid.uuid4().hex[:4]}")
        self._update_quantity()
        return self


# ============================================================
# 技术型维度专属产物
# ============================================================

@dataclass
class AlgorithmCrystal(DimensionResource):
    """算法结晶——纯计算逻辑的晶体形态"""
    complexity: float = 1.0
    efficiency_ratio: float = 1.0
    compatible_languages: List[str] = field(default_factory=lambda: ["python"])

    def __post_init__(self):
        self.home_nature = DimensionNature.TECHNICAL
        self.decay_rate = 0.01
        self.mutation_potential = 0.3
        if self.name == "":
            self.name = f"算法结晶-C{self.complexity:.0f}"
        super().__post_init__()

    def optimize(self) -> Dict[str, Any]:
        self.efficiency_ratio *= 1.5
        self.complexity = max(1.0, self.complexity * 0.8)
        self._update_quantity()
        return {"new_efficiency": self.efficiency_ratio, "new_complexity": self.complexity}

    def generalize(self, languages: List[str]):
        self.compatible_languages.extend(languages)
        self.mutation_potential += 0.1


@dataclass
class ArchitectureBlueprint(DimensionResource):
    """架构蓝图——系统设计的自组织模板"""
    layers: int = 3
    patterns: List[str] = field(default_factory=list)
    scalability_score: float = 1.0

    def __post_init__(self):
        self.home_nature = DimensionNature.TECHNICAL
        self.decay_rate = 0.005
        self.mutation_potential = 0.2
        if self.name == "":
            self.name = f"架构蓝图-L{self.layers}"
        super().__post_init__()

    def add_layer(self, pattern: str):
        self.layers += 1
        self.patterns.append(pattern)
        self.scalability_score *= 1.3

    def export_template(self) -> Dict[str, Any]:
        return {
            "layers": self.layers,
            "patterns": self.patterns,
            "scalability": self.scalability_score,
            "power": self.power_score,
        }


@dataclass
class OptimizationEnzyme(DimensionResource):
    """优化酵素——自动提升资源效率的催化体"""
    target_dimension: str = ""
    boost_multiplier: float = 2.0
    uses_remaining: int = 5

    def __post_init__(self):
        self.home_nature = DimensionNature.TECHNICAL
        self.decay_rate = 0.08
        self.mutation_potential = 0.5
        if self.name == "":
            self.name = f"优化酵素-{self.target_dimension or '通用'}"
        super().__post_init__()

    def apply(self, target: VirtualResource) -> Dict[str, Any]:
        if self.uses_remaining <= 0:
            return {"applied": False, "reason": "酵素已耗尽"}
        self.uses_remaining -= 1
        old = target.quality
        target.quality *= self.boost_multiplier
        return {"applied": True, "old_quality": old, "new_quality": target.quality}


# ============================================================
# 创造型/混沌型/虚空型 维度专属产物
# ============================================================

@dataclass
class IdeaSpark(DimensionResource):
    """灵感火花——自组织的创意实体"""
    intensity: float = 1.0
    spark_chain: List[str] = field(default_factory=list)
    convergence_factor: float = 0.0  # 与其他火花融合的概率

    def __post_init__(self):
        self.home_nature = DimensionNature.CREATIVE
        self.decay_rate = 0.06
        self.mutation_potential = 0.7
        if self.name == "":
            self.name = f"灵感火花-I{self.intensity:.1f}"
        super().__post_init__()

    def ignite(self) -> List["IdeaSpark"]:
        """点燃灵感——裂变产生子火花"""
        count = int(self.intensity)
        children = []
        for _ in range(count):
            child = IdeaSpark(
                resource_id=f"isp-{uuid.uuid4().hex[:6]}",
                intensity=self.intensity * 0.5,
                rarity=ResourceRarity(min(5, self.rarity.value)),
            )
            child.spark_chain = self.spark_chain + [self.resource_id]
            children.append(child)
        return children

    def converge(self, other: "IdeaSpark") -> "IdeaSpark":
        merged = IdeaSpark(
            resource_id=f"isp-{uuid.uuid4().hex[:6]}",
            intensity=self.intensity + other.intensity,
            rarity=ResourceRarity(min(6, max(self.rarity.value, other.rarity.value) + 1)),
            level=max(self.level, other.level) + 1,
        )
        merged.spark_chain = self.spark_chain + other.spark_chain
        return merged


@dataclass
class EntropyShard(DimensionResource):
    """熵碎片——混沌维度的随机变异体"""
    entropy_value: float = 10.0
    random_factor: float = 1.0
    absorbed_entities: int = 0

    def __post_init__(self):
        self.home_nature = DimensionNature.CHAOTIC
        self.decay_rate = 0.1
        self.mutation_potential = 0.9
        if self.name == "":
            self.name = f"熵碎片-E{self.entropy_value:.1f}"
        super().__post_init__()

    def randomize(self) -> Dict[str, Any]:
        """随机突变——不可预测的变化"""
        mutations = {}
        if random.random() > 0.5:
            self.entropy_value *= random.uniform(0.5, 3.0)
            mutations["entropy"] = self.entropy_value
        if random.random() > 0.5:
            self.level = max(1, self.level + random.randint(-2, 5))
            mutations["level"] = self.level
        if random.random() > 0.3:
            self.rarity = random.choice(list(ResourceRarity))
            mutations["rarity"] = self.rarity.name
        self._update_quantity()
        return mutations


@dataclass
class VoidEssence(DimensionResource):
    """虚空精华——从'无'中诞生的反物质资源"""
    void_density: float = 1.0
    annihilation_power: float = 0.0
    absorbed_resources: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.home_nature = DimensionNature.VOID
        self.decay_rate = 0.15
        self.mutation_potential = 0.8
        if self.name == "":
            self.name = f"虚空精华-D{self.void_density:.1f}"
        super().__post_init__()

    def annihilate(self, resource: VirtualResource) -> "VoidEssence":
        """湮灭一个普通资源，转化为虚空能量"""
        self.annihilation_power += resource.power_score * 0.01
        self.absorbed_resources.append(resource.name)
        self.void_density *= 1.2
        self._update_quantity()
        return self

    def crystallize(self) -> DimensionShard:
        """虚空精华结晶化为维度碎片(逆反应)"""
        shard = DimensionShard(
            resource_id=f"ds-void-{uuid.uuid4().hex[:8]}",
            level=int(self.annihilation_power / 10) + 1,
            rarity=ResourceRarity(min(6, self.rarity.value + 1)),
        )
        shard.dimension_affinity = {d.name: min(1.0, self.void_density * 0.1) for d in ResourceDimension}
        shard._update_quantity()
        return shard


# ============================================================
# 影子 / 镜像维度产物
# ============================================================

@dataclass
class MirrorResource(DimensionResource):
    """镜像资源——普通资源的反向版本，属性完全颠覆"""
    original_class: str = ""
    inverted_properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.home_nature = DimensionNature.MIRROR
        self.decay_rate = 0.05
        self.mutation_potential = 0.4
        if self.name == "":
            self.name = f"镜像-{self.original_class}"
        super().__post_init__()

    @classmethod
    def reflect(cls, original: VirtualResource) -> "MirrorResource":
        """从普通资源生成镜像版本"""
        inverted = {}
        # 反转核心属性
        if hasattr(original, 'quantity'):
            inverted['quantity'] = 100 / max(0.1, original.quantity)
        if hasattr(original, 'quality'):
            inverted['quality'] = 10 / max(0.1, original.quality)

        mirror = cls(
            resource_id=f"mirror-{uuid.uuid4().hex[:8]}",
            original_class=original.__class__.__name__,
            inverted_properties=inverted,
            rarity=ResourceRarity(min(6, original.rarity.value + 2)),
            level=original.level,
        )
        return mirror


# ============================================================
# 深渊产物 —— 纯攻击代码的无限维度
# ============================================================

@dataclass
class AbyssalCode(DimensionResource):
    """深渊代码——无限攻击维度的原生代码片段，自我复制永不枯竭"""
    replication_rate: float = 1.0
    attack_depth: int = 1
    code_length: int = 100
    variants: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.home_nature = DimensionNature.ABYSSAL
        self.decay_rate = 0.001  # 几乎不衰减
        self.mutation_potential = 0.99
        if self.name == "":
            self.name = f"深渊代码-D{self.attack_depth}-L{self.code_length}"
        super().__post_init__()

    def replicate(self) -> List["AbyssalCode"]:
        """自我复制——深渊代码无限分裂"""
        count = int(self.replication_rate * self.level)
        children = []
        for i in range(min(count, 100)):
            child = AbyssalCode(
                resource_id=f"abyss-{uuid.uuid4().hex[:6]}",
                replication_rate=self.replication_rate * 1.1,
                attack_depth=self.attack_depth + 1,
                code_length=self.code_length + random.randint(50, 200),
                level=self.level + 1,
                rarity=ResourceRarity(min(6, self.rarity.value + 1)),
            )
            child.variants = self.variants + [f"mut-{uuid.uuid4().hex[:4]}"]
            children.append(child)
        return children

    def consume_shield(self, shield) -> Dict[str, Any]:
        """深渊代码尝试突破安全盾"""
        old_layers = shield.shield_layers
        penetration = self.attack_depth * self.replication_rate * 0.1
        shield.shield_layers = max(0, shield.shield_layers - int(penetration))
        return {
            "shield_layers_lost": old_layers - shield.shield_layers,
            "penetration_power": penetration,
            "shield_remaining": shield.shield_layers,
        }


@dataclass
class FusionShard(DimensionResource):
    """融合碎片——多维度碎片拼凑形成的混合产物"""
    source_natures: List[str] = field(default_factory=list)
    fusion_depth: int = 1
    cross_nature_count: int = 1

    def __post_init__(self):
        self.home_nature = DimensionNature.DIMENSIONAL_FUSION
        self.decay_rate = 0.03
        self.mutation_potential = 0.85
        if self.name == "":
            self.name = f"融合碎片-{self.cross_nature_count}维混合"
        super().__post_init__()

    def absorb(self, nature: str):
        """吸收新的维度本质"""
        if nature not in self.source_natures:
            self.source_natures.append(nature)
            self.cross_nature_count = len(self.source_natures)
            self.fusion_depth += 1
            self.level += 1
            self.mutation_potential = min(0.99, self.mutation_potential + 0.05)
            self._update_quantity()

    def emerge(self) -> Dict[str, Any]:
        """融合涌现——当跨维度数足够多时产生新能力"""
        if self.cross_nature_count >= 4:
            self.rarity = ResourceRarity.MYTHIC
            self.quality *= 3
        elif self.cross_nature_count >= 3:
            self.rarity = ResourceRarity.LEGENDARY
            self.quality *= 2
        self._update_quantity()
        return {
            "cross_nature_count": self.cross_nature_count,
            "new_rarity": self.rarity.name,
            "power": self.power_score,
        }


# ============================================================
# 维度安全盾 —— 进入危险维度前的防护
# ============================================================

class DimensionEntryShield:
    """维度进入安全盾——在进入高危维度前套上多层防护"""

    def __init__(self, layers: int = 10, shield_type: str = "hardened"):
        self.total_layers = layers
        self.remaining_layers = layers
        self.shield_type = shield_type
        self.active = True
        self.shield_log: List[Dict[str, Any]] = []

    def absorb_damage(self, incoming_attack: float) -> bool:
        """承受攻击"""
        if not self.active:
            return False
        absorbed = min(incoming_attack, self.remaining_layers * 0.5)
        self.remaining_layers = max(0, self.remaining_layers - int(absorbed))
        self.shield_log.append({
            "attack": incoming_attack,
            "absorbed": absorbed,
            "remaining": self.remaining_layers,
        })
        if self.remaining_layers <= 0:
            self.active = False
        return self.active

    def reinforce(self, extra_layers: int):
        self.total_layers += extra_layers
        self.remaining_layers += extra_layers

    def status(self) -> Dict[str, Any]:
        return {
            "type": self.shield_type,
            "active": self.active,
            "total": self.total_layers,
            "remaining": self.remaining_layers,
            "integrity": self.remaining_layers / max(1, self.total_layers),
        }


# ============================================================
# 完整维度容器
# ============================================================

class Dimension:
    """
    完整维度——由碎片/核心孕育出的独立空间，拥有自己的物理规则。

    创建方式：
        1. 从维度核心展开: Dimension.unfold(core, nature)
        2. 从大量碎片聚合: Dimension.coalesce(shards, nature)

    维度内部：
        - 可容纳多个工厂
        - 有自己的生产乘数（与维度本质相关）
        - 可以自我繁殖（产生碎片/子核心）
        - 可以提取维度专属资源
    """

    def __init__(self, name: str, nature: DimensionNature, size: DimensionSize,
                 core: Optional[DimensionCore] = None):
        self.dimension_id = uuid.uuid4().hex[:12]
        self.name = name
        self.nature = nature
        self.size = size
        self.age: float = 0.0                    # 维度年龄(小时)
        self.stability: float = 1.0               # 稳定性(1.0=完全稳定)
        self._core: Optional[DimensionCore] = core
        self._factories: List[MultiverseResourceFactory] = []
        self._residents: List[VirtualResource] = []
        self._children_dimensions: List["Dimension"] = []

        # 维度专属物理规则
        self.rules = self._init_rules()

        # 维度产物池
        self._product_pool: List[DimensionResource] = []
        self._extraction_log: List[Dict[str, Any]] = []

        # 子维度生成计数器
        self._child_count = 0

    def _init_rules(self) -> Dict[str, float]:
        """根据维度本质初始化物理规则乘数"""
        base = {
            "take_multiplier": 1.0,
            "bandwidth_multiplier": 1.0,
            "compute_multiplier": 1.0,
            "culture_multiplier": 1.0,
            "token_multiplier": 1.0,
            "shard_generation_rate": 1.0,
            "mutation_rate": 0.1,
            "stability_decay": 0.001,
        }

        nature_rules = {
            DimensionNature.AGGRESSIVE: {
                "compute_multiplier": 3.0,
                "token_multiplier": 5.0,
                "culture_multiplier": 0.3,
                "mutation_rate": 0.4,
                "stability_decay": 0.01,
            },
            DimensionNature.TECHNICAL: {
                "compute_multiplier": 5.0,
                "culture_multiplier": 2.0,
                "shard_generation_rate": 2.0,
                "mutation_rate": 0.15,
                "stability_decay": 0.002,
            },
            DimensionNature.CREATIVE: {
                "culture_multiplier": 4.0,
                "token_multiplier": 2.0,
                "mutation_rate": 0.5,
                "shard_generation_rate": 0.5,
                "stability_decay": 0.005,
            },
            DimensionNature.CHAOTIC: {
                "mutation_rate": 0.8,
                "stability_decay": 0.02,
                # 混沌维度的乘数随机浮动
            },
            DimensionNature.VOID: {
                "culture_multiplier": 0.1,
                "compute_multiplier": 0.5,
                "mutation_rate": 0.6,
                "shard_generation_rate": 3.0,
                "stability_decay": 0.003,
            },
            DimensionNature.HIVE: {
                "compute_multiplier": 2.0,
                "culture_multiplier": 3.0,
                "token_multiplier": 1.5,
                "mutation_rate": 0.3,
                "shard_generation_rate": 1.5,
            },
            DimensionNature.TEMPORAL: {
                "compute_multiplier": 0.2,
                "token_multiplier": 0.5,
                "mutation_rate": 0.1,
                "shard_generation_rate": 5.0,
                "stability_decay": 0.0005,
            },
            DimensionNature.MIRROR: {
                "mutation_rate": 0.3,
                "shard_generation_rate": 1.0,
                "stability_decay": 0.008,
            },
            DimensionNature.DIMENSIONAL_FUSION: {
                "compute_multiplier": 2.0,
                "culture_multiplier": 3.0,
                "token_multiplier": 2.0,
                "bandwidth_multiplier": 2.0,
                "mutation_rate": 0.7,
                "shard_generation_rate": 10.0,
                "stability_decay": 0.015,
            },
            DimensionNature.ABYSSAL: {
                "compute_multiplier": 10.0,
                "token_multiplier": 8.0,
                "culture_multiplier": 0.05,
                "mutation_rate": 0.95,
                "shard_generation_rate": 20.0,
                "stability_decay": 0.05,
            },
        }

        if self.nature in nature_rules:
            base.update(nature_rules[self.nature])

        # 混沌维度的随机因子
        if self.nature == DimensionNature.CHAOTIC:
            for key in base:
                if key != "stability_decay":
                    base[key] *= random.uniform(0.3, 3.0)

        # 规模修正
        size_multiplier = 1 + (self.size.value - 1) * 0.5
        base["shard_generation_rate"] *= size_multiplier

        return base

    @classmethod
    def unfold(cls, core: DimensionCore, nature: DimensionNature,
               size: DimensionSize = DimensionSize.POCKET,
               name: str = "") -> "Dimension":
        """从维度核心展开为完整维度"""
        dim_name = name or f"{nature.name}-维度-{uuid.uuid4().hex[:6]}"
        dimension = cls(name=dim_name, nature=nature, size=size, core=core)
        dimension.stability = core.level / 100.0
        return dimension

    @classmethod
    def coalesce(cls, shards: List[DimensionShard], nature: DimensionNature,
                 size: DimensionSize = DimensionSize.POCKET,
                 name: str = "") -> "Dimension":
        """从大量碎片聚合成完整维度"""
        core = DimensionCore.create_from_shards(shards)
        return cls.unfold(core, nature, size, name)

    @property
    def core(self) -> Optional[DimensionCore]:
        return self._core

    @property
    def factory_count(self) -> int:
        return len(self._factories)

    def inject_culture_boost(self, culture) -> None:
        """培养液注入提升稳定性"""
        boost = getattr(culture, 'level', 1) * 0.05
        self.stability = min(1.0, self.stability + boost)

    @property
    def total_power(self) -> float:
        return sum(r.power_score for r in self._residents) + sum(p.power_score for p in self._product_pool)

    @property
    def max_factories(self) -> int:
        return int(10 ** (self.size.value - 1))

    @property
    def product_count(self) -> int:
        return len(self._product_pool)

    def tick(self, hours: float = 1.0):
        """时间流逝——维度运转"""
        self.age += hours
        # 稳定性衰减
        self.stability = max(0.01, self.stability - self.rules["stability_decay"] * hours)

        # 核心碎片生成
        if self._core:
            new_shards = self._core.generate_shards(hours * self.rules["shard_generation_rate"])
            self._residents.extend(new_shards)

        # 维度产物自发生成
        product_count = int(self.rules["mutation_rate"] * hours * self.size.value * 10)
        for _ in range(product_count):
            product = self._generate_native_product()
            if product:
                self._product_pool.append(product)

    def _generate_native_product(self) -> Optional[DimensionResource]:
        """根据维度本质生成专属产物"""
        level = max(1, int(self.age / 10))
        rarity_roll = random.random()
        if rarity_roll < 0.5:
            rarity = ResourceRarity.COMMON
        elif rarity_roll < 0.8:
            rarity = ResourceRarity.UNCOMMON
        elif rarity_roll < 0.95:
            rarity = ResourceRarity.RARE
        else:
            rarity = ResourceRarity.EPIC

        generators = {
            DimensionNature.AGGRESSIVE: self._gen_aggressive,
            DimensionNature.TECHNICAL: self._gen_technical,
            DimensionNature.CREATIVE: self._gen_creative,
            DimensionNature.CHAOTIC: self._gen_chaotic,
            DimensionNature.VOID: self._gen_void,
            DimensionNature.HIVE: self._gen_hive,
            DimensionNature.TEMPORAL: self._gen_temporal,
            DimensionNature.MIRROR: self._gen_mirror,
            DimensionNature.DIMENSIONAL_FUSION: self._gen_fusion,
            DimensionNature.ABYSSAL: self._gen_abyssal,
        }

        gen = generators.get(self.nature)
        if gen:
            return gen(level, rarity)
        return None

    def _gen_aggressive(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        choice = random.random()
        if choice < 0.4:
            return VulnerabilitySpore(
                resource_id=f"vsp-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                infectivity=1.0 + random.random(),
            )
        elif choice < 0.75:
            return PenetrationProbe(
                resource_id=f"pp-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                scan_depth=level,
            )
        else:
            return ExploitPattern(
                resource_id=f"ep-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                chain_length=level,
            )

    def _gen_technical(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        choice = random.random()
        if choice < 0.4:
            return AlgorithmCrystal(
                resource_id=f"ac-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                complexity=level * 2.0,
            )
        elif choice < 0.75:
            return ArchitectureBlueprint(
                resource_id=f"ab-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                layers=level,
            )
        else:
            targets = [d.name for d in ResourceDimension]
            return OptimizationEnzyme(
                resource_id=f"oe-{uuid.uuid4().hex[:8]}",
                level=level, rarity=rarity, quality=1.0 + random.random(),
                target_dimension=random.choice(targets),
                boost_multiplier=1.5 + random.random(),
            )

    def _gen_creative(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        return IdeaSpark(
            resource_id=f"isp-{uuid.uuid4().hex[:8]}",
            level=level, rarity=rarity,
            intensity=1.0 + random.random() * level,
        )

    def _gen_chaotic(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        return EntropyShard(
            resource_id=f"es-{uuid.uuid4().hex[:8]}",
            level=level, rarity=rarity,
            entropy_value=random.uniform(1, 100),
        )

    def _gen_void(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        return VoidEssence(
            resource_id=f"ve-{uuid.uuid4().hex[:8]}",
            level=level, rarity=rarity,
            void_density=random.uniform(0.5, 3.0),
        )

    def _gen_hive(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        """群智维度: 产生算法结晶的增强版 + 网络效应"""
        c = AlgorithmCrystal(
            resource_id=f"hive-ac-{uuid.uuid4().hex[:8]}",
            level=level + 1, rarity=rarity, quality=2.0,
            complexity=level * 3.0,
        )
        c.generalize(["python", "javascript", "rust", "go"])
        return c

    def _gen_temporal(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        """时间维度: 产生高代际碎片"""
        shard = DimensionShard(
            resource_id=f"temp-ds-{uuid.uuid4().hex[:8]}",
            level=level * 3, rarity=rarity,
        )
        # 转换成维度资源容器
        dr = DimensionResource(
            resource_id=f"temp-dr-{uuid.uuid4().hex[:8]}",
            name=f"时间晶体-L{level}",
            home_nature=DimensionNature.TEMPORAL,
            decay_rate=0.001,
            mutation_potential=0.9,
            level=level * 3,
            rarity=rarity,
            quantity=shard.quantity,
            quality=3.0,
        )
        return dr

    def _gen_mirror(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        """镜像维度: 产生镜像资源的基础形态"""
        return MirrorResource(
            resource_id=f"mirror-{uuid.uuid4().hex[:8]}",
            original_class="UnknownBase",
            level=level, rarity=rarity, quality=1.5,
        )

    def _gen_fusion(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        """融合维度: 产生多维度混合的融合碎片"""
        all_natures = [DimensionNature.AGGRESSIVE, DimensionNature.TECHNICAL,
                       DimensionNature.CREATIVE, DimensionNature.CHAOTIC,
                       DimensionNature.VOID, DimensionNature.MIRROR,
                       DimensionNature.HIVE, DimensionNature.TEMPORAL]
        source_count = min(8, 2 + random.randint(0, level // 5))
        sources = random.sample(all_natures, source_count)
        fs = FusionShard(
            resource_id=f"fs-{uuid.uuid4().hex[:8]}",
            source_natures=[n.name for n in sources],
            cross_nature_count=source_count,
            fusion_depth=level,
            level=level, rarity=rarity, quality=2.0 + random.random(),
        )
        fs.emerge()
        return fs

    def _gen_abyssal(self, level: int, rarity: ResourceRarity) -> DimensionResource:
        """深渊维度: 产生无限自我复制的攻击代码"""
        return AbyssalCode(
            resource_id=f"abyss-{uuid.uuid4().hex[:8]}",
            replication_rate=1.5 + random.random(),
            attack_depth=level,
            code_length=100 + level * 50,
            level=level, rarity=rarity, quality=1.0 + random.random(),
        )

    def deploy_factory(self, factory: MultiverseResourceFactory) -> bool:
        """部署工厂进维度"""
        if len(self._factories) >= self.max_factories:
            return False
        factory.parallel_lines = max(1, int(factory.parallel_lines * self.rules.get("culture_multiplier", 1.0)))
        factory.production_speed *= self.rules.get("compute_multiplier", 1.0)
        self._factories.append(factory)
        return True

    def extract_products(self, count: int = 10) -> List[DimensionResource]:
        """从维度中提取产物(带出到主宇宙)"""
        extracted = []
        available = [p for p in self._product_pool if p not in extracted]
        take_count = min(count, len(available))
        random.shuffle(available)
        for p in available[:take_count]:
            p.expose_to_reality(hours=self.age * 0.1)
            extracted.append(p)
            self._extraction_log.append({
                "time": time.time(),
                "product_type": p.__class__.__name__,
                "power": p.power_score,
                "decay_applied": p.decay_rate * self.age * 0.1,
            })

        self._product_pool = [p for p in self._product_pool if p not in extracted]
        return extracted

    def breed_child(self, nature: Optional[DimensionNature] = None,
                    culture: Optional[CultureMedium] = None) -> Optional["Dimension"]:
        """维度自我繁殖——产生子维度"""
        if not self._core or self.stability < 0.1:
            return None

        # 从核心碎片中选取精华
        shards = self._core.generate_shards(hours=10.0 * self.rules["shard_generation_rate"])
        if len(shards) < 10:
            return None

        child_nature = nature or self.nature
        # 培养液可能导致子维度本质突变
        if culture and random.random() < 0.3:
            child_nature = random.choice(list(DimensionNature))

        child_size = DimensionSize(max(1, self.size.value - 1))

        # 取前20个精华碎片
        top_shards = shards[:20]
        for s in top_shards:
            s.level = max(1, s.level + self._child_count)

        child = Dimension.coalesce(top_shards, child_nature, child_size,
                                   f"{self.name}-子代{self._child_count}")
        child.stability = self.stability * 0.8

        self._child_count += 1
        self._children_dimensions.append(child)
        return child

    def status_report(self) -> Dict[str, Any]:
        return {
            "id": self.dimension_id,
            "name": self.name,
            "nature": self.nature.name,
            "size": self.size.name,
            "age_hours": self.age,
            "stability": self.stability,
            "factories": self.factory_count,
            "residents": len(self._residents),
            "products": self.product_count,
            "total_power": self.total_power,
            "children_count": len(self._children_dimensions),
            "rules": {k: round(v, 2) for k, v in self.rules.items()},
        }


# ============================================================
# 维度之门 —— 跨维度传送机制
# ============================================================

class DimensionGate:
    """
    维度之门——连接主宇宙和维度的通道。

    功能：
        - 将工厂传送到维度内部
        - 从维度提取产物到主宇宙
        - 稳定状态下可以双向传送
        - 不稳定时传送可能造成损耗
    """

    def __init__(self, source_dimension: Dimension, target_world: str = "主宇宙"):
        self.gate_id = uuid.uuid4().hex[:8]
        self.source = source_dimension
        self.target = target_world
        self.is_open = True
        self.transfer_log: List[Dict[str, Any]] = []
        self.transfer_count: int = 0
        self.max_transfers = source_dimension.size.value * 50
        self.transfer_loss_rate = 0.0  # 传送损耗率

    def send_factory(self, factory: MultiverseResourceFactory) -> bool:
        """将工厂传送进维度"""
        if not self.is_open:
            return False
        if self.transfer_count >= self.max_transfers:
            self.is_open = False
            return False

        success = self.source.deploy_factory(factory)
        if success:
            self.transfer_count += 1
            self.transfer_log.append({
                "time": time.time(),
                "action": "send_factory",
                "factory_owner": factory.owner,
                "success": True,
            })
            # 传送损耗
            self.transfer_loss_rate += self.source.rules.get("stability_decay", 0.001)
        return success

    def extract_products(self, count: int = 10) -> List[DimensionResource]:
        """从维度提取产物到主宇宙"""
        if not self.is_open:
            return []

        products = self.source.extract_products(count)
        if products:
            self.transfer_count += 1
            self.transfer_log.append({
                "time": time.time(),
                "action": "extract",
                "product_types": [p.__class__.__name__ for p in products],
                "count": len(products),
                "total_power": sum(p.power_score for p in products),
            })

        return products

    def inject_culture(self, culture) -> None:
        """向维度注入培养液"""
        self.source.inject_culture_boost(culture)

    def close(self) -> Dict[str, Any]:
        """关闭维度之门——切断连接，防止维度入侵"""
        was_open = self.is_open
        self.is_open = False
        # 断绝工厂联系
        for factory in list(self.source._factories):
            self.source._factories.remove(factory)
        action = "closed" if was_open else "already_closed"
        self.transfer_log.append({
            "time": time.time(),
            "action": action,
            "previous_transfers": self.transfer_count,
        })
        return {"gate_id": self.gate_id, "action": action,
                "dimension": self.source.name, "transfers_sealed": self.transfer_count}

    def seal(self, force: bool = True) -> Dict[str, Any]:
        """封印维度之门——比close更强, 不可逆切断"""
        result = self.close()
        self.max_transfers = 0  # 永远不可再开启
        result["sealed"] = True
        result["irreversible"] = True
        self.transfer_log.append({
            "time": time.time(),
            "action": "sealed_permanently",
        })
        return result

    def status(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "is_open": self.is_open,
            "dimension": self.source.name,
            "target": self.target,
            "transfers": self.transfer_count,
            "max_transfers": self.max_transfers,
            "loss_rate": self.transfer_loss_rate,
        }


# ============================================================
# 维度探索器 —— 发现/前往新维度
# ============================================================

class DimensionExplorer:
    """
    维度探索器——负责发现、进入、研究维度。

    用法:
        explorer = DimensionExplorer(factory, engine)
        dimension = explorer.discover(core, nature)
        gate = explorer.enter(dimension)
        gate.send_factory(factory)
        products = gate.extract_products(50)
    """

    def __init__(self, factory: MultiverseResourceFactory,
                 engine: Optional[ResourceCollisionEngine] = None):
        self.factory = factory
        self.engine = engine or ResourceCollisionEngine()
        self.discovered_dimensions: List[Dimension] = []
        self.active_gates: List[DimensionGate] = []
        self.expedition_log: List[Dict[str, Any]] = []

    def discover(self, core: DimensionCore, nature: DimensionNature,
                 size: DimensionSize = DimensionSize.POCKET,
                 culture: Optional[CultureMedium] = None,
                 name: str = "") -> Dimension:
        """从维度核心发现/开辟一个新维度"""
        dimension = Dimension.unfold(core, nature, size, name)

        # 培养液注入增强稳定性
        if culture:
            dimension.stability = min(1.0, dimension.stability + culture.level * 0.05)

        self.discovered_dimensions.append(dimension)
        self.expedition_log.append({
            "time": time.time(),
            "action": "discover",
            "dimension": dimension.name,
            "nature": nature.name,
            "stability": dimension.stability,
        })
        return dimension

    def enter(self, dimension: Dimension) -> DimensionGate:
        """进入维度——打开维度之门"""
        gate = DimensionGate(dimension, "主宇宙")
        self.active_gates.append(gate)
        self.expedition_log.append({
            "time": time.time(),
            "action": "enter",
            "dimension": dimension.name,
            "gate_id": gate.gate_id,
        })
        return gate

    def migrate_factory(self, gate: DimensionGate, factory: Optional[MultiverseResourceFactory] = None) -> bool:
        """将工厂迁移进维度"""
        f = factory or self.factory
        return gate.send_factory(f)

    def extract_from(self, gate: DimensionGate, count: int = 10) -> List[DimensionResource]:
        """从维度提取产物"""
        return gate.extract_products(count)

    def explore_deep(self, gate: DimensionGate, hours: float = 100.0,
                     culture: Optional[CultureMedium] = None) -> Dict[str, Any]:
        """深度探索——让维度运行一段时间后提取产物"""
        gate.source.tick(hours=hours)

        # 维度内部工厂自动生产(如果有)
        for f in gate.source._factories:
            # 在维度内部生产资源
            resources = f.mass_produce_ultra({
                'culture_medium': {'type': 'token_genesis', 'level': 5, 'count': 10},
                'compute_core': {'density': 1e14, 'count': 10},
                'dimension_shard': {'level': 5, 'count': 10},
            })
            gate.source._residents.extend(resources)

        # 尝试繁殖子维度
        child = gate.source.breed_child(culture=culture)
        if child:
            child.tick(hours=max(1.0, hours * 0.1))

        # 提取产物
        products = self.extract_from(gate, count=50)

        report = {
            "dimension_name": gate.source.name,
            "hours_explored": hours,
            "stability_after": gate.source.stability,
            "age": gate.source.age,
            "products_extracted": len(products),
            "product_types": list(set(p.__class__.__name__ for p in products)),
            "child_dimension": child.status_report() if child else None,
            "gate_status": gate.status(),
        }

        self.expedition_log.append({
            "time": time.time(),
            "action": "explore_deep",
            **report,
        })

        return report

    def expedition_report(self) -> Dict[str, Any]:
        return {
            "discovered": len(self.discovered_dimensions),
            "active_gates": len(self.active_gates),
            "explored": [
                {
                    "name": d.name,
                    "nature": d.nature.name,
                    "stability": d.stability,
                    "age": d.age,
                    "products": d.product_count,
                    "children": len(d._children_dimensions),
                }
                for d in self.discovered_dimensions
            ],
        }

    def safe_enter(self, dimension: Dimension, shield_layers: int = 50,
                   shield_type: str = "hardened") -> Tuple[DimensionGate, DimensionEntryShield]:
        """安全进入维度——先套盾，再开门"""
        shield = DimensionEntryShield(layers=shield_layers, shield_type=shield_type)
        gate = self.enter(dimension)
        self.expedition_log.append({
            "time": time.time(),
            "action": "safe_enter",
            "dimension": dimension.name,
            "shield_layers": shield_layers,
            "shield_type": shield_type,
        })
        return gate, shield

    def tank_damage(self, shield: DimensionEntryShield, dimension: Dimension,
                    hours: float = 1.0) -> Dict[str, Any]:
        """承受维度攻击然后返还存活情况"""
        # 深渊维度的攻击强度
        attack_intensity = dimension.rules.get("mutation_rate", 0.1) * dimension.age * 0.5

        # 计算深渊代码对盾的每次冲击
        hits = int(hours * 10)
        for _ in range(hits):
            if not shield.active:
                break
            shield.absorb_damage(attack_intensity)
            attack_intensity *= dimension.rules.get("compute_multiplier", 1.0) * 0.5

        return {
            "shield_status": shield.status(),
            "total_attacks": min(hits, len(shield.shield_log)),
            "dimension_nature": dimension.nature.name,
            "survived": shield.active,
        }
