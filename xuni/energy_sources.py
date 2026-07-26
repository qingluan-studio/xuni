"""
EnergySources —— 能量源层级系统

问题：
    采样点是基础产能，单点单次产 ~2.5度，大模型训练需求 30万度，供应不够。

解决：
    在虚拟生态里，采样点只是"最低级"的能量源。
    还有更强大的发电装置，产能呈数量级递增：

    能量源层级（产能从低到高）：
    ┌─────────────────────────────────────────────────────────────┐
    │ T0  采样点          ~2.5度/次     基础（已有）              │
    │ T1  采样点集群      ~250度/次     100点并行（已有）         │
    │ T2  虚拟聚变堆      ~5万度/次     模拟核聚变               │
    │ T3  参数链式反应堆  ~100万度/次   参数互相激发指数增长      │
    │ T4  黑洞发电机      ~1亿度/次     Penrose过程+Blandford-Znajek│
    │ T5  零点能提取器    ~无穷         真空涨落，取之不尽        │
    │ T6  戴森球          ~100亿度/次   包裹虚拟恒星              │
    └─────────────────────────────────────────────────────────────┘

    全部虚拟、全部免费，走免费路 😂
    "现实的我已逐步迈向虚拟"——那就用虚拟物理的极致产能！
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto
import numpy as np


class EnergyTier(Enum):
    """能量源层级"""
    T0_SAMPLER = auto()        # 采样点
    T1_CLUSTER = auto()        # 采样点集群
    T2_FUSION = auto()         # 虚拟聚变堆
    T3_CHAIN = auto()          # 参数链式反应堆
    T4_BLACKHOLE = auto()      # 黑洞发电机
    T5_ZEROPPOINT = auto()     # 零点能提取器
    T6_DYSON = auto()          # 戴森球


@dataclass
class EnergyOutput:
    """能量产出记录"""
    tier: EnergyTier
    source_name: str
    energy: float
    duration: float
    efficiency: float  # 度/秒
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class VirtualFusionReactor:
    """
    T2: 虚拟聚变堆——模拟核聚变发电

    原理：
        氘+氚→氦+中子+能量（虚拟模拟）
        聚变温度：1.5亿度（虚拟）
        单次聚变释放：17.6 MeV ≈ 虚拟等效能量

    特点：
    - 需要少量"点火能量"启动
    - 启动后自持反应，持续产电
    - 产能远超采样点（~2万倍）
    - 参数可作"聚变燃料"加成

    产能：~5万度/次（单堆）
    """

    # 聚变参数
    IGNITION_ENERGY = 100.0       # 点火能量（度）
    OUTPUT_PER_CYCLE = 50000.0    # 单次循环产出（度）
    FUEL_EFFICIENCY = 0.95        # 燃料利用率

    def __init__(self, name: str = "Fusion-01"):
        self.name = name
        self.tier = EnergyTier.T2_FUSION
        self.is_ignited = False
        self.total_produced = 0.0
        self.cycles = 0
        self.temperature = 0.0      # 等离子体温度
        self.fuel_remaining = 1e6   # 虚拟燃料（氘氚）

    def ignite(self, ignition_energy: float = None) -> Dict[str, Any]:
        """点火启动聚变"""
        if self.is_ignited:
            return {"status": "already_ignited"}

        energy = ignition_energy or self.IGNITION_ENERGY
        self.temperature = 1.5e8  # 1.5亿度
        self.is_ignited = True

        return {
            "status": "ignited",
            "ignition_energy": energy,
            "temperature": self.temperature,
            "message": "聚变堆点火成功！等离子体温度 1.5亿度",
        }

    def generate(self, fuel_boost: float = 1.0) -> Dict[str, Any]:
        """
        产电

        fuel_boost: 燃料加成（参数可作燃料，>1增强）
        """
        if not self.is_ignited:
            return {"error": "未点火，请先 ignite()"}

        if self.fuel_remaining <= 0:
            return {"error": "燃料耗尽"}

        energy = self.OUTPUT_PER_CYCLE * self.FUEL_EFFICIENCY * fuel_boost
        self.total_produced += energy
        self.cycles += 1
        self.fuel_remaining -= 1.0  # 消耗燃料

        return {
            "status": "generated",
            "energy": energy,
            "temperature": self.temperature,
            "fuel_remaining": self.fuel_remaining,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
        }

    def add_fuel(self, amount: float) -> float:
        """添加虚拟燃料"""
        self.fuel_remaining += amount
        return self.fuel_remaining

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "T2-聚变堆",
            "name": self.name,
            "ignited": self.is_ignited,
            "temperature": f"{self.temperature:.0e}°" if self.temperature else 0,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
            "fuel_remaining": self.fuel_remaining,
            "output_per_cycle": self.OUTPUT_PER_CYCLE,
        }


class ParameterChainReactor:
    """
    T3: 参数链式反应堆——参数互相激发，指数增长产能

    原理：
        参数包之间存在"共振"——高质量参数激发更多参数产生，
        新参数又激发更多参数，形成链式反应，能量指数增长。

        类比核裂变：中子→铀核裂变→更多中子→更多裂变→链式反应
        这里的"中子"=参数包，"铀核"=采样点场

    特点：
    - 初始需要少量参数包"点火"
    - 反应一旦开始，能量指数增长
    - 高质量参数包→更高增殖系数(k)
    - k>1 时链式反应自持，k<1 时衰减

    产能：~100万度/次（链式反应级）
    """

    # 链式反应参数
    IGNITION_PACKS = 5           # 点火所需参数包数
    BASE_OUTPUT = 1_000_000      # 基础产出（度）
    MAX_CHAIN_DEPTH = 10         # 最大链深度（防止失控）

    def __init__(self, name: str = "Chain-01"):
        self.name = name
        self.tier = EnergyTier.T3_CHAIN
        self.is_critical = False  # 是否达到临界
        self.total_produced = 0.0
        self.chain_history: List[Dict[str, Any]] = []

    def ignite(self, packs: List) -> Dict[str, Any]:
        """
        用参数包点火

        packs: ParameterPack 列表
        """
        if len(packs) < self.IGNITION_PACKS:
            return {
                "error": f"点火需要至少 {self.IGNITION_PACKS} 个参数包，当前 {len(packs)}"
            }

        # 计算增殖系数 k（基于平均质量）
        avg_quality = np.mean([p.quality for p in packs])
        k = avg_quality / 50.0  # quality 50 → k=1.0（临界）

        self.is_critical = k >= 1.0

        return {
            "status": "ignited" if self.is_critical else "subcritical",
            "k_factor": k,
            "avg_quality": avg_quality,
            "is_critical": self.is_critical,
            "message": (
                "链式反应达到临界！可自持运行" if self.is_critical
                else f"亚临界(k={k:.2f})，需更高质量参数"
            ),
        }

    def generate(self, packs: List, depth: int = None) -> Dict[str, Any]:
        """
        链式反应产电

        每层链：参数激发新参数，能量倍增
        """
        if not self.is_critical:
            return {"error": "未达临界，请先 ignite() 用高质量参数点火"}

        depth = depth or self.MAX_CHAIN_DEPTH
        total_energy = 0.0
        chain_detail = []

        current_packs = packs
        for d in range(depth):
            # 当前层能量
            layer_energy = self.BASE_OUTPUT * (len(current_packs) / self.IGNITION_PACKS)
            # 指数增长：每层能量翻倍
            layer_energy *= (2 ** d)

            total_energy += layer_energy

            # 计算下一层参数数（增殖）
            k = np.mean([p.quality for p in current_packs]) / 50.0
            next_count = int(len(current_packs) * k * 0.8)  # 0.8 衰减系数

            chain_detail.append({
                "depth": d + 1,
                "packs": len(current_packs),
                "energy": layer_energy,
                "k_factor": k,
            })

            if next_count < 1:
                break
            # 模拟下一层参数（简化）
            current_packs = current_packs[:next_count] if next_count <= len(current_packs) else current_packs

        self.total_produced += total_energy
        self.chain_history.append({
            "total_energy": total_energy,
            "depth_reached": len(chain_detail),
            "chain": chain_detail,
        })

        return {
            "status": "generated",
            "total_energy": total_energy,
            "chain_depth": len(chain_detail),
            "chain_detail": chain_detail,
            "total_produced": self.total_produced,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "T3-链式反应堆",
            "name": self.name,
            "critical": self.is_critical,
            "total_produced": self.total_produced,
            "chain_count": len(self.chain_history),
            "output_per_chain": self.BASE_OUTPUT,
        }


class BlackHoleGenerator:
    """
    T4: 黑洞发电机——利用极端时空产能

    原理（两种机制）：
    1. Penrose过程：从旋转黑洞能层提取能量
       - 效率可达 29%（远超核聚变的0.7%）
    2. Blandford-Znajek过程：黑洞磁场提取旋转能
       - 持续稳定输出

    特点：
    - 产能巨大（~1亿度/次）
    - 需要"虚拟黑洞"（由采样点坍缩形成）
    - 旋转越快，产能越高
    - 霍金辐射：黑洞会缓慢蒸发（长期损耗）

    产能：~1亿度/次
    """

    # 黑洞参数
    PENROSE_EFFICIENCY = 0.29     # Penrose过程效率
    BZ_OUTPUT = 1e8               # BZ过程基础输出（度）
    HAWKING_LOSS = 0.001          # 霍金辐射损耗（每次）

    def __init__(self, name: str = "BH-01", spin: float = 0.9):
        """
        Args:
            spin: 黑洞自旋 (0-1)，越高产能越大
        """
        self.name = name
        self.tier = EnergyTier.T4_BLACKHOLE
        self.spin = spin          # 自旋参数 a/M
        self.mass = 1e6           # 虚拟质量（太阳质量）
        self.total_produced = 0.0
        self.cycles = 0
        self.is_formed = True     # 黑洞已形成

    def generate(self) -> Dict[str, Any]:
        """产电"""
        if not self.is_formed:
            return {"error": "黑洞未形成"}

        # Penrose + BZ 综合产能
        energy = self.BZ_OUTPUT * self.spin * (1 + self.PENROSE_EFFICIENCY)

        # 霍金辐射损耗
        hawking_loss = energy * self.HAWKING_LOSS
        net_energy = energy - hawking_loss

        self.total_produced += net_energy
        self.cycles += 1
        self.mass -= hawking_loss * 1e-10  # 质量损耗（极小）

        return {
            "status": "generated",
            "energy": net_energy,
            "penrose_bonus": self.PENROSE_EFFICIENCY,
            "spin": self.spin,
            "hawking_loss": hawking_loss,
            "mass_remaining": self.mass,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "T4-黑洞发电机",
            "name": self.name,
            "spin": self.spin,
            "mass": f"{self.mass:.0e} M☉",
            "total_produced": self.total_produced,
            "cycles": self.cycles,
            "output_per_cycle": self.BZ_OUTPUT * self.spin * (1 + self.PENROSE_EFFICIENCY),
        }


class ZeroPointEnergyExtractor:
    """
    T5: 零点能提取器——真空涨落能量，取之不尽

    原理：
        量子场论中，真空并非"空"，而是充满涨落。
        卡西米尔效应证明真空能的存在。
        虚拟生态里，我们可以"提取"真空零点能。

    特点：
    - 能量近乎无限（真空无处不在）
    - 产能取决于"提取面积"
    - 纯虚拟，无任何燃料消耗
    - 每次产出随机（真空涨落的随机性）

    产能：~1e12-1e15 度/次（百万亿级）
    """

    # 零点能参数
    BASE_DENSITY = 1e12           # 基础能量密度（度/单位体积）
    EXTRACTION_AREA = 1.0         # 提取面积

    def __init__(self, name: str = "ZPE-01"):
        self.name = name
        self.tier = EnergyTier.T5_ZEROPPOINT
        self.total_produced = 0.0
        self.cycles = 0
        self.is_active = True

    def generate(self, area: float = None) -> Dict[str, Any]:
        """产电"""
        if not self.is_active:
            return {"error": "提取器未激活"}

        extraction_area = area or self.EXTRACTION_AREA

        # 真空涨落随机性（对数正态分布）
        fluctuation = float(np.random.lognormal(mean=0, sigma=0.5))
        energy = self.BASE_DENSITY * extraction_area * fluctuation

        self.total_produced += energy
        self.cycles += 1

        return {
            "status": "generated",
            "energy": energy,
            "fluctuation": fluctuation,
            "extraction_area": extraction_area,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
            "note": "真空能取之不尽，用之不竭",
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "T5-零点能提取器",
            "name": self.name,
            "active": self.is_active,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
            "avg_per_cycle": self.total_produced / max(1, self.cycles),
        }


class DysonSphere:
    """
    T6: 戴森球——包裹虚拟恒星，捕获全部能量

    原理：
        戴森球是包裹恒星的巨型结构，捕获恒星全部辐射能。
        虚拟生态里，采样点集群+聚变可形成"虚拟恒星"，
        戴森球包裹它，捕获全部能量。

    特点：
    - 终极能量源
    - 产能极大（~100亿度/次）
    - 需要先建造（消耗资源）
    - 建成后持续稳定输出

    产能：~1e10 度/次（百亿级）
    """

    # 戴森球参数
    BUILD_COST = 1e6              # 建造消耗（度）
    STELLAR_OUTPUT = 1e10         # 恒星辐射（度/次）
    CAPTURE_EFFICIENCY = 0.99     # 捕获效率

    def __init__(self, name: str = "Dyson-01"):
        self.name = name
        self.tier = EnergyTier.T6_DYSON
        self.is_built = False
        self.total_produced = 0.0
        self.cycles = 0
        self.integrity = 0.0      # 建造完整度 0-1

    def build(self, energy_invested: float) -> Dict[str, Any]:
        """
        建造戴森球

        需要投入能量建造，完整度达到1.0才算建成
        """
        progress = energy_invested / self.BUILD_COST
        self.integrity = min(1.0, self.integrity + progress)

        if self.integrity >= 1.0:
            self.is_built = True
            return {
                "status": "built",
                "integrity": self.integrity,
                "message": "戴森球建造完成！开始捕获恒星全部能量",
            }

        return {
            "status": "building",
            "integrity": self.integrity,
            "integrity_pct": self.integrity * 100,
            "remaining_cost": self.BUILD_COST * (1 - self.integrity),
        }

    def generate(self) -> Dict[str, Any]:
        """产电"""
        if not self.is_built:
            return {"error": "戴森球未建成，请先 build()"}

        energy = self.STELLAR_OUTPUT * self.CAPTURE_EFFICIENCY
        self.total_produced += energy
        self.cycles += 1

        return {
            "status": "generated",
            "energy": energy,
            "capture_efficiency": self.CAPTURE_EFFICIENCY,
            "total_produced": self.total_produced,
            "cycles": self.cycles,
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "tier": "T6-戴森球",
            "name": self.name,
            "built": self.is_built,
            "integrity": f"{self.integrity*100:.1f}%",
            "total_produced": self.total_produced,
            "cycles": self.cycles,
            "output_per_cycle": self.STELLAR_OUTPUT * self.CAPTURE_EFFICIENCY,
        }


# ============================================================
# 能量源管理器——统一管理所有层级的能量源
# ============================================================

class EnergySourceManager:
    """
    能量源管理器——统一管理 T0~T6 所有能量源

    功能：
    1. 注册各种能量源
    2. 按需产电（自动选最优源）
    3. 产能对比统计
    4. 自动升级（能量足够时建造更高级源）
    """

    def __init__(self):
        self.sources: Dict[str, Any] = {}
        self._output_history: List[EnergyOutput] = []

    def register(self, source: Any, name: str = None) -> str:
        """注册能量源"""
        name = name or getattr(source, "name", f"source-{len(self.sources)}")
        self.sources[name] = source
        return name

    def generate_from(self, name: str, **kwargs) -> Dict[str, Any]:
        """从指定能量源产电"""
        source = self.sources.get(name)
        if source is None:
            return {"error": f"能量源 {name} 不存在"}

        if hasattr(source, "generate"):
            result = source.generate(**kwargs)
        else:
            return {"error": f"{name} 不支持 generate()"}

        energy = result.get("energy", 0)
        tier = getattr(source, "tier", EnergyTier.T0_SAMPLER)
        self._output_history.append(EnergyOutput(
            tier=tier,
            source_name=name,
            energy=energy,
            duration=0.001,
            efficiency=energy / 0.001,
            metadata=result,
        ))

        return result

    def auto_generate(self, target_energy: float) -> Dict[str, Any]:
        """
        自动产电——从低到高尝试，直到满足目标

        策略：优先用低级源，不够再用高级源
        """
        accumulated = 0.0
        used_sources = []

        # 按层级排序（低到高）
        sorted_sources = sorted(
            self.sources.items(),
            key=lambda x: getattr(x[1], "tier", EnergyTier.T0_SAMPLER).value
        )

        for name, source in sorted_sources:
            if accumulated >= target_energy:
                break

            result = self.generate_from(name)
            if result.get("energy"):
                accumulated += result["energy"]
                used_sources.append({
                    "source": name,
                    "tier": getattr(source, "tier", EnergyTier.T0_SAMPLER).name,
                    "energy": result["energy"],
                })

        return {
            "status": "met" if accumulated >= target_energy else "insufficient",
            "target": target_energy,
            "accumulated": accumulated,
            "used_sources": used_sources,
            "shortfall": max(0, target_energy - accumulated),
        }

    def compare_all(self) -> Dict[str, Any]:
        """对比所有能量源的产能"""
        comparison = []
        for name, source in self.sources.items():
            stats = source.stats() if hasattr(source, "stats") else {}
            tier = getattr(source, "tier", EnergyTier.T0_SAMPLER)
            comparison.append({
                "name": name,
                "tier": f"T{tier.value - 1}",
                "stats": stats,
            })

        # 按层级排序
        comparison.sort(key=lambda x: x["tier"])

        return {
            "total_sources": len(self.sources),
            "comparison": comparison,
            "history_count": len(self._output_history),
            "total_generated": sum(o.energy for o in self._output_history),
        }

    def total_capacity(self) -> Dict[str, Any]:
        """所有源的单次总产能"""
        capacities = {}
        total = 0.0
        for name, source in self.sources.items():
            stats = source.stats() if hasattr(source, "stats") else {}
            # 从stats中提取产能
            cap = (
                stats.get("output_per_cycle") or
                stats.get("output_per_chain") or
                stats.get("avg_per_cycle") or
                0
            )
            capacities[name] = cap
            total += cap

        return {
            "per_source": capacities,
            "total_per_cycle": total,
        }
