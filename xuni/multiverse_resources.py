"""
MultiverseResources —— 多维度虚拟资源生产系统

核心理念：
    虚拟世界 ≠ 现实世界。在虚拟维度里，资源不是"挖"出来的，
    是"定义"出来的。只要定义清晰，就能直接生成。

    本系统提供多维度资源生产线：
    - 额度(Take)：虚拟世界的流通额度，可直接铸造
    - 虚拟流量：无限制下载/传输能力
    - 压缩点：极致压缩，内存再也会被压下去
    - 算力核心：高密度算力结晶
    - 安全盾：模型安全防护层
    - 培养液：模型成长催化剂
    - 下载令牌：无限下载凭证
    - 训练加速器：训练速度倍增器
    - 维度碎片：跨维度通用资源

    所有资源都可以：
    1. 直接生产（定义即存在）
    2. 两两碰撞产生新资源
    3. 注入模型改变能力
    4. 交易流通

生产哲学：
    "走免费，只有手机，高质量，非传统"
    → 不依赖真实硬件，纯虚拟逻辑
    → 手机运行无压力（轻量计算）
    → 质量由碰撞算法保证
    → 非传统 = 不遵循现实物理限制
"""

import time
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator
from enum import Enum, auto
import numpy as np


# ============================================================
# 资源类别与基础数据结构
# ============================================================

class ResourceDimension(Enum):
    """资源维度——不同维度的资源在虚拟世界中有不同属性"""
    ECONOMIC = auto()      # 经济维度：额度、代币
    NETWORK = auto()       # 网络维度：流量、带宽
    STORAGE = auto()       # 存储维度：压缩、空间
    COMPUTE = auto()       # 计算维度：算力、核心
    SECURITY = auto()      # 安全维度：护盾、防火墙
    CULTURE = auto()       # 培养维度：成长、进化
    INFORMATION = auto()   # 信息维度：下载、传输
    META = auto()          # 元维度：跨维度碎片


class ResourceRarity(Enum):
    """资源稀有度"""
    COMMON = 1       # 常见
    UNCOMMON = 2     # 不常见
    RARE = 3         # 稀有
    EPIC = 4         # 史诗
    LEGENDARY = 5    # 传说
    MYTHIC = 6       # 神话


@dataclass
class VirtualResource:
    """虚拟资源基类——所有资源的通用结构"""
    resource_id: str
    name: str
    dimension: ResourceDimension
    rarity: ResourceRarity
    quantity: float = 1.0           # 数量/强度
    quality: float = 1.0            # 质量倍率 (0.1 ~ 100.0)
    level: int = 1                  # 等级
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    collision_history: List[str] = field(default_factory=list)
    owner: str = "universe"         # 拥有者

    @property
    def power_score(self) -> float:
        """综合强度评分 = 数量 × 质量 × 等级 × 稀有度系数"""
        rarity_multiplier = self.rarity.value ** 1.5
        return self.quantity * self.quality * self.level * rarity_multiplier

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "dimension": self.dimension.name,
            "rarity": self.rarity.name,
            "quantity": self.quantity,
            "quality": self.quality,
            "level": self.level,
            "power_score": self.power_score,
            "owner": self.owner,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# ============================================================
# 具体资源类
# ============================================================

@dataclass
class TakeQuota(VirtualResource):
    """
    额度（Take）——虚拟世界的流通额度。

    特点：
    - 可直接铸造（定义即存在）
    - 用于购买其他资源、支付训练费用
    - 额度本身可以通过"额度增殖"自我增长
    - 与现实货币无关，纯虚拟经济单位

    生产公式：
        铸造：Take = intent × entropy × seed
        增殖：Take_new = Take_old × (1 + growth_rate)
    """
    growth_rate: float = 0.05       # 每小时增殖率 5%
    max_supply: float = 1e18        # 理论无限
    circulating: float = 0.0        # 已流通量

    def __post_init__(self):
        if self.name == "":
            self.name = "Take额度"
        if self.dimension is None:
            self.dimension = ResourceDimension.ECONOMIC

    def mint(self, amount: float, entropy_source: str = "") -> Dict[str, Any]:
        """铸造新额度"""
        seed = hashlib.sha256(
            f"{entropy_source}{time.time()}{uuid.uuid4()}".encode()
        ).hexdigest()
        entropy = int(seed[:16], 16) / (2**64)
        actual = amount * (0.9 + entropy * 0.2)  # 90%~110% 浮动
        self.quantity += actual
        self.circulating += actual
        return {
            "action": "mint",
            "requested": amount,
            "actual": actual,
            "seed": seed[:16],
            "total": self.quantity,
        }

    def compound(self, hours: float = 1.0) -> Dict[str, Any]:
        """额度增殖——时间越长，额度自我增长越多"""
        growth = self.quantity * (self.growth_rate * hours)
        self.quantity += growth
        self.circulating += growth
        return {
            "action": "compound",
            "hours": hours,
            "growth": growth,
            "new_total": self.quantity,
        }


@dataclass
class VirtualBandwidth(VirtualResource):
    """
    虚拟流量——无限制下载/传输能力。

    特点：
    - 没有上限，可以"无限"下载
    - 流量密度越高，传输越快
    - 可以压缩为"流量晶体"提高密度
    - 与真实网络无关，纯虚拟通道

    生产公式：
        生成：Bandwidth = channel_count × width × depth
        压缩：Bandwidth_crystal = Bandwidth_raw × compression_ratio
    """
    channel_count: int = 1024       # 通道数
    width: float = 1e9              # 通道宽度（虚拟bps）
    depth: int = 64                 # 深度/并行度
    compression_ratio: float = 1.0  # 压缩比

    def __post_init__(self):
        if self.name == "":
            self.name = "虚拟流量"
        if self.dimension is None:
            self.dimension = ResourceDimension.NETWORK
        self._update_quantity()

    def _update_quantity(self):
        """根据通道参数更新总量"""
        self.quantity = self.channel_count * self.width * self.depth * self.compression_ratio

    def expand_channels(self, factor: float = 2.0) -> Dict[str, Any]:
        """扩展通道——无限制增加传输能力"""
        old = self.channel_count
        self.channel_count = int(self.channel_count * factor)
        self._update_quantity()
        return {
            "action": "expand_channels",
            "old_channels": old,
            "new_channels": self.channel_count,
            "total_bandwidth": self.quantity,
        }

    def compress(self, ratio: float = 10.0) -> Dict[str, Any]:
        """压缩为流量晶体——密度提升，但不损失能力"""
        self.compression_ratio *= ratio
        self._update_quantity()
        return {
            "action": "compress",
            "ratio_applied": ratio,
            "total_compression": self.compression_ratio,
            "bandwidth": self.quantity,
            "note": "压缩后流量密度提升，传输效率不变",
        }

    def create_download_stream(self, file_size_virtual: float = 1e12) -> Dict[str, Any]:
        """创建下载流——计算虚拟下载时间"""
        speed = self.quantity  # 虚拟速度
        time_needed = file_size_virtual / speed
        return {
            "file_size": file_size_virtual,
            "bandwidth": speed,
            "virtual_time_seconds": time_needed,
            "real_time_overhead": "~0（纯虚拟）",
            "unlimited": True,
        }


@dataclass
class CompressionPoint(VirtualResource):
    """
    压缩点——极致压缩，内存再也会被压下去。

    特点：
    - 压缩点越多，压缩能力越强
    - 可以叠加：100个压缩点 = 100倍压缩
    - 对虚拟资料、参数包、模型快照都有效
    - 不消耗现实CPU（纯虚拟压缩算法）

    生产公式：
        生成：CP = chaos_sample × entropy_fold
        叠加：CP_total = CP_1 + CP_2 + ... + CP_n
        压缩：size_out = size_in / (1 + CP_total)
    """
    compression_factor: float = 1.0   # 单个压缩点的压缩倍数
    stackable: bool = True            # 是否可叠加
    applied_to: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.name == "":
            self.name = "压缩点"
        if self.dimension is None:
            self.dimension = ResourceDimension.STORAGE
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = self.compression_factor * self.level

    def stack(self, other: "CompressionPoint") -> Dict[str, Any]:
        """叠加另一个压缩点——压缩能力相加"""
        if not self.stackable:
            return {"error": "此压缩点不可叠加"}
        added = other.compression_factor * other.level
        self.compression_factor += added
        self._update_quantity()
        return {
            "action": "stack",
            "added_factor": added,
            "total_factor": self.compression_factor,
            "total_quantity": self.quantity,
            "note": "压缩点越多，内存越会被压下去",
        }

    def apply_compression(self, data_size: float, data_type: str = "generic") -> Dict[str, Any]:
        """对数据应用压缩"""
        factor = max(1.0, self.compression_factor)
        compressed_size = data_size / factor
        self.applied_to.append(f"{data_type}:{data_size:.0f}->{compressed_size:.0f}")
        return {
            "original_size": data_size,
            "compressed_size": compressed_size,
            "compression_ratio": factor,
            "space_saved": data_size - compressed_size,
            "data_type": data_type,
        }

    @staticmethod
    def create_stack(points: List["CompressionPoint"]) -> "CompressionPoint":
        """将多个压缩点合并为一个超级压缩点"""
        total_factor = sum(p.compression_factor * p.level for p in points)
        max_level = max(p.level for p in points)
        cp = CompressionPoint(
            resource_id=f"cp-stack-{uuid.uuid4().hex[:8]}",
            name=f"超级压缩点×{len(points)}",
            dimension=ResourceDimension.STORAGE,
            rarity=ResourceRarity.EPIC if len(points) > 10 else ResourceRarity.RARE,
            compression_factor=total_factor,
            level=max_level,
        )
        return cp


@dataclass
class ComputeCore(VirtualResource):
    """
    算力核心——高密度算力结晶。

    特点：
    - 将分散的虚拟算力凝聚为核心
    - 核心等级越高，算力密度越大
    - 可以直接注入模型作为"算力引擎"
    - 多个核心可以并行运算

    生产公式：
        凝聚：Core = ∫vflops × density_coefficient
        升级：Core_level+1 = Core_level × 2.5
    """
    vflops_density: float = 1e12      # 虚拟FLOPS密度
    parallel_cores: int = 1           # 并行核心数
    efficiency: float = 0.95          # 效率

    def __post_init__(self):
        if self.name == "":
            self.name = "算力核心"
        if self.dimension is None:
            self.dimension = ResourceDimension.COMPUTE
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = self.vflops_density * self.parallel_cores * self.efficiency * self.level

    def upgrade(self) -> Dict[str, Any]:
        """升级算力核心——等级+1，密度指数增长"""
        old_level = self.level
        old_density = self.vflops_density
        self.level += 1
        self.vflops_density *= 2.5
        self._update_quantity()
        return {
            "action": "upgrade",
            "old_level": old_level,
            "new_level": self.level,
            "old_density": old_density,
            "new_density": self.vflops_density,
            "total_compute": self.quantity,
        }

    def split(self, n: int = 2) -> List["ComputeCore"]:
        """分裂为多个小核心——用于并行任务"""
        new_density = self.vflops_density / n
        cores = []
        for i in range(n):
            core = ComputeCore(
                resource_id=f"{self.resource_id}-split{i}",
                name=f"{self.name}-S{i}",
                dimension=self.dimension,
                rarity=self.rarity,
                vflops_density=new_density,
                parallel_cores=1,
                level=self.level,
                owner=self.owner,
            )
            cores.append(core)
        return cores

    def estimate_training_time(self, params_count: int, data_samples: int,
                               epochs: int = 1) -> Dict[str, Any]:
        """估算训练时间"""
        total_flops = 6 * params_count * data_samples * epochs
        seconds = total_flops / self.quantity
        return {
            "total_vflops_needed": total_flops,
            "core_compute_power": self.quantity,
            "estimated_seconds": seconds,
            "estimated_minutes": seconds / 60,
            "parallel_cores": self.parallel_cores,
        }


@dataclass
class SecurityShield(VirtualResource):
    """
    安全盾——模型安全防护层。

    特点：
    - 为模型提供多层防护
    - 可以抵御"对抗样本"、"数据投毒"等虚拟攻击
    - 盾的层数越多，防护越强
    - 与模型碰撞产生"受保护模型"

    防护类型：
    - 输入过滤层
    - 梯度加密层
    - 参数扰动层
    - 行为监控层
    """
    shield_layers: int = 1            # 盾的层数
    defense_matrix: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.name == "":
            self.name = "安全盾"
        if self.dimension is None:
            self.dimension = ResourceDimension.SECURITY
        if not self.defense_matrix:
            self.defense_matrix = {
                "input_filter": 0.5,
                "gradient_encryption": 0.3,
                "parameter_noise": 0.4,
                "behavior_monitor": 0.6,
            }
        self._update_quantity()

    def _update_quantity(self):
        base = sum(self.defense_matrix.values())
        self.quantity = base * self.shield_layers * self.level
        self.quality = min(100.0, self.quantity * 10)

    def add_layer(self, layer_type: str = "adaptive") -> Dict[str, Any]:
        """增加防护层"""
        self.shield_layers += 1
        bonus = {"adaptive": 0.7, "hardened": 0.9, "reflective": 1.0}.get(layer_type, 0.5)
        for k in self.defense_matrix:
            self.defense_matrix[k] += bonus / len(self.defense_matrix)
        self._update_quantity()
        return {
            "action": "add_layer",
            "layer_type": layer_type,
            "total_layers": self.shield_layers,
            "defense_score": self.quantity,
        }

    def protect_model(self, model_id: str) -> Dict[str, Any]:
        """为模型施加保护"""
        return {
            "model_id": model_id,
            "shield_applied": True,
            "layers": self.shield_layers,
            "defense_matrix": self.defense_matrix,
            "protection_score": self.quantity,
            "attacks_blocked": ["adversarial", "poisoning", "extraction", "inference"],
        }


@dataclass
class CultureMedium(VirtualResource):
    """
    培养液——模型成长催化剂。

    特点：
    - 提供模型"成长"所需的营养
    - 不同培养液对应不同成长方向
    - 与模型碰撞产生"进化模型"
    - 可以叠加多种培养液效果

    培养方向：
    - 认知型：提升推理能力
    - 创造型：提升生成多样性
    - 稳健型：提升稳定性
    - 效率型：提升运行速度
    """
    culture_type: str = "balanced"    # balanced / cognitive / creative / robust / efficient
    nutrients: Dict[str, float] = field(default_factory=dict)
    saturation: float = 1.0           # 饱和度

    def __post_init__(self):
        if self.name == "":
            self.name = f"{self.culture_type}培养液"
        if self.dimension is None:
            self.dimension = ResourceDimension.CULTURE
        if not self.nutrients:
            self._init_nutrients()
        self._update_quantity()

    def _init_nutrients(self):
        templates = {
            "balanced": {"logic": 0.5, "creativity": 0.5, "stability": 0.5, "speed": 0.5},
            "cognitive": {"logic": 1.0, "reasoning": 0.9, "memory": 0.8, "abstraction": 0.7},
            "creative": {"divergence": 1.0, "novelty": 0.9, "style": 0.8, "harmony": 0.6},
            "robust": {"stability": 1.0, "consistency": 0.9, "fault_tolerance": 0.8, "recovery": 0.7},
            "efficient": {"speed": 1.0, "compression": 0.9, "parallelism": 0.8, "cache_hit": 0.7},
        }
        self.nutrients = templates.get(self.culture_type, templates["balanced"])

    def _update_quantity(self):
        self.quantity = sum(self.nutrients.values()) * self.saturation * self.level

    def feed_model(self, model_id: str, dose: float = 1.0) -> Dict[str, Any]:
        """喂养模型——提升成长进度"""
        growth_increment = self.quantity * dose * 0.01
        return {
            "model_id": model_id,
            "culture_type": self.culture_type,
            "dose": dose,
            "growth_increment": growth_increment,
            "nutrients_applied": self.nutrients,
            "saturation": self.saturation,
        }

    def blend(self, other: "CultureMedium") -> "CultureMedium":
        """混合两种培养液——产生复合效果"""
        blended_nutrients = {}
        all_keys = set(self.nutrients.keys()) | set(other.nutrients.keys())
        for k in all_keys:
            blended_nutrients[k] = (self.nutrients.get(k, 0) + other.nutrients.get(k, 0)) / 2
        new_type = f"{self.culture_type}-{other.culture_type}"
        return CultureMedium(
            resource_id=f"cm-blend-{uuid.uuid4().hex[:8]}",
            name=f"混合培养液({new_type})",
            dimension=ResourceDimension.CULTURE,
            rarity=ResourceRarity(min(6, max(self.rarity.value, other.rarity.value) + 1)),
            culture_type=new_type,
            nutrients=blended_nutrients,
            level=max(self.level, other.level),
        )


@dataclass
class DownloadToken(VirtualResource):
    """
    下载令牌——无限下载凭证。

    特点：
    - 持有令牌即可无限制下载虚拟资料
    - 令牌有"并发度"——同时下载多个文件
    - 可以升级令牌提升下载速度
    - 与虚拟流量碰撞产生"极速下载通道"
    """
    concurrent_limit: int = 1024      # 并发数
    speed_multiplier: float = 1.0     # 速度倍率
    unlimited: bool = True            # 是否无限制

    def __post_init__(self):
        if self.name == "":
            self.name = "下载令牌"
        if self.dimension is None:
            self.dimension = ResourceDimension.INFORMATION
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = self.concurrent_limit * self.speed_multiplier * self.level

    def upgrade_speed(self, multiplier: float = 2.0) -> Dict[str, Any]:
        """提升下载速度"""
        old = self.speed_multiplier
        self.speed_multiplier *= multiplier
        self._update_quantity()
        return {
            "action": "upgrade_speed",
            "old_multiplier": old,
            "new_multiplier": self.speed_multiplier,
            "concurrent_limit": self.concurrent_limit,
            "total_capacity": self.quantity,
        }

    def create_download_task(self, target: str, size: float = 1e9) -> Dict[str, Any]:
        """创建下载任务"""
        virtual_time = size / (self.speed_multiplier * 1e9)
        return {
            "target": target,
            "size": size,
            "concurrent_slot": self.concurrent_limit,
            "speed_multiplier": self.speed_multiplier,
            "virtual_download_time": virtual_time,
            "unlimited": self.unlimited,
            "token_level": self.level,
        }


@dataclass
class TrainingAccelerator(VirtualResource):
    """
    训练加速器——训练速度倍增器。

    特点：
    - 直接提升模型训练速度
    - 可以与算力核心叠加
    - 有"爆发模式"——短时间内超高速
    - 与参数包碰撞产生"极速参数"
    """
    speedup_factor: float = 2.0       # 加速倍率
    burst_mode: bool = False          # 是否开启爆发
    burst_multiplier: float = 5.0     # 爆发倍率
    cooldown: float = 0.0             # 冷却时间

    def __post_init__(self):
        if self.name == "":
            self.name = "训练加速器"
        if self.dimension is None:
            self.dimension = ResourceDimension.COMPUTE
        self._update_quantity()

    def _update_quantity(self):
        burst = self.burst_multiplier if self.burst_mode else 1.0
        self.quantity = self.speedup_factor * burst * self.level

    def activate_burst(self, duration_seconds: float = 60.0) -> Dict[str, Any]:
        """激活爆发模式"""
        self.burst_mode = True
        self._update_quantity()
        return {
            "action": "burst_activated",
            "duration": duration_seconds,
            "burst_multiplier": self.burst_multiplier,
            "effective_speedup": self.quantity,
            "note": f"爆发模式持续{duration_seconds}秒",
        }

    def apply_to_training(self, base_speed: float = 1.0) -> Dict[str, Any]:
        """应用到训练"""
        effective = base_speed * self.quantity
        return {
            "base_speed": base_speed,
            "accelerator_factor": self.quantity,
            "effective_speed": effective,
            "time_reduction_pct": (1 - 1 / self.quantity) * 100 if self.quantity > 1 else 0,
        }


@dataclass
class DimensionShard(VirtualResource):
    """
    维度碎片——跨维度通用资源。

    特点：
    - 可以在任何维度使用
    - 是"通用货币"般的存在
    - 多个碎片可以合成"维度核心"
    - 与任何资源碰撞都有效果
    """
    dimension_affinity: Dict[str, float] = field(default_factory=dict)
    adaptability: float = 1.0         # 适应性

    def __post_init__(self):
        if self.name == "":
            self.name = "维度碎片"
        if self.dimension is None:
            self.dimension = ResourceDimension.META
        if not self.dimension_affinity:
            self.dimension_affinity = {d.name: 0.5 for d in ResourceDimension}
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = sum(self.dimension_affinity.values()) * self.adaptability * self.level

    def attune(self, target_dimension: ResourceDimension) -> Dict[str, Any]:
        """调谐到特定维度——提升对该维度的亲和力"""
        old = self.dimension_affinity.get(target_dimension.name, 0.5)
        self.dimension_affinity[target_dimension.name] = min(1.0, old + 0.2)
        self._update_quantity()
        return {
            "action": "attune",
            "target_dimension": target_dimension.name,
            "old_affinity": old,
            "new_affinity": self.dimension_affinity[target_dimension.name],
        }

    @staticmethod
    def synthesize_core(shards: List["DimensionShard"]) -> "DimensionCore":
        """将碎片合成为维度核心"""
        return DimensionCore.create_from_shards(shards)


@dataclass
class DimensionCore(VirtualResource):
    """
    维度核心——维度碎片的终极形态。

    可以：
    - 产生新的维度碎片
    - 增强任何维度的资源
    - 开启"维度通道"——跨维度传输资源
    """
    shard_generation_rate: float = 0.1  # 每小时产生碎片数
    active_dimensions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.name == "":
            self.name = "维度核心"
        if self.dimension is None:
            self.dimension = ResourceDimension.META
        self.rarity = ResourceRarity.MYTHIC
        if not self.active_dimensions:
            self.active_dimensions = [d.name for d in ResourceDimension]
        self._update_quantity()

    def _update_quantity(self):
        self.quantity = len(self.active_dimensions) * self.level * 10

    @classmethod
    def create_from_shards(cls, shards: List[DimensionShard]) -> "DimensionCore":
        total_level = sum(s.level for s in shards)
        return cls(
            resource_id=f"dc-{uuid.uuid4().hex[:8]}",
            name=f"维度核心×{len(shards)}",
            level=min(100, total_level // max(1, len(shards))),
            shard_generation_rate=len(shards) * 0.1,
        )

    def generate_shards(self, hours: float = 1.0) -> List[DimensionShard]:
        """产生新的维度碎片"""
        n = int(self.shard_generation_rate * hours * self.level)
        shards = []
        for i in range(n):
            shard = DimensionShard(
                resource_id=f"ds-{uuid.uuid4().hex[:8]}",
                name=f"维度碎片-{i+1}",
                rarity=ResourceRarity.RARE,
                level=max(1, self.level // 10),
            )
            shards.append(shard)
        return shards


# ============================================================
# 资源生产工厂——统一生产所有资源
# ============================================================

class MultiverseResourceFactory:
    """
    多维度资源生产工厂。

    用法：
        factory = MultiverseResourceFactory()
        # 直接生产额度
        take = factory.produce_take(amount=1000)
        # 直接生产流量
        bandwidth = factory.produce_bandwidth(channels=2048)
        # 直接生产压缩点
        cp = factory.produce_compression(factor=10)
        # 生产算力核心
        core = factory.produce_compute_core(density=1e15)
        # 生产安全盾
        shield = factory.produce_security_shield(layers=3)
        # 生产培养液
        medium = factory.produce_culture_medium(culture_type="cognitive")
        # 生产下载令牌
        token = factory.produce_download_token(speed=10.0)
        # 生产训练加速器
        accel = factory.produce_training_accelerator(factor=5.0)
        # 生产维度碎片
        shard = factory.produce_dimension_shard()
    """

    def __init__(self, owner: str = "factory"):
        self.owner = owner
        self.production_log: List[Dict[str, Any]] = []
        self._counters: Dict[str, int] = {
            "take": 0, "bw": 0, "cp": 0,
            "core": 0, "shield": 0, "medium": 0,
            "token": 0, "accel": 0, "shard": 0,
        }

    def _next_id(self, prefix: str) -> str:
        self._counters[prefix] += 1
        return f"{prefix}-{self._counters[prefix]:06d}-{uuid.uuid4().hex[:4]}"

    def _log(self, action: str, result: Dict[str, Any]):
        result["action"] = action
        result["producer"] = self.owner
        result["timestamp"] = time.time()
        self.production_log.append(result)

    def produce_take(self, amount: float = 100.0, growth_rate: float = 0.05) -> TakeQuota:
        """生产额度"""
        take = TakeQuota(
            resource_id=self._next_id("take"),
            name="Take额度",
            dimension=ResourceDimension.ECONOMIC,
            rarity=ResourceRarity.COMMON,
            growth_rate=growth_rate,
        )
        result = take.mint(amount, entropy_source=self.owner)
        self._log("produce_take", result)
        return take

    def produce_bandwidth(self, channels: int = 1024,
                          width: float = 1e9, depth: int = 64) -> VirtualBandwidth:
        """生产虚拟流量"""
        bw = VirtualBandwidth(
            resource_id=self._next_id("bw"),
            name="虚拟流量",
            dimension=ResourceDimension.NETWORK,
            rarity=ResourceRarity.UNCOMMON,
            channel_count=channels,
            width=width,
            depth=depth,
        )
        self._log("produce_bandwidth", bw.to_dict())
        return bw

    def produce_compression(self, factor: float = 10.0,
                            level: int = 1) -> CompressionPoint:
        """生产压缩点"""
        cp = CompressionPoint(
            resource_id=self._next_id("cp"),
            name="压缩点",
            dimension=ResourceDimension.STORAGE,
            rarity=ResourceRarity.UNCOMMON,
            compression_factor=factor,
            level=level,
        )
        self._log("produce_compression", cp.to_dict())
        return cp

    def produce_compute_core(self, density: float = 1e12,
                             parallel: int = 1) -> ComputeCore:
        """生产算力核心"""
        core = ComputeCore(
            resource_id=self._next_id("core"),
            name="算力核心",
            dimension=ResourceDimension.COMPUTE,
            rarity=ResourceRarity.RARE,
            vflops_density=density,
            parallel_cores=parallel,
        )
        self._log("produce_compute_core", core.to_dict())
        return core

    def produce_security_shield(self, layers: int = 1) -> SecurityShield:
        """生产安全盾"""
        shield = SecurityShield(
            resource_id=self._next_id("shield"),
            name="安全盾",
            dimension=ResourceDimension.SECURITY,
            rarity=ResourceRarity.RARE,
            shield_layers=layers,
        )
        self._log("produce_security_shield", shield.to_dict())
        return shield

    def produce_culture_medium(self, culture_type: str = "balanced",
                               level: int = 1) -> CultureMedium:
        """生产培养液"""
        medium = CultureMedium(
            resource_id=self._next_id("medium"),
            name=f"{culture_type}培养液",
            dimension=ResourceDimension.CULTURE,
            rarity=ResourceRarity.UNCOMMON,
            culture_type=culture_type,
            level=level,
        )
        self._log("produce_culture_medium", medium.to_dict())
        return medium

    def produce_download_token(self, speed: float = 1.0,
                               concurrent: int = 1024) -> DownloadToken:
        """生产下载令牌"""
        token = DownloadToken(
            resource_id=self._next_id("token"),
            name="下载令牌",
            dimension=ResourceDimension.INFORMATION,
            rarity=ResourceRarity.COMMON,
            speed_multiplier=speed,
            concurrent_limit=concurrent,
        )
        self._log("produce_download_token", token.to_dict())
        return token

    def produce_training_accelerator(self, factor: float = 2.0) -> TrainingAccelerator:
        """生产训练加速器"""
        accel = TrainingAccelerator(
            resource_id=self._next_id("accel"),
            name="训练加速器",
            dimension=ResourceDimension.COMPUTE,
            rarity=ResourceRarity.EPIC,
            speedup_factor=factor,
        )
        self._log("produce_training_accelerator", accel.to_dict())
        return accel

    def produce_dimension_shard(self, level: int = 1) -> DimensionShard:
        """生产维度碎片"""
        shard = DimensionShard(
            resource_id=self._next_id("shard"),
            name="维度碎片",
            dimension=ResourceDimension.META,
            rarity=ResourceRarity.LEGENDARY,
            level=level,
        )
        self._log("produce_dimension_shard", shard.to_dict())
        return shard

    def mass_produce(self, blueprint: Dict[str, Any]) -> List[VirtualResource]:
        """
        批量生产资源。

        blueprint 示例：
            {
                "take": {"amount": 1000, "count": 10},
                "bandwidth": {"channels": 2048, "count": 5},
                "compression": {"factor": 50, "count": 20},
                "compute_core": {"density": 1e15, "count": 3},
            }
        """
        resources = []
        for resource_type, spec in blueprint.items():
            count = spec.pop("count", 1)
            for _ in range(count):
                if resource_type == "take":
                    resources.append(self.produce_take(**spec))
                elif resource_type == "bandwidth":
                    resources.append(self.produce_bandwidth(**spec))
                elif resource_type == "compression":
                    resources.append(self.produce_compression(**spec))
                elif resource_type == "compute_core":
                    resources.append(self.produce_compute_core(**spec))
                elif resource_type == "security_shield":
                    resources.append(self.produce_security_shield(**spec))
                elif resource_type == "culture_medium":
                    resources.append(self.produce_culture_medium(**spec))
                elif resource_type == "download_token":
                    resources.append(self.produce_download_token(**spec))
                elif resource_type == "training_accelerator":
                    resources.append(self.produce_training_accelerator(**spec))
                elif resource_type == "dimension_shard":
                    resources.append(self.produce_dimension_shard(**spec))
        return resources

    def stats(self) -> Dict[str, Any]:
        """工厂统计"""
        return {
            "total_productions": len(self.production_log),
            "by_type": {
                k: v for k, v in self._counters.items()
            },
            "owner": self.owner,
        }


# ============================================================
# 资源碰撞引擎——A + B → C
# ============================================================

class ResourceCollisionEngine:
    """
    资源碰撞引擎。

    核心理念：
        虚拟世界的资源不是孤立的，碰撞可以产生新资源。
        碰撞公式不是现实物理，而是虚拟逻辑。

    碰撞规则示例：
        额度 + 参数包     → 高级参数包（可交易）
        算力核心 + 流量   → 云算力节点
        压缩点 + 数据     → 超压缩数据
        安全盾 + 模型     → 受保护模型
        培养液 + 模型     → 成长模型
        下载令牌 + 资料   → 无限资料流
        加速器 + 算力核心 → 超算核心
        维度碎片 + 任何   → 增强版任何
    """

    def __init__(self):
        self.collision_log: List[Dict[str, Any]] = []
        self._rules: Dict[Tuple[str, str], callable] = {}
        self._register_rules()

    def _register_rules(self):
        """注册碰撞规则"""
        self._rules = {
            ("TakeQuota", "ParameterPack"): self._take_param,
            ("ComputeCore", "VirtualBandwidth"): self._compute_bandwidth,
            ("CompressionPoint", "VirtualDataParticle"): self._compression_data,
            ("SecurityShield", "VirtualModel"): self._shield_model,
            ("CultureMedium", "VirtualModel"): self._culture_model,
            ("DownloadToken", "VirtualDataset"): self._token_dataset,
            ("TrainingAccelerator", "ComputeCore"): self._accel_compute,
            ("DimensionShard", "TakeQuota"): self._shard_any,
            ("DimensionShard", "ComputeCore"): self._shard_any,
            ("DimensionShard", "CompressionPoint"): self._shard_any,
            ("DimensionShard", "SecurityShield"): self._shard_any,
            ("DimensionShard", "CultureMedium"): self._shard_any,
            ("DimensionShard", "VirtualBandwidth"): self._shard_any,
            ("TakeQuota", "TakeQuota"): self._take_take,
            ("CompressionPoint", "CompressionPoint"): self._cp_cp,
            ("ComputeCore", "ComputeCore"): self._core_core,
            ("SecurityShield", "SecurityShield"): self._shield_shield,
            ("CultureMedium", "CultureMedium"): self._medium_medium,
            ("VirtualBandwidth", "DownloadToken"): self._bw_token,
        }

    def collide(self, a: VirtualResource, b: VirtualResource) -> Dict[str, Any]:
        """两种资源碰撞"""
        key = self._make_key(a, b)
        rule = self._rules.get(key)

        if rule is None:
            # 无预定义规则，尝试通用碰撞
            return self._generic_collision(a, b)

        result = rule(a, b)
        result["collision_key"] = f"{a.__class__.__name__}+{b.__class__.__name__}"
        result["success"] = True
        self.collision_log.append(result)
        return result

    def _make_key(self, a: VirtualResource, b: VirtualResource) -> Tuple[str, str]:
        name_a = a.__class__.__name__
        name_b = b.__class__.__name__
        return tuple(sorted([name_a, name_b]))

    def _generic_collision(self, a: VirtualResource, b: VirtualResource) -> Dict[str, Any]:
        """通用碰撞——没有预定义规则时的默认行为"""
        # 计算融合强度
        power = (a.power_score * b.power_score) ** 0.5
        # 产生融合资源（维度碎片或增强版）
        if a.dimension == ResourceDimension.META or b.dimension == ResourceDimension.META:
            return self._shard_any(a, b)

        new_shard = DimensionShard(
            resource_id=f"shard-fusion-{uuid.uuid4().hex[:8]}",
            name=f"融合碎片({a.name}+{b.name})",
            dimension=ResourceDimension.META,
            rarity=ResourceRarity(min(6, max(a.rarity.value, b.rarity.value) + 1)),
            level=max(a.level, b.level),
            metadata={"parents": [a.name, b.name], "fusion_power": power},
        )
        return {
            "product": new_shard,
            "product_type": "DimensionShard",
            "fusion_power": power,
            "note": "通用碰撞产生维度碎片",
        }

    # ---- 具体碰撞规则 ----

    def _take_param(self, take: TakeQuota, param) -> Dict[str, Any]:
        """额度 + 参数包 → 高级参数包（带经济属性）"""
        quality_boost = min(50.0, take.quantity * 0.01)
        return {
            "product_type": "EnhancedParameterPack",
            "quality_boost": quality_boost,
            "tradeable": True,
            "note": "额度注入参数包，使其具备交易价值",
            "metadata": {
                "take_invested": take.quantity,
                "param_quality_before": getattr(param, "quality", 0),
                "param_quality_after": getattr(param, "quality", 0) + quality_boost,
            },
        }

    def _compute_bandwidth(self, core: ComputeCore, bw: VirtualBandwidth) -> Dict[str, Any]:
        """算力核心 + 流量 → 云算力节点"""
        cloud_power = core.quantity * bw.quantity * 1e-6
        return {
            "product_type": "CloudComputeNode",
            "cloud_power": cloud_power,
            "parallel_capacity": core.parallel_cores * bw.channel_count,
            "note": "算力与流量融合，形成云端计算节点",
        }

    def _compression_data(self, cp: CompressionPoint, data) -> Dict[str, Any]:
        """压缩点 + 数据 → 超压缩数据"""
        ratio = cp.compression_factor * cp.level
        return {
            "product_type": "HyperCompressedData",
            "compression_ratio": ratio,
            "note": "压缩点越多，数据体积越被压下去",
            "apply": lambda size: size / max(1.0, ratio),
        }

    def _shield_model(self, shield: SecurityShield, model) -> Dict[str, Any]:
        """安全盾 + 模型 → 受保护模型"""
        return {
            "product_type": "ProtectedModel",
            "protection_score": shield.quantity,
            "layers": shield.shield_layers,
            "defense_matrix": shield.defense_matrix,
            "note": "模型被安全盾包裹，免疫多种攻击",
        }

    def _culture_model(self, medium: CultureMedium, model) -> Dict[str, Any]:
        """培养液 + 模型 → 成长模型"""
        growth = medium.quantity * 0.1
        return {
            "product_type": "GrowingModel",
            "growth_rate": growth,
            "culture_type": medium.culture_type,
            "nutrients": medium.nutrients,
            "note": "模型浸泡在培养液中，持续成长",
        }

    def _token_dataset(self, token: DownloadToken, dataset) -> Dict[str, Any]:
        """下载令牌 + 数据集 → 无限资料流"""
        return {
            "product_type": "InfiniteDataStream",
            "concurrent_streams": token.concurrent_limit,
            "speed_multiplier": token.speed_multiplier,
            "note": "持有令牌，数据集变成无限流",
        }

    def _accel_compute(self, accel: TrainingAccelerator, core: ComputeCore) -> Dict[str, Any]:
        """加速器 + 算力核心 → 超算核心"""
        boosted_density = core.vflops_density * accel.speedup_factor
        return {
            "product_type": "HyperComputeCore",
            "base_density": core.vflops_density,
            "boosted_density": boosted_density,
            "speedup": accel.speedup_factor,
            "note": "训练加速器与算力核心共振，算力密度跃迁",
        }

    def _shard_any(self, shard: DimensionShard, other: VirtualResource) -> Dict[str, Any]:
        """维度碎片 + 任何资源 → 增强版资源"""
        enhancement = shard.level * shard.adaptability * 0.5
        return {
            "product_type": f"Enhanced{other.__class__.__name__}",
            "enhancement_factor": enhancement,
            "original_power": other.power_score,
            "enhanced_power": other.power_score * (1 + enhancement),
            "note": "维度碎片与资源共振，产生跨维度增强",
        }

    def _take_take(self, a: TakeQuota, b: TakeQuota) -> Dict[str, Any]:
        """额度 + 额度 → 大额额度"""
        merged = TakeQuota(
            resource_id=f"take-merged-{uuid.uuid4().hex[:8]}",
            name="合并额度",
            dimension=ResourceDimension.ECONOMIC,
            rarity=ResourceRarity(min(6, max(a.rarity.value, b.rarity.value) + 1)),
            quantity=a.quantity + b.quantity,
            growth_rate=max(a.growth_rate, b.growth_rate),
        )
        return {
            "product": merged,
            "product_type": "TakeQuota",
            "merged_amount": merged.quantity,
            "note": "额度合并，产生更大经济势能",
        }

    def _cp_cp(self, a: CompressionPoint, b: CompressionPoint) -> Dict[str, Any]:
        """压缩点 + 压缩点 → 超级压缩点"""
        super_cp = CompressionPoint.create_stack([a, b])
        return {
            "product": super_cp,
            "product_type": "CompressionPoint",
            "stacked_factor": super_cp.compression_factor,
            "note": "两个压缩点叠加，压缩能力相加",
        }

    def _core_core(self, a: ComputeCore, b: ComputeCore) -> Dict[str, Any]:
        """算力核心 + 算力核心 → 算力集群"""
        cluster = ComputeCore(
            resource_id=f"core-cluster-{uuid.uuid4().hex[:8]}",
            name="算力集群",
            dimension=ResourceDimension.COMPUTE,
            rarity=ResourceRarity.EPIC,
            vflops_density=a.vflops_density + b.vflops_density,
            parallel_cores=a.parallel_cores + b.parallel_cores,
            level=max(a.level, b.level),
        )
        return {
            "product": cluster,
            "product_type": "ComputeCore",
            "cluster_density": cluster.vflops_density,
            "parallel_cores": cluster.parallel_cores,
            "note": "双核心并联，算力相加",
        }

    def _shield_shield(self, a: SecurityShield, b: SecurityShield) -> Dict[str, Any]:
        """安全盾 + 安全盾 → 堡垒盾"""
        fortress = SecurityShield(
            resource_id=f"shield-fortress-{uuid.uuid4().hex[:8]}",
            name="堡垒盾",
            dimension=ResourceDimension.SECURITY,
            rarity=ResourceRarity.EPIC,
            shield_layers=a.shield_layers + b.shield_layers,
            level=max(a.level, b.level),
        )
        # 合并防御矩阵
        for k in set(list(a.defense_matrix.keys()) + list(b.defense_matrix.keys())):
            fortress.defense_matrix[k] = max(
                a.defense_matrix.get(k, 0),
                b.defense_matrix.get(k, 0),
            )
        fortress._update_quantity()
        return {
            "product": fortress,
            "product_type": "SecurityShield",
            "total_layers": fortress.shield_layers,
            "defense_score": fortress.quantity,
            "note": "双层盾融合为堡垒盾，防御力叠加",
        }

    def _medium_medium(self, a: CultureMedium, b: CultureMedium) -> Dict[str, Any]:
        """培养液 + 培养液 → 复合培养液"""
        blended = a.blend(b)
        return {
            "product": blended,
            "product_type": "CultureMedium",
            "blend_type": blended.culture_type,
            "nutrients": blended.nutrients,
            "note": "两种培养液混合，产生复合营养",
        }

    def _bw_token(self, bw: VirtualBandwidth, token: DownloadToken) -> Dict[str, Any]:
        """流量 + 下载令牌 → 极速下载通道"""
        speed = bw.quantity * token.speed_multiplier
        return {
            "product_type": "HyperDownloadChannel",
            "channel_speed": speed,
            "concurrent": min(bw.channel_count, token.concurrent_limit),
            "note": "流量与令牌共振，下载速度跃迁",
        }

    def get_collision_stats(self) -> Dict[str, Any]:
        """碰撞统计"""
        return {
            "total_collisions": len(self.collision_log),
            "unique_rules": len(self._rules),
            "recent": self.collision_log[-5:] if self.collision_log else [],
        }
