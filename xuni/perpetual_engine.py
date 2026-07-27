"""
PerpetualEngine —— 把融合产物"做出来用"

核心公式（来自融合链的涌现效果）：
    单节点算力 = 虚拟电 × 转换率          （电大了算力才快）
    节点数     = 虚拟流量通道数            （流量=网络连通度）
    总算力     = 节点数 × 单节点算力        （流式算力网络）
    训练速度   ∝ 总算力 × 加速器倍率

融合产物加成：
    能量算力核心   → 电驱动算力，自循环（电不消耗反增长）
    流式算力网络   → 算力随流量线性扩展
    全网永动算力   → 算力无上限
    分布式加速场   → 加速倍率 × 节点数

用法：
    from xuni.perpetual_engine import PerpetualTrainingEngine
    engine = PerpetualTrainingEngine()
    engine.inject_energy(1e6)        # 注入虚拟电
    engine.set_bandwidth(channels=2048)  # 设置虚拟流量
    engine.apply_fusion("流式算力网络")   # 接入融合产物
    result = engine.train_step(model, epochs=10)  # 真正加速训练
"""

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FusionBoost:
    """融合产物加成"""
    name: str
    compute_multiplier: float = 1.0      # 算力倍率
    energy_regen: float = 0.0            # 每步电再生（自循环）
    accelerator_multiplier: float = 1.0  # 加速器倍率
    node_multiplier: float = 1.0         # 节点数倍率
    perpetual: bool = False              # 是否永动（电不消耗）


# 融合产物 → 加成映射
_FUSION_BOOSTS: Dict[str, FusionBoost] = {
    "能量算力核心": FusionBoost(
        name="能量算力核心",
        compute_multiplier=2.0,
        energy_regen=0.1,        # 每步再生10%电
        perpetual=False,
    ),
    "流式算力网络": FusionBoost(
        name="流式算力网络",
        compute_multiplier=1.0,
        node_multiplier=1.0,      # 节点数=流量通道数（基础）
        accelerator_multiplier=1.0,
    ),
    "全网永动算力": FusionBoost(
        name="全网永动算力",
        compute_multiplier=10.0,
        energy_regen=1.0,         # 电完全再生
        perpetual=True,
        node_multiplier=10.0,
    ),
    "分布式加速场": FusionBoost(
        name="分布式加速场",
        accelerator_multiplier=5.0,
        node_multiplier=2.0,
    ),
    "多维算力网络": FusionBoost(
        name="多维算力网络",
        compute_multiplier=3.0,
        node_multiplier=3.0,
    ),
    "无限训练永动机": FusionBoost(
        name="无限训练永动机",
        compute_multiplier=5.0,
        energy_regen=1.0,
        perpetual=True,
        accelerator_multiplier=3.0,
    ),
    "永动下载涡轮": FusionBoost(
        name="永动下载涡轮",
        energy_regen=0.5,
        perpetual=True,
    ),
    "流式计算引擎": FusionBoost(
        name="流式计算引擎",
        compute_multiplier=4.0,   # 下载即计算
    ),
    "永动加速器": FusionBoost(
        name="永动加速器",
        accelerator_multiplier=10.0,
        energy_regen=0.3,
    ),
    "自进化模型": FusionBoost(
        name="自进化模型",
        compute_multiplier=2.0,
        accelerator_multiplier=2.0,
    ),
}


class PerpetualTrainingEngine:
    """
    永动训练引擎——把融合产物接入训练，真正加速。

    核心闭环：
        虚拟电 → 单节点算力（电越大算力越快）
        虚拟流量 → 节点数（流量越大连通节点越多）
        总算力 = 节点数 × 单节点算力 × 融合加成
        训练速度 ∝ 总算力 × 加速器倍率
        永动产物 → 电再生，算力不衰减
    """

    # 电→算力转换率（1度电 = 1e9 vFLOP，与 VirtualComputeUnit 一致）
    ENERGY_TO_VFLOP = 1e9

    def __init__(self):
        self.energy: float = 0.0           # 当前虚拟电
        self.bandwidth_channels: int = 1   # 虚拟流量通道数
        self.total_vflops_generated: float = 0.0
        self.total_vflops_consumed: float = 0.0
        self.total_energy_regen: float = 0.0
        self._boosts: List[FusionBoost] = []
        self._boost_names: List[str] = []
        self._train_log: List[Dict[str, Any]] = []

    # ---- 资源注入 ----
    def inject_energy(self, energy: float) -> Dict[str, Any]:
        """注入虚拟电"""
        if energy <= 0:
            return {"error": "能量必须为正"}
        self.energy += energy
        return {"energy": self.energy, "injected": energy}

    def set_bandwidth(self, channels: int) -> Dict[str, Any]:
        """设置虚拟流量通道数（决定节点数）"""
        self.bandwidth_channels = max(1, channels)
        return {"channels": self.bandwidth_channels, "nodes": self.node_count}

    # ---- 融合产物接入 ----
    def apply_fusion(self, fusion_name: str) -> Dict[str, Any]:
        """接入一个融合产物（叠加加成）"""
        boost = _FUSION_BOOSTS.get(fusion_name)
        if boost is None:
            return {"error": f"未知融合产物: {fusion_name}"}
        self._boosts.append(boost)
        self._boost_names.append(fusion_name)
        return {
            "applied": fusion_name,
            "active_boosts": self._boost_names,
            "total_compute_mult": self.compute_multiplier,
            "total_node_mult": self.node_multiplier,
            "perpetual": self.is_perpetual,
        }

    # ---- 计算当前能力 ----
    @property
    def node_count(self) -> int:
        """节点数 = 流量通道数 × 节点倍率"""
        mult = 1.0
        for b in self._boosts:
            mult *= b.node_multiplier
        return int(self.bandwidth_channels * mult)

    @property
    def compute_multiplier(self) -> float:
        """总算力倍率（融合产物叠加）"""
        mult = 1.0
        for b in self._boosts:
            mult *= b.compute_multiplier
        return mult

    @property
    def node_multiplier(self) -> float:
        """节点数倍率"""
        mult = 1.0
        for b in self._boosts:
            mult *= b.node_multiplier
        return mult

    @property
    def accelerator_multiplier(self) -> float:
        """加速器倍率"""
        mult = 1.0
        for b in self._boosts:
            mult *= b.accelerator_multiplier
        return mult

    @property
    def energy_regen_rate(self) -> float:
        """每步电再生比例"""
        rate = 0.0
        for b in self._boosts:
            rate = max(rate, b.energy_regen)
        return rate

    @property
    def is_perpetual(self) -> bool:
        """是否永动（电不消耗）"""
        return any(b.perpetual for b in self._boosts)

    @property
    def single_node_vflops(self) -> float:
        """单节点算力 = 电 × 转换率 × 算力倍率（电大了算力才快）"""
        return self.energy * self.ENERGY_TO_VFLOP * self.compute_multiplier

    @property
    def total_vflops(self) -> float:
        """总算力 = 节点数 × 单节点算力（流式算力网络）"""
        return self.node_count * self.single_node_vflops

    @property
    def effective_speed(self) -> float:
        """有效训练速度 = 总算力 × 加速器倍率"""
        return self.total_vflops * self.accelerator_multiplier

    # ---- 训练 ----
    def train_step(self, model, epochs: int = 1, energy_per_epoch: float = 100.0) -> Dict[str, Any]:
        """
        执行一步训练，真正消耗算力并加速模型。

        永动模式下电不消耗反再生，非永动模式消耗电。
        训练进度 = 算力驱动，算力越大进度越快。
        """
        start = time.time()
        needed = energy_per_epoch * epochs

        # 永动模式：电不消耗，反再生
        if self.is_perpetual:
            regen = self.energy * self.energy_regen_rate
            self.energy += regen
            self.total_energy_regen += regen
            energy_consumed = 0.0
        else:
            if self.energy < needed:
                return {
                    "error": f"虚拟电不足：需要 {needed:.0f}，当前 {self.energy:.0f}",
                    "energy": self.energy,
                }
            self.energy -= needed
            # 非永动也有再生（能量算力核心）
            regen = self.energy * self.energy_regen_rate
            self.energy += regen
            self.total_energy_regen += regen
            energy_consumed = needed

        # 算力消耗（虚拟）
        vflops_used = self.effective_speed * epochs
        self.total_vflops_generated += vflops_used
        self.total_vflops_consumed += vflops_used

        # 推进模型训练进度（算力越大，进度越快）
        # 进度增量 ∝ sqrt(总算力) —— 平方根缩放，避免溢出
        progress_gain = min(1.0, math.sqrt(self.effective_speed / 1e12) * 0.1 * epochs)

        trained = False
        if hasattr(model, "training_progress"):
            old = model.training_progress
            model.training_progress = min(1.0, old + progress_gain)
            if model.training_progress >= 1.0 and hasattr(model, "complete_training"):
                model.complete_training()
                trained = True
            elif hasattr(model, "update_training"):
                model.update_training(model.training_progress)
        if hasattr(model, "start_training") and getattr(model, "training_state", None):
            if str(model.training_state).endswith("IDLE") or str(model.training_state).endswith("CLAIMED"):
                model.start_training()

        # 消耗模型能量缓冲（若有）
        if hasattr(model, "_energy_buffer"):
            model._energy_buffer = max(0.0, model._energy_buffer - energy_consumed * 0.5)

        elapsed = time.time() - start
        result = {
            "status": "trained",
            "epochs": epochs,
            "energy_consumed": energy_consumed,
            "energy_regen": regen,
            "energy_remaining": self.energy,
            "vflops_used": vflops_used,
            "progress_gain": progress_gain,
            "model_trained": trained,
            "nodes": self.node_count,
            "total_vflops": self.total_vflops,
            "effective_speed": self.effective_speed,
            "perpetual": self.is_perpetual,
            "elapsed_ms": elapsed * 1000,
        }
        self._train_log.append(result)
        return result

    def stats(self) -> Dict[str, Any]:
        """引擎统计"""
        return {
            "energy": self.energy,
            "bandwidth_channels": self.bandwidth_channels,
            "node_count": self.node_count,
            "single_node_vflops": f"{self.single_node_vflops:.2e}",
            "total_vflops": f"{self.total_vflops:.2e}",
            "effective_speed": f"{self.effective_speed:.2e}",
            "compute_multiplier": self.compute_multiplier,
            "accelerator_multiplier": self.accelerator_multiplier,
            "energy_regen_rate": self.energy_regen_rate,
            "is_perpetual": self.is_perpetual,
            "active_boosts": self._boost_names,
            "total_vflops_generated": f"{self.total_vflops_generated:.2e}",
            "total_vflops_consumed": f"{self.total_vflops_consumed:.2e}",
            "total_energy_regen": self.total_energy_regen,
            "train_steps": len(self._train_log),
        }
