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
    # ---- 参数融合链加成 ----
    "参数流式训练场": FusionBoost(
        name="参数流式训练场",
        compute_multiplier=1.0,
        # train_with_params 中检测此名称启用节点并行注入
    ),
    "能量参数核心": FusionBoost(
        name="能量参数核心",
        compute_multiplier=1.5,
        energy_regen=0.15,
    ),
    "无限参数流": FusionBoost(
        name="无限参数流",
        compute_multiplier=2.0,
        energy_regen=0.2,
        perpetual=False,
    ),
    "超频参数训练": FusionBoost(
        name="超频参数训练",
        accelerator_multiplier=5.0,
        compute_multiplier=2.0,
    ),
    "多维参数训练": FusionBoost(
        name="多维参数训练",
        compute_multiplier=3.0,
        node_multiplier=3.0,
    ),
    "参数自进化体": FusionBoost(
        name="参数自进化体",
        compute_multiplier=3.0,
        accelerator_multiplier=3.0,
        energy_regen=0.3,
    ),
    "永动参数引擎": FusionBoost(
        name="永动参数引擎",
        compute_multiplier=10.0,
        accelerator_multiplier=10.0,
        energy_regen=1.0,
        perpetual=True,
        node_multiplier=5.0,
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

        trained = self._apply_progress(model, progress_gain)

        # 消耗模型能量缓冲（若有）
        if hasattr(model, "_energy_buffer"):
            model._energy_buffer = max(0.0, model._energy_buffer - energy_consumed * 0.5)

        elapsed = time.time() - start
        result = {
            "status": "trained",
            "method": "compute",
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

    def train_with_params(self, model, pack, energy_cost: float = None) -> Dict[str, Any]:
        """
        用参数包 + 流式算力网络训练模型（指数级加速）。

        核心公式（来自融合链涌现效果）：
            原始增量 = 参数质量 × 0.005          （ParameterTrainer 线性）
            流式增量 = 原始增量 × 节点数          （参数流式训练场：N节点并行注入）
            超频增量 = 流式增量 × 加速器倍率       （超频参数训练）
            永动增量 = 超频增量 × 算力倍率         （永动参数引擎）

        没有融合产物时退化为原始 ParameterTrainer 的线性效果。
        """
        start = time.time()

        # 检查模型认领
        if getattr(model, "owner", None) is None:
            return {"error": "模型未被认领，无法训练"}

        # 检查训练状态（用 .name 精确匹配，避免 UNTRAINED 误判）
        state = getattr(model, "training_state", None)
        state_name = state.name if state else ""
        if state_name == "TRAINED":
            return {"error": "模型已训练完成", "progress": 1.0}

        # 参数质量
        quality = max(0.0, min(100.0, getattr(pack, "quality", 50.0)))
        param_count = len(getattr(pack, "params", {}))

        # 基础增量（与 ParameterTrainer 一致）
        base_increment = quality * 0.005
        diversity_bonus = min(0.05, param_count * 0.002)
        base_increment += diversity_bonus

        # ---- 融合产物放大 ----
        # 参数流式训练场：N 节点并行注入，增量 × 节点数
        node_boost = 1.0
        if "参数流式训练场" in self._boost_names or "流式算力网络" in self._boost_names:
            node_boost = float(self.node_count)

        # 超频参数训练：增量 × 加速器倍率
        accel_boost = 1.0
        if "超频参数训练" in self._boost_names:
            accel_boost = self.accelerator_multiplier

        # 永动参数引擎：增量 × 算力倍率
        compute_boost = 1.0
        if "永动参数引擎" in self._boost_names:
            compute_boost = self.compute_multiplier

        # 能量参数核心：参数质量随能量增长
        energy_quality_boost = 1.0
        if "能量参数核心" in self._boost_names:
            # 电越大参数质量越高（对数缩放）
            energy_quality_boost = 1.0 + math.log10(max(1.0, self.energy)) * 0.1

        # 总增量（乘法扩展，从线性→指数）
        total_increment = (
            base_increment
            * node_boost
            * accel_boost
            * compute_boost
            * energy_quality_boost
        )
        # 平方根缩放防止溢出（增量太大也最多一步完成）
        total_increment = min(1.0, total_increment)

        # 激活能量
        if energy_cost is None:
            energy_cost = max(1.0, quality * 0.1)

        energy_consumed = 0.0
        regen = 0.0
        if self.is_perpetual:
            regen = self.energy * self.energy_regen_rate
            self.energy += regen
            self.total_energy_regen += regen
        else:
            if self.energy < energy_cost:
                return {
                    "error": f"激活能量不足：需 {energy_cost:.1f}，当前 {self.energy:.1f}",
                }
            self.energy -= energy_cost
            regen = self.energy * self.energy_regen_rate
            self.energy += regen
            self.total_energy_regen += regen
            energy_consumed = energy_cost

        # 注入参数到模型
        injected = 0
        if hasattr(model, "_energy_buffer"):
            model._energy_buffer = max(0.0, model._energy_buffer - energy_consumed * 0.5)
        try:
            from .parameter import ParameterInjector
            injected = ParameterInjector.inject(model, pack)
        except Exception:
            injected = param_count

        # 应用训练进度
        trained = self._apply_progress(model, total_increment)

        # 记录训练样本
        if hasattr(model, "training_samples_seen"):
            model.training_samples_seen += param_count * max(1, self.node_count)
        if hasattr(model, "training_epochs_done"):
            model.training_epochs_done += 1

        elapsed = time.time() - start
        result = {
            "status": "trained",
            "method": "parameter_fusion",
            "pack_id": getattr(pack, "pack_id", "?"),
            "pack_quality": quality,
            "param_count": param_count,
            "base_increment": base_increment,
            "node_boost": node_boost,
            "accel_boost": accel_boost,
            "compute_boost": compute_boost,
            "energy_quality_boost": energy_quality_boost,
            "total_increment": total_increment,
            "progress_before": getattr(model, "training_progress", 0.0) - total_increment,
            "progress_after": getattr(model, "training_progress", 0.0),
            "energy_consumed": energy_consumed,
            "energy_regen": regen,
            "nodes": self.node_count,
            "effective_speed": self.effective_speed,
            "perpetual": self.is_perpetual,
            "model_trained": trained,
            "elapsed_ms": elapsed * 1000,
        }
        self._train_log.append(result)
        return result

    def _apply_progress(self, model, increment: float) -> bool:
        """应用训练进度到模型，返回是否训练完成"""
        # 开始训练
        state = getattr(model, "training_state", None)
        state_name = state.name if state else ""
        if state_name in ("IDLE", "CLAIMED", "UNTRAINED"):
            if hasattr(model, "start_training"):
                model.start_training()

        old = getattr(model, "training_progress", 0.0)
        new_progress = min(1.0, old + increment)
        if hasattr(model, "update_training"):
            model.update_training(new_progress)
        else:
            model.training_progress = new_progress

        if new_progress >= 1.0 and hasattr(model, "complete_training"):
            model.complete_training()
            return True
        return False

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
