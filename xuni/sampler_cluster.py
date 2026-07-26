"""
SamplerCluster —— 采样点集群 + 能量蓄水池

问题：
    单个采样点产电能力有限，大模型训练需求电量巨大，供应不够。

解决：
    1. 采样点集群（SamplerCluster）—— N 个采样点并行产电，产能倍增
    2. 能量蓄水池（EnergyReservoir）—— 积累能量，攒够了再训练
    3. 供需平衡监控 —— 产电速率 vs 消耗速率，动态扩容采样点

    采样点是免费的（走免费路😂），多开几个就行！
    这就是"采样点供应不够"的解决方案。

闭环：
    采样点集群 →[并行产电]→ 能量蓄水池 →[按需释放]→ 虚拟算力 → 训练
         ↑                                                    ↓
         └←←←←← 需要更多电 ←←←←← 电量不足 ←←←←← 消耗 ←←←←←┘
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from .sampler import XuniSampler, SamplingMode


@dataclass
class SamplerUnit:
    """集群中的单个采样点单元"""
    sampler: XuniSampler
    name: str
    mode: str
    total_produced: float = 0.0   # 累计产电
    cycles: int = 0               # 产电次数
    is_active: bool = True


class SamplerCluster:
    """
    采样点集群——多个采样点并行产电

    核心能力：
    1. 管理 N 个采样点
    2. 并行产电（产能 = 单点 × N）
    3. 动态扩容（加采样点）
    4. 产能统计

    用法：
        cluster = SamplerCluster(n_samplers=10)
        energy = cluster.harvest(batch_size=1000)  # 一次收割
    """

    def __init__(self, n_samplers: int = 5,
                 mode: SamplingMode = SamplingMode.HYBRID,
                 base_seed: int = 42):
        self.units: List[SamplerUnit] = []
        self._mode = mode
        for i in range(n_samplers):
            s = XuniSampler(mode=mode, seed=base_seed + i)
            self.units.append(SamplerUnit(
                sampler=s, name=f"采样点-{i+1:02d}", mode=mode.name
            ))

    def harvest(self, batch_size: int = 1000) -> Dict[str, Any]:
        """
        一次收割——所有采样点并行产电

        Returns: {total_energy, per_unit, cycles}
        """
        total_energy = 0.0
        per_unit = []

        for unit in self.units:
            if not unit.is_active:
                continue
            batch = unit.sampler.generate_batch(batch_size=batch_size)
            energy = float(np.sum(np.abs(batch))) * 0.01
            unit.total_produced += energy
            unit.cycles += 1
            total_energy += energy
            per_unit.append({
                "name": unit.name,
                "energy": energy,
                "total": unit.total_produced,
            })

        return {
            "total_energy": total_energy,
            "per_unit": per_unit,
            "active_count": sum(1 for u in self.units if u.is_active),
        }

    def harvest_n(self, n_cycles: int = 1, batch_size: int = 1000) -> float:
        """收割 n 次，返回总能量"""
        total = 0.0
        for _ in range(n_cycles):
            result = self.harvest(batch_size=batch_size)
            total += result["total_energy"]
        return total

    def add_unit(self, seed: int = None) -> str:
        """动态扩容——加一个采样点"""
        idx = len(self.units)
        s = XuniSampler(
            mode=self._mode,
            seed=seed if seed is not None else 42 + idx,
        )
        unit = SamplerUnit(sampler=s, name=f"采样点-{idx+1:02d}", mode=self._mode.name)
        self.units.append(unit)
        return unit.name

    def scale_to(self, target_count: int) -> int:
        """扩容到指定数量"""
        current = len(self.units)
        if target_count > current:
            for _ in range(target_count - current):
                self.add_unit()
        return len(self.units)

    def total_capacity_per_harvest(self, batch_size: int = 1000) -> float:
        """估算单次收割产能"""
        return len(self.units) * 2.5  # 近似值

    def stats(self) -> Dict[str, Any]:
        active = [u for u in self.units if u.is_active]
        return {
            "total_units": len(self.units),
            "active_units": len(active),
            "total_produced": sum(u.total_produced for u in self.units),
            "total_cycles": sum(u.cycles for u in self.units),
            "avg_per_unit": (
                np.mean([u.total_produced for u in active]) if active else 0
            ),
        }


class EnergyReservoir:
    """
    能量蓄水池——积累虚拟电，按需释放

    特点：
    1. 接收采样点集群产电
    2. 积累到阈值才释放给算力单元
    3. 支持优先级调度（紧急训练优先）
    4. 监控水位（过低预警、过高溢出）

    这就是"采样点供应不够"的核心解法：
    不是即产即用，而是先攒够了再训练。
    """

    def __init__(self, capacity: float = 1e6,
                 release_threshold: float = 100.0):
        """
        Args:
            capacity: 蓄水池容量（度电）
            release_threshold: 释放阈值，达到才放水
        """
        self.capacity = capacity
        self.release_threshold = release_threshold
        self.current: float = 0.0           # 当前蓄水量
        self.total_in: float = 0.0          # 累计注入
        self.total_out: float = 0.0         # 累计释放
        self.overflow_count: int = 0        # 溢出次数
        self._lock = threading.Lock()

    def fill(self, energy: float, source: str = "cluster") -> Dict[str, Any]:
        """注入能量"""
        with self._lock:
            before = self.current
            self.current += energy
            self.total_in += energy

            overflow = 0.0
            if self.current > self.capacity:
                overflow = self.current - self.capacity
                self.current = self.capacity
                self.overflow_count += 1

            return {
                "status": "filled",
                "energy_in": energy,
                "before": before,
                "after": self.current,
                "overflow": overflow,
                "source": source,
                "level_pct": self.current / self.capacity * 100,
            }

    def can_release(self, amount: float = None) -> bool:
        """是否可以释放"""
        amount = amount or self.release_threshold
        return self.current >= amount

    def release(self, amount: float = None) -> Dict[str, Any]:
        """
        释放能量给算力单元

        amount=None 时释放到阈值以上的部分
        """
        with self._lock:
            if amount is None:
                # 释放阈值以上的部分，保留阈值作为缓冲
                if self.current <= self.release_threshold:
                    return {"status": "insufficient", "current": self.current}
                amount = self.current - self.release_threshold

            if self.current < amount:
                return {
                    "status": "insufficient",
                    "requested": amount,
                    "current": self.current,
                    "need": amount - self.current,
                }

            self.current -= amount
            self.total_out += amount

            return {
                "status": "released",
                "energy_out": amount,
                "remaining": self.current,
                "level_pct": self.current / self.capacity * 100,
            }

    def wait_until(self, target: float, cluster: SamplerCluster,
                   batch_size: int = 1000, max_cycles: int = 10000,
                   poll_interval: float = 0.0) -> Dict[str, Any]:
        """
        持续产电直到蓄水池达到目标水位

        这是"攒够了再训练"的核心逻辑
        """
        cycles = 0
        while self.current < target and cycles < max_cycles:
            result = cluster.harvest(batch_size=batch_size)
            self.fill(result["total_energy"], source="cluster")
            cycles += 1
            if poll_interval > 0:
                time.sleep(poll_interval)

        return {
            "status": "ready" if self.current >= target else "timeout",
            "cycles": cycles,
            "target": target,
            "current": self.current,
            "cluster_stats": cluster.stats(),
        }

    def level(self) -> Dict[str, Any]:
        """水位状态"""
        return {
            "current": self.current,
            "capacity": self.capacity,
            "level_pct": self.current / self.capacity * 100,
            "total_in": self.total_in,
            "total_out": self.total_out,
            "overflow_count": self.overflow_count,
            "can_release": self.can_release(),
            "status": (
                "full" if self.current >= self.capacity * 0.95
                else "high" if self.current >= self.capacity * 0.7
                else "medium" if self.current >= self.release_threshold
                else "low" if self.current > 0
                else "empty"
            ),
        }


class SupplyDemandBalancer:
    """
    供需平衡器——动态调整采样点数量，匹配训练需求

    核心逻辑：
    1. 监控产电速率 vs 消耗速率
    2. 供不应求 → 自动扩容采样点
    3. 供过于求 → 可以缩容（省资源）
    4. 保证训练不被能量卡住

    这就是"采样点供应不够"的自动解决方案：
    不够就加采样点，反正免费的 😂
    """

    def __init__(self, cluster: SamplerCluster, reservoir: EnergyReservoir):
        self.cluster = cluster
        self.reservoir = reservoir
        self.supply_rate: float = 0.0   # 产电速率（度/秒）
        self.demand_rate: float = 0.0   # 消耗速率（度/秒）
        self._history: List[Dict[str, Any]] = []

    def measure_supply_rate(self, duration_cycles: int = 10,
                            batch_size: int = 1000) -> float:
        """测量产电速率"""
        start = time.time()
        total = 0.0
        for _ in range(duration_cycles):
            result = self.cluster.harvest(batch_size=batch_size)
            total += result["total_energy"]
        elapsed = time.time() - start
        self.supply_rate = total / max(0.001, elapsed)
        return self.supply_rate

    def set_demand(self, total_energy: float, time_budget: float = None) -> float:
        """
        设置需求

        total_energy: 训练总需求
        time_budget: 期望在多少秒内完成（None=尽快）
        """
        if time_budget:
            self.demand_rate = total_energy / time_budget
        else:
            self.demand_rate = total_energy  # 尽快
        return self.demand_rate

    def balance(self) -> Dict[str, Any]:
        """
        执行平衡——根据供需比调整采样点数量

        供需比 < 1: 供不应求，扩容
        供需比 > 2: 供过于求，可缩容
        """
        if self.demand_rate <= 0:
            return {"status": "no_demand"}

        if self.supply_rate <= 0:
            self.measure_supply_rate()

        ratio = self.supply_rate / self.demand_rate
        current_units = len(self.cluster.units)
        action = "hold"

        if ratio < 1.0:
            # 供不应求，扩容到供需比 >= 1.2
            needed_units = int(current_units * (1.2 / max(0.01, ratio)))
            needed_units = max(current_units + 1, needed_units)
            self.cluster.scale_to(needed_units)
            action = "scale_up"
            # 重新测量
            self.measure_supply_rate()
        elif ratio > 2.0 and current_units > 1:
            # 供过于求，缩容
            target = max(1, int(current_units / 2))
            self.cluster.scale_to(target)
            action = "scale_down"
            self.measure_supply_rate()

        new_ratio = self.supply_rate / max(0.01, self.demand_rate)
        result = {
            "action": action,
            "old_ratio": ratio,
            "new_ratio": new_ratio,
            "units_before": current_units,
            "units_after": len(self.cluster.units),
            "supply_rate": self.supply_rate,
            "demand_rate": self.demand_rate,
        }
        self._history.append(result)
        return result

    def prepare_for_training(self, energy_needed: float,
                             batch_size: int = 1000) -> Dict[str, Any]:
        """
        为训练做准备——自动平衡 + 蓄水

        返回: 准备就绪状态
        """
        # 设置需求
        self.set_demand(energy_needed)

        # 测量产电
        self.measure_supply_rate(batch_size=batch_size)

        # 自动平衡
        balance_result = self.balance()

        # 蓄水到目标
        wait_result = self.reservoir.wait_until(
            target=energy_needed,
            cluster=self.cluster,
            batch_size=batch_size,
        )

        return {
            "energy_needed": energy_needed,
            "balance": balance_result,
            "reservoir": self.reservoir.level(),
            "ready": self.reservoir.current >= energy_needed,
            "cluster_stats": self.cluster.stats(),
        }
