"""
XuniVitality —— 活力系统（资源 × 信息 = 自由能）

核心理念（来自物理/生命科学）：
  自由能 F = 内能 U - 温度 × 熵 S
            ↑资源      ↑信息(负熵)
  
  采样点的电 = 无序能量（资源），只能"发热"
  参数的信息 = 负熵（序），给能量加"图纸"
  两者融合 = 智能电（活力）= 能做"有用功"的能量

活力（Vitality）是这个生态的高阶产物：
  普通虚拟电 → 只能"驱动"模型（被动）
  智能电     → 能"养育"模型（主动）—— 模型从工具变成生命体

涌现阈值：
  vitality > 30: 自繁殖（采样点分裂）
  vitality > 50: 自迁移（向高活力区聚集）
  vitality > 70: 自组队（多个采样点聚合为簇）
  vitality > 90: 自创造（涌现新参数包）

闭环：
  采样点 → 电 + 参数
            ↓ 融合
         智能电(活力)
            ↓
      驱动模型自演化
            ↓
      模型产出新参数
            ↓
      反哺采样点（活力改变分布）
            ↓
         回到起点 ↑ ← 自循环
"""

import time
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple
from enum import Enum, auto

import numpy as np


class EmergenceType(Enum):
    """涌现行为类型"""
    NONE = auto()           # 无涌现
    SELF_REPLICATE = auto() # 自繁殖：采样点分裂
    SELF_MIGRATE = auto()   # 自迁移：向高活力区移动
    SELF_CLUSTER = auto()   # 自组队：聚合为簇
    SELF_CREATE = auto()    # 自创造：涌现新参数包


@dataclass
class VitalityCell:
    """
    活力单元。
    
    记录一个空间区域的活力状态。
    """
    grid_x: int
    grid_y: int
    grid_z: int
    energy: float = 0.0           # 资源（虚拟电）
    info_potential: float = 0.0   # 信息势（参数密度）
    vitality: float = 0.0         # 活力（融合产物）
    emergence: EmergenceType = EmergenceType.NONE
    last_updated: float = 0.0


class FusionReactor:
    """
    融合反应器。
    
    将资源（电）和信息（参数势）融合为活力（智能电）。
    
    公式：
      vitality = energy * (1 + info_potential) / 2
      
    纯能量（info=0）：vitality = energy/2，活力低
    能量+信息（info=1）：vitality = energy，活力满
    超量信息（info>1）：vitality > energy，信息加成
    """
    
    def __init__(self, info_weight: float = 1.0):
        self.info_weight = info_weight

    def fuse(self, energy: float, info_potential: float) -> float:
        """
        融合资源与信息，产出活力。
        
        energy: 虚拟电（资源）
        info_potential: 信息势（参数密度/质量，0-1+）
        return: 活力值（智能电）
        """
        if energy <= 0:
            return 0.0
        weighted_info = info_potential * self.info_weight
        vitality = energy * (1.0 + weighted_info) / 2.0
        return round(vitality, 4)

    def batch_fuse(self, cells: List[VitalityCell]) -> None:
        """批量融合，更新每个单元的活力"""
        for cell in cells:
            cell.vitality = self.fuse(cell.energy, cell.info_potential)
            cell.last_updated = time.time()


class VitalityField:
    """
    活力场。
    
    活力在三维空间上的分布，形成势场。
    高活力区 = 富庶区，低活力区 = 荒芜区。
    活力会扩散（从高到低），形成流动。
    """
    
    def __init__(self, grid_size: Tuple[int, int, int] = (16, 16, 16)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.float32)
        self.cells: Dict[Tuple[int, int, int], VitalityCell] = {}
        self._diffusion_rate = 0.1  # 扩散率

    def deposit(self, x: int, y: int, z: int, vitality: float):
        """在某个网格点存入活力"""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
            self.grid[x, y, z] += vitality
            key = (x, y, z)
            if key not in self.cells:
                self.cells[key] = VitalityCell(grid_x=x, grid_y=y, grid_z=z)
            self.cells[key].vitality = self.grid[x, y, z]

    def get_vitality(self, x: int, y: int, z: int) -> float:
        """获取某点活力"""
        if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
            return float(self.grid[x, y, z])
        return 0.0

    def diffuse(self, steps: int = 1):
        """
        活力扩散。
        
        活力从高区流向低区，模拟"智能电的流动"。
        """
        for _ in range(steps):
            # 简单的3D扩散：与6个邻居取加权平均
            shifted = []
            for axis in range(3):
                for direction in (-1, 1):
                    s = np.roll(self.grid, direction, axis=axis)
                    shifted.append(s)
            neighbor_avg = np.mean(shifted, axis=0)
            self.grid = self.grid * (1 - self._diffusion_rate) + neighbor_avg * self._diffusion_rate

    def get_total_vitality(self) -> float:
        """总活力"""
        return float(np.sum(self.grid))

    def get_hotspots(self, threshold: float = 30.0) -> List[Tuple[int, int, int, float]]:
        """获取活力热点（高活力区）"""
        indices = np.argwhere(self.grid >= threshold)
        return [(int(i[0]), int(i[1]), int(i[2]), float(self.grid[i[0], i[1], i[2]]))
                for i in indices]

    def get_gradient(self, x: int, y: int, z: int) -> Tuple[float, float, float]:
        """获取某点的活力梯度（指向高活力方向）"""
        gx = self.get_vitality(x+1, y, z) - self.get_vitality(x-1, y, z)
        gy = self.get_vitality(x, y+1, z) - self.get_vitality(x, y-1, z)
        gz = self.get_vitality(x, y, z+1) - self.get_vitality(x, y, z-1)
        return (gx, gy, gz)


class EmergenceEngine:
    """
    涌现引擎。
    
    根据活力值判断是否出现涌现行为。
    活力越高，涌现越复杂。
    """
    
    def __init__(
        self,
        replicate_threshold: float = 30.0,  # 自繁殖阈值
        migrate_threshold: float = 50.0,    # 自迁移阈值
        cluster_threshold: float = 70.0,    # 自组队阈值
        create_threshold: float = 90.0,     # 自创造阈值
    ):
        self.replicate_threshold = replicate_threshold
        self.migrate_threshold = migrate_threshold
        self.cluster_threshold = cluster_threshold
        self.create_threshold = create_threshold

    def check_emergence(self, vitality: float) -> EmergenceType:
        """检查活力值对应的涌现行为"""
        if vitality >= self.create_threshold:
            return EmergenceType.SELF_CREATE
        if vitality >= self.cluster_threshold:
            return EmergenceType.SELF_CLUSTER
        if vitality >= self.migrate_threshold:
            return EmergenceType.SELF_MIGRATE
        if vitality >= self.replicate_threshold:
            return EmergenceType.SELF_REPLICATE
        return EmergenceType.NONE

    def apply_emergence(
        self,
        cell: VitalityCell,
        field: VitalityField,
        existing_samples: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        对一个活力单元应用涌现行为。
        
        返回涌现事件列表。
        """
        events = []
        emergence = self.check_emergence(cell.vitality)
        cell.emergence = emergence

        if emergence == EmergenceType.NONE:
            return events

        if emergence == EmergenceType.SELF_REPLICATE:
            # 自繁殖：采样点分裂为2个
            events.append({
                "type": "self_replicate",
                "location": (cell.grid_x, cell.grid_y, cell.grid_z),
                "vitality": cell.vitality,
                "action": "sample_point_split",
                "description": "活力充足，采样点自繁殖分裂",
            })

        elif emergence == EmergenceType.SELF_MIGRATE:
            # 自迁移：沿活力梯度移动
            gx, gy, gz = field.get_gradient(cell.grid_x, cell.grid_y, cell.grid_z)
            # 归一化为方向
            mag = math.sqrt(gx*gx + gy*gy + gz*gz) + 1e-8
            direction = (gx/mag, gy/mag, gz/mag)
            events.append({
                "type": "self_migrate",
                "location": (cell.grid_x, cell.grid_y, cell.grid_z),
                "vitality": cell.vitality,
                "direction": direction,
                "description": "采样点沿活力梯度自迁移",
            })

        elif emergence == EmergenceType.SELF_CLUSTER:
            # 自组队：聚合为簇
            events.append({
                "type": "self_cluster",
                "location": (cell.grid_x, cell.grid_y, cell.grid_z),
                "vitality": cell.vitality,
                "action": "form_cluster",
                "description": "高活力区采样点自发组队聚合",
            })

        elif emergence == EmergenceType.SELF_CREATE:
            # 自创造：涌现新参数包
            new_params = {
                "emerged_vitality": cell.vitality,
                "emerged_location_x": float(cell.grid_x),
                "emerged_location_y": float(cell.grid_y),
                "emerged_location_z": float(cell.grid_z),
                "emerged_timestamp": time.time(),
                "emerged_seed": int(hashlib.md5(
                    f"{cell.grid_x}{cell.grid_y}{cell.grid_z}{cell.vitality}".encode()
                ).hexdigest(), 16) % 1000000,
            }
            events.append({
                "type": "self_create",
                "location": (cell.grid_x, cell.grid_y, cell.grid_z),
                "vitality": cell.vitality,
                "action": "create_parameter_pack",
                "new_params": new_params,
                "description": "活力极高，自发涌现新参数包",
            })

        return events


class VitalitySystem:
    """
    活力系统。
    
    整合融合反应器、活力场、涌现引擎。
    实现采样点→电+参数→活力→涌现→反哺 的完整闭环。
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (16, 16, 16),
        reactor: Optional[FusionReactor] = None,
        emergence: Optional[EmergenceEngine] = None,
    ):
        self.reactor = reactor or FusionReactor()
        self.field = VitalityField(grid_size=grid_size)
        self.emergence = emergence or EmergenceEngine()
        self.grid_size = grid_size
        self.total_emergence_events = 0
        self.emergence_log: List[Dict[str, Any]] = []

    def inject_energy(self, x: int, y: int, z: int, energy: float):
        """注入资源（虚拟电）"""
        key = (x, y, z)
        if key not in self.field.cells:
            self.field.cells[key] = VitalityCell(grid_x=x, grid_y=y, grid_z=z)
        self.field.cells[key].energy += energy

    def inject_info(self, x: int, y: int, z: int, info: float):
        """注入信息势（参数）"""
        key = (x, y, z)
        if key not in self.field.cells:
            self.field.cells[key] = VitalityCell(grid_x=x, grid_y=y, grid_z=z)
        self.field.cells[key].info_potential += info

    def fuse_all(self):
        """对所有单元执行融合"""
        self.reactor.batch_fuse(list(self.field.cells.values()))
        # 更新场
        for cell in self.field.cells.values():
            self.field.grid[cell.grid_x, cell.grid_y, cell.grid_z] = cell.vitality

    def run_emergence(self) -> List[Dict[str, Any]]:
        """运行涌现检查，返回涌现事件"""
        all_events = []
        for cell in self.field.cells.values():
            events = self.emergence.apply_emergence(cell, self.field)
            all_events.extend(events)
        
        self.total_emergence_events += len(all_events)
        self.emergence_log.extend(all_events)
        if len(self.emergence_log) > 1000:
            self.emergence_log = self.emergence_log[-500:]
        return all_events

    def evolve_step(self, diffuse_steps: int = 1) -> Dict[str, Any]:
        """
        执行一个演化步骤。
        
        1. 融合
        2. 扩散
        3. 涌现
        """
        # 1. 融合
        self.fuse_all()
        # 2. 扩散
        self.field.diffuse(steps=diffuse_steps)
        # 同步扩散后的值到 cells
        for cell in self.field.cells.values():
            cell.vitality = self.field.get_vitality(cell.grid_x, cell.grid_y, cell.grid_z)
        # 3. 涌现
        events = self.run_emergence()
        
        return {
            "total_vitality": self.field.get_total_vitality(),
            "active_cells": len(self.field.cells),
            "emergence_events": len(events),
            "events": events,
            "emergence_breakdown": self._count_emergence_types(events),
        }

    def _count_emergence_types(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        breakdown = {}
        for e in events:
            t = e.get("type", "unknown")
            breakdown[t] = breakdown.get(t, 0) + 1
        return breakdown

    def feed_from_samples(self, samples: List, info_scale: float = 1.0, energy_scale: float = 0.001):
        """
        从采样点喂入资源和信息。
        
        采样点同时提供：
        - 资源：w（能量维）→ 虚拟电
        - 信息：entropy（熵维）→ 参数势
        
        energy_scale: 能量缩放因子（采样点能量大，需缩放到合理范围）
        info_scale: 信息缩放因子
        """
        gs = self.grid_size
        for s in samples:
            # 映射到网格坐标
            x = int((s.x % 1.0) * gs[0]) if hasattr(s, "x") else 0
            y = int((s.y % 1.0) * gs[1]) if hasattr(s, "y") else 0
            z = int((s.z % 1.0) * gs[2]) if hasattr(s, "z") else 0
            x = max(0, min(gs[0]-1, x))
            y = max(0, min(gs[1]-1, y))
            z = max(0, min(gs[2]-1, z))
            
            energy = abs(getattr(s, "w", 0.0)) * energy_scale
            info = abs(getattr(s, "entropy", 0.0)) * info_scale
            self.inject_energy(x, y, z, energy)
            self.inject_info(x, y, z, info)

    def statistics(self) -> Dict[str, Any]:
        """统计"""
        cells = list(self.field.cells.values())
        vitalities = [c.vitality for c in cells] if cells else [0]
        energies = [c.energy for c in cells] if cells else [0]
        infos = [c.info_potential for c in cells] if cells else [0]
        
        return {
            "grid_size": list(self.grid_size),
            "active_cells": len(cells),
            "total_energy": round(sum(energies), 2),
            "total_info": round(sum(infos), 4),
            "total_vitality": round(sum(vitalities), 2),
            "avg_vitality": round(np.mean(vitalities), 2) if cells else 0,
            "max_vitality": round(max(vitalities), 2) if cells else 0,
            "hotspots": len(self.field.get_hotspots()),
            "total_emergence_events": self.total_emergence_events,
            "emergence_types": self._count_emergence_types(self.emergence_log),
        }

    def visualize(self) -> str:
        """可视化"""
        lines = []
        lines.append("=" * 60)
        lines.append("VITALITY FIELD (智能电 / 活力场)")
        lines.append("=" * 60)
        
        stats = self.statistics()
        lines.append(f"Grid: {stats['grid_size']}")
        lines.append(f"Active cells: {stats['active_cells']}")
        lines.append(f"Total energy (资源): {stats['total_energy']}")
        lines.append(f"Total info (信息势): {stats['total_info']}")
        lines.append(f"Total vitality (活力): {stats['total_vitality']}")
        lines.append(f"Avg vitality: {stats['avg_vitality']} | Max: {stats['max_vitality']}")
        lines.append(f"Hotspots (>30): {stats['hotspots']}")
        lines.append(f"Emergence events: {stats['total_emergence_events']}")
        if stats["emergence_types"]:
            lines.append("Emergence breakdown:")
            for t, c in stats["emergence_types"].items():
                lines.append(f"  {t}: {c}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
