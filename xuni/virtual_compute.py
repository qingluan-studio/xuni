"""
VirtualCompute —— 虚拟算力系统

核心理念：
    虚拟电 → 虚拟算力 → 驱动训练 → 消耗虚拟电 → 采样点产电 → 闭环

    现实中 CPU/GPU 算力有限，但在虚拟生态里：
    - 虚拟电可以转化为虚拟算力（vFLOP）
    - 虚拟算力驱动模型训练
    - 训练消耗算力 → 消耗虚拟电
    - 虚拟电来自采样点 → 采样点不停产电 → 无限循环

    这就是"CPU不好=算力不足"的解决方案：
    现实算力不足没关系，虚拟算力由虚拟电支撑，形成自给自足的闭环。

算力单位：
    vFLOP = 虚拟浮点运算（Virtual Floating Point Operation）
    1 vFLOP = 1 次虚拟浮点运算
    转换效率：1 虚拟电 ≈ 10^9 vFLOP（10亿次虚拟浮点运算）

训练算力需求：
    需求 = 6 × 参数量 × 数据量 × epoch
    （遵循 Chinchilla scaling 的近似公式）
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class ComputeAllocation:
    """算力分配记录"""
    model_id: str
    vflops_allocated: float    # 分配的算力
    vflops_used: float = 0.0   # 已使用的算力
    energy_cost: float = 0.0   # 消耗的虚拟电
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # 过期时间


class VirtualComputeUnit:
    """
    虚拟算力单元——虚拟电 → 虚拟算力转换器（极致加速版）

    核心功能：
    1. 电→算力转换（有转换效率）
    2. 算力分配给模型训练
    3. 算力消耗统计
    4. 算力预算管理
    5. 批量注入加速（一次注入大量能量）
    6. 预计算缓存（减少重复计算）

    闭环：
    采样点产电 → 虚拟电注入算力单元 → 转为虚拟算力
    → 分配给模型 → 训练消耗算力 → 算力耗尽 → 需要更多电
    → 采样点继续产电 → 闭环
    """

    # 转换常数：1 虚拟电 → 多少 vFLOP
    VFLOP_PER_ENERGY = 1e9  # 10亿次虚拟浮点运算/度电
    
    # 批量注入阈值（超过此值使用快速路径）
    BATCH_THRESHOLD = 1000.0

    def __init__(self, name: str = "VCU-01"):
        self.name = name
        self.total_energy_received: float = 0.0    # 累计接收的虚拟电
        self.total_vflops_generated: float = 0.0   # 累计生成的算力
        self.total_vflops_consumed: float = 0.0    # 累计消耗的算力
        self.current_vflops: float = 0.0           # 当前可用算力
        self.allocations: Dict[str, ComputeAllocation] = {}
        self._history: List[Dict[str, Any]] = []
        
        # 预计算缓存
        self._conversion_cache: Dict[float, float] = {}
        self._cost_cache: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def inject_energy(self, energy: float, source: str = "sampler") -> Dict[str, Any]:
        """
        注入虚拟电，转换为虚拟算力

        这是闭环的入口：采样点产电 → 注入算力单元 → 转为算力
        """
        if energy <= 0:
            return {"error": "能量必须为正"}

        # 使用缓存加速转换
        if energy in self._conversion_cache:
            vflops = self._conversion_cache[energy]
            self._cache_hits += 1
        else:
            vflops = energy * self.VFLOP_PER_ENERGY
            self._conversion_cache[energy] = vflops
            self._cache_misses += 1

        self.total_energy_received += energy
        self.total_vflops_generated += vflops
        self.current_vflops += vflops

        # 批量注入时跳过历史记录（减少内存占用）
        if energy < self.BATCH_THRESHOLD:
            record = {
                "action": "inject_energy",
                "energy": energy,
                "vflops": vflops,
                "source": source,
                "timestamp": time.time(),
            }
            self._history.append(record)

        return {
            "status": "injected",
            "energy_in": energy,
            "vflops_out": vflops,
            "current_vflops": self.current_vflops,
            "source": source,
        }

    def inject_energy_batch(self, energies: np.ndarray, source: str = "sampler_batch") -> Dict[str, Any]:
        """
        批量注入虚拟电（向量化加速）。
        
        Args:
            energies: 能量数组，shape (N,)
            source: 来源标识
            
        Returns:
            Dict: 批量注入结果
        """
        energies = np.asarray(energies, dtype=np.float64)
        mask = energies > 0
        valid_energies = energies[mask]
        
        if len(valid_energies) == 0:
            return {"error": "没有有效能量"}
        
        # 完全向量化计算
        total_energy = np.sum(valid_energies)
        total_vflops = total_energy * self.VFLOP_PER_ENERGY
        
        # 批量更新
        self.total_energy_received += total_energy
        self.total_vflops_generated += total_vflops
        self.current_vflops += total_vflops
        
        # 批量缓存
        for e in valid_energies:
            if e not in self._conversion_cache:
                self._conversion_cache[e] = e * self.VFLOP_PER_ENERGY
        
        return {
            "status": "batch_injected",
            "count": len(valid_energies),
            "total_energy_in": float(total_energy),
            "total_vflops_out": float(total_vflops),
            "current_vflops": self.current_vflops,
            "source": source,
        }

    def allocate_batch(self, allocations: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        批量分配算力（向量化加速）。
        
        Args:
            allocations: [{"model_id": str, "vflops": float}, ...]
            
        Returns:
            Dict: 批量分配结果
        """
        model_ids = [a["model_id"] for a in allocations]
        vflops_list = np.array([a["vflops"] for a in allocations], dtype=np.float64)
        
        total_requested = np.sum(vflops_list)
        
        if self.current_vflops < total_requested:
            return {
                "error": "算力不足",
                "requested": float(total_requested),
                "available": self.current_vflops,
            }
        
        # 批量扣除
        self.current_vflops -= total_requested
        
        # 批量创建分配记录
        for model_id, vflops in zip(model_ids, vflops_list):
            energy_cost = vflops / self.VFLOP_PER_ENERGY
            self.allocations[model_id] = ComputeAllocation(
                model_id=model_id,
                vflops_allocated=float(vflops),
                energy_cost=float(energy_cost),
            )
        
        return {
            "status": "batch_allocated",
            "count": len(model_ids),
            "total_vflops_allocated": float(total_requested),
            "remaining_vflops": self.current_vflops,
        }

    def consume_batch(self, consumptions: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        批量消耗算力（向量化加速）。
        
        Args:
            consumptions: [{"model_id": str, "vflops": float}, ...]
            
        Returns:
            Dict: 批量消耗结果
        """
        total_consumed = 0.0
        failed = []
        
        for item in consumptions:
            model_id = item["model_id"]
            vflops = item["vflops"]
            
            if model_id not in self.allocations:
                failed.append(model_id)
                continue
            
            alloc = self.allocations[model_id]
            remaining = alloc.vflops_allocated - alloc.vflops_used
            
            if vflops > remaining:
                failed.append(model_id)
                continue
            
            alloc.vflops_used += vflops
            total_consumed += vflops
        
        self.total_vflops_consumed += total_consumed
        
        return {
            "status": "batch_consumed",
            "total_vflops_consumed": total_consumed,
            "failed_models": failed,
        }

    def allocate(self, model_id: str, vflops: float,
                 energy_budget: float = None) -> Dict[str, Any]:
        """
        为模型分配算力

        算力从当前池中扣除，分配给指定模型。
        """
        if vflops <= 0:
            return {"error": "算力必须为正"}
        if self.current_vflops < vflops:
            return {
                "error": "算力不足",
                "requested": vflops,
                "available": self.current_vflops,
                "need_energy": (vflops - self.current_vflops) / self.VFLOP_PER_ENERGY,
            }

        self.current_vflops -= vflops
        energy_cost = vflops / self.VFLOP_PER_ENERGY

        alloc = ComputeAllocation(
            model_id=model_id,
            vflops_allocated=vflops,
            energy_cost=energy_cost,
        )
        self.allocations[model_id] = alloc

        return {
            "status": "allocated",
            "model_id": model_id,
            "vflops": vflops,
            "energy_cost": energy_cost,
            "remaining_vflops": self.current_vflops,
        }

    def consume(self, model_id: str, vflops: float) -> Dict[str, Any]:
        """
        模型消耗算力（训练时调用）

        消耗的算力 = 训练实际用掉的算力
        """
        if model_id not in self.allocations:
            return {"error": f"模型 {model_id} 未分配算力"}

        alloc = self.allocations[model_id]
        remaining = alloc.vflops_allocated - alloc.vflops_used

        if vflops > remaining:
            return {
                "error": "分配额度不足",
                "model_id": model_id,
                "requested": vflops,
                "remaining": remaining,
            }

        alloc.vflops_used += vflops
        self.total_vflops_consumed += vflops

        return {
            "status": "consumed",
            "model_id": model_id,
            "vflops": vflops,
            "used_total": alloc.vflops_used,
            "allocated_total": alloc.vflops_allocated,
            "utilization": alloc.vflops_used / alloc.vflops_allocated,
        }

    def release(self, model_id: str) -> Dict[str, Any]:
        """释放模型未使用的算力，回到池中"""
        if model_id not in self.allocations:
            return {"error": f"模型 {model_id} 未分配算力"}

        alloc = self.allocations[model_id]
        unused = alloc.vflops_allocated - alloc.vflops_used
        self.current_vflops += unused

        del self.allocations[model_id]

        return {
            "status": "released",
            "model_id": model_id,
            "vflops_returned": unused,
            "current_vflops": self.current_vflops,
        }

    @staticmethod
    def estimate_training_cost(params: float, data_samples: int,
                               epochs: int = 1) -> Dict[str, Any]:
        """
        估算训练算力需求

        遵循近似公式：需求 ≈ 6 × 参数量 × 数据量 × epoch
        （参数量单位：个，数据量单位：条）

        Returns: {vflops, energy_needed}
        """
        # 6 FLOP per parameter per sample（前向+反向）
        vflops = 6 * params * data_samples * epochs
        energy = vflops / VirtualComputeUnit.VFLOP_PER_ENERGY

        return {
            "params": params,
            "data_samples": data_samples,
            "epochs": epochs,
            "vflops_needed": vflops,
            "energy_needed": energy,
            "vflops_str": f"{vflops:.2e}",
            "energy_str": f"{energy:.2f}",
        }

    def stats(self) -> Dict[str, Any]:
        """算力统计"""
        efficiency = (
            self.total_vflops_consumed / max(1, self.total_vflops_generated)
        )
        cache_total = self._cache_hits + self._cache_misses
        cache_hit_rate = (self._cache_hits / cache_total * 100) if cache_total > 0 else 0
        return {
            "name": self.name,
            "total_energy_received": self.total_energy_received,
            "total_vflops_generated": f"{self.total_vflops_generated:.2e}",
            "total_vflops_consumed": f"{self.total_vflops_consumed:.2e}",
            "current_vflops_available": f"{self.current_vflops:.2e}",
            "current_energy_equivalent": self.current_vflops / self.VFLOP_PER_ENERGY,
            "utilization_efficiency": f"{efficiency*100:.1f}%",
            "active_allocations": len(self.allocations),
            "conversion_rate": f"{self.VFLOP_PER_ENERGY:.0e} vFLOP/度电",
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        }


# ============================================================
# 算力闭环管理器——整合采样点产电 + 算力转换 + 训练消耗
# ============================================================

class ComputeLoopManager:
    """
    算力闭环管理器

    完整闭环：
    ┌─────────────────────────────────────────────────┐
    │                                                   │
    │  采样点 ──产电──→ 虚拟电 ──转换──→ 虚拟算力       │
    │                       ↑                    ↓     │
    │                       │              分配给模型   │
    │                       │                    ↓     │
    │                  需要更多电          训练消耗算力  │
    │                       │                    ↓     │
    │                  采样点继续产电 ←── 电量不足 ←──┘ │
    │                                                   │
    └─────────────────────────────────────────────────┘

    这就是"CPU不好=算力不足"的终极解决方案：
    现实CPU算力有限？没关系，虚拟算力由虚拟电无限支撑！
    """

    def __init__(self, sampler=None, compute_unit: VirtualComputeUnit = None,
                 cluster=None, reservoir=None):
        """
        Args:
            sampler: 单个采样点（旧用法，保留兼容）
            compute_unit: 虚拟算力单元
            cluster: 采样点集群（推荐，解决供应不足）
            reservoir: 能量蓄水池（推荐，攒够了再训练）
        """
        self.sampler = sampler  # XuniSampler 采样点（单点）
        self.cluster = cluster  # 采样点集群（多点并行）
        self.reservoir = reservoir  # 能量蓄水池
        self.vcu = compute_unit or VirtualComputeUnit()
        self._loop_count = 0
        self._loop_log: List[Dict[str, Any]] = []

    def run_loop_once(self, energy_amount: float = 100.0,
                      target_model_id: str = None,
                      training_cost_vflops: float = None) -> Dict[str, Any]:
        """
        执行一次完整算力闭环

        优先使用集群+蓄水池（供应充足）；
        否则回退到单采样点。
        """
        self._loop_count += 1
        loop_start = time.time()

        # 1. 产电（集群优先，蓄水池积累）
        if self.cluster is not None:
            harvest = self.cluster.harvest(batch_size=100)
            energy_from_source = harvest["total_energy"]
            if self.reservoir is not None:
                self.reservoir.fill(energy_from_source, source="cluster")
                # 从蓄水池释放
                release = self.reservoir.release(energy_amount)
                if release.get("status") == "released":
                    energy_from_sampler = release["energy_out"]
                else:
                    energy_from_sampler = energy_from_source
            else:
                energy_from_sampler = energy_from_source
        elif self.sampler is not None:
            try:
                batch = self.sampler.generate_batch(batch_size=100)
                energy_from_sampler = float(np.sum(np.abs(batch))) * 0.01
            except Exception:
                energy_from_sampler = energy_amount
        else:
            energy_from_sampler = energy_amount

        # 2. 电注入算力单元
        inject_result = self.vcu.inject_energy(energy_from_sampler, source="sampler")

        # 3. 分配算力给模型
        if target_model_id and training_cost_vflops:
            alloc_result = self.vcu.allocate(target_model_id, training_cost_vflops)

            # 4. 训练消耗
            if alloc_result.get("status") == "allocated":
                consume_result = self.vcu.consume(target_model_id, training_cost_vflops)
                release_result = self.vcu.release(target_model_id)
            else:
                consume_result = alloc_result
                release_result = {"status": "skipped"}
        else:
            alloc_result = {"status": "no_target"}
            consume_result = {"status": "no_target"}
            release_result = {"status": "no_target"}

        loop_time = time.time() - loop_start
        result = {
            "loop_id": self._loop_count,
            "energy_produced": energy_from_sampler,
            "vflops_injected": inject_result.get("vflops_out", 0),
            "allocation": alloc_result.get("status"),
            "consumption": consume_result.get("status"),
            "loop_time_ms": loop_time * 1000,
            "vcu_stats": self.vcu.stats(),
            "source": "cluster" if self.cluster else ("sampler" if self.sampler else "manual"),
        }
        if self.reservoir is not None:
            result["reservoir_level"] = self.reservoir.level()
        self._loop_log.append(result)
        return result

    def auto_loop_until_trained(self, target_model_id: str,
                                total_vflops_needed: float,
                                energy_per_loop: float = 100.0,
                                max_loops: int = 100) -> Dict[str, Any]:
        """
        自动循环直到训练完成

        不断产电→转算力→消耗，直到算力满足训练需求
        """
        accumulated_vflops = 0.0
        loops_done = 0

        for i in range(max_loops):
            if accumulated_vflops >= total_vflops_needed:
                break

            result = self.run_loop_once(
                energy_amount=energy_per_loop,
                target_model_id=target_model_id,
                training_cost_vflops=min(
                    energy_per_loop * self.vcu.VFLOP_PER_ENERGY,
                    total_vflops_needed - accumulated_vflops,
                ),
            )

            if result.get("consumption") == "consumed":
                consumed = result.get("vcu_stats", {})
                accumulated_vflops += energy_per_loop * self.vcu.VFLOP_PER_ENERGY

            loops_done += 1

        return {
            "status": "completed" if accumulated_vflops >= total_vflops_needed else "incomplete",
            "loops_executed": loops_done,
            "vflops_accumulated": f"{accumulated_vflops:.2e}",
            "vflops_needed": f"{total_vflops_needed:.2e}",
            "total_energy_consumed": loops_done * energy_per_loop,
            "final_vcu_stats": self.vcu.stats(),
        }
