"""
XuniParameter —— 参数系统

核心理念：
  参数是采样点和模型之间的桥梁。
  采样点产生参数 → 参数包 → 注入模型 / 交易 / 导入导出。

参数的来源：
  1. 采样点 → 抽取位置/速度/能量/熵 → 参数包
  2. 模型   → 抽取内部状态 → 参数包（模型导出）
  3. 场能量 → 抽取总能量/梯度/密度 → 参数包

参数包的用途：
  1. 注入模型：改变模型行为（强化/变异）
  2. 导入导出：JSON 序列化，跨实例转移
  3. 交易市场：AI 之间用能量买卖
  4. 质量评级：评估参数包的价值

一个 ParameterPack 就是一组参数，是模型的"本质"。
"""

import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Tuple

import numpy as np


@dataclass
class ParameterPack:
    """
    参数包。
    
    一组可交易、可序列化、可注入模型的参数。
    """
    pack_id: str
    source: str                          # 来源："sampler" / "model" / "field" / "ai"
    params: Dict[str, float]             # 参数字典
    vector: Optional[np.ndarray] = None  # 参数向量（用于相似度计算）
    quality: float = 0.0                 # 质量评分 (0-100)
    created_at: float = 0.0
    origin_info: Dict[str, Any] = field(default_factory=dict)  # 来源详情
    injection_history: List[Dict[str, Any]] = field(default_factory=list)  # 注入历史

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.vector is None and self.params:
            self.vector = np.array(list(self.params.values()), dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "pack_id": self.pack_id,
            "source": self.source,
            "params": self.params,
            "quality": self.quality,
            "created_at": self.created_at,
            "origin_info": self.origin_info,
            "injection_history": self.injection_history,
        }

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def save(self, filepath: str) -> bool:
        """保存到文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.to_json())
            return True
        except Exception:
            return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterPack":
        """从字典反序列化"""
        pack = cls(
            pack_id=data["pack_id"],
            source=data["source"],
            params={k: float(v) for k, v in data["params"].items()},
            quality=data.get("quality", 0.0),
            created_at=data.get("created_at", time.time()),
            origin_info=data.get("origin_info", {}),
            injection_history=data.get("injection_history", []),
        )
        return pack

    @classmethod
    def from_json(cls, json_str: str) -> "ParameterPack":
        """从JSON字符串反序列化"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, filepath: str) -> Optional["ParameterPack"]:
        """从文件加载"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return cls.from_json(f.read())
        except Exception:
            return None

    def similarity(self, other: "ParameterPack") -> float:
        """计算与另一个参数包的相似度 (0-1)"""
        if self.vector is None or other.vector is None:
            return 0.0
        # 余弦相似度
        a, b = self.vector, other.vector
        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a, b = a[:min_len], b[:min_len]
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

    def summary(self) -> str:
        """摘要信息"""
        return (f"ParameterPack({self.pack_id}) "
                f"source={self.source} "
                f"params={len(self.params)} "
                f"quality={self.quality:.1f}")


class ParameterExtractor:
    """
    参数抽取器。
    
    从采样点、模型、场能量中抽取参数包。
    """

    @staticmethod
    def from_samples(samples, count: int = 100) -> ParameterPack:
        """
        从采样点抽取参数包。
        
        samples: SamplePoint 列表或数值数组
        """
        # 取前 count 个
        data = list(samples)[:count]
        n = len(data)
        if n == 0:
            data = list(samples)
            n = len(data)
        
        params = {}
        # 尝试按对象属性抽取
        if hasattr(data[0], "x"):
            xs = [s.x for s in data]
            ys = [s.y for s in data]
            zs = [s.z for s in data]
            ws = [getattr(s, "w", 0.0) for s in data]
            cs = [getattr(s, "charge", 1.0) for s in data]
            es = [getattr(s, "entropy", 0.0) for s in data]
            params = {
                "mean_x": float(np.mean(xs)), "mean_y": float(np.mean(ys)), "mean_z": float(np.mean(zs)),
                "std_x": float(np.std(xs)), "std_y": float(np.std(ys)), "std_z": float(np.std(zs)),
                "mean_energy": float(np.mean(ws)), "std_energy": float(np.std(ws)),
                "mean_charge": float(np.mean(cs)), "mean_entropy": float(np.mean(es)),
                "std_entropy": float(np.std(es)),
                "sample_count": float(n),
            }
        else:
            # 数值数组
            arr = np.array(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            ncols = arr.shape[1]
            params = {f"mean_{i}": float(np.mean(arr[:, i])) for i in range(min(ncols, 6))}
            for i in range(min(ncols, 6)):
                params[f"std_{i}"] = float(np.std(arr[:, i]))
            params["sample_count"] = float(n)
            params["dimensionality"] = float(arr.shape[1])
        
        pack_id = f"pk-sampler-{int(time.time()*1000)}-{hash(str(params)) % 10000:04d}"
        pack = ParameterPack(
            pack_id=pack_id,
            source="sampler",
            params=params,
            origin_info={"sample_count": n, "dimensions": len(params)},
        )
        pack.quality = ParameterExtractor._compute_quality(pack)
        return pack

    @staticmethod
    def from_model(model) -> ParameterPack:
        """
        从模型抽取参数包（模型导出）。
        
        模型的本质就是一组参数。
        """
        params = {
            "energy_requirement": float(model.energy_requirement),
            "energy_buffer": float(model._energy_buffer),
            "training_progress": float(model.training_progress),
            "total_calls": float(model.stats.total_calls),
            "total_energy_consumed": float(model.stats.total_energy_consumed),
            "avg_latency": float(model.stats.avg_latency_ms),
            "model_type_hash": float(hash(model.model_type.name) % 1000),
            "capabilities_count": float(len(model.capabilities)),
            "rng_seed": float(int(hashlib.md5(model.model_id.encode()).hexdigest(), 16) % 1000000),
        }
        
        # 模型特定参数
        if hasattr(model, "max_length"):
            params["max_length"] = float(model.max_length)
        if hasattr(model, "steps"):
            params["steps"] = float(model.steps)
        if hasattr(model, "horizon"):
            params["horizon"] = float(model.horizon)
        if hasattr(model, "encode_dim"):
            params["encode_dim"] = float(model.encode_dim)
        if hasattr(model, "personality"):
            params["personality_hash"] = float(hash(model.personality) % 1000)
        
        pack_id = f"pk-model-{model.model_id}-{int(time.time()*1000) % 100000:05d}"
        pack = ParameterPack(
            pack_id=pack_id,
            source="model",
            params=params,
            origin_info={
                "model_id": model.model_id,
                "model_type": model.model_type.name,
                "owner": model.owner,
                "training_state": model.training_state.name,
            },
        )
        pack.quality = ParameterExtractor._compute_quality(pack)
        return pack

    @staticmethod
    def from_field(field_obj) -> ParameterPack:
        """从场能量抽取参数包"""
        params = {
            "total_energy": float(field_obj.get_total_energy()),
        }
        
        # 尝试获取更多场信息
        if hasattr(field_obj, "grid_size"):
            gs = field_obj.grid_size
            if isinstance(gs, (tuple, list)):
                params["grid_x"] = float(gs[0])
                params["grid_y"] = float(gs[1])
                params["grid_z"] = float(gs[2])
            else:
                params["grid_size"] = float(gs)
        
        if hasattr(field_obj, "_charge_density"):
            cd = field_obj._charge_density
            if hasattr(cd, "mean"):
                params["charge_mean"] = float(np.mean(cd))
                params["charge_std"] = float(np.std(cd))
                params["charge_max"] = float(np.max(cd))
        
        if hasattr(field_obj, "_potential"):
            pot = field_obj._potential
            if hasattr(pot, "mean"):
                params["potential_mean"] = float(np.mean(pot))
                params["potential_std"] = float(np.std(pot))
        
        pack_id = f"pk-field-{int(time.time()*1000)}-{hash(str(params)) % 10000:04d}"
        pack = ParameterPack(
            pack_id=pack_id,
            source="field",
            params=params,
            origin_info={"field_type": type(field_obj).__name__},
        )
        pack.quality = ParameterExtractor._compute_quality(pack)
        return pack

    @staticmethod
    def _compute_quality(pack: ParameterPack) -> float:
        """
        计算参数包质量评分 (0-100)。
        
        基于：
        1. 参数维度丰富度
        2. 参数熵（多样性）
        3. 参数幅值（信息量）
        """
        if pack.vector is None:
            return 0.0
        
        v = pack.vector
        # 1. 维度丰富度（越多越好，上限100）
        dim_score = min(100.0, len(v) * 10)
        
        # 2. 熵（多样性）
        if len(v) > 1:
            # 归一化后计算熵
            v_norm = np.abs(v) / (np.max(np.abs(v)) + 1e-8)
            entropy = -np.sum(v_norm * np.log(v_norm + 1e-8))
            entropy_score = min(100.0, entropy * 20)
        else:
            entropy_score = 0.0
        
        # 3. 信息量（非零比例）
        nonzero_ratio = np.count_nonzero(np.abs(v) > 1e-6) / len(v)
        info_score = nonzero_ratio * 100
        
        return round((dim_score + entropy_score + info_score) / 3, 1)


class ParameterInjector:
    """
    参数注入器。
    
    将参数包注入模型，改变模型行为。
    """

    @staticmethod
    def inject(model, pack: ParameterPack) -> bool:
        """
        将参数包注入模型。
        
        注入会影响：
        1. 模型的随机种子（影响输出风格）
        2. 能量需求（影响效率）
        3. 输出长度/步数等（影响输出）
        """
        if pack.source not in ("sampler", "model", "field", "ai"):
            return False
        
        injected = []
        
        # 1. 注入随机种子（改变输出风格）
        if "rng_seed" in pack.params:
            new_seed = int(pack.params["rng_seed"]) % 1000000
            model._rng = np.random.default_rng(new_seed)
            injected.append("rng_seed")
        elif pack.vector is not None:
            # 用参数向量的哈希作为种子
            new_seed = int(hashlib.md5(pack.vector.tobytes()).hexdigest(), 16) % 1000000
            model._rng = np.random.default_rng(new_seed)
            injected.append("rng_seed(via_vector)")
        
        # 2. 注入能量需求
        if "energy_requirement" in pack.params:
            model.energy_requirement = max(1.0, pack.params["energy_requirement"])
            injected.append("energy_requirement")
        
        # 3. 注入模型特定参数
        if hasattr(model, "max_length") and "max_length" in pack.params:
            model.max_length = int(pack.params["max_length"])
            injected.append("max_length")
        
        if hasattr(model, "steps") and "steps" in pack.params:
            model.steps = int(pack.params["steps"])
            injected.append("steps")
        
        if hasattr(model, "horizon") and "horizon" in pack.params:
            model.horizon = int(pack.params["horizon"])
            injected.append("horizon")
        
        if hasattr(model, "encode_dim") and "encode_dim" in pack.params:
            model.encode_dim = int(pack.params["encode_dim"])
            injected.append("encode_dim")
        
        # 4. 注入能量缓冲
        if "energy_buffer" in pack.params:
            model._energy_buffer += pack.params["energy_buffer"]
            injected.append("energy_buffer")
        
        # 记录注入历史
        pack.injection_history.append({
            "model_id": model.model_id,
            "injected_params": injected,
            "timestamp": time.time(),
        })
        
        return len(injected) > 0

    @staticmethod
    def merge_packs(packs: List[ParameterPack]) -> ParameterPack:
        """合并多个参数包为一个"""
        if not packs:
            raise ValueError("Cannot merge empty pack list")
        
        merged_params = {}
        for pack in packs:
            for k, v in pack.params.items():
                if k in merged_params:
                    # 取平均
                    merged_params[k] = (merged_params[k] + v) / 2
                else:
                    merged_params[k] = v
        
        pack_id = f"pk-merged-{int(time.time()*1000)}-{hash(str(merged_params)) % 10000:04d}"
        merged = ParameterPack(
            pack_id=pack_id,
            source="merged",
            params=merged_params,
            origin_info={
                "source_packs": [p.pack_id for p in packs],
                "merge_count": len(packs),
            },
        )
        merged.quality = ParameterExtractor._compute_quality(merged)
        return merged


# ============================================================
# 参数训练器——参数直接驱动模型训练（无需算力，最短路径）
# ============================================================

class ParameterTrainer:
    """
    参数训练器——用参数包直接训练模型

    核心理念：
        采样点产参数 → 参数包 → 直接注入训练 → 模型能力提升

        这是最短路径，不需要绕道"电→算力→训练"。
        参数本身就是模型权重的"原料"，直接喂就能训练。

    三条训练路径对比：
        1. 参数训练（本类）：参数→注入→进度提升   [直接、快]
        2. 算力训练：电→算力→替代物→权重更新      [间接、深]
        3. 融合训练：电+参数→活力→涌现训练        [最强]

    训练效果由参数包质量决定：
        - quality 高 → 训练进度大幅提升
        - quality 低 → 训练进度小幅提升
        - 多个参数包叠加 → 进度累积
    """

    # 质量到训练增量的映射系数
    # quality 0-100 → 增量 0.0-0.5
    QUALITY_TO_INCREMENT = 0.005  # 每1点质量 = 0.5% 进度

    @staticmethod
    def train_with_params(model, pack: ParameterPack,
                          energy_cost: float = None) -> Dict[str, Any]:
        """
        用单个参数包直接训练模型

        流程：
        1. 检查模型是否被认领
        2. 消耗少量虚拟电（参数训练也需一点电作"激活"）
        3. 根据参数质量计算训练增量
        4. 注入参数（调超参数）
        5. 提升训练进度
        6. 记录训练历史

        Args:
            model: XuniModel 虚拟模型
            pack: ParameterPack 参数包
            energy_cost: 激活能量（默认按质量计算）
        """
        # 检查认领
        if getattr(model, "owner", None) is None:
            return {"error": "模型未被认领，无法训练"}

        # 检查训练状态
        training_state = getattr(model, "training_state", None)
        state_name = training_state.name if training_state else "UNKNOWN"
        if state_name == "TRAINED":
            return {"error": "模型已训练完成", "progress": 1.0}

        # 激活能量（参数训练需要少量电作"激活"，远少于算力训练）
        if energy_cost is None:
            energy_cost = max(1.0, pack.quality * 0.1)  # 质量越高激活耗电略多
        if hasattr(model, "_energy_buffer"):
            if model._energy_buffer < energy_cost:
                return {
                    "error": f"激活能量不足：需 {energy_cost:.1f}，当前 {model._energy_buffer:.1f}",
                }
            model._energy_buffer -= energy_cost

        # 开始训练（如果还没开始）
        if state_name == "UNTRAINED" or state_name == "CLAIMED":
            if hasattr(model, "start_training"):
                model.start_training()

        # 计算训练增量（基于参数质量）
        quality = max(0.0, min(100.0, pack.quality))
        increment = quality * ParameterTrainer.QUALITY_TO_INCREMENT

        # 参数维度丰富度加成（参数越多，训练越全面）
        param_count = len(pack.params)
        diversity_bonus = min(0.05, param_count * 0.002)
        increment += diversity_bonus

        # 注入参数（调超参数，改变模型行为）
        injected = ParameterInjector.inject(model, pack)

        # 提升训练进度
        old_progress = getattr(model, "training_progress", 0.0)
        new_progress = min(1.0, old_progress + increment)
        if hasattr(model, "update_training"):
            model.update_training(new_progress)
        else:
            model.training_progress = new_progress

        # 记录训练样本
        if hasattr(model, "training_samples_seen"):
            model.training_samples_seen += param_count
        if hasattr(model, "training_epochs_done"):
            model.training_epochs_done += 1

        # 检查是否训练完成
        completed = new_progress >= 1.0
        if completed and hasattr(model, "complete_training"):
            model.complete_training()

        return {
            "status": "trained",
            "method": "parameter",
            "pack_id": pack.pack_id,
            "pack_quality": quality,
            "increment": increment,
            "progress_before": old_progress,
            "progress_after": new_progress,
            "progress_pct": new_progress * 100,
            "energy_cost": energy_cost,
            "params_injected": injected,
            "completed": completed,
        }

    @staticmethod
    def train_with_batch(model, packs: List[ParameterPack]) -> Dict[str, Any]:
        """
        用多个参数包批量训练模型

        每个参数包依次注入，训练进度累积。
        高质量参数包排前面效果更好。
        """
        if getattr(model, "owner", None) is None:
            return {"error": "模型未被认领，无法训练"}

        # 按质量降序排列（高质量先训）
        sorted_packs = sorted(packs, key=lambda p: p.quality, reverse=True)

        results = []
        total_increment = 0.0
        total_energy = 0.0

        for pack in sorted_packs:
            result = ParameterTrainer.train_with_params(model, pack)
            results.append(result)
            if result.get("status") == "trained":
                total_increment += result["increment"]
                total_energy += result["energy_cost"]
            if result.get("completed"):
                break
            if result.get("error"):
                break

        return {
            "status": "batch_trained",
            "method": "parameter_batch",
            "packs_used": len([r for r in results if r.get("status") == "trained"]),
            "packs_total": len(packs),
            "total_increment": total_increment,
            "total_energy": total_energy,
            "final_progress": getattr(model, "training_progress", 0.0),
            "completed": getattr(model, "training_progress", 0.0) >= 1.0,
            "details": results,
        }

    @staticmethod
    def train_from_sampler(model, sampler, n_packs: int = 10,
                           samples_per_pack: int = 100) -> Dict[str, Any]:
        """
        直接从采样点产参数训练模型——最短闭环

        采样点 → 产参数 → 训练模型
        不经过电、不经过算力，最直接的路径！

        Args:
            model: 虚拟模型
            sampler: XuniSampler 采样点
            n_packs: 产生多少个参数包
            samples_per_pack: 每个参数包用多少采样点
        """
        if getattr(model, "owner", None) is None:
            return {"error": "模型未被认领，无法训练"}

        packs = []
        for i in range(n_packs):
            batch = sampler.generate_batch(batch_size=samples_per_pack)
            pack = ParameterExtractor.from_samples(batch, count=samples_per_pack)
            packs.append(pack)

        return ParameterTrainer.train_with_batch(model, packs)


# ============================================================
# 三路径训练器——统一参数训练 + 算力训练 + 融合训练
# ============================================================

class MultiPathTrainer:
    """
    三路径训练器——统一管理三种训练方式

    路径1·参数训练（直接）：
        采样点 → 参数 → 注入 → 进度提升
        特点：快，但受参数质量限制

    路径2·算力训练（间接）：
        采样点 → 电 → 算力 → 替代物 → 权重更新
        特点：深，能优化权重，但耗电多

    路径3·融合训练（最强）：
        电 + 参数 → 活力 → 涌现训练
        特点：最强，参数+电融合产生涌现效果

    用法：
        trainer = MultiPathTrainer(model)
        trainer.train_via_parameter(pack)     # 路径1
        trainer.train_via_compute(vcu, data)  # 路径2
        trainer.train_via_fusion(pack, vcu)   # 路径3
    """

    def __init__(self, model):
        self.model = model

    def train_via_parameter(self, pack: ParameterPack,
                            energy_cost: float = None) -> Dict[str, Any]:
        """路径1：参数直接训练"""
        result = ParameterTrainer.train_with_params(
            self.model, pack, energy_cost=energy_cost
        )
        result["path"] = "parameter"
        return result

    def train_via_compute(self, dual_state_manager, training_data=None,
                          epochs: int = 1, compute_unit=None) -> Dict[str, Any]:
        """路径2：算力训练（通过双态管理器的替代物）"""
        result = dual_state_manager.train_with_surrogate(
            training_data=training_data,
            epochs=epochs,
        )
        result["path"] = "compute"
        return result

    def train_via_fusion(self, pack: ParameterPack, energy: float,
                         fusion_reactor=None) -> Dict[str, Any]:
        """
        路径3：融合训练（电+参数→活力→训练）

        参数 + 电 融合产生活力，活力驱动最高效训练。
        需要 FusionReactor（活力系统）。
        """
        if getattr(self.model, "owner", None) is None:
            return {"error": "模型未被认领，无法训练"}

        # 融合：电 + 参数 → 活力
        # 活力 = 参数质量 × 能量 的几何平均
        quality = max(0.0, min(100.0, pack.quality))
        vitality = (quality * energy) ** 0.5  # 几何平均

        # 活力驱动的训练增量（比单独参数或单独电都高）
        # 融合加成：1.5x
        increment = vitality * ParameterTrainer.QUALITY_TO_INCREMENT * 1.5

        # 消耗电
        if hasattr(self.model, "_energy_buffer"):
            if self.model._energy_buffer < energy:
                return {
                    "error": f"能量不足：需 {energy:.1f}，当前 {self.model._energy_buffer:.1f}",
                }
            self.model._energy_buffer -= energy

        # 开始训练
        training_state = getattr(self.model, "training_state", None)
        state_name = training_state.name if training_state else "UNKNOWN"
        if state_name in ("UNTRAINED", "CLAIMED"):
            if hasattr(self.model, "start_training"):
                self.model.start_training()

        # 注入参数
        ParameterInjector.inject(self.model, pack)

        # 提升进度
        old_progress = getattr(self.model, "training_progress", 0.0)
        new_progress = min(1.0, old_progress + increment)
        if hasattr(self.model, "update_training"):
            self.model.update_training(new_progress)

        completed = new_progress >= 1.0
        if completed and hasattr(self.model, "complete_training"):
            self.model.complete_training()

        return {
            "status": "trained",
            "path": "fusion",
            "pack_quality": quality,
            "energy_in": energy,
            "vitality": vitality,
            "increment": increment,
            "progress_before": old_progress,
            "progress_after": new_progress,
            "progress_pct": new_progress * 100,
            "fusion_bonus": 1.5,
            "completed": completed,
        }

    def auto_train(self, packs: List[ParameterPack] = None,
                   sampler=None, energy: float = 100.0,
                   prefer: str = "auto") -> Dict[str, Any]:
        """
        自动选择最佳路径训练

        prefer: "auto" / "parameter" / "compute" / "fusion"
        auto 策略：
        - 有参数包 + 有电 → 融合训练（最强）
        - 有参数包 + 无电 → 参数训练
        - 无参数包 + 有电 → 算力训练（需双态管理器）
        """
        has_packs = packs is not None and len(packs) > 0
        has_energy = (
            hasattr(self.model, "_energy_buffer")
            and self.model._energy_buffer >= energy
        )

        if prefer == "auto":
            if has_packs and has_energy:
                prefer = "fusion"
            elif has_packs:
                prefer = "parameter"
            else:
                prefer = "compute"

        if prefer == "fusion" and has_packs and has_energy:
            best_pack = max(packs, key=lambda p: p.quality)
            return self.train_via_fusion(best_pack, energy)
        elif prefer == "parameter" and has_packs:
            return ParameterTrainer.train_with_batch(self.model, packs)
        elif prefer == "compute":
            return {"error": "算力训练需 DualStateManager，请用 train_via_compute()"}
        else:
            return {"error": f"路径 {prefer} 条件不满足"}
