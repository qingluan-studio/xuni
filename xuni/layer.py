"""
XuniLayer —— 分层模型系统

核心理念：
  模型按层组织，每层有特定类型和数量。
  每个 AI 可以认领（claim）一个模型进行训练，上面有归属名。
  层间数据流动：上层输出 → 下层输入。

分层结构（可扩展）：
  Layer 1: 音乐模型层（5个）  — 每个 AI 认领一个训练
  Layer 2: 扩散模型层（5个）  — 图像生成
  Layer 3: 对话模型层（5个）  — 聊天机器人
  Layer 4: 文本模型层（5个）  — 文本生成
  Layer 5: 分类模型层（5个）  — 分类预测
  ...

每层的模型：
  - 有归属名（owner）：哪个 AI 认领训练
  - 有训练状态：未训练→已认领→训练中→训练完成
  - 消耗采样点能量
  - 可被上层输出驱动
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Callable

import numpy as np

from .model import (
    XuniModel, XuniTextGenerator, XuniImageDescriber, XuniMusicComposer,
    XuniClassifier, XuniChatBot, XuniDiffusion, XuniPredictor, XuniAutoencoder,
    ModelType, ModelStatus, TrainingState, ModelInput, ModelOutput,
)


class LayerType(Enum):
    """层类型"""
    MUSIC = auto()        # 音乐层
    DIFFUSION = auto()    # 扩散层
    CHAT = auto()         # 对话层
    TEXT = auto()         # 文本层
    CLASSIFIER = auto()   # 分类层
    IMAGE = auto()        # 图像描述层
    PREDICTOR = auto()    # 预测层
    AUTOENCODER = auto()  # 自编码层
    CUSTOM = auto()       # 自定义层


# AI 名称池（可供认领使用）
AI_NAME_POOL = [
    "Aria", "Bolt", "Coda", "Dusk", "Echo",
    "Flux", "Glow", "Halo", "Iris", "Jade",
    "Kite", "Luna", "Mist", "Nova", "Onyx",
    "Pulse", "Quartz", "Rune", "Sage", "Tide",
    "Umber", "Vex", "Wisp", "Xen", "Yarn", "Zen",
]


@dataclass
class LayerConfig:
    """层配置"""
    layer_id: str
    layer_name: str
    layer_type: LayerType
    level: int                              # 层级 1,2,3...
    model_count: int = 5                    # 模型数量
    model_factory: Optional[Callable[[str], XuniModel]] = None  # 模型工厂
    description: str = ""


class ModelLayer:
    """
    模型层。
    
    管理一层内的所有模型：
    1. 创建/删除模型
    2. AI 认领/释放/转移
    3. 批量训练
    4. 层内统计
    """

    def __init__(self, config: LayerConfig):
        self.config = config
        self.models: Dict[str, XuniModel] = {}
        self._create_models()

    def _create_models(self):
        """创建该层的所有模型"""
        factory = self.config.model_factory or self._default_factory()
        for i in range(self.config.model_count):
            idx = i + 1
            model_id = f"{self.config.layer_id}-m{idx:02d}"
            model = factory(model_id)
            self.models[model_id] = model

    def _default_factory(self) -> Callable[[str], XuniModel]:
        """默认模型工厂：根据层类型创建模型"""
        layer_type = self.config.layer_type

        def factory(model_id: str) -> XuniModel:
            if layer_type == LayerType.MUSIC:
                return XuniMusicComposer(model_id)
            elif layer_type == LayerType.DIFFUSION:
                return XuniDiffusion(model_id)
            elif layer_type == LayerType.CHAT:
                personalities = ["friendly", "professional", "creative", "technical"]
                personality = personalities[hash(model_id) % len(personalities)]
                return XuniChatBot(model_id, personality)
            elif layer_type == LayerType.TEXT:
                return XuniTextGenerator(model_id)
            elif layer_type == LayerType.CLASSIFIER:
                return XuniClassifier(model_id, ["class_a", "class_b", "class_c"])
            elif layer_type == LayerType.IMAGE:
                return XuniImageDescriber(model_id)
            elif layer_type == LayerType.PREDICTOR:
                return XuniPredictor(model_id)
            elif layer_type == LayerType.AUTOENCODER:
                return XuniAutoencoder(model_id)
            else:
                return XuniTextGenerator(model_id)

        return factory

    # ===== AI 认领机制 =====

    def claim_model(self, model_id: str, owner_name: str) -> bool:
        """AI 认领模型"""
        model = self.models.get(model_id)
        if model is None:
            return False
        return model.claim(owner_name)

    def release_model(self, model_id: str) -> bool:
        """释放模型认领"""
        model = self.models.get(model_id)
        if model is None:
            return False
        return model.release()

    def transfer_model(self, model_id: str, new_owner: str) -> bool:
        """转移模型认领"""
        model = self.models.get(model_id)
        if model is None:
            return False
        return model.transfer(new_owner)

    def auto_assign(self, ai_names: List[str]) -> Dict[str, str]:
        """
        自动分配：每个 AI 认领一个未认领的模型。
        
        返回: {ai_name: model_id}
        """
        assignments = {}
        unclaimed = [m for m in self.models.values() if m.owner is None]
        
        for i, ai_name in enumerate(ai_names):
            if i >= len(unclaimed):
                break
            model = unclaimed[i]
            if model.claim(ai_name):
                assignments[ai_name] = model.model_id
        
        return assignments

    # ===== 训练管理 =====

    def train_model(self, model_id: str, progress: float = 1.0) -> bool:
        """训练指定模型"""
        model = self.models.get(model_id)
        if model is None:
            return False
        if model.training_state == TrainingState.CLAIMED:
            model.start_training()
        model.update_training(progress)
        return model.training_state == TrainingState.TRAINED

    def train_all_claimed(self, progress: float = 1.0) -> Dict[str, bool]:
        """训练所有已认领的模型"""
        results = {}
        for model_id, model in self.models.items():
            if model.training_state in (TrainingState.CLAIMED, TrainingState.TRAINING):
                if model.training_state == TrainingState.CLAIMED:
                    model.start_training()
                model.update_training(progress)
                results[model_id] = model.training_state == TrainingState.TRAINED
        return results

    def batch_train_step(self, step_progress: float = 0.1) -> Dict[str, float]:
        """
        批量训练一步（增量进度）。
        
        每个 TRAINING 中的模型增加 step_progress 进度。
        """
        results = {}
        for model_id, model in self.models.items():
            if model.training_state == TrainingState.TRAINING:
                new_progress = model.training_progress + step_progress
                model.update_training(new_progress)
                results[model_id] = model.training_progress
        return results

    # ===== 能量管理 =====

    def charge_all(self, energy: float) -> float:
        """给层内所有模型充能"""
        total = 0.0
        for model in self.models.values():
            model.charge(energy)
            total += energy
        return total

    def charge_trained_only(self, energy: float) -> float:
        """只给已训练的模型充能"""
        total = 0.0
        for model in self.models.values():
            if model.training_state == TrainingState.TRAINED:
                model.charge(energy)
                total += energy
        return total

    # ===== 调用 =====

    def predict(self, model_id: str, input_data: ModelInput) -> Optional[ModelOutput]:
        """调用指定模型"""
        model = self.models.get(model_id)
        if model is None:
            return None
        return model.predict(input_data)

    def predict_all_trained(self, input_data: ModelInput) -> Dict[str, ModelOutput]:
        """调用所有已训练的模型"""
        results = {}
        for model_id, model in self.models.items():
            if model.training_state == TrainingState.TRAINED:
                results[model_id] = model.predict(input_data)
        return results

    def predict_best(self, input_data: ModelInput) -> Optional[ModelOutput]:
        """调用训练度最高的模型"""
        trained = [m for m in self.models.values() if m.training_state == TrainingState.TRAINED]
        if not trained:
            return None
        # 选择调用次数最多（经验最丰富）的模型
        best = max(trained, key=lambda m: m.stats.total_calls)
        return best.predict(input_data)

    def ensemble_predict(self, input_data: ModelInput) -> Optional[ModelOutput]:
        """
        集成预测：所有已训练模型投票/汇总。
        
        - 分类层：多数投票
        - 预测层：取平均
        - 其他层：选择置信度最高的
        """
        trained = [m for m in self.models.values() if m.training_state == TrainingState.TRAINED]
        if not trained:
            return None

        outputs = [m.predict(input_data) for m in trained]

        # 分类层：多数投票
        if self.config.layer_type == LayerType.CLASSIFIER:
            votes: Dict[str, int] = {}
            for o in outputs:
                if o.classification:
                    votes[o.classification] = votes.get(o.classification, 0) + 1
            if votes:
                winner = max(votes, key=votes.get)
                return ModelOutput(
                    classification=winner,
                    json={"votes": votes, "ensemble": True},
                    metadata={"method": "majority_vote", "voters": len(trained)},
                )

        # 预测层：取平均
        if self.config.layer_type == LayerType.PREDICTOR:
            preds = [o.prediction for o in outputs if o.prediction is not None]
            if preds:
                avg = sum(preds) / len(preds)
                return ModelOutput(
                    prediction=avg,
                    json={"individual_predictions": preds, "ensemble": True, "method": "average"},
                    metadata={"voters": len(trained)},
                )

        # 其他层：选第一个（可扩展）
        return outputs[0]

    def collaborative_train(self, step_progress: float = 0.2, mentor_bonus: float = 0.0) -> Dict[str, float]:
        """
        协作训练：同层已认领的模型互相帮助训练。
        
        每个正在训练的模型获得进度增量，
        且如果同层有已训练的模型，进度增量翻倍（知识共享）。
        mentor_bonus: 导师模型额外加成（来自评估系统）。
        """
        trained_count = len(self.get_trained())
        bonus = 1.0 + (0.1 * trained_count) + mentor_bonus  # 每个已训练模型给10%加成 + 导师加成

        results = {}
        for model_id, model in self.models.items():
            if model.training_state == TrainingState.TRAINING:
                effective_progress = step_progress * bonus
                new_progress = model.training_progress + effective_progress
                model.update_training(new_progress)
                results[model_id] = model.training_progress
        return results

    # ===== 查询 =====

    def get_model(self, model_id: str) -> Optional[XuniModel]:
        return self.models.get(model_id)

    def get_by_owner(self, owner_name: str) -> List[XuniModel]:
        """按归属名查询"""
        return [m for m in self.models.values() if m.owner == owner_name]

    def get_unclaimed(self) -> List[XuniModel]:
        """获取未认领的模型"""
        return [m for m in self.models.values() if m.owner is None]

    def get_claimed(self) -> List[XuniModel]:
        """获取已认领的模型"""
        return [m for m in self.models.values() if m.owner is not None]

    def get_trained(self) -> List[XuniModel]:
        """获取已训练的模型"""
        return [m for m in self.models.values() if m.training_state == TrainingState.TRAINED]

    # ===== 统计 =====

    def statistics(self) -> Dict[str, Any]:
        total = len(self.models)
        claimed = len(self.get_claimed())
        trained = len(self.get_trained())
        owners = set(m.owner for m in self.models.values() if m.owner)

        return {
            "layer_id": self.config.layer_id,
            "layer_name": self.config.layer_name,
            "level": self.config.level,
            "layer_type": self.config.layer_type.name,
            "total_models": total,
            "claimed": claimed,
            "unclaimed": total - claimed,
            "trained": trained,
            "training": len([m for m in self.models.values() if m.training_state == TrainingState.TRAINING]),
            "owners": list(owners),
            "total_calls": sum(m.stats.total_calls for m in self.models.values()),
            "total_energy": round(sum(m.stats.total_energy_consumed for m in self.models.values()), 2),
        }

    def visualize(self) -> str:
        """可视化层内模型状态"""
        lines = []
        lines.append(f"  Layer {self.config.level}: {self.config.layer_name} ({self.config.layer_type.name})")
        lines.append(f"  {'Model ID':<20} {'Owner':<15} {'Training':<12} {'Progress':<10} {'Calls':<6}")
        lines.append(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*10} {'-'*6}")
        for model in self.models.values():
            owner = model.owner or "---"
            state = model.training_state.name
            progress = f"{model.training_progress*100:.0f}%"
            calls = str(model.stats.total_calls)
            lines.append(f"  {model.model_id:<20} {owner:<15} {state:<12} {progress:<10} {calls:<6}")
        return "\n".join(lines)


class LayeredModelSystem:
    """
    分层模型系统。
    
    管理所有层：
    1. 创建/删除层
    2. 层间数据流动（上层输出 → 下层输入）
    3. 全局统计
    4. 全局可视化
    """

    def __init__(self):
        self.layers: Dict[str, ModelLayer] = {}
        self._layer_order: List[str] = []

    def add_layer(self, config: LayerConfig) -> bool:
        """添加层"""
        if config.layer_id in self.layers:
            return False
        layer = ModelLayer(config)
        self.layers[config.layer_id] = layer
        self._layer_order.append(config.layer_id)
        return True

    def remove_layer(self, layer_id: str) -> bool:
        """删除层"""
        if layer_id not in self.layers:
            return False
        del self.layers[layer_id]
        self._layer_order.remove(layer_id)
        return True

    def get_layer(self, layer_id: str) -> Optional[ModelLayer]:
        return self.layers.get(layer_id)

    def get_layer_by_level(self, level: int) -> Optional[ModelLayer]:
        """按层级获取"""
        for layer in self.layers.values():
            if layer.config.level == level:
                return layer
        return None

    def get_layers_ordered(self) -> List[ModelLayer]:
        """按层级顺序获取所有层"""
        return sorted(self.layers.values(), key=lambda l: l.config.level)

    # ===== 默认分层结构 =====

    def setup_default_layers(self, models_per_layer: int = 5):
        """
        设置默认分层结构：
          Layer 1: 音乐模型（5个）
          Layer 2: 扩散模型（5个）
          Layer 3: 对话模型（5个）
          Layer 4: 文本模型（5个）
          Layer 5: 分类模型（5个）
          Layer 6: 图像描述模型（5个）
          Layer 7: 预测模型（5个）
          Layer 8: 自编码器模型（5个）
        """
        defaults = [
            LayerConfig("L1-music", "音乐模型层", LayerType.MUSIC, 1, models_per_layer,
                        description="音乐作曲模型，每个AI认领一个训练"),
            LayerConfig("L2-diffusion", "扩散模型层", LayerType.DIFFUSION, 2, models_per_layer,
                        description="图像扩散生成模型"),
            LayerConfig("L3-chat", "对话模型层", LayerType.CHAT, 3, models_per_layer,
                        description="聊天对话模型"),
            LayerConfig("L4-text", "文本模型层", LayerType.TEXT, 4, models_per_layer,
                        description="文本生成模型"),
            LayerConfig("L5-classifier", "分类模型层", LayerType.CLASSIFIER, 5, models_per_layer,
                        description="分类预测模型"),
            LayerConfig("L6-image", "图像描述层", LayerType.IMAGE, 6, models_per_layer,
                        description="图像描述生成模型"),
            LayerConfig("L7-predictor", "预测模型层", LayerType.PREDICTOR, 7, models_per_layer,
                        description="时间序列预测模型"),
            LayerConfig("L8-autoencoder", "自编码层", LayerType.AUTOENCODER, 8, models_per_layer,
                        description="自编码器模型，编码解码"),
        ]
        for config in defaults:
            self.add_layer(config)

    # ===== 层间数据流动 =====

    def flow_down(self, from_level: int, input_data: ModelInput) -> Dict[str, Any]:
        """
        数据向下流动：指定层的输出 → 下一层的输入。
        
        返回各模型的输出结果。
        """
        current_layer = self.get_layer_by_level(from_level)
        next_layer = self.get_layer_by_level(from_level + 1)

        if current_layer is None:
            return {"error": f"Layer {from_level} not found"}

        # 当前层预测
        current_outputs = current_layer.predict_all_trained(input_data)

        result = {
            "from_level": from_level,
            "from_layer": current_layer.config.layer_name,
            "outputs": {},
        }

        for model_id, output in current_outputs.items():
            result["outputs"][model_id] = {
                "text": output.text,
                "classification": output.classification,
                "json": output.json,
            }

        # 如果有下一层，将输出作为输入传递
        if next_layer is not None and current_outputs:
            # 将所有输出合并为下一层的 prompt
            combined_text = " | ".join(
                o.text for o in current_outputs.values() if o.text
            )
            next_input = ModelInput(
                prompt=combined_text[:500],  # 限制长度
                parameters={"source_layer": from_level},
            )
            next_outputs = next_layer.predict_all_trained(next_input)
            result["next_level"] = from_level + 1
            result["next_layer"] = next_layer.config.layer_name
            result["next_outputs"] = {
                mid: {"text": o.text, "json": o.json}
                for mid, o in next_outputs.items()
            }

        return result

    def flow_through_all(self, initial_input: ModelInput) -> List[Dict[str, Any]]:
        """
        数据从第1层流到最后一层。
        
        每层的输出作为下一层的输入。
        """
        results = []
        current_input = initial_input

        for layer in self.get_layers_ordered():
            layer_outputs = layer.predict_all_trained(current_input)
            
            layer_result = {
                "level": layer.config.level,
                "layer_name": layer.config.layer_name,
                "outputs": {},
            }

            # 收集输出
            combined_text = []
            for model_id, output in layer_outputs.items():
                layer_result["outputs"][model_id] = {
                    "text": output.text,
                    "classification": output.classification,
                    "json": output.json,
                }
                if output.text:
                    combined_text.append(output.text)

            results.append(layer_result)

            # 组合输出作为下一层输入
            if combined_text:
                current_input = ModelInput(
                    prompt=" | ".join(combined_text)[:500],
                    parameters={"source_level": layer.config.level},
                )

        return results

    # ===== 全局操作 =====

    def charge_all_layers(self, energy: float):
        """给所有层的所有模型充能"""
        for layer in self.layers.values():
            layer.charge_all(energy)

    def train_all_layers(self, progress: float = 1.0):
        """训练所有层中已认领的模型"""
        results = {}
        for layer_id, layer in self.layers.items():
            results[layer_id] = layer.train_all_claimed(progress)
        return results

    def auto_assign_all(self, ai_assignments: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """
        全局自动分配。
        
        ai_assignments: {layer_id: [ai_name1, ai_name2, ...]}
        返回: {layer_id: {ai_name: model_id}}
        """
        results = {}
        for layer_id, ai_names in ai_assignments.items():
            layer = self.layers.get(layer_id)
            if layer:
                results[layer_id] = layer.auto_assign(ai_names)
        return results

    def auto_assign_from_pool(self, ai_names: Optional[List[str]] = None) -> Dict[str, Dict[str, str]]:
        """
        从 AI 名称池自动认领所有层的所有模型。
        
        每个 AI 会跨层认领多个模型（一个 AI 可以训练多层的模型）。
        """
        pool = ai_names if ai_names else AI_NAME_POOL.copy()
        results = {}
        ai_idx = 0

        for layer in self.get_layers_ordered():
            layer_results = {}
            unclaimed = layer.get_unclaimed()
            for model in unclaimed:
                ai_name = pool[ai_idx % len(pool)]
                if model.claim(ai_name):
                    layer_results[ai_name] = model.model_id
                    ai_idx += 1
            results[layer.config.layer_id] = layer_results

        return results

    def collaborative_train_all(self, step_progress: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        全局协作训练：所有层同时进行协作训练。
        
        每层的已训练模型会给同层正在训练的模型提供加成。
        """
        results = {}
        for layer_id, layer in self.layers.items():
            results[layer_id] = layer.collaborative_train(step_progress)
        return results

    def train_until_complete(self, step_progress: float = 0.25, max_steps: int = 10) -> Dict[str, Any]:
        """
        训练直到所有已认领的模型完成训练。
        
        返回训练过程统计。
        """
        history = []
        for step in range(max_steps):
            step_result = self.collaborative_train_all(step_progress)
            stats = self.statistics()
            history.append({
                "step": step + 1,
                "trained": stats["total_trained"],
                "claimed": stats["total_claimed"],
            })
            # 检查是否所有已认领的都训练完成
            if stats["total_trained"] >= stats["total_claimed"]:
                break

        return {
            "total_steps": len(history),
            "history": history,
            "final_trained": self.statistics()["total_trained"],
        }

    def ensemble_all_layers(self, input_data: ModelInput) -> Dict[str, Any]:
        """
        每层都做集成预测，返回所有层的集成结果。
        """
        results = {}
        for layer in self.get_layers_ordered():
            ensemble = layer.ensemble_predict(input_data)
            if ensemble:
                results[layer.config.layer_id] = {
                    "layer_name": layer.config.layer_name,
                    "level": layer.config.level,
                    "classification": ensemble.classification,
                    "prediction": ensemble.prediction,
                    "text": ensemble.text,
                    "json": ensemble.json,
                    "metadata": ensemble.metadata,
                }
        return results

    # ===== 全局统计 =====

    def statistics(self) -> Dict[str, Any]:
        total_models = 0
        total_claimed = 0
        total_trained = 0
        total_calls = 0
        total_energy = 0.0
        all_owners = set()

        layer_stats = []
        for layer in self.get_layers_ordered():
            stat = layer.statistics()
            layer_stats.append(stat)
            total_models += stat["total_models"]
            total_claimed += stat["claimed"]
            total_trained += stat["trained"]
            total_calls += stat["total_calls"]
            total_energy += stat["total_energy"]
            all_owners.update(stat["owners"])

        return {
            "total_layers": len(self.layers),
            "total_models": total_models,
            "total_claimed": total_claimed,
            "total_trained": total_trained,
            "total_unclaimed": total_models - total_claimed,
            "total_calls": total_calls,
            "total_energy_consumed": round(total_energy, 2),
            "unique_owners": len(all_owners),
            "owners": list(all_owners),
            "layers": layer_stats,
        }

    def visualize(self) -> str:
        """可视化整个分层系统"""
        lines = []
        lines.append("=" * 70)
        lines.append("XUNI LAYERED MODEL SYSTEM")
        lines.append("=" * 70)

        stats = self.statistics()
        lines.append(f"Layers: {stats['total_layers']} | Models: {stats['total_models']} | "
                     f"Claimed: {stats['total_claimed']} | Trained: {stats['total_trained']}")
        lines.append("")

        for layer in self.get_layers_ordered():
            lines.append(layer.visualize())
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ===== 持久化存储 =====

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典（可转JSON）"""
        return {
            "version": "1.0",
            "timestamp": time.time(),
            "layers": [
                {
                    "config": {
                        "layer_id": layer.config.layer_id,
                        "layer_name": layer.config.layer_name,
                        "layer_type": layer.config.layer_type.name,
                        "level": layer.config.level,
                        "model_count": layer.config.model_count,
                        "description": layer.config.description,
                    },
                    "models": [
                        {
                            "model_id": m.model_id,
                            "model_type": m.model_type.name,
                            "owner": m.owner,
                            "training_state": m.training_state.name,
                            "training_progress": m.training_progress,
                            "claimed_at": m.claimed_at,
                            "trained_at": m.trained_at,
                            "energy_buffer": m._energy_buffer,
                            "energy_requirement": m.energy_requirement,
                            "stats": {
                                "total_calls": m.stats.total_calls,
                                "total_energy_consumed": m.stats.total_energy_consumed,
                                "total_latency_ms": m.stats.total_latency_ms,
                            },
                            "training_history": m.training_history,
                        }
                        for m in layer.models.values()
                    ],
                }
                for layer in self.get_layers_ordered()
            ],
        }

    def save(self, filepath: str) -> bool:
        """保存到JSON文件"""
        import json
        try:
            data = self.to_dict()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, filepath: str) -> Optional["LayeredModelSystem"]:
        """从JSON文件加载"""
        import json
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            system = cls()
            for layer_data in data.get("layers", []):
                cfg = layer_data["config"]
                layer_type = LayerType[cfg["layer_type"]]
                config = LayerConfig(
                    layer_id=cfg["layer_id"],
                    layer_name=cfg["layer_name"],
                    layer_type=layer_type,
                    level=cfg["level"],
                    model_count=cfg["model_count"],
                    description=cfg.get("description", ""),
                )
                layer = ModelLayer(config)

                # 恢复模型状态
                for model_data in layer_data.get("models", []):
                    model = layer.models.get(model_data["model_id"])
                    if model:
                        model.owner = model_data.get("owner")
                        state_name = model_data.get("training_state", "UNTRAINED")
                        model.training_state = TrainingState[state_name]
                        model.training_progress = model_data.get("training_progress", 0.0)
                        model.claimed_at = model_data.get("claimed_at", 0.0)
                        model.trained_at = model_data.get("trained_at", 0.0)
                        model._energy_buffer = model_data.get("energy_buffer", 0.0)
                        stats = model_data.get("stats", {})
                        model.stats.total_calls = stats.get("total_calls", 0)
                        model.stats.total_energy_consumed = stats.get("total_energy_consumed", 0.0)
                        model.stats.total_latency_ms = stats.get("total_latency_ms", 0.0)
                        model.training_history = model_data.get("training_history", [])

                system.layers[config.layer_id] = layer
                system._layer_order.append(config.layer_id)

            return system
        except Exception:
            return None
