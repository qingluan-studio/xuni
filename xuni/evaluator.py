"""
XuniEvaluator —— 模型评估淘汰系统

核心理念：
  每个模型都要被评估，根据输出质量打分。
  分数低的模型被淘汰（释放认领），让其他 AI 接手。
  分数高的模型可以带新模型（知识传承）。

评估指标：
  1. 输出质量（是否有有效输出）
  2. 响应速度（延迟越低越好）
  3. 能量效率（输出/能量比）
  4. 稳定性（多次调用结果一致性）
  5. 经验值（调用次数）

淘汰机制：
  - 分数低于阈值 → 释放认领，重置训练
  - 分数最高 → 成为"导师模型"，带新模型训练加成
  
晋升机制：
  - 连续N次评估高分 → 晋升为导师
  - 导师模型给同层新模型训练加成
"""

import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List, Tuple

import numpy as np

from .model import XuniModel, ModelInput, ModelOutput, TrainingState, ModelType


class EvalMetric(Enum):
    """评估指标"""
    QUALITY = auto()       # 输出质量
    LATENCY = auto()       # 响应速度
    EFFICIENCY = auto()    # 能量效率
    STABILITY = auto()     # 稳定性
    EXPERIENCE = auto()    # 经验值


class ModelRole(Enum):
    """模型角色"""
    NORMAL = auto()      # 普通模型
    MENTOR = auto()      # 导师模型（高分晋升）
    PROBATION = auto()   # 观察期（刚淘汰重训）


@dataclass
class EvalRecord:
    """单次评估记录"""
    timestamp: float
    scores: Dict[str, float]     # 各项分数
    total_score: float           # 总分
    output_sample: Optional[str]  # 输出样本


@dataclass
class ModelEvaluation:
    """模型评估汇总"""
    model_id: str
    current_score: float = 0.0
    avg_score: float = 0.0
    max_score: float = 0.0
    min_score: float = 0.0
    eval_count: int = 0
    consecutive_high: int = 0    # 连续高分次数
    consecutive_low: int = 0     # 连续低分次数
    role: ModelRole = ModelRole.NORMAL
    history: List[EvalRecord] = field(default_factory=list)
    last_eval_time: float = 0.0


class ModelEvaluator:
    """
    模型评估器。
    
    对每个模型进行多维度评估，决定淘汰/保留/晋升。
    """

    def __init__(
        self,
        low_score_threshold: float = 30.0,    # 低于此分淘汰
        high_score_threshold: float = 80.0,   # 高于此分计连续高分
        mentor_required_streak: int = 3,      # 连续N次高分晋升导师
        probation_score: float = 20.0,        # 低于此分立即淘汰
        max_history: int = 20,                # 保留最近N次评估
    ):
        self.low_score_threshold = low_score_threshold
        self.high_score_threshold = high_score_threshold
        self.mentor_required_streak = mentor_required_streak
        self.probation_score = probation_score
        self.max_history = max_history
        
        self.evaluations: Dict[str, ModelEvaluation] = {}
        self._test_prompts = [
            "生成一段音乐",
            "描述这个场景",
            "预测未来趋势",
            "分类这个输入",
            "回答问题",
        ]

    def evaluate_model(self, model: XuniModel, test_input: Optional[ModelInput] = None) -> ModelEvaluation:
        """
        评估单个模型。
        
        执行测试调用，打分，更新评估记录。
        """
        if model.model_id not in self.evaluations:
            self.evaluations[model.model_id] = ModelEvaluation(model_id=model.model_id)
        
        ev = self.evaluations[model.model_id]
        
        # 准备测试输入
        if test_input is None:
            prompt = self._test_prompts[hash(model.model_id) % len(self._test_prompts)]
            test_input = ModelInput(prompt=prompt)
        
        # 确保有能量
        if model._energy_buffer < model.energy_requirement:
            model.charge(model.energy_requirement * 2)
        
        # 执行调用
        start_time = time.time()
        output = model.predict(test_input)
        latency = (time.time() - start_time) * 1000
        
        # 计算各项分数
        scores = self._compute_scores(model, output, latency)
        total = sum(scores.values()) / len(scores)
        
        # 记录
        output_sample = None
        if output.text:
            output_sample = output.text[:100]
        elif output.json:
            output_sample = str(output.json)[:100]
        
        record = EvalRecord(
            timestamp=time.time(),
            scores=scores,
            total_score=total,
            output_sample=output_sample,
        )
        ev.history.append(record)
        if len(ev.history) > self.max_history:
            ev.history.pop(0)
        
        # 更新汇总
        ev.current_score = total
        ev.eval_count += 1
        ev.last_eval_time = time.time()
        ev.max_score = max(ev.max_score, total)
        ev.min_score = min(ev.min_score, total) if ev.min_score > 0 else total
        ev.avg_score = sum(r.total_score for r in ev.history) / len(ev.history)
        
        # 更新连续计数
        if total >= self.high_score_threshold:
            ev.consecutive_high += 1
            ev.consecutive_low = 0
        elif total < self.low_score_threshold:
            ev.consecutive_low += 1
            ev.consecutive_high = 0
        else:
            # 中间分数不连续累计
            pass
        
        # 角色判定
        self._update_role(ev)
        
        return ev

    def _compute_scores(self, model: XuniModel, output: ModelOutput, latency: float) -> Dict[str, float]:
        """计算各项分数（0-100）"""
        scores = {}
        
        # 1. 输出质量（0-100）
        quality = 0.0
        if output.text and len(output.text) > 5:
            quality += 40
        if output.text and len(output.text) > 50:
            quality += 20
        if output.json:
            quality += 20
        if output.classification:
            quality += 10
        if output.prediction is not None:
            quality += 10
        if output.metadata.get("error"):
            quality = 0.0
        scores["quality"] = min(100.0, quality)
        
        # 2. 响应速度（0-100，越快越高）
        if latency < 1:
            scores["latency"] = 100.0
        elif latency < 10:
            scores["latency"] = 90.0
        elif latency < 50:
            scores["latency"] = 70.0
        elif latency < 100:
            scores["latency"] = 50.0
        else:
            scores["latency"] = 30.0
        
        # 3. 能量效率（0-100）
        energy = model.energy_requirement
        if energy <= 3:
            scores["efficiency"] = 100.0
        elif energy <= 5:
            scores["efficiency"] = 80.0
        elif energy <= 8:
            scores["efficiency"] = 60.0
        elif energy <= 12:
            scores["efficiency"] = 40.0
        else:
            scores["efficiency"] = 20.0
        
        # 4. 稳定性（0-100，基于历史方差）
        ev = self.evaluations.get(model.model_id)
        if ev and len(ev.history) >= 3:
            recent_scores = [r.total_score for r in ev.history[-5:]]
            variance = np.var(recent_scores)
            scores["stability"] = max(0.0, 100.0 - variance)
        else:
            scores["stability"] = 50.0  # 没有足够历史，给中间分
        
        # 5. 经验值（0-100，基于调用次数）
        calls = model.stats.total_calls
        if calls >= 100:
            scores["experience"] = 100.0
        elif calls >= 50:
            scores["experience"] = 80.0
        elif calls >= 20:
            scores["experience"] = 60.0
        elif calls >= 10:
            scores["experience"] = 40.0
        elif calls >= 1:
            scores["experience"] = 20.0
        else:
            scores["experience"] = 0.0
        
        return scores

    def _update_role(self, ev: ModelEvaluation):
        """更新模型角色"""
        # 晋升导师
        if ev.consecutive_high >= self.mentor_required_streak:
            ev.role = ModelRole.MENTOR
        # 降为观察期
        elif ev.consecutive_low >= 2:
            ev.role = ModelRole.PROBATION
        # 恢复普通
        elif ev.role == ModelRole.PROBATION and ev.consecutive_high >= 1:
            ev.role = ModelRole.NORMAL

    def should_retire(self, model_id: str) -> bool:
        """是否应该淘汰"""
        ev = self.evaluations.get(model_id)
        if ev is None:
            return False
        # 立即淘汰：分数极低
        if ev.current_score < self.probation_score:
            return True
        # 连续低分淘汰
        if ev.consecutive_low >= 3:
            return True
        return False

    def retire_model(self, model: XuniModel) -> bool:
        """淘汰模型：释放认领，重置训练"""
        if not self.should_retire(model.model_id):
            return False
        
        model.release()
        ev = self.evaluations.get(model.model_id)
        if ev:
            ev.role = ModelRole.PROBATION
            ev.consecutive_low = 0
        return True

    def get_mentors(self) -> List[str]:
        """获取所有导师模型ID"""
        return [mid for mid, ev in self.evaluations.items() if ev.role == ModelRole.MENTOR]

    def evaluate_layer(self, layer) -> Dict[str, ModelEvaluation]:
        """评估整个层"""
        results = {}
        for model in layer.models.values():
            if model.training_state == TrainingState.TRAINED:
                ev = self.evaluate_model(model)
                results[model.model_id] = ev
        return results

    def evaluate_system(self, system) -> Dict[str, Any]:
        """评估整个分层系统"""
        all_evals = {}
        mentors = []
        retired = []
        
        for layer in system.get_layers_ordered():
            layer_evals = self.evaluate_layer(layer)
            all_evals.update(layer_evals)
            
            # 检查淘汰
            for model in layer.models.values():
                if model.training_state == TrainingState.TRAINED and self.should_retire(model.model_id):
                    if self.retire_model(model):
                        retired.append(model.model_id)
        
        mentors = self.get_mentors()
        
        # 统计
        scores = [ev.current_score for ev in all_evals.values()]
        
        return {
            "total_evaluated": len(all_evals),
            "avg_score": round(np.mean(scores), 2) if scores else 0,
            "max_score": round(max(scores), 2) if scores else 0,
            "min_score": round(min(scores), 2) if scores else 0,
            "mentors": mentors,
            "mentor_count": len(mentors),
            "retired": retired,
            "retired_count": len(retired),
            "probation_count": sum(1 for ev in all_evals.values() if ev.role == ModelRole.PROBATION),
            "evaluations": {
                mid: {
                    "score": ev.current_score,
                    "avg": ev.avg_score,
                    "role": ev.role.name,
                    "consecutive_high": ev.consecutive_high,
                    "consecutive_low": ev.consecutive_low,
                    "eval_count": ev.eval_count,
                }
                for mid, ev in all_evals.items()
            },
        }

    def get_ranking(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """获取排名"""
        ranked = sorted(
            self.evaluations.items(),
            key=lambda x: x[1].current_score,
            reverse=True,
        )
        return [(mid, ev.current_score) for mid, ev in ranked[:top_n]]
