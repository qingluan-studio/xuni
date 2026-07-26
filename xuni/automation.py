"""
XuniAutomation —— 自动化运行系统

核心理念：
  不眠不休自动运行：
  1. 训练循环：认领→训练→评估→奖励/淘汰
  2. 能量循环：赚能量→训练更多→赚更多
  3. 进化循环：差模型淘汰→好模型带新模型→整体提升

一个 cycle 包含：
  Phase 1: AI 认领未认领的模型
  Phase 2: 训练所有已认领的模型
  Phase 3: 评估所有已训练的模型
  Phase 4: 根据评估结果发放能量奖励
  Phase 5: 淘汰低分模型
  Phase 6: 导师模型给新模型加成
  Phase 7: 保存状态
"""

import time
from typing import Dict, Any, List, Optional

from .layer import LayeredModelSystem, AI_NAME_POOL
from .model import TrainingState, ModelInput
from .evaluator import ModelEvaluator, ModelRole
from .economy import EnergyEconomy


class AutomationRunner:
    """
    自动化运行器。
    
    整合分层系统 + 评估 + 经济，自动循环运行。
    """

    def __init__(
        self,
        system: Optional[LayeredModelSystem] = None,
        evaluator: Optional[ModelEvaluator] = None,
        economy: Optional[EnergyEconomy] = None,
        state_file: str = "xuni_layers.json",
    ):
        self.system = system or LayeredModelSystem()
        if not self.system.layers:
            self.system.setup_default_layers()
        
        self.evaluator = evaluator or ModelEvaluator()
        self.economy = economy or EnergyEconomy()
        self.state_file = state_file
        
        self.cycle_count = 0
        self.history: List[Dict[str, Any]] = []
        self._running = False

    def run_cycle(self) -> Dict[str, Any]:
        """
        运行一个完整周期。
        
        返回周期报告。
        """
        self.cycle_count += 1
        cycle_start = time.time()
        report = {
            "cycle": self.cycle_count,
            "timestamp": cycle_start,
            "phases": {},
        }

        # Phase 1: 认领
        phase1 = self._phase_claim()
        report["phases"]["claim"] = phase1

        # Phase 2: 训练
        phase2 = self._phase_train()
        report["phases"]["train"] = phase2

        # Phase 3: 评估
        phase3 = self._phase_evaluate()
        report["phases"]["evaluate"] = phase3

        # Phase 4: 奖励
        phase4 = self._phase_reward(phase3)
        report["phases"]["reward"] = phase4

        # Phase 5: 淘汰
        phase5 = self._phase_retire()
        report["phases"]["retire"] = phase5

        # Phase 6: 导师加成
        phase6 = self._phase_mentor()
        report["phases"]["mentor"] = phase6

        # Phase 7: 保存
        self.system.save(self.state_file)

        # 汇总
        report["duration_ms"] = round((time.time() - cycle_start) * 1000, 1)
        report["summary"] = {
            "total_models": self.system.statistics()["total_models"],
            "total_claimed": self.system.statistics()["total_claimed"],
            "total_trained": self.system.statistics()["total_trained"],
            "economy": self.economy.statistics(),
            "mentors": len(self.evaluator.get_mentors()),
        }
        
        self.history.append(report)
        if len(self.history) > 100:
            self.history.pop(0)
        
        return report

    def _phase_claim(self) -> Dict[str, Any]:
        """Phase 1: AI 认领未认领的模型"""
        claimed = 0
        ai_idx = 0
        
        for layer in self.system.get_layers_ordered():
            for model in layer.get_unclaimed():
                # 找一个有足够能量的 AI
                for _ in range(len(AI_NAME_POOL)):
                    ai_name = AI_NAME_POOL[ai_idx % len(AI_NAME_POOL)]
                    self.economy.register_ai(ai_name)
                    
                    if self.economy.can_afford_training(ai_name, model.energy_requirement):
                        if self.economy.charge_training(ai_name, model.model_id, model.energy_requirement):
                            model.claim(ai_name)
                            claimed += 1
                            ai_idx += 1
                            break
                    ai_idx += 1
        
        return {"claimed": claimed}

    def _phase_train(self) -> Dict[str, Any]:
        """Phase 2: 训练所有已认领的模型"""
        for layer in self.system.get_layers_ordered():
            for model in layer.models.values():
                if model.training_state == TrainingState.CLAIMED:
                    model.start_training()
        
        result = self.system.train_until_complete(step_progress=0.3, max_steps=10)
        return {
            "steps": result["total_steps"],
            "trained": result["final_trained"],
        }

    def _phase_evaluate(self) -> Dict[str, Any]:
        """Phase 3: 评估所有已训练的模型"""
        eval_result = self.evaluator.evaluate_system(self.system)
        return eval_result

    def _phase_reward(self, eval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: 根据评估结果发放能量奖励"""
        mentors = set(eval_result.get("mentors", []))
        total_rewarded = 0
        
        for model_id, eval_info in eval_result.get("evaluations", {}).items():
            score = eval_info["score"]
            is_mentor = model_id in mentors
            
            # 找到模型归属
            for layer in self.system.get_layers_ordered():
                model = layer.get_model(model_id)
                if model and model.owner:
                    self.economy.reward_evaluation(model.owner, model_id, score, is_mentor)
                    total_rewarded += 1
                    break
        
        return {
            "rewarded": total_rewarded,
            "mentors": len(mentors),
        }

    def _phase_retire(self) -> Dict[str, Any]:
        """Phase 5: 淘汰低分模型"""
        retired = []
        
        for layer in self.system.get_layers_ordered():
            for model in layer.models.values():
                if model.training_state == TrainingState.TRAINED:
                    if self.evaluator.should_retire(model.model_id):
                        # 淘汰前先惩罚 AI
                        if model.owner:
                            self.economy.penalize_retirement(model.owner, model.model_id)
                        self.evaluator.retire_model(model)
                        retired.append(model.model_id)
        
        return {"retired": len(retired), "models": retired}

    def _phase_mentor(self) -> Dict[str, Any]:
        """Phase 6: 导师模型给新模型训练加成"""
        mentors = self.evaluator.get_mentors()
        mentor_bonus = len(mentors) * 0.05  # 每个导师给5%加成
        
        # 对正在训练的模型应用导师加成
        for layer in self.system.get_layers_ordered():
            layer.collaborative_train(step_progress=0.0, mentor_bonus=mentor_bonus)
        
        return {
            "mentors": len(mentors),
            "bonus": mentor_bonus,
        }

    def run_cycles(self, n: int = 3, verbose: bool = True) -> List[Dict[str, Any]]:
        """运行多个周期"""
        reports = []
        for i in range(n):
            report = self.run_cycle()
            reports.append(report)
            
            if verbose:
                s = report["summary"]
                print(f"Cycle {report['cycle']}: "
                      f"trained={s['total_trained']}/{s['total_claimed']} "
                      f"mentors={s['mentors']} "
                      f"retired={report['phases']['retire']['retired']} "
                      f"energy={s['economy']['total_energy']}")
        
        return reports

    def get_report(self) -> Dict[str, Any]:
        """获取当前状态报告"""
        system_stats = self.system.statistics()
        economy_stats = self.economy.statistics()
        
        return {
            "cycles_run": self.cycle_count,
            "system": system_stats,
            "economy": economy_stats,
            "mentors": self.evaluator.get_mentors(),
            "leaderboard": self.economy.get_leaderboard()[:5],
            "model_ranking": self.evaluator.get_ranking(5),
            "recent_history": self.history[-5:],
        }

    def visualize(self) -> str:
        """可视化当前状态"""
        lines = []
        lines.append("=" * 60)
        lines.append("XUNI AUTOMATION RUNNER")
        lines.append("=" * 60)
        
        stats = self.system.statistics()
        econ = self.economy.statistics()
        
        lines.append(f"Cycles run: {self.cycle_count}")
        lines.append(f"Models: {stats['total_models']} | Trained: {stats['total_trained']}")
        lines.append(f"Total AI: {econ['total_ai']} | Total energy: {econ['total_energy']}")
        lines.append(f"Mentors: {len(self.evaluator.get_mentors())}")
        
        lines.append("\n--- Energy Leaderboard ---")
        for i, acc in enumerate(self.economy.get_leaderboard()[:5]):
            if acc:
                lines.append(f"  {i+1}. {acc['owner']}: {acc['balance']} energy "
                           f"(earned={acc['total_earned']}, retired={acc['models_retired']})")
        
        lines.append("\n--- Model Ranking ---")
        for i, (mid, score) in enumerate(self.evaluator.get_ranking(5)):
            lines.append(f"  {i+1}. {mid}: {score:.1f} pts")
        
        lines.append("=" * 60)
        return "\n".join(lines)
