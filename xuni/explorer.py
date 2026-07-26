"""
XuniExplorer —— 探索-利用控制器（Explore-Exploit Controller）

从 CEE 的 explore_exploit.py 提取核心思想：
让 Xuni 系统在"探索新声音"和"利用已知好模式"之间自动平衡。

策略：
    1. Epsilon-Greedy：以 ε 概率随机探索，以 1-ε 概率选择历史最佳
    2. Thompson Sampling：贝叶斯 Bandit，自动平衡探索与利用
    3. NoveltyTracker：检测意外成功的参数组合，标记为"黑马"
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time


class SamplingStrategy(Enum):
    """采样策略枚举"""
    HYPER_CHAOS = "hyper_chaos"
    LORENZ_96 = "lorenz_96"
    MANDELBULB = "mandelbulb"
    NOISE_FIELD = "noise_field"
    HYBRID = "hybrid"


@dataclass
class StrategyRecord:
    """策略记录"""
    strategy: str
    params: dict  # 如 {"seed": 42, "field_coupling": 0.5}
    score: float = 0.0  # 评估得分
    trials: int = 0
    successes: int = 0
    last_used: float = field(default_factory=time.time)
    novelty_flag: bool = False  # 是否为意外成功（黑马）


class EpsilonGreedyExplorer:
    """
    Epsilon-Greedy 探索器。

    以概率 ε 随机探索新策略，以概率 1-ε 选择历史平均得分最高的策略。
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.records: Dict[str, StrategyRecord] = {}
        self.rng = np.random.default_rng()

    def register_strategy(self, name: str, params: dict):
        """注册一个可选策略"""
        if name not in self.records:
            self.records[name] = StrategyRecord(strategy=name, params=params)

    def select(self) -> StrategyRecord:
        """选择一个策略"""
        if not self.records:
            raise ValueError("No strategies registered")

        if self.rng.random() < self.epsilon:
            # 探索：随机选择
            name = self.rng.choice(list(self.records.keys()))
            return self.records[name]
        else:
            # 利用：选择平均得分最高的
            best = max(self.records.values(), key=lambda r: r.score / max(r.trials, 1))
            return best

    def update(self, name: str, score: float):
        """更新策略的得分记录"""
        if name not in self.records:
            return
        rec = self.records[name]
        rec.score += score
        rec.trials += 1
        rec.last_used = time.time()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_best(self) -> Optional[StrategyRecord]:
        """返回当前最佳策略"""
        if not self.records:
            return None
        return max(self.records.values(), key=lambda r: r.score / max(r.trials, 1))


class ThompsonExplorer:
    """
    Thompson Sampling 探索器（贝叶斯 Bandit）。

    假设每个策略的得分服从 Beta 分布，
    从分布中采样后选择采样值最高的策略。
    天然平衡探索（不确定的策略采样方差大）和利用（好策略均值高）。
    """

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}
        self.rng = np.random.default_rng()

    def register_strategy(self, name: str, params: dict):
        """注册策略，用 Beta(1,1) = 均匀分布作为先验"""
        if name not in self.records:
            self.records[name] = {
                "params": params,
                "alpha": 1.0,  # 成功计数 + 1
                "beta": 1.0,   # 失败计数 + 1
            }

    def select(self) -> tuple:
        """返回 (策略名, 参数)"""
        if not self.records:
            raise ValueError("No strategies registered")

        samples = {}
        for name, rec in self.records.items():
            # 从 Beta(α, β) 采样
            samples[name] = self.rng.beta(rec["alpha"], rec["beta"])

        best_name = max(samples, key=samples.get)
        return best_name, self.records[best_name]["params"]

    def update(self, name: str, score: float):
        """
        更新 Beta 分布参数。
        score 在 0~1 之间，视为"成功概率"的观测。
        """
        if name not in self.records:
            return
        rec = self.records[name]
        # score 越高，越倾向于增加 alpha
        rec["alpha"] += score
        rec["beta"] += (1.0 - score)


class NoveltyTracker:
    """
    新颖性追踪器。

    检测"意外成功"：那些之前表现一般、突然产生高分输出的参数组合。
    标记为"黑马"策略，给予额外探索权重。
    """

    def __init__(
        self,
        surprise_threshold: float = 0.3,  # 超出预期多少算意外
        window_size: int = 5,
    ):
        self.surprise_threshold = surprise_threshold
        self.window_size = window_size
        self.history: Dict[str, List[float]] = {}  # 每个策略的近期得分历史

    def record(self, name: str, score: float):
        """记录一次得分"""
        if name not in self.history:
            self.history[name] = []
        self.history[name].append(score)
        if len(self.history[name]) > self.window_size:
            self.history[name].pop(0)

    def is_novelty(self, name: str, score: float) -> bool:
        """判断这次得分是否为意外成功"""
        if name not in self.history or len(self.history[name]) < 2:
            return False

        past_scores = self.history[name][:-1]
        mean_past = np.mean(past_scores)
        std_past = np.std(past_scores) + 1e-12

        # z-score 检验
        z_score = (score - mean_past) / std_past
        return z_score > self.surprise_threshold

    def get_novelty_bonus(self, name: str) -> float:
        """返回新颖性奖励（0~1）"""
        if name not in self.history:
            return 0.0
        # 历史越短、方差越大，新颖性奖励越高
        scores = self.history[name]
        if len(scores) < 2:
            return 0.5
        return float(np.clip(np.std(scores), 0.0, 1.0))


class XuniExplorer:
    """
    Xuni 统一的探索-利用控制器。

    组合 Epsilon-Greedy + Thompson Sampling + NoveltyTracker，
    为采样模式、Brain 参数、合成器参数提供自适应选择。
    """

    def __init__(
        self,
        epsilon: float = 0.3,
        use_thompson: bool = True,
        use_novelty: bool = True,
    ):
        self.epsilon_greedy = EpsilonGreedyExplorer(epsilon=epsilon)
        self.thompson = ThompsonExplorer() if use_thompson else None
        self.novelty = NoveltyTracker() if use_novelty else None
        self.rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # 注册策略
    # ------------------------------------------------------------------
    def register_sampling_mode(self, mode: SamplingStrategy, seed_range: tuple = (0, 1000)):
        """注册采样模式策略"""
        name = f"sample_{mode.value}"
        params = {"mode": mode.value, "seed_range": seed_range}
        self.epsilon_greedy.register_strategy(name, params)
        if self.thompson:
            self.thompson.register_strategy(name, params)

    def register_brain_config(self, n_neurons: int, field_coupling: float, label: str = ""):
        """注册 Brain 配置策略"""
        name = f"brain_{n_neurons}_{field_coupling}" if not label else label
        params = {"n_neurons": n_neurons, "field_coupling": field_coupling}
        self.epsilon_greedy.register_strategy(name, params)
        if self.thompson:
            self.thompson.register_strategy(name, params)

    # ------------------------------------------------------------------
    # 选择与更新
    # ------------------------------------------------------------------
    def select_strategy(self, category: str = "sample") -> tuple:
        """
        选择策略。

        Returns:
            (策略名, 参数字典)
        """
        category_candidates = [
            k for k in self.epsilon_greedy.records
            if k.startswith(category)
        ]
        if not category_candidates:
            raise ValueError(f"No strategies in category '{category}'")

        if self.thompson:
            thompson_candidates = [
                k for k in self.thompson.records
                if k.startswith(category)
            ]
            if thompson_candidates and self.rng.random() < 0.5:
                name, params = self.thompson.select()
                if name.startswith(category):
                    return name, params

        rec = self.epsilon_greedy.select()
        while not rec.strategy.startswith(category):
            rec = self.epsilon_greedy.select()
        return rec.strategy, rec.params

    def feedback(self, name: str, score: float):
        """
        接收评估反馈，更新所有控制器。

        Args:
            name: 策略名
            score: 评估得分（0~1）
        """
        self.epsilon_greedy.update(name, score)
        if self.thompson:
            self.thompson.update(name, score)
        if self.novelty:
            is_surprise = self.novelty.is_novelty(name, score)
            self.novelty.record(name, score)
            if is_surprise and name in self.epsilon_greedy.records:
                self.epsilon_greedy.records[name].novelty_flag = True

    def get_report(self) -> dict:
        """获取探索报告"""
        report = {
            "epsilon": self.epsilon_greedy.epsilon,
            "strategies": [],
            "novelties": [],
        }
        for rec in self.epsilon_greedy.records.values():
            report["strategies"].append({
                "name": rec.strategy,
                "avg_score": round(rec.score / max(rec.trials, 1), 4),
                "trials": rec.trials,
                "novelty": rec.novelty_flag,
            })
            if rec.novelty_flag:
                report["novelties"].append(rec.strategy)
        return report
