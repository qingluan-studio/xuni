"""
Tests for XuniExplorer
"""

import pytest
from xuni.explorer import (
    EpsilonGreedyExplorer,
    ThompsonExplorer,
    NoveltyTracker,
    XuniExplorer,
    SamplingStrategy,
)


class TestEpsilonGreedyExplorer:
    def test_register_and_select(self):
        exp = EpsilonGreedyExplorer(epsilon=0.0)  # 纯利用
        exp.register_strategy("a", {"x": 1})
        exp.register_strategy("b", {"x": 2})
        rec = exp.select()
        assert rec.strategy in ["a", "b"]

    def test_update(self):
        exp = EpsilonGreedyExplorer(epsilon=0.0)
        exp.register_strategy("a", {})
        exp.register_strategy("b", {})
        exp.update("a", 0.9)
        exp.update("b", 0.1)
        best = exp.get_best()
        assert best.strategy == "a"

    def test_epsilon_decay(self):
        exp = EpsilonGreedyExplorer(epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.5)
        exp.register_strategy("a", {})
        exp.update("a", 0.5)
        assert exp.epsilon == 0.9


class TestThompsonExplorer:
    def test_select(self):
        exp = ThompsonExplorer()
        exp.register_strategy("a", {"x": 1})
        exp.register_strategy("b", {"x": 2})
        name, params = exp.select()
        assert name in ["a", "b"]

    def test_update(self):
        exp = ThompsonExplorer()
        exp.register_strategy("a", {})
        exp.update("a", 0.8)
        assert exp.records["a"]["alpha"] > 1.0


class TestNoveltyTracker:
    def test_is_novelty(self):
        nt = NoveltyTracker(surprise_threshold=0.3)
        nt.record("a", 0.5)
        nt.record("a", 0.5)
        assert not nt.is_novelty("a", 0.5)
        assert nt.is_novelty("a", 0.9)

    def test_bonus(self):
        nt = NoveltyTracker()
        assert nt.get_novelty_bonus("a") == 0.0
        nt.record("a", 0.5)
        assert nt.get_novelty_bonus("a") == 0.5


class TestXuniExplorer:
    def test_register_and_feedback(self):
        xe = XuniExplorer(epsilon=0.0)
        xe.register_sampling_mode(SamplingStrategy.HYPER_CHAOS)
        name, params = xe.select_strategy("sample")
        assert "hyper_chaos" in name
        xe.feedback(name, 0.7)
        report = xe.get_report()
        assert report["strategies"][0]["trials"] == 1
