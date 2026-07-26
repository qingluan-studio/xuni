"""
Tests for XuniCritic
"""

import numpy as np
import pytest

from xuni.critic import XuniCritic, MusicInvariantScores


class TestXuniCritic:
    def test_evaluate_sine_wave(self):
        critic = XuniCritic(sample_rate=22050)
        t = np.linspace(0, 0.5, int(22050 * 0.5))
        audio = np.sin(2 * np.pi * 440 * t)
        scores = critic.evaluate(audio)
        assert 0.0 <= scores.itc <= 1.0
        assert 0.0 <= scores.scs <= 1.0
        assert 0.0 <= scores.iec <= 1.0
        assert 0.0 <= scores.pfft <= 1.0
        assert 0.0 <= scores.overall <= 1.0

    def test_evaluate_noise(self):
        critic = XuniCritic(sample_rate=22050)
        audio = np.random.randn(int(22050 * 0.3))
        scores = critic.evaluate(audio)
        # 噪声的 ITC 应该较低（频谱分散）
        assert scores.itc < 0.5
        assert scores.overall < 0.8

    def test_suggestions(self):
        critic = XuniCritic(sample_rate=22050)
        scores = MusicInvariantScores(itc=0.2, scs=0.9, iec=0.3, pfft=0.9, overall=0.5)
        suggestions = critic.suggest_optimization(scores)
        assert "itc" in suggestions
        assert "iec" in suggestions

    def test_harmonic_mean(self):
        critic = XuniCritic()
        assert critic._harmonic_mean([0.5, 0.5]) == 0.5
        assert critic._harmonic_mean([1.0, 1.0]) == 1.0
        # 0 值被过滤，只计算非零值
        assert critic._harmonic_mean([0.0, 1.0]) == 1.0
