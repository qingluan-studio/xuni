"""
Tests for XuniBrain, XuniTrainer, XuniMemory
"""

import numpy as np
import pytest

from xuni.brain import XuniBrain
from xuni.trainer import XuniTrainer, TrainingConfig
from xuni.memory import XuniMemory


class TestXuniBrain:
    def test_init(self):
        brain = XuniBrain(n_neurons=64, seed=1)
        assert brain.n == 64
        assert brain.sr == 22050
        assert brain.phi.shape == (64,)
        assert brain.freq.shape == (64,)
        assert brain.amp.shape == (64,)
        assert brain.W.shape == (64, 64)

    def test_stimulate(self):
        brain = XuniBrain(n_neurons=32, seed=2)
        audio = brain.stimulate(duration=0.1, field_energy=0.0)
        assert len(audio) == int(brain.sr * 0.1)
        assert np.all(np.isfinite(audio))
        assert np.max(np.abs(audio)) <= 1.0 + 1e-9

    def test_field_energy_modulation(self):
        brain = XuniBrain(n_neurons=32, seed=3, field_coupling=1.0)
        audio_low = brain.stimulate(duration=0.05, field_energy=0.0)
        brain.reset()
        audio_high = brain.stimulate(duration=0.05, field_energy=10.0)
        assert len(audio_low) == len(audio_high)
        # 高场能量应产生不同输出
        assert not np.allclose(audio_low, audio_high)

    def test_get_set_state(self):
        brain = XuniBrain(n_neurons=32, seed=4)
        state = brain.get_state()
        brain.stimulate(duration=0.05)
        brain.set_state(state)
        assert np.allclose(brain.phi, state["phi"])
        assert np.allclose(brain.freq, state["freq"])

    def test_brain_summary(self):
        brain = XuniBrain(n_neurons=32, seed=5)
        summary = brain.brain_summary()
        assert "n_neurons" in summary
        assert "synchronization" in summary
        assert 0.0 <= summary["synchronization"] <= 1.0


class TestXuniTrainer:
    def test_cultivate(self):
        brain = XuniBrain(n_neurons=32, seed=6)
        trainer = XuniTrainer(brain, config=TrainingConfig())
        target = np.sin(2 * np.pi * 440 * np.linspace(0, 0.2, int(brain.sr * 0.2)))
        output = trainer.cultivate(target_audio=target, duration=0.2, epochs=1)
        assert len(output) == int(brain.sr * 0.2 * trainer.config.spontaneous_ratio)
        assert np.all(np.isfinite(output))

    def test_imprint_evoke(self):
        brain = XuniBrain(n_neurons=32, seed=7)
        trainer = XuniTrainer(brain)
        imprint = trainer.imprint_pattern("test_pattern", duration=0.1)
        assert imprint["name"] == "test_pattern"
        audio = trainer.evoke_pattern(imprint, duration=0.1)
        assert len(audio) == int(brain.sr * 0.1)

    def test_training_report(self):
        brain = XuniBrain(n_neurons=32, seed=8)
        trainer = XuniTrainer(brain)
        target = np.sin(2 * np.pi * 330 * np.linspace(0, 0.1, int(brain.sr * 0.1)))
        trainer.cultivate(target_audio=target, duration=0.1, epochs=1)
        report = trainer.get_training_report()
        assert report["epochs"] == 1
        assert "progression" in report


class TestXuniMemory:
    def test_capture_recall(self):
        brain = XuniBrain(n_neurons=32, seed=9)
        memory = XuniMemory(brain)
        brain.stimulate(duration=0.05)
        mem = memory.capture("test_mem", tags=["demo"])
        assert mem.name == "test_mem"
        assert mem.tags == ["demo"]
        audio = memory.recall("test_mem", duration=0.05, perturbation=0.0)
        assert len(audio) == int(brain.sr * 0.05)

    def test_dream(self):
        brain = XuniBrain(n_neurons=32, seed=10)
        memory = XuniMemory(brain)
        memory.capture("m1")
        memory.capture("m2")
        audio = memory.dream(duration=0.1, n_memories=2)
        assert len(audio) == int(brain.sr * 0.1)
        assert np.all(np.isfinite(audio))

    def test_fuse(self):
        brain = XuniBrain(n_neurons=32, seed=11)
        memory = XuniMemory(brain)
        memory.capture("m1")
        memory.capture("m2")
        fused = memory.fuse(["m1", "m2"], weights=[0.5, 0.5])
        assert "phi" in fused
        assert "freq" in fused
        assert "W" in fused

    def test_save_load(self, tmp_path):
        brain = XuniBrain(n_neurons=32, seed=12)
        memory = XuniMemory(brain)
        memory.capture("saved_mem")
        path = tmp_path / "memories.json"
        memory.save_to_file(str(path))
        memory2 = XuniMemory(brain)
        memory2.load_from_file(str(path))
        assert len(memory2.memories) == 1
        assert memory2.memories[0].name == "saved_mem"
