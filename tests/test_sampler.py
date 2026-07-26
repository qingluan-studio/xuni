import numpy as np
import pytest

from xuni.sampler import XuniSampler, SamplingMode


class TestSampler:
    def test_hyper_chaos_batch(self):
        sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        batch = sampler.generate_batch(1000)
        assert batch.shape == (1000, 6)
        assert np.all(np.isfinite(batch))

    def test_lorenz_96_batch(self):
        sampler = XuniSampler(mode=SamplingMode.LORENZ_96, seed=42)
        batch = sampler.generate_batch(1000)
        assert batch.shape == (1000, 6)
        assert np.all(np.isfinite(batch))

    def test_mandelbulb_batch(self):
        sampler = XuniSampler(mode=SamplingMode.MANDELBULB, seed=42)
        batch = sampler.generate_batch(1000)
        assert batch.shape == (1000, 6)
        assert np.all(np.isfinite(batch))

    def test_hybrid_batch(self):
        sampler = XuniSampler(mode=SamplingMode.HYBRID, seed=42)
        batch = sampler.generate_batch(1000)
        assert batch.shape == (1000, 6)
        assert np.all(np.isfinite(batch))

    def test_stream(self):
        sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        points = list(sampler.generate_stream(100))
        assert len(points) == 100
        assert sampler.total_generated == 100

    def test_estimate_capacity(self):
        sampler = XuniSampler()
        cap = sampler.estimate_capacity(1.0)
        assert cap > 1_000_000
