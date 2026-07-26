import numpy as np
import pytest

from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField


class TestField:
    def test_field_compute(self):
        sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        field = XuniField(grid_size=(8, 8, 8))
        batch = sampler.generate_batch(5000)
        field.ingest_batch(batch)
        field.compute_field()
        summary = field.field_summary()
        assert summary["total_samples"] == 5000
        assert summary["total_energy"] >= 0

    def test_field_cells(self):
        sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        field = XuniField(grid_size=(8, 8, 8))
        batch = sampler.generate_batch(10000)
        field.ingest_batch(batch)
        cells = field.get_cells(threshold=0.01)
        assert len(cells) >= 0

    def test_dominant_vector(self):
        sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        field = XuniField(grid_size=(8, 8, 8))
        batch = sampler.generate_batch(10000)
        field.ingest_batch(batch)
        vec = field.get_dominant_vector()
        assert len(vec) == 3
        assert all(np.isfinite(v) for v in vec)
