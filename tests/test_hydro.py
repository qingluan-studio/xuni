import numpy as np
import pytest

from xuni.hydro import XuniHydro


class TestHydro:
    def test_init(self):
        hydro = XuniHydro(n_particles=100, seed=42)
        assert len(hydro.particles) == 100

    def test_step(self):
        hydro = XuniHydro(n_particles=100, seed=42)
        energy = hydro._step()
        assert energy >= 0
        summary = hydro.hydro_summary()
        assert summary["particle_count"] > 0

    def test_sample_batch(self):
        hydro = XuniHydro(n_particles=256, seed=42)
        batch = hydro.get_sample_batch(1000)
        assert batch.shape == (1000, 6)
        assert np.all(np.isfinite(batch))

    def test_evaporation(self):
        hydro = XuniHydro(n_particles=100, seed=42, evap_threshold=100.0)
        for _ in range(100):
            hydro._step()
        summary = hydro.hydro_summary()
        assert summary["particle_count"] <= 100
