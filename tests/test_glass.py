import numpy as np
import pytest

from xuni.glass import XuniGlass, OpticalMedium


class TestGlass:
    def test_shine(self):
        glass = XuniGlass("test")
        glass.add_element("lens", OpticalMedium.LENS, focus=0.2)
        glass.add_element("transform", OpticalMedium.GLASS, func=lambda x: x * 2)
        data = np.array([1.0, 2.0, 3.0])
        ray = glass.shine(data)
        assert ray.intensity > 0
        assert ray.get_path_length() == 2

    def test_prism_dispersion(self):
        glass = XuniGlass("prism_test")
        glass.add_element("prism", OpticalMedium.PRISM, dispersion=0.5)
        data = np.array([1.0, 2.0, 3.0])
        ray = glass.shine(data)
        assert ray.payload is not None

    def test_resonance(self):
        glass = XuniGlass("res_test")
        glass.add_element("mirror1", OpticalMedium.MIRROR, reflectivity=0.3)
        glass.add_element("transform", OpticalMedium.GLASS, func=lambda x: x + 1)
        glass.add_element("mirror2", OpticalMedium.MIRROR, reflectivity=0.3)
        data = np.array([0.0, 0.0, 0.0])
        rays = glass.resonance_loop(data, iterations=3, feedback_gain=0.3)
        assert len(rays) == 3

    def test_optical_report(self):
        glass = XuniGlass("report_test")
        glass.add_element("elem", OpticalMedium.GLASS)
        glass.shine(42)
        report = glass.get_optical_report()
        assert report["system_name"] == "report_test"
        assert report["latest_ray"]["path_length"] == 1
