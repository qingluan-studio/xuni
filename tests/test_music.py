import numpy as np
import pytest

from xuni.converter import XuniConverter, MusicParams
from xuni.music import XuniMusic


class TestMusic:
    def test_synthesize(self):
        params = MusicParams(base_frequency=440.0, amplitude=0.5, harmonics=4)
        music = XuniMusic(sample_rate=22050)
        audio = music.synthesize(params, duration=0.5)
        assert audio.duration == 0.5
        assert len(audio.data) == int(22050 * 0.5)
        assert audio.data.ndim == 2

    def test_sequence(self):
        params_list = [
            MusicParams(base_frequency=220.0, amplitude=0.5, harmonics=3),
            MusicParams(base_frequency=440.0, amplitude=0.6, harmonics=5),
        ]
        music = XuniMusic(sample_rate=22050)
        audio = music.synthesize_sequence(params_list, segment_duration=0.5, overlap=0.1)
        assert audio.duration > 0.5
        assert len(audio.data) > 0

    def test_wav_bytes(self):
        params = MusicParams(base_frequency=440.0, amplitude=0.5)
        music = XuniMusic(sample_rate=22050)
        audio = music.synthesize(params, duration=0.3)
        wav = music.to_wav_bytes(audio)
        assert len(wav) > 0
        assert wav[:4] == b"RIFF"
