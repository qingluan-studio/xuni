"""
XuniMusic —— 原创虚拟音乐模型

一个完全原创的物理建模合成器，不依赖任何现成神经网络或采样库。
核心原理：用"虚拟电"参数驱动数字振荡器、共鸣滤波器和粒子泛音系统。

架构：
1. 振荡器层：多个基础波形（正弦、三角、方波），由虚拟电频率参数驱动
2. 谐波层：粒子系统根据电荷密度产生泛音列
3. 共鸣层：模拟物理共鸣体（弦、管、膜），由电场能量调制
4. 空间层：3D 声像定位，由电场方向驱动
5. 包络层：ADSR，由场的弛豫特性决定

输出：原始音频波形（NumPy 数组），可直接保存为 WAV 或通过 API 流式传输。
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from .converter import MusicParams


@dataclass
class AudioBuffer:
    """音频缓冲区"""
    sample_rate: int
    data: np.ndarray  # shape (samples,) 或 (samples, channels)
    duration: float

    def to_mono(self) -> np.ndarray:
        if self.data.ndim == 1:
            return self.data
        return np.mean(self.data, axis=1)

    def to_stereo(self) -> np.ndarray:
        if self.data.ndim == 2 and self.data.shape[1] >= 2:
            return self.data[:, :2]
        mono = self.to_mono()
        return np.stack([mono, mono], axis=1)


class _Oscillator:
    """数字振荡器"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def sine(self, freq: float, duration: float, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        return amp * np.sin(2 * np.pi * freq * t + phase)

    def saw(self, freq: float, duration: float, amp: float = 1.0) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        # 带限锯齿波（叠加谐波的近似）
        wave = np.zeros_like(t)
        for n in range(1, 20):
            wave += ((-1)**(n+1)) * np.sin(2 * np.pi * freq * n * t) / n
        return amp * (2.0 / np.pi) * wave

    def square(self, freq: float, duration: float, amp: float = 1.0) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = np.zeros_like(t)
        for n in range(1, 20, 2):
            wave += np.sin(2 * np.pi * freq * n * t) / n
        return amp * (4.0 / np.pi) * wave

    def triangle(self, freq: float, duration: float, amp: float = 1.0) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        wave = np.zeros_like(t)
        for n in range(1, 20, 2):
            wave += ((-1)**((n-1)//2)) * np.sin(2 * np.pi * freq * n * t) / (n**2)
        return amp * (8.0 / (np.pi**2)) * wave


class _Resonator:
    """共鸣滤波器（模拟物理共鸣体）"""

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def filter(self, signal: np.ndarray, cutoff: float, resonance: float) -> np.ndarray:
        """
        简单的二阶状态变量滤波器（SVF）。
        可以同时输出低通、高通、带通信号。
        """
        f = 2.0 * np.sin(np.pi * cutoff / self.sr)
        q = 1.0 - resonance
        q = np.clip(q, 0.01, 1.0)

        low = np.zeros_like(signal)
        band = np.zeros_like(signal)
        high = np.zeros_like(signal)

        z1 = 0.0
        z2 = 0.0

        for i in range(len(signal)):
            low[i] = z2 + f * z1
            high[i] = signal[i] - low[i] - q * z1
            band[i] = f * high[i] + z1
            z1 = band[i]
            z2 = low[i]

        return low  # 返回低通输出

    def comb_resonator(self, signal: np.ndarray, delay_ms: float, feedback: float) -> np.ndarray:
        """梳状滤波共鸣器，模拟弦和管"""
        delay_samples = int(delay_ms * self.sr / 1000.0)
        output = np.zeros_like(signal)
        buffer = np.zeros(delay_samples)
        idx = 0
        for i in range(len(signal)):
            delayed = buffer[idx]
            output[i] = signal[i] + feedback * delayed
            buffer[idx] = output[i]
            idx = (idx + 1) % delay_samples
        return output


class _ParticleHarmonics:
    """
    粒子泛音系统。

    用"粒子"模拟泛音的产生：每个粒子有频率、振幅、衰减时间，
    由虚拟电的电荷密度决定粒子数量和分布。
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

    def generate(self, base_freq: float, duration: float, params: MusicParams) -> np.ndarray:
        t = np.linspace(0, duration, int(self.sr * duration), endpoint=False)
        signal = np.zeros_like(t)

        n_particles = params.harmonics
        rng = np.random.default_rng(int(base_freq * 1000))

        for i in range(n_particles):
            # 泛音频率：整数倍 + 微小失谐（模拟物理非理想性）
            harmonic_n = i + 1
            detune = rng.normal(0, 0.5)  #  cents 级别失谐
            freq = base_freq * harmonic_n * (1 + detune / 1200.0)

            # 振幅：按谐波衰减 + 随机波动
            amp = (params.harmonic_decay ** i) * rng.uniform(0.7, 1.0)
            amp *= params.amplitude / n_particles

            # 相位随机
            phase = rng.uniform(0, 2 * np.pi)

            # 衰减包络（每个粒子独立）
            decay_time = rng.uniform(0.1, duration)
            env = np.exp(-t / decay_time)

            # 波形混合：正弦 + 少量噪声
            wave = np.sin(2 * np.pi * freq * t + phase)
            if params.noise_ratio > 0:
                wave += params.noise_ratio * rng.normal(0, 1, size=len(t))

            signal += amp * wave * env

        return signal


class XuniMusic:
    """
    Xuni 虚拟音乐模型。

    输入 MusicParams，输出音频波形。
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        self.osc = _Oscillator(sample_rate)
        self.res = _Resonator(sample_rate)
        self.particles = _ParticleHarmonics(sample_rate)

    def synthesize(self, params: MusicParams, duration: float = 2.0) -> AudioBuffer:
        """
        合成单段音频。

        Args:
            params: 音乐参数
            duration: 音频时长（秒）

        Returns:
            AudioBuffer
        """
        samples = int(self.sr * duration)
        t = np.linspace(0, duration, samples, endpoint=False)

        # 1. 基础振荡器（混合波形）
        base = self.osc.sine(params.base_frequency, duration, amp=0.5)
        base += self.osc.triangle(params.base_frequency, duration, amp=0.3)

        # 2. FM 调制（如果 fm_depth > 0）
        if params.fm_depth > 0:
            mod_freq = params.base_frequency * 0.5
            mod_amp = params.fm_depth
            fm_signal = mod_amp * np.sin(2 * np.pi * mod_freq * t)
            base = np.sin(2 * np.pi * params.base_frequency * t + fm_signal)

        # 3. 粒子泛音
        harmonics = self.particles.generate(params.base_frequency, duration, params)

        # 4. 混合基础波与泛音
        mix = base * 0.4 + harmonics * 0.6

        # 5. 共鸣滤波（模拟物理空间）
        filtered = self.res.filter(mix, params.filter_cutoff, params.filter_resonance)

        # 6. 梳状共鸣（增加空间感）
        delay_ms = 1000.0 / params.base_frequency  # 与基频周期相关
        resonated = self.res.comb_resonator(filtered, delay_ms, feedback=0.4)

        # 7. ADSR 包络
        env = self._adsr_envelope(duration, params)
        final = resonated * env * params.amplitude

        # 8. 3D 声像（立体声）
        stereo = self._pan_3d(final, params)

        return AudioBuffer(
            sample_rate=self.sr,
            data=stereo,
            duration=duration,
        )

    def synthesize_sequence(
        self,
        params_list: List[MusicParams],
        segment_duration: float = 1.0,
        overlap: float = 0.2,
    ) -> AudioBuffer:
        """
        合成参数序列，生成长段音乐。

        Args:
            params_list: 参数列表（每段一个）
            segment_duration: 每段时长
            overlap: 交叉淡化重叠比例
        """
        if not params_list:
            return AudioBuffer(self.sr, np.zeros(0), 0.0)

        segment_samples = int(self.sr * segment_duration)
        overlap_samples = int(self.sr * overlap)
        total_samples = segment_samples + (len(params_list) - 1) * (segment_samples - overlap_samples)
        output = np.zeros((total_samples, 2))

        pos = 0
        for i, params in enumerate(params_list):
            buf = self.synthesize(params, duration=segment_duration)
            stereo = buf.to_stereo()

            # 交叉淡化
            if i > 0 and overlap_samples > 0:
                fade_in = np.linspace(0, 1, overlap_samples).reshape(-1, 1)
                fade_out = np.linspace(1, 0, overlap_samples).reshape(-1, 1)
                output[pos:pos+overlap_samples] *= fade_out
                stereo[:overlap_samples] *= fade_in

            end = min(pos + segment_samples, total_samples)
            stereo_len = end - pos
            output[pos:end] += stereo[:stereo_len]
            pos += segment_samples - overlap_samples

        # 防止削波
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output /= max_val

        return AudioBuffer(sample_rate=self.sr, data=output, duration=total_samples / self.sr)

    def _adsr_envelope(self, duration: float, params: MusicParams) -> np.ndarray:
        """ADSR 包络生成"""
        samples = int(self.sr * duration)
        env = np.zeros(samples)
        t = np.linspace(0, duration, samples, endpoint=False)

        a_samples = int(self.sr * params.attack)
        d_samples = int(self.sr * params.decay)
        r_samples = int(self.sr * params.release)

        # Attack
        if a_samples > 0:
            env[:a_samples] = np.linspace(0, 1, a_samples)

        # Decay
        if d_samples > 0:
            start = a_samples
            end = min(start + d_samples, samples)
            env[start:end] = np.linspace(1, params.sustain, end - start)

        # Sustain
        sustain_start = a_samples + d_samples
        if sustain_start < samples:
            env[sustain_start:-r_samples if r_samples > 0 else samples] = params.sustain

        # Release
        if r_samples > 0 and samples > r_samples:
            env[-r_samples:] = np.linspace(params.sustain, 0, r_samples)

        return env

    def _pan_3d(self, mono: np.ndarray, params: MusicParams) -> np.ndarray:
        """3D 声像：将单声道映射为立体声，用 pan_x 控制左右"""
        stereo = np.zeros((len(mono), 2))
        # 简单的 VBAP 近似
        angle = np.arctan2(params.pan_y, params.pan_x)
        left_gain = np.clip(0.5 + 0.5 * np.cos(angle + np.pi/4), 0, 1)
        right_gain = np.clip(0.5 + 0.5 * np.cos(angle - np.pi/4), 0, 1)

        stereo[:, 0] = mono * left_gain
        stereo[:, 1] = mono * right_gain
        return stereo

    @staticmethod
    def to_wav_bytes(buffer: AudioBuffer) -> bytes:
        """将音频缓冲区转换为 WAV 格式字节（用于 API 传输）"""
        import io
        import wave

        mono = buffer.to_mono()
        # 归一化到 16-bit
        mono = np.clip(mono, -1.0, 1.0)
        pcm = (mono * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(buffer.sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()
