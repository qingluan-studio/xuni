"""
XuniConverter —— 能量转换系统

将虚拟电场的能量转换为音乐生成模型的控制参数。
映射哲学：
- 场能量 → 振幅包络（能量越高，声音越响）
- 电势梯度 → 频率调制（电势变化快，频率跳变）
- 电荷密度 → 谐波丰富度（电荷密集，泛音多）
- 电场方向 → 声像定位（3D 空间音频）
- 熵 → 随机性/噪声成分
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class MusicParams:
    """从场能量转换得到的音乐参数"""
    # 基础频率 (Hz)
    base_frequency: float = 440.0
    # 频率偏移/调制深度
    fm_depth: float = 0.0
    # 振幅 (0-1)
    amplitude: float = 0.5
    # 谐波数量
    harmonics: int = 4
    # 谐波衰减系数
    harmonic_decay: float = 0.5
    # 噪声/非谐波成分比例
    noise_ratio: float = 0.1
    # 攻击时间 (s)
    attack: float = 0.01
    # 衰减时间 (s)
    decay: float = 0.1
    #  sustain 电平
    sustain: float = 0.5
    # 释放时间 (s)
    release: float = 0.3
    # 空间位置 (-1 ~ 1)
    pan_x: float = 0.0
    pan_y: float = 0.0
    pan_z: float = 0.0
    # 滤波器截止频率
    filter_cutoff: float = 2000.0
    # 滤波器共振
    filter_resonance: float = 0.5
    # 速度/BPM
    tempo: float = 120.0
    # 节奏复杂度
    rhythmic_complexity: float = 0.5


class XuniConverter:
    """
    能量转换器。

    将 XuniField 的能量分布映射为 MusicParams，驱动 XuniMusic 发声。
    """

    def __init__(
        self,
        base_freq_range: Tuple[float, float] = (55.0, 880.0),  # A1 ~ A5
        tempo_range: Tuple[float, float] = (60.0, 180.0),
        max_harmonics: int = 16,
    ):
        self.f_min, self.f_max = base_freq_range
        self.t_min, self.t_max = tempo_range
        self.max_harmonics = max_harmonics

    def convert(self, field_summary: dict, energy_distribution: np.ndarray) -> MusicParams:
        """
        将场摘要和能量分布转换为音乐参数。
        """
        total_e = field_summary.get("total_energy", 1.0)
        emax = np.max(energy_distribution) if len(energy_distribution) > 0 else 1.0
        emean = np.mean(energy_distribution) if len(energy_distribution) > 0 else 0.0
        estd = np.std(energy_distribution) if len(energy_distribution) > 0 else 0.0

        # 能量归一化
        norm_energy = emean / (emax + 1e-12)
        peak_ratio = emax / (total_e + 1e-12)

        # 1. 基础频率：用能量分布的"质心"映射到对数频率空间
        # 能量越高 → 频率越高（但非线性）
        freq_ratio = np.clip(norm_energy * 2.0, 0.0, 1.0)
        base_freq = self.f_min * (self.f_max / self.f_min) ** freq_ratio

        # 2. FM 深度：能量标准差反映场的不稳定性 → 频率抖动
        fm_depth = np.clip(estd / (emean + 1e-12) * 100.0, 0.0, 500.0)

        # 3. 振幅：总能量直接映射
        amplitude = np.clip(np.log1p(total_e) / 10.0, 0.05, 1.0)

        # 4. 谐波数：峰值能量比 → 谐波丰富度
        harmonics = int(1 + peak_ratio * self.max_harmonics)
        harmonics = max(1, min(self.max_harmonics, harmonics))

        # 5. 谐波衰减：能量均匀度 → 衰减速度
        harmonic_decay = np.clip(1.0 - norm_energy, 0.1, 0.9)

        # 6. 噪声比例：场的"混乱度"
        noise_ratio = np.clip(estd / (emax + 1e-12), 0.0, 0.5)

        # 7. 包络参数：场的响应特性
        attack = np.clip(0.001 + norm_energy * 0.05, 0.001, 0.1)
        decay = np.clip(0.05 + peak_ratio * 0.3, 0.05, 0.5)
        sustain = np.clip(amplitude * 0.7, 0.1, 0.9)
        release = np.clip(0.1 + norm_energy * 0.5, 0.1, 1.0)

        # 8. 空间位置：从场摘要中解析（如果有主导方向）
        pan_x = field_summary.get("dominant_ex", 0.0)
        pan_y = field_summary.get("dominant_ey", 0.0)
        pan_z = field_summary.get("dominant_ez", 0.0)

        # 9. 滤波器：能量均值 → 截止频率
        filter_cutoff = 200.0 + norm_energy * 8000.0
        filter_resonance = np.clip(peak_ratio * 2.0, 0.1, 0.95)

        # 10. 速度：总能量 → BPM
        tempo = self.t_min + norm_energy * (self.t_max - self.t_min)

        # 11. 节奏复杂度：标准差/均值 → 变化丰富度
        rhythmic_complexity = np.clip(estd / (emean + 1e-12) * 0.5, 0.0, 1.0)

        return MusicParams(
            base_frequency=base_freq,
            fm_depth=fm_depth,
            amplitude=amplitude,
            harmonics=harmonics,
            harmonic_decay=harmonic_decay,
            noise_ratio=noise_ratio,
            attack=attack,
            decay=decay,
            sustain=sustain,
            release=release,
            pan_x=pan_x,
            pan_y=pan_y,
            pan_z=pan_z,
            filter_cutoff=filter_cutoff,
            filter_resonance=filter_resonance,
            tempo=tempo,
            rhythmic_complexity=rhythmic_complexity,
        )

    def convert_sequence(
        self,
        field_summaries: List[dict],
        energy_distributions: List[np.ndarray],
    ) -> List[MusicParams]:
        """批量转换，用于生成一段音乐的参数序列"""
        return [
            self.convert(s, e)
            for s, e in zip(field_summaries, energy_distributions)
        ]
