"""
XuniCritic —— 音乐认知不变量评估器（Music Cognitive Invariants）

从 CEE 的 T6 认知几何不变量引擎提取核心思想，适配到音频波形评估：

四大不变量：
    ITC  - Information Topological Compactness（频谱拓扑紧凑度）
    SCS  - Surface Curvature Smoothness（时频曲率平滑度）
    IEC  - Information Entropy Criticality（信息熵临界性）
    PFFT - Projection Fidelity-Flexibility Tradeoff（保真-灵活权衡）

评估流程：
    音频波形 → STFT 分帧 → 特征提取 → 四维不变量 → 综合评分
    低分触发闭环优化：调整采样模式 / Brain 参数 / 合成器参数
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MusicInvariantScores:
    """音乐认知不变量评分"""
    itc: float   # 0~1，频谱能量集中程度
    scs: float   # 0~1，时频过渡平滑度
    iec: float   # 0~1，熵临界性（越接近临界值越高）
    pfft: float  # 0~1，保真-灵活权衡
    overall: float  # 综合评分

    def to_dict(self) -> dict:
        return {
            "itc": round(self.itc, 4),
            "scs": round(self.scs, 4),
            "iec": round(self.iec, 4),
            "pfft": round(self.pfft, 4),
            "overall": round(self.overall, 4),
        }


class XuniCritic:
    """
    音乐评论家。

    不依赖外部神经网络，纯数学/物理方法评估音频质量。
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        frame_size: int = 1024,
        hop_size: int = 512,
        iec_critical: float = 4.0,
        iec_sigma: float = 2.0,
    ):
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.iec_critical = iec_critical
        self.iec_sigma = iec_sigma

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def evaluate(self, audio: np.ndarray) -> MusicInvariantScores:
        """
        评估音频波形，返回四维不变量评分。
        """
        audio = self._preprocess(audio)
        frames = self._stft_frames(audio)
        spectrogram = np.abs(frames)  # (n_frames, n_freqs)

        itc = self._compute_itc(spectrogram)
        scs = self._compute_scs(spectrogram)
        iec = self._compute_iec(spectrogram)
        pfft = self._compute_pfft(audio, spectrogram)

        # 综合评分：四维的谐波均值（避免单一维度垄断）
        overall = self._harmonic_mean([itc, scs, iec, pfft])

        return MusicInvariantScores(itc=itc, scs=scs, iec=iec, pfft=pfft, overall=overall)

    def suggest_optimization(self, scores: MusicInvariantScores) -> Dict[str, str]:
        """
        根据低分维度给出优化建议（闭环优化）。
        """
        suggestions = {}
        if scores.itc < 0.4:
            suggestions["itc"] = "频谱过于分散，建议增加谐波数量或提高基频清晰度"
        if scores.scs < 0.4:
            suggestions["scs"] = "时频过渡过于突兀，建议降低 FM 深度或增加混响"
        if scores.iec < 0.4:
            suggestions["iec"] = "音频过于单调或过于嘈杂，建议调节噪声比例或谐波衰减"
        if scores.pfft < 0.4:
            suggestions["pfft"] = "结构保真与创新失衡，建议调整采样模式或培养更多 epoch"
        if not suggestions:
            suggestions["general"] = "质量良好，可尝试增加场能量或探索新采样模式"
        return suggestions

    # ------------------------------------------------------------------
    # 底层计算
    # ------------------------------------------------------------------
    def _preprocess(self, audio: np.ndarray) -> np.ndarray:
        """预处理：归一化、去直流"""
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def _stft_frames(self, audio: np.ndarray) -> np.ndarray:
        """简化的短时傅里叶变换（矩形窗）"""
        n = len(audio)
        frames = []
        for start in range(0, n - self.frame_size + 1, self.hop_size):
            frame = audio[start:start + self.frame_size]
            spectrum = np.fft.rfft(frame)
            frames.append(spectrum)
        if not frames:
            # 音频太短，直接整段 FFT
            spectrum = np.fft.rfft(audio)
            frames.append(spectrum)
        return np.array(frames)

    def _compute_itc(self, spectrogram: np.ndarray) -> float:
        """
        ITC: Information Topological Compactness
        频谱能量的拓扑紧凑度——能量越集中在少数频段，ITC 越高。
        """
        # 计算平均频谱
        mean_spec = np.mean(spectrogram, axis=0)
        if np.sum(mean_spec) < 1e-12:
            return 0.0

        # 基尼系数：衡量能量集中度（0=均匀分布，1=完全集中）
        gini = self._gini_coefficient(mean_spec)

        # 谐波集中度：检测基频整数倍的能量占比
        harmonic_concentration = self._harmonic_concentration(mean_spec)

        # ITC = 1 - (1 - gini) * (1 - harmonic_concentration)
        # 即：能量越集中、谐波越清晰，ITC 越高
        itc = 1.0 - (1.0 - gini) * (1.0 - harmonic_concentration)
        return float(np.clip(itc, 0.0, 1.0))

    def _compute_scs(self, spectrogram: np.ndarray) -> float:
        """
        SCS: Surface Curvature Smoothness
        时频曲率平滑度——相邻帧频谱变化越平滑，SCS 越高。
        """
        if len(spectrogram) < 2:
            return 0.5

        # 计算相邻帧的余弦相似度
        similarities = []
        for i in range(len(spectrogram) - 1):
            a = spectrogram[i]
            b = spectrogram[i + 1]
            sim = self._cosine_similarity(a, b)
            similarities.append(sim)

        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # SCS = 平均相似度 × (1 - 标准差/2)
        # 高平均相似度 + 低波动 = 高 SCS
        scs = mean_sim * (1.0 - std_sim / 2.0)
        return float(np.clip(scs, 0.0, 1.0))

    def _compute_iec(self, spectrogram: np.ndarray) -> float:
        """
        IEC: Information Entropy Criticality
        信息熵临界性——音频熵越接近临界值（有序-混沌边缘），IEC 越高。
        """
        # 计算每帧的功率谱熵
        entropies = []
        for frame in spectrogram:
            power = frame ** 2
            total = np.sum(power)
            if total < 1e-12:
                entropies.append(0.0)
                continue
            probs = power / total
            entropy = -np.sum(probs * np.log2(probs + 1e-12))
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)

        # IEC = exp(-|H - H_crit| / sigma)
        # H_crit 是临界熵（有序-混沌边缘）
        iec = np.exp(-abs(mean_entropy - self.iec_critical) / self.iec_sigma)
        return float(np.clip(iec, 0.0, 1.0))

    def _compute_pfft(self, audio: np.ndarray, spectrogram: np.ndarray) -> float:
        """
        PFFT: Projection Fidelity-Flexibility Tradeoff
        保真-灵活权衡——音乐既要有结构（周期性/自相关）又要有变化（频谱演变）。
        """
        # 保真度 Fidelity：自相关峰值（检测周期性/重复结构）
        fidelity = self._autocorrelation_fidelity(audio)

        # 灵活度 Flexibility：频谱变化率的标准差
        flexibility = self._spectral_flexibility(spectrogram)

        # PFFT = 2 * F * D / (F + D)  谐波均值
        if fidelity + flexibility < 1e-12:
            return 0.0
        pfft = 2.0 * fidelity * flexibility / (fidelity + flexibility)
        return float(np.clip(pfft, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 辅助函数
    # ------------------------------------------------------------------
    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """基尼系数，衡量分布集中度"""
        values = np.array(values).flatten()
        if np.sum(values) < 1e-12:
            return 0.0
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        return float(np.clip(gini, 0.0, 1.0))

    @staticmethod
    def _harmonic_concentration(spectrum: np.ndarray) -> float:
        """
        检测基频整数倍的能量占比。
        简化版：假设基频在能量最大的 bin，计算其前 8 个谐波的能量占比。
        """
        if np.sum(spectrum) < 1e-12:
            return 0.0
        peak_idx = np.argmax(spectrum)
        if peak_idx == 0:
            return 0.0

        harmonic_energy = 0.0
        for h in range(1, 9):
            idx = peak_idx * h
            if idx >= len(spectrum):
                break
            harmonic_energy += spectrum[idx]

        return float(np.clip(harmonic_energy / np.sum(spectrum), 0.0, 1.0))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0))

    @staticmethod
    def _autocorrelation_fidelity(audio: np.ndarray, max_lag: int = 2000) -> float:
        """
        自相关峰值作为结构保真度度量。
        峰值越高，说明音频周期性越强（有旋律/节奏结构）。
        """
        audio = audio - np.mean(audio)
        n = len(audio)
        max_lag = min(max_lag, n // 2)
        if max_lag < 2:
            return 0.5

        # 快速自相关（利用 FFT）
        fft_audio = np.fft.fft(audio, n=2 * n)
        autocorr = np.fft.ifft(fft_audio * np.conj(fft_audio)).real[:max_lag]
        autocorr = autocorr / autocorr[0]  # 归一化

        # 排除零滞后，找最大峰值
        peak = np.max(autocorr[1:]) if len(autocorr) > 1 else 0.0
        return float(np.clip(peak, 0.0, 1.0))

    @staticmethod
    def _spectral_flexibility(spectrogram: np.ndarray) -> float:
        """
        频谱灵活性：衡量频谱随时间的变化丰富度。
        用相邻帧频谱差异的标准差，归一化到 0~1。
        """
        if len(spectrogram) < 2:
            return 0.5

        diffs = []
        for i in range(len(spectrogram) - 1):
            diff = np.mean(np.abs(spectrogram[i + 1] - spectrogram[i]))
            diffs.append(diff)

        mean_diff = np.mean(diffs)
        # 归一化：差异太大也不好，用 tanh 压缩
        flexibility = np.tanh(mean_diff * 2.0)
        return float(np.clip(flexibility, 0.0, 1.0))

    @staticmethod
    def _harmonic_mean(values: list) -> float:
        """谐波均值，避免单一维度垄断"""
        values = [v for v in values if v > 1e-12]
        if not values:
            return 0.0
        return len(values) / sum(1.0 / v for v in values)
