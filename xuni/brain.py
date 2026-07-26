"""
XuniBrain —— 共振神经网络（Resonant Neural Network）

一个完全原创的、基于物理共振原理的神经网络。
它不是传统深度学习（没有反向传播、没有梯度下降），
而是将每个神经元建模为一个耦合振荡器，通过共振同步来"学习"和"记忆"音乐模式。

核心方程（扩展 Kuramoto 模型）：
    dφᵢ/dt = 2πfᵢ + Σⱼ Wᵢⱼ Aⱼ sin(φⱼ - φᵢ) + ξ(t)
    dfᵢ/dt = ε · (f_targetᵢ - fᵢ)
    dAᵢ/dt = -γᵢAᵢ + Σⱼ |Wᵢⱼ| Aⱼ cos(φⱼ - φᵢ)

其中：
- φᵢ: 神经元 i 的瞬时相位
- fᵢ: 固有频率（Hz）
- Aᵢ: 振幅
- Wᵢⱼ: 连接权重（兴奋性为正，抑制性为负）
- γᵢ: 阻尼
- ξ(t): 虚拟电驱动的噪声

输出：O(t) = Σᵢ Aᵢ(t) · sin(φᵢ(t))

创新点：
1. 虚拟电调制：外部场能量实时改变连接权重
2. 频率可塑性：神经元频率不是固定的，会向输入信号靠拢
3. 振幅竞争：类似"赢者通吃"，但由相位相干性决定
4. 连续时间：不是离散前向传播，而是微分方程演化
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class NeuronState:
    """单个神经元的完整状态"""
    phi: float = 0.0      # 相位 [0, 2π)
    freq: float = 440.0   # 固有频率 Hz
    amp: float = 0.1      # 振幅
    gamma: float = 0.05   # 阻尼
    plasticity: float = 0.1  # 可塑性速率
    excitability: float = 1.0  # 兴奋性（调制响应强度）


class XuniBrain:
    """
    共振神经网络。

    参数：
        n_neurons: 神经元数量（建议 64~1024）
        sample_rate: 音频采样率
        dt: 仿真步长（秒）
        connection_density: 连接密度（0~1）
        field_coupling: 虚拟电场对权重的耦合强度
    """

    def __init__(
        self,
        n_neurons: int = 256,
        sample_rate: int = 22050,
        dt: float = 1.0 / 22050.0,
        connection_density: float = 0.3,
        field_coupling: float = 0.5,
        seed: int = 42,
    ):
        self.n = n_neurons
        self.sr = sample_rate
        self.dt = dt
        self.field_coupling = field_coupling
        self.rng = np.random.default_rng(seed)

        # 神经元状态数组
        self.phi = self.rng.random(n_neurons) * 2.0 * np.pi
        self.freq = self._init_frequencies(n_neurons)
        self.amp = np.ones(n_neurons) * 0.05
        self.gamma = self.rng.uniform(0.02, 0.15, n_neurons)
        self.plasticity = self.rng.uniform(0.05, 0.2, n_neurons)

        # 连接权重矩阵 W[i,j]：从 j 到 i 的连接
        self.W = self._init_weights(n_neurons, connection_density)
        self.W_structural = self.W.copy()  # 结构性权重（不变）

        # 记录
        self.history = {
            "freq": [],
            "amp": [],
            "sync": [],
        }
        self._step_count = 0

    def _init_frequencies(self, n: int) -> np.ndarray:
        """
        初始化频率分布。
        使用对数均匀分布覆盖可听范围（20Hz ~ 8kHz），
        并在音乐上重要的频率附近增加密度。
        """
        # 基础对数分布
        log_f = np.linspace(np.log10(55), np.log10(7040), n)
        freqs = 10.0 ** log_f

        # 在音乐三度、五度附近增加密度（谐波吸引子）
        for center in [110, 220, 440, 880, 1760]:
            distances = np.abs(freqs - center)
            nearby = distances < center * 0.1
            freqs[nearby] += self.rng.normal(0, center * 0.02, np.sum(nearby))

        return np.clip(freqs, 20.0, 8000.0)

    def _init_weights(self, n: int, density: float) -> np.ndarray:
        """初始化稀疏连接权重"""
        W = np.zeros((n, n))
        n_connections = int(n * n * density)
        for _ in range(n_connections):
            i = self.rng.integers(0, n)
            j = self.rng.integers(0, n)
            if i == j:
                continue
            # 兴奋性连接为主，少量抑制性
            if self.rng.random() < 0.8:
                W[i, j] = self.rng.exponential(0.3)
            else:
                W[i, j] = -self.rng.exponential(0.15)
        return W

    # ------------------------------------------------------------------
    # 核心动力学
    # ------------------------------------------------------------------
    def _dynamics_step(
        self,
        target_signal: Optional[np.ndarray] = None,
        field_energy: float = 0.0,
    ):
        """
        单步动力学演化。

        Args:
            target_signal: 训练信号，shape (n,) 或 None
            field_energy: 外部虚拟场能量，调制连接权重
        """
        n = self.n
        phi = self.phi
        freq = self.freq
        amp = self.amp
        gamma = self.gamma
        dt = self.dt

        # 1. 虚拟电场调制连接权重
        # 场能量越高，连接越强（类似神经调制物质）
        W_effective = self.W_structural * (1.0 + self.field_coupling * np.tanh(field_energy * 0.1))

        # 2. 相位演化（扩展 Kuramoto）
        # 计算耦合项：sum_j W_ij * A_j * sin(phi_j - phi_i)
        phase_diff = phi[:, None] - phi[None, :]  # (n, n)
        coupling = np.sum(W_effective * amp[None, :] * np.sin(-phase_diff), axis=1)

        # 虚拟电噪声（场能量越大，噪声越有序——类似相干噪声）
        noise_amp = 0.01 + field_energy * 0.001
        noise = self.rng.normal(0, noise_amp, n)

        dphi = (2.0 * np.pi * freq + coupling + noise) * dt
        self.phi = np.mod(phi + dphi, 2.0 * np.pi)

        # 3. 频率可塑性（向训练信号靠拢）
        if target_signal is not None:
            # 提取训练信号中的"主导频率"
            # 简化为目标信号的平均功率谱密度峰值
            f_target = self._extract_target_freq(target_signal)
            # 每个神经元根据与目标的距离调整频率
            dist = np.abs(freq - f_target)
            attraction = np.exp(-dist / (f_target * 0.1))  # 距离越近，吸引力越强
            df = self.plasticity * attraction * (f_target - freq) * dt * 10.0
            self.freq = np.clip(freq + df, 20.0, 8000.0)

        # 4. 振幅演化（竞争 + 阻尼）
        amp_coupling = np.sum(np.abs(W_effective) * amp[None, :] * np.cos(-phase_diff), axis=1)
        damp = -gamma * amp
        damp = (damp + amp_coupling) * dt
        self.amp = np.clip(amp + damp, 0.0, 1.0)

        # 5. 记录历史
        if self._step_count % 100 == 0:
            self.history["freq"].append(self.freq.copy())
            self.history["amp"].append(self.amp.copy())
            # 同步性：R = |<e^{iφ}>|
            sync = np.abs(np.mean(np.exp(1j * self.phi)))
            self.history["sync"].append(float(sync))

        self._step_count += 1

    def _extract_target_freq(self, signal: np.ndarray) -> float:
        """从信号中提取主导频率（简化版自相关法）"""
        # 用零交叉率估计基频
        crossings = np.sum((signal[:-1] * signal[1:]) < 0)
        zcr_freq = crossings * self.sr / (2.0 * len(signal))
        return np.clip(zcr_freq, 50.0, 4000.0)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def stimulate(
        self,
        duration: float,
        target_audio: Optional[np.ndarray] = None,
        field_energy: float = 0.0,
    ) -> np.ndarray:
        """
        刺激网络演化，生成输出音频。

        Args:
            duration: 演化时长（秒）
            target_audio: 训练目标音频（用于培养），None 则为自由生成
            field_energy: 外部虚拟场能量

        Returns:
            输出音频波形
        """
        samples = int(self.sr * duration)
        output = np.zeros(samples)

        # 如果提供目标音频，分帧处理
        frame_size = 256
        if target_audio is not None:
            # 插值或裁剪到所需长度
            if len(target_audio) < samples:
                target_audio = np.tile(target_audio, int(np.ceil(samples / len(target_audio))))[:samples]
            else:
                target_audio = target_audio[:samples]

        for i in range(samples):
            target_frame = None
            if target_audio is not None:
                start = i - frame_size // 2
                end = i + frame_size // 2
                start = max(0, start)
                end = min(len(target_audio), end)
                target_frame = target_audio[start:end]

            self._dynamics_step(
                target_signal=target_frame,
                field_energy=field_energy,
            )
            output[i] = np.sum(self.amp * np.sin(self.phi))

        # 归一化
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val
        return output

    def get_state(self) -> dict:
        """获取当前网络状态快照"""
        return {
            "phi": self.phi.copy(),
            "freq": self.freq.copy(),
            "amp": self.amp.copy(),
            "W": self.W.copy(),
            "W_structural": self.W_structural.copy(),
        }

    def set_state(self, state: dict):
        """加载网络状态快照"""
        self.phi = state["phi"].copy()
        self.freq = state["freq"].copy()
        self.amp = state["amp"].copy()
        self.W = state["W"].copy()
        self.W_structural = state["W_structural"].copy()

    def reset(self):
        """重置到初始混沌状态（不清除学习到的权重）"""
        self.phi = self.rng.random(self.n) * 2.0 * np.pi
        self.amp = np.ones(self.n) * 0.05

    def brain_summary(self) -> dict:
        """网络摘要"""
        sync = np.abs(np.mean(np.exp(1j * self.phi)))
        return {
            "n_neurons": self.n,
            "mean_freq": float(np.mean(self.freq)),
            "freq_std": float(np.std(self.freq)),
            "mean_amp": float(np.mean(self.amp)),
            "max_amp": float(np.max(self.amp)),
            "synchronization": float(sync),
            "W_mean": float(np.mean(np.abs(self.W))),
            "W_max": float(np.max(np.abs(self.W))),
            "steps": self._step_count,
        }
