"""
XuniTrainer —— 共振网络培养器

负责"培养"XuniBrain，使其学会产生特定风格的音乐。
培养不是训练，因为没有损失函数和反向传播。
而是通过以下机制让网络与音乐共振：

1. Hebbian 相位学习：一起共振的神经元加强连接
2. 频率吸引：神经元的固有频率向输入音乐的主导频率靠拢
3. 振幅标记：重要时刻的神经元振幅被强化
4. 场能量注入：虚拟电作为"神经调制物质"改变网络全局状态

培养阶段：
- 聆听期：网络被动接收目标音乐，调整内部频率分布
- 共鸣期：网络开始与目标音乐同步振荡，连接权重重塑
- 自发期：移除目标音乐，网络仅凭内部动力学继续生成
- 固化期：高振幅连接被固化，低振幅连接衰减
"""

import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass

from .brain import XuniBrain


@dataclass
class TrainingConfig:
    """培养配置"""
    listen_ratio: float = 0.3      # 聆听期占比
    resonate_ratio: float = 0.4    # 共鸣期占比
    spontaneous_ratio: float = 0.3 # 自发期占比
    hebbian_rate: float = 0.001    # Hebbian 学习率
    weight_decay: float = 0.0001   # 权重衰减
    field_boost: float = 2.0       # 培养时的场能量增强


class XuniTrainer:
    """
    共振网络培养器。
    """

    def __init__(
        self,
        brain: XuniBrain,
        config: Optional[TrainingConfig] = None,
    ):
        self.brain = brain
        self.config = config or TrainingConfig()
        self.training_log: List[Dict] = []

    def cultivate(
        self,
        target_audio: np.ndarray,
        duration: Optional[float] = None,
        field_energy: float = 0.0,
        epochs: int = 1,
    ) -> np.ndarray:
        """
        培养网络。

        Args:
            target_audio: 目标音乐（单声道，-1~1）
            duration: 培养时长，None 则用音频长度
            field_energy: 基础场能量
            epochs: 重复培养的轮数

        Returns:
            培养结束后网络的自发输出
        """
        if duration is None:
            duration = len(target_audio) / self.brain.sr

        total_samples = int(self.brain.sr * duration)
        listen_samples = int(total_samples * self.config.listen_ratio)
        resonate_samples = int(total_samples * self.config.resonate_ratio)
        spontaneous_samples = int(total_samples * self.config.spontaneous_ratio)

        # 准备目标音频
        if len(target_audio) < total_samples:
            repeats = int(np.ceil(total_samples / len(target_audio)))
            target_audio = np.tile(target_audio, repeats)[:total_samples]
        else:
            target_audio = target_audio[:total_samples]

        # 增强场能量以促进学习
        boosted_field = field_energy + self.config.field_boost

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # 阶段 1：聆听期（被动调整频率）
            print("  [Listen] Network adjusting frequencies...")
            self._phase_listen(target_audio[:listen_samples], boosted_field)

            # 阶段 2：共鸣期（主动同步 + Hebbian 学习）
            print("  [Resonate] Synchronizing and rewiring...")
            self._phase_resonate(
                target_audio[listen_samples:listen_samples + resonate_samples],
                boosted_field,
            )

            # 阶段 3：自发期（移除目标，测试网络内部动力学）
            print("  [Spontaneous] Free generation...")
            output = self._phase_spontaneous(
                spontaneous_samples,
                field_energy,
            )

            # 记录
            summary = self.brain.brain_summary()
            self.training_log.append({
                "epoch": epoch,
                "sync": summary["synchronization"],
                "mean_freq": summary["mean_freq"],
                "W_mean": summary["W_mean"],
            })
            print(f"  Sync: {summary['synchronization']:.4f}, "
                  f"Mean freq: {summary['mean_freq']:.1f} Hz")

        return output

    def _phase_listen(self, audio: np.ndarray, field_energy: float):
        """聆听期：频率调整，不改连接权重"""
        frame_size = 512
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            self.brain._dynamics_step(
                target_signal=frame,
                field_energy=field_energy * 0.5,  # 较低能量，被动接收
            )

    def _phase_resonate(self, audio: np.ndarray, field_energy: float):
        """共鸣期：同步 + Hebbian 学习"""
        frame_size = 256
        W = self.brain.W
        eta = self.config.hebbian_rate
        decay = self.config.weight_decay
        n = self.brain.n

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            old_phi = self.brain.phi.copy()
            old_amp = self.brain.amp.copy()

            self.brain._dynamics_step(
                target_signal=frame,
                field_energy=field_energy,
            )

            # Hebbian 学习：一起共振的神经元加强连接
            # ΔW_ij = η · A_i · A_j · sin(Δφ_ij) - decay · W_ij
            phase_diff = np.sin(self.brain.phi[:, None] - self.brain.phi[None, :])
            amp_product = self.brain.amp[:, None] * self.brain.amp[None, :]

            dW = eta * amp_product * phase_diff - decay * W
            W += dW

            # 限制权重范围
            W = np.clip(W, -2.0, 2.0)
            # 弱化自连接
            np.fill_diagonal(W, 0.0)

        self.brain.W = W
        self.brain.W_structural = W.copy()

    def _phase_spontaneous(self, n_samples: int, field_energy: float) -> np.ndarray:
        """自发期：无目标，自由生成"""
        output = np.zeros(n_samples)
        for i in range(n_samples):
            self.brain._dynamics_step(
                target_signal=None,
                field_energy=field_energy,
            )
            output[i] = np.sum(self.brain.amp * np.sin(self.brain.phi))

        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val
        return output

    def cultivate_from_field(
        self,
        field_energy: float = 0.0,
        duration: float = 5.0,
        epochs: int = 1,
    ) -> np.ndarray:
        """
        纯采样点/场能量驱动的自培养模式——不需要目标音频。

        原理：场能量作为"神经调制物质"注入到 Brain 中，
        网络在能量驱动下自发学习结构，通过 Hebbian 学习强化共振模式。

        Args:
            field_energy: 虚拟场能量（来自采样点发电）
            duration: 每次培养时长（秒）
            epochs: 重复培养轮数

        Returns:
            培养结束后网络的自发输出
        """
        total_samples = int(self.brain.sr * duration)
        samples_per_phase = total_samples // 3

        for epoch in range(epochs):
            # 阶段 1：能量注入 + 频率调整
            self.brain._dynamics_step(
                target_signal=np.sin(2 * np.pi * np.array([440]) * np.arange(128) / self.brain.sr),
                field_energy=field_energy * 0.3,
            )

            # 阶段 2：共振期 - 高能量注入 + Hebbian
            W = self.brain.W
            eta = self.config.hebbian_rate * 2.0
            decay = self.config.weight_decay
            for _ in range(samples_per_phase // 64):
                old_phi = self.brain.phi.copy()
                old_amp = self.brain.amp.copy()
                self.brain._dynamics_step(
                    target_signal=None,
                    field_energy=field_energy,
                )
                phase_diff = np.sin(self.brain.phi[:, None] - self.brain.phi[None, :])
                amp_product = self.brain.amp[:, None] * self.brain.amp[None, :]
                dW = eta * amp_product * phase_diff - decay * W
                W += dW
                W = np.clip(W, -2.0, 2.0)
                np.fill_diagonal(W, 0.0)
            self.brain.W = W
            self.brain.W_structural = W.copy()

            # 阶段 3：自发期 - 自由生成
            output = np.zeros(samples_per_phase)
            for i in range(samples_per_phase):
                self.brain._dynamics_step(
                    target_signal=None,
                    field_energy=field_energy * 0.5,
                )
                output[i] = np.sum(self.brain.amp * np.sin(self.brain.phi))
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output /= max_val

            summary = self.brain.brain_summary()
            self.training_log.append({
                "epoch": epoch,
                "field_energy": field_energy,
                "sync": summary["synchronization"],
                "mean_freq": summary["mean_freq"],
                "W_mean": summary["W_mean"],
            })

        return output

    def imprint_pattern(self, pattern_name: str, duration: float = 3.0) -> dict:
        """
        将当前网络状态固化为一个"印记"（记忆快照）。
        """
        snapshot = self.brain.get_state()
        snapshot["name"] = pattern_name
        snapshot["duration"] = duration
        snapshot["summary"] = self.brain.brain_summary()
        return snapshot

    def evoke_pattern(self, snapshot: dict, duration: float = 5.0) -> np.ndarray:
        """
        唤起一个印记：加载快照并让其自由演化。
        """
        self.brain.set_state(snapshot)
        # 轻微扰动以打破完美对称
        self.brain.phi += np.random.default_rng().normal(0, 0.01, self.brain.n)
        return self.brain.stimulate(duration=duration)

    def get_training_report(self) -> dict:
        """培养报告"""
        return {
            "epochs": len(self.training_log),
            "config": {
                "listen_ratio": self.config.listen_ratio,
                "resonate_ratio": self.config.resonate_ratio,
                "hebbian_rate": self.config.hebbian_rate,
            },
            "progression": self.training_log,
            "final_state": self.brain.brain_summary(),
        }
