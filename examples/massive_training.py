"""
Xuni 海量训练引擎 —— 虚拟音乐模型的大规模自培养

核心理念：
    采样点 → 发电（虚拟电场）→ 注入 Brain（共振网络）
    → Hebbian 自组织 → Critic 评估 → Explorer 策略优化
    → Memory 记忆最佳模式 → 输出训练后的音乐

三阶段海量管线：
    Phase 1: 海量采样发电（千万级采样点，多模态并行）
    Phase 2: 共振网络培养（数千步演化，场能量调制）
    Phase 3: 质量闭环优化（Critic + Explorer 自动调优）
"""

import numpy as np
import time
import os
from datetime import datetime

from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.converter import XuniConverter
from xuni.music import XuniMusic
from xuni.brain import XuniBrain
from xuni.trainer import XuniTrainer, TrainingConfig
from xuni.critic import XuniCritic
from xuni.explorer import XuniExplorer, SamplingStrategy
from xuni.memory import XuniMemory, MemoryBank, MemoryType
from xuni.overseer import XuniOverseer, OverseerConfig


class MassiveTrainingEngine:
    """
    海量训练引擎 —— 将采样点的"电"注入虚拟音乐模型，
    通过大规模自组织训练产出高质量音乐。
    """

    def __init__(
        self,
        total_samples: int = 10_000_000,
        batch_size: int = 200_000,
        n_neurons: int = 512,
        grid_size: int = 32,
        field_iterations: int = 80,
        training_epochs: int = 100,
        music_duration: float = 8.0,
        output_dir: str = "massive_output",
        seed: int = 42,
    ):
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.grid_size = grid_size
        self.field_iterations = field_iterations
        self.training_epochs = training_epochs
        self.music_duration = music_duration
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        os.makedirs(output_dir, exist_ok=True)

        self.field = XuniField(
            grid_size=(grid_size, grid_size, grid_size),
            bounds=(-80.0, 80.0),
            smooth_sigma=1.8,
        )

        self.converter = XuniConverter(
            base_freq_range=(55.0, 1320.0),
        )

        self.music = XuniMusic(sample_rate=22050)

        self.brain = XuniBrain(
            n_neurons=n_neurons,
            sample_rate=22050,
            connection_density=0.35,
            field_coupling=0.6,
            seed=seed,
        )

        self.trainer = XuniTrainer(
            self.brain,
            config=TrainingConfig(
                listen_ratio=0.25,
                resonate_ratio=0.45,
                spontaneous_ratio=0.30,
                hebbian_rate=0.002,
                weight_decay=0.00005,
                field_boost=3.0,
            ),
        )

        self.critic = XuniCritic(sample_rate=22050)
        self.memory = MemoryBank(stm_capacity=30)
        self.resonance_memory = XuniMemory(self.brain)

        self.explorer = XuniExplorer(epsilon=0.4, use_thompson=True, use_novelty=True)
        self._register_strategies()

        self.overseer = XuniOverseer(OverseerConfig(
            auto_intervention=True,
            intervention_cooldown=3,
            max_consecutive_crashes=8,
        ))

        self.history = []
        self.best_score = 0.0
        self.best_audio = None
        self.start_time = None

    def _register_strategies(self):
        """注册所有可探索的策略"""
        for mode in SamplingStrategy:
            self.explorer.register_sampling_mode(mode)

        brain_configs = [
            (256, 0.3, "brain_small"),
            (512, 0.6, "brain_medium"),
            (1024, 0.8, "brain_large"),
        ]
        for n, fc, label in brain_configs:
            self.explorer.register_brain_config(n, fc, label)

    def run(self):
        """启动海量训练主循环"""
        self.start_time = time.time()
        self._print_header()

        for epoch in range(self.training_epochs):
            phase_start = time.time()

            sampling_mode, params = self.explorer.select_strategy("sample")
            mode_enum = SamplingMode[sampling_mode.replace("sample_", "").upper()]

            print(f"\n{'='*60}")
            print(f"  EPOCH {epoch + 1}/{self.training_epochs}")
            print(f"  Mode: {mode_enum.value} | Strategy: {sampling_mode}")
            print(f"{'='*60}")

            field_energy = self._phase_generate_electricity(mode_enum)

            audio = self._phase_train_brain(field_energy, epoch)

            scores = self._phase_evaluate(audio, epoch)

            self._phase_memorize(audio, scores, epoch, sampling_mode, field_energy)

            self.explorer.feedback(sampling_mode, scores.overall)

            watch_result = self.overseer.watch(
                epoch=epoch,
                field_energy=field_energy,
                scores=scores,
                brain_summary=self.brain.brain_summary(),
                audio=audio,
                current_strategy=sampling_mode,
            )
            self.overseer.print_watch_summary(watch_result)

            if watch_result.intervention != "NONE":
                self._handle_intervention(watch_result, epoch)

            if self.overseer.should_early_stop():
                print("\n  [监管] 触发早停，训练终止")
                break

            elapsed = time.time() - phase_start
            self.history.append({
                "epoch": epoch,
                "mode": mode_enum.value,
                "field_energy": field_energy,
                "scores": scores.to_dict(),
                "time": elapsed,
            })

            self._print_epoch_summary(epoch, field_energy, scores, elapsed)

        self._final_render()
        self.overseer.print_final_report()

    def _handle_intervention(self, watch_result, epoch):
        from xuni.overseer import InterventionType

        intervention = watch_result.intervention
        if intervention == InterventionType.NONE:
            return

        print(f"  [监管] 执行干预: {intervention.name}")

        result = self.overseer.execute_intervention(
            intervention, self.brain, self.field, self.explorer
        )
        print(f"  [监管] 干预结果: {result}")

    def _phase_generate_electricity(self, mode: SamplingMode) -> float:
        """
        Phase 1: 海量采样发电
        用数百万采样点构建虚拟电场，提取场能量。
        """
        print(f"  [GEN] 采样发电中... ({self.total_samples:,} points)")

        self.field.reset()
        sampler = XuniSampler(
            mode=mode,
            seed=self.rng.integers(0, 100000),
            dt=0.0005,
        )
        stream = sampler.generate_stream()

        n_batches = self.total_samples // self.batch_size
        for i in range(n_batches):
            batch = np.zeros((self.batch_size, 6), dtype=np.float64)
            for j in range(self.batch_size):
                try:
                    pt = next(stream)
                    batch[j] = pt.to_array()
                except StopIteration:
                    batch = batch[:j]
                    break
            self.field.ingest_batch(batch)
            if (i + 1) % max(1, n_batches // 5) == 0:
                dots = "." * ((i + 1) * 50 // n_batches)
                print(f"\r    [{dots:<50}] {i+1}/{n_batches} batches", end="", flush=True)

        print()
        self.field.compute_field(iterations=self.field_iterations)
        total_energy = self.field.get_total_energy()
        summary = self.field.field_summary()

        print(f"    场能量: {total_energy:,.2f} | 采样点: {self.field._total_samples:,} "
              f"| 最大场强: {summary['max_field_strength']:.4f}")
        return float(np.log1p(total_energy) * 0.5)

    def _phase_train_brain(self, field_energy: float, epoch: int) -> np.ndarray:
        """
        Phase 2: 共振网络培养
        场能量作为神经调制物质注入 Brain，驱动 Hebbian 自组织学习。
        """
        print(f"  [TRAIN] 场能量 {field_energy:.2f} 注入共振网络...")

        selected_duration = self.music_duration * (1.0 + 0.5 * np.sin(epoch * 0.3))
        audio = self.trainer.cultivate_from_field(
            field_energy=field_energy,
            duration=selected_duration,
            epochs=1,
        )
        return audio

    def _phase_evaluate(self, audio: np.ndarray, epoch: int):
        """
        Phase 3: 质量评估
        四维认知不变量评估 + 闭环优化建议。
        """
        scores = self.critic.evaluate(audio)
        suggestions = self.critic.suggest_optimization(scores)

        print(f"  [EVAL] ITC={scores.itc:.3f} SCS={scores.scs:.3f} "
              f"IEC={scores.iec:.3f} PFFT={scores.pfft:.3f} "
              f"=> Overall={scores.overall:.3f}")

        if suggestions:
            for k, v in suggestions.items():
                print(f"         [{k}] {v}")

        if scores.overall > self.best_score:
            self.best_score = scores.overall
            self.best_audio = audio.copy()
            print(f"         *** 新最佳评分 {self.best_score:.4f}! ***")

        return scores

    def _phase_memorize(
        self,
        audio: np.ndarray,
        scores,
        epoch: int,
        strategy_name: str,
        field_energy: float,
    ):
        """Phase 4: 记忆最佳模式"""
        summary = self.brain.brain_summary()
        importance = scores.overall * 0.7 + 0.3 * np.tanh(field_energy * 0.1)

        self.memory.memorize(
            content=f"epoch_{epoch}_sync={summary['synchronization']:.4f}_score={scores.overall:.3f}",
            memory_type=MemoryType.EXPERIENCE,
            importance=importance,
            tags=[strategy_name, f"score_{scores.overall:.2f}"],
            metadata={
                "epoch": epoch,
                "field_energy": field_energy,
                "scores": scores.to_dict(),
                "sync": summary["synchronization"],
                "mean_freq": summary["mean_freq"],
            },
        )

        if epoch % 10 == 0 or scores.overall > 0.6:
            capture_name = f"best_epoch_{epoch}_score_{scores.overall:.3f}"
            self.resonance_memory.capture(
                name=capture_name,
                tags=["best", f"epoch_{epoch}", f"score_{scores.overall:.2f}"],
            )

        if epoch % 20 == 0:
            self.memory.consolidate()

    def _print_header(self):
        print("\n" + "=" * 60)
        print("  XUNI 海量训练引擎")
        print("  虚拟音乐模型 — 大规模自培养")
        print("=" * 60)
        print(f"  总采样点: {self.total_samples:,}")
        print(f"  神经元:   {self.n_neurons}")
        print(f"  网格:     {self.grid_size}x{self.grid_size}x{self.grid_size}")
        print(f"  Epochs:   {self.training_epochs}")
        print(f"  输出目录: {self.output_dir}/")
        print("=" * 60)

    def _print_epoch_summary(self, epoch: int, energy: float, scores, elapsed: float):
        total_elapsed = time.time() - self.start_time
        est_remaining = (total_elapsed / (epoch + 1)) * (self.training_epochs - epoch - 1)
        print(f"    耗时: {elapsed:.1f}s | 累计: {total_elapsed/60:.1f}min "
              f"| 预计剩余: {est_remaining/60:.1f}min")

    def _final_render(self):
        """最终渲染：输出最佳音频 + 训练报告"""
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"  海量训练完成！")
        print(f"  总耗时: {total_time/60:.1f} min")
        print(f"  总Epochs: {self.training_epochs}")
        print(f"  最佳评分: {self.best_score:.4f}")
        print(f"  记忆库: STM={self.memory.stm.size}, LTM={len(self.memory.ltm._store)}")
        print(f"  共振记忆: {len(self.resonance_memory.memories)} 个")
        print(f"{'='*60}")

        explorer_report = self.explorer.get_report()
        print(f"\n策略探索报告 (epsilon={explorer_report['epsilon']:.3f}):")
        for s in explorer_report["strategies"]:
            novelty_flag = " [NOVELTY!]" if s["novelty"] else ""
            print(f"  {s['name']}: avg={s['avg_score']:.4f} trials={s['trials']}{novelty_flag}")

        if self.best_audio is not None and len(self.best_audio) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"massive_trained_{timestamp}.wav")
            buf = self.music.__class__(sample_rate=22050)
            from xuni.music import AudioBuffer
            audio_buf = AudioBuffer(
                sample_rate=22050,
                data=self.best_audio.reshape(-1, 1),
                duration=len(self.best_audio) / 22050,
            )
            wav = XuniMusic.to_wav_bytes(audio_buf)
            with open(output_path, "wb") as f:
                f.write(wav)
            print(f"\n最佳音频已保存: {output_path}")

        memory_report = self.memory.report()
        print(f"\n记忆银行状态: {memory_report}")

        print(f"\n海量训练引擎关闭。\n")


def quick_massive():
    """快速海量训练（测试用，较小规模）"""
    engine = MassiveTrainingEngine(
        total_samples=1_000_000,
        batch_size=100_000,
        n_neurons=256,
        grid_size=24,
        field_iterations=50,
        training_epochs=30,
        music_duration=4.0,
        output_dir="massive_output",
    )
    engine.run()


def standard_massive():
    """标准海量训练（千万级采样点）"""
    engine = MassiveTrainingEngine(
        total_samples=10_000_000,
        batch_size=200_000,
        n_neurons=512,
        grid_size=32,
        field_iterations=80,
        training_epochs=100,
        music_duration=8.0,
        output_dir="massive_output",
    )
    engine.run()


def extreme_massive():
    """究极海量训练（亿级采样点）"""
    engine = MassiveTrainingEngine(
        total_samples=100_000_000,
        batch_size=500_000,
        n_neurons=1024,
        grid_size=40,
        field_iterations=120,
        training_epochs=200,
        music_duration=12.0,
        output_dir="massive_output",
    )
    engine.run()


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "standard"

    modes = {
        "quick": quick_massive,
        "standard": standard_massive,
        "extreme": extreme_massive,
    }

    if mode in modes:
        print(f"启动海量训练模式: {mode}")
        modes[mode]()
    else:
        print(f"未知模式: {mode}")
        print(f"可选: {list(modes.keys())}")
        print("默认运行标准模式...")
        standard_massive()
