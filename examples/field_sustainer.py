"""
FieldSustainer —— 后台持续发电引擎

维持虚拟电场能量，定时产出采样点并注入场中。
支持多模式轮换、场能量监控、自动调参。
"""

import time
import signal
import sys
import threading
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, "/workspace")
from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField


@dataclass
class SustainerState:
    running: bool = True
    total_samples: int = 0
    total_energy: float = 0.0
    current_mode: str = "hyper_chaos"
    peak_energy: float = 0.0
    batch_count: int = 0
    last_report: float = 0.0
    errors: list = field(default_factory=list)


_state = SustainerState()


def _report(field: XuniField):
    """输出当前状态"""
    summary = field.field_summary()
    energy = float(summary["total_energy"])
    _state.total_energy = energy
    if energy > _state.peak_energy:
        _state.peak_energy = energy

    elapsed = time.time() - _state.last_report
    _state.last_report = time.time()

    print(
        f"[{'SUSTAIN':>7}] "
        f"batch=#{_state.batch_count:04d} "
        f"samples={_state.total_samples:>10,.0f} "
        f"energy={energy:>10.2f} "
        f"peak={_state.peak_energy:>10.2f} "
        f"mode={_state.current_mode:>16s} "
        f"dt={elapsed:.1f}s"
    )


def sustain_loop(
    mode_sequence: list = None,
    batch_size: int = 100000,
    grid_size: tuple = (16, 16, 16),
    report_interval: float = 2.0,
):
    """
    主循环：持续产出采样点并注入场，维持能量。

    mode_sequence: 模式轮换列表，None 则只用 hyper_chaos
    """
    if mode_sequence is None:
        mode_sequence = [SamplingMode.NOISE_FIELD, SamplingMode.HYBRID,
                         SamplingMode.LORENZ_96, SamplingMode.MANDELBULB,
                         SamplingMode.HYPER_CHAOS]

    field = XuniField(grid_size=grid_size)
    mode_idx = 0
    _state.last_report = time.time()

    print("=" * 60)
    print("FieldSustainer 启动")
    print(f"  批次大小: {batch_size:,}")
    print(f"  场网格: {grid_size}")
    print(f"  模式轮换: {[m.name for m in mode_sequence]}")
    print(f"  报告间隔: {report_interval}s")
    print("=" * 60)

    while _state.running:
        try:
            mode = mode_sequence[mode_idx % len(mode_sequence)]
            _state.current_mode = mode.name.lower()
            mode_idx += 1

            rng = np.random.default_rng()
            sampler = XuniSampler(mode=mode, seed=rng.integers(0, 2**31))
            batch = sampler.generate_batch(batch_size)

            mask = np.isfinite(batch[:, 0]) & np.isfinite(batch[:, 1]) & np.isfinite(batch[:, 2])
            valid = batch[mask]
            if len(valid) == 0:
                _state.errors.append(f"batch #{_state.batch_count}: all NaN")
                time.sleep(1.0)
                continue

            field.ingest_batch(valid)
            field.compute_field()

            _state.total_samples += len(valid)
            _state.batch_count += 1

            _report(field)
            time.sleep(report_interval)

        except Exception as e:
            _state.errors.append(str(e))
            print(f"[ERROR] {e}")
            time.sleep(1.0)


def main():
    signal.signal(signal.SIGTERM, lambda *_: setattr(_state, "running", False))
    signal.signal(signal.SIGINT, lambda *_: setattr(_state, "running", False))

    try:
        sustain_loop()
    except KeyboardInterrupt:
        pass

    print(f"\nFieldSustainer 停止。总计 {_state.total_samples:,} 采样点，峰值能量 {_state.peak_energy:.2f}")


if __name__ == "__main__":
    main()
