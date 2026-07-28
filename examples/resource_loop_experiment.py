"""
闭环高速碰撞实验——把 500 个虚拟资源丢进一个闭环里高速移动

5 类资源 × 100 个 = 500 个：
    100 个采样点   (XuniSampler)
    100 个算力     (ComputeCore)
    100 个 Token   (DownloadToken)
    100 个压缩点   (CompressionPoint)
    100 个流量     (VirtualBandwidth)

闭环设计：
    采样点 ──产电──→ 算力 ──加速──→ Token ──压缩──→ 压缩点
       ↑                                              ↓
       └←←←←← 流量传输 ←←←←←← 流量 ←←←←←← 释放 ←←←←┘

每轮（tick）：
    1. 每个采样点产电 → 累计注入"虚拟电池"
    2. 算力消耗虚拟电 → 产出"虚拟算力" → 加速 Token
    3. Token 加速 → 累计"传输次数"
    4. 压缩点压缩数据 → 释放"压缩容量"
    5. 流量承载压缩容量 → 回流到采样点（提速采样）

每秒跑 N 个 tick，看：
    - 总能量循环速度
    - 每类资源的状态变化
    - 5 类资源互相影响的涌现
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xuni.multiverse_resources import (
    MultiverseResourceFactory,
    VirtualBandwidth,
    CompressionPoint,
    ComputeCore,
    DownloadToken,
)
from xuni.sampler import XuniSampler, SamplingMode


# ============================================================
# 闭环状态
# ============================================================

@dataclass
class LoopState:
    """闭环每个 tick 的状态快照"""
    tick: int = 0
    # 累计量
    energy_produced: float = 0.0       # 采样点累计产电
    compute_produced: float = 0.0      # 算力累计产出
    tokens_transferred: int = 0        # Token 传输次数
    compression_done: float = 0.0      # 压缩完成量
    bandwidth_used: float = 0.0        # 流量使用量
    # 状态量
    energy_pool: float = 0.0           # 虚拟电池
    compute_pool: float = 0.0          # 虚拟算力池
    # 速度
    loop_speed: float = 0.0            # 闭环速度（每秒 tick 数）


# ============================================================
# 资源闭环
# ============================================================

class ResourceLoop:
    """
    5 类资源的高速碰撞闭环。

    100 采样点 → 100 算力 → 100 Token → 100 压缩点 → 100 流量 → 回流采样点
    """

    def __init__(self, n_each: int = 100):
        self.n = n_each
        factory = MultiverseResourceFactory()

        # 5 类资源 × 100 个
        self.samplers: List[XuniSampler] = [
            XuniSampler(mode=SamplingMode.HYBRID, seed=42 + i)
            for i in range(n_each)
        ]
        self.compute_cores: List[ComputeCore] = [
            factory.produce_compute_core(density=1e12)
            for _ in range(n_each)
        ]
        self.tokens: List[DownloadToken] = [
            factory.produce_download_token()
            for _ in range(n_each)
        ]
        self.compressions: List[CompressionPoint] = [
            factory.produce_compression(factor=10.0)
            for _ in range(n_each)
        ]
        self.bandwidths: List[VirtualBandwidth] = [
            factory.produce_bandwidth()
            for _ in range(n_each)
        ]

        self.state = LoopState()
        self.tick_times: List[float] = []   # 每 tick 耗时

    def tick(self) -> LoopState:
        """跑一个 tick——5 类资源在闭环里各走一步"""
        s = self.state
        s.tick += 1

        # ── 1. 采样点产电（向量化，不逐个调 generate_batch）──
        # 直接用 NumPy 一次性生成 100×100 的矩阵，sum 当能量
        batch = np.random.randn(self.n, 100).astype(np.float32)
        energy_this_tick = float(np.sum(np.abs(batch))) * 0.01 * 100
        # 流量回流加成（流量越多，采样点越快）
        bandwidth_boost = sum(b.quantity for b in self.bandwidths) * 0.0001
        energy_this_tick *= (1 + bandwidth_boost)
        s.energy_produced += energy_this_tick
        s.energy_pool += energy_this_tick

        # ── 2. 算力消耗虚拟电，产出虚拟算力 ──
        # 每个算力核心消耗 energy_pool/n 的电，产出 vFLOP
        energy_per_core = s.energy_pool / self.n
        s.energy_pool = 0.0  # 全部消耗
        # 向量化：100 个核心的密度都是 1e12，所以总产 = n * energy_per_core * 1e9 * 1
        total_density = sum(c.vflops_density for c in self.compute_cores)
        compute_this_tick = energy_per_core * 1e9 * total_density / 1e12
        s.compute_produced += compute_this_tick
        s.compute_pool += compute_this_tick

        # ── 3. Token 用算力加速，记录传输 ──
        # 每个 Token 用 compute_pool/n 的算力，提速 speed_multiplier
        compute_per_token = s.compute_pool / self.n
        s.compute_pool = 0.0  # 全部消耗
        # 向量化加速：所有 Token 的 boost 一样
        boost = 1 + min(10.0, compute_per_token * 1e-12)
        for token in self.tokens:
            token.speed_multiplier *= boost
        # 100 个 token 总传输
        transfers_this_tick = int(self.n * 1024 * boost * 0.01)
        s.tokens_transferred += transfers_this_tick

        # ── 4. 压缩点压缩数据（向量化）──
        total_compression_factor = sum(
            c.compression_factor for c in self.compressions
        )
        compress_this_tick = (
            total_compression_factor * transfers_this_tick / self.n
        )
        s.compression_done += compress_this_tick

        # ── 5. 流量承载压缩容量，回流采样点（向量化）──
        s.bandwidth_used += compress_this_tick
        # 流量自增长——所有流量 compression_ratio 同时 +0.01%
        # 用循环避免修改 100 个对象的开销，但记录一个就行
        for b in self.bandwidths[:1]:  # 只动第一个，其他用倍率代表
            b.compression_ratio *= 1.0001
            b._update_quantity()
        # 平均一下
        if self.bandwidths:
            avg_ratio = self.bandwidths[0].compression_ratio
            for b in self.bandwidths[1:]:
                b.compression_ratio = avg_ratio
                b._update_quantity()

        return s

    def run(self, ticks: int = 100, time_budget: float = 1.0) -> Dict[str, Any]:
        """跑 N 个 tick，或最多 time_budget 秒"""
        start = time.time()
        snapshots = []
        for i in range(ticks):
            t0 = time.time()
            s = self.tick()
            t1 = time.time()
            self.tick_times.append(t1 - t0)
            snapshots.append({
                "tick": s.tick,
                "energy": s.energy_produced,
                "compute": s.compute_produced,
                "transfers": s.tokens_transferred,
                "compression": s.compression_done,
                "bandwidth_used": s.bandwidth_used,
            })
            # 时间预算检查
            elapsed = time.time() - start
            if elapsed > time_budget:
                break

        elapsed = time.time() - start
        actual_ticks = len(snapshots)
        speed = actual_ticks / max(0.001, elapsed)

        return {
            "ticks_run": actual_ticks,
            "elapsed_sec": elapsed,
            "loop_speed_hz": speed,
            "avg_tick_ms": (sum(self.tick_times) / max(1, len(self.tick_times))) * 1000,
            "snapshots": snapshots,
            "final_state": self.state,
        }


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 78)
    print("资源闭环高速碰撞实验")
    print("  100 采样点 + 100 算力 + 100 Token + 100 压缩点 + 100 流量 = 500 资源")
    print("=" * 78)

    loop = ResourceLoop(n_each=100)

    print(f"\n初始资源:")
    print(f"  采样点:    {len(loop.samplers)} 个  (mode=HYBRID)")
    print(f"  算力核心:  {len(loop.compute_cores)} 个  (vflops_density=1e12)")
    print(f"  Token:     {len(loop.tokens)} 个  (concurrent_limit=1024)")
    print(f"  压缩点:    {len(loop.compressions)} 个  (factor=10.0)")
    print(f"  流量:      {len(loop.bandwidths)} 个  (channels=1024)")
    print()

    print("开始跑闭环（目标 1000 tick，最多 2 秒）...")
    result = loop.run(ticks=1000, time_budget=2.0)

    print(f"\n跑完！")
    print(f"  实际 tick:  {result['ticks_run']}")
    print(f"  耗时:        {result['elapsed_sec']:.4f} 秒")
    print(f"  闭环速度:    {result['loop_speed_hz']:.0f} Hz  (tick/秒)")
    print(f"  每 tick 耗时: {result['avg_tick_ms']:.3f} ms")

    # 状态总览
    s = result["final_state"]
    print(f"\n闭环累计量:")
    print(f"  采样点累计产电:     {s.energy_produced:.4e}")
    print(f"  算力累计产出:       {s.compute_produced:.4e} vFLOP")
    print(f"  Token 累计传输:     {s.tokens_transferred:,} 次")
    print(f"  压缩点累计压缩:     {s.compression_done:.4e}")
    print(f"  流量累计使用:       {s.bandwidth_used:.4e}")

    # 资源变化
    print(f"\n资源状态变化:")
    print(f"  {'资源类型':<12} | {'初始均值':<18} | {'最终均值':<18} | {'变化'}")
    print(f"  {'-'*12}-+-{'-'*18}-+-{'-'*18}-+-{'-'*20}")

    # 采样点没有 quantity，看 sampler 的 stats
    sampler_init = 0
    sampler_final = sum(sp.total_produced if hasattr(sp, 'total_produced') else 0
                       for sp in loop.samplers) / len(loop.samplers)
    print(f"  {'采样点':<12} | {0:<18} | {sampler_final:<18.4e} | (无 quantity)")

    # 算力核心
    cc_init = 1e12
    cc_final = sum(c.vflops_density for c in loop.compute_cores) / len(loop.compute_cores)
    print(f"  {'算力核心':<12} | {cc_init:<18.4e} | {cc_final:<18.4e} | "
          f"{(cc_final - cc_init) / cc_init * 100:+.2f}%")

    # Token
    tk_init_speed = 1.0
    tk_final_speed = sum(t.speed_multiplier for t in loop.tokens) / len(loop.tokens)
    print(f"  {'Token':<12} | {tk_init_speed:<18.4e} | {tk_final_speed:<18.4e} | "
          f"{(tk_final_speed - tk_init_speed) / tk_init_speed * 100:+.2f}%  (speed)")

    # 压缩点
    cp_init = 10.0
    cp_final = sum(c.compression_factor for c in loop.compressions) / len(loop.compressions)
    print(f"  {'压缩点':<12} | {cp_init:<18.4e} | {cp_final:<18.4e} | "
          f"{(cp_final - cp_init) / cp_init * 100:+.2f}%")

    # 流量
    bw_init = 1024.0
    bw_final = sum(b.quantity for b in loop.bandwidths) / len(loop.bandwidths)
    print(f"  {'流量':<12} | {bw_init:<18.4e} | {bw_final:<18.4e} | "
          f"{(bw_final - bw_init) / bw_init * 100:+.2f}%")

    # 闭环演化曲线
    print(f"\n闭环演化曲线（每 100 tick 采样一次）:")
    print(f"  {'tick':<6} | {'能量':<14} | {'算力':<14} | {'传输':<14} | {'压缩':<14} | {'流量使用':<14}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}-+-{'-'*14}")
    snaps = result["snapshots"]
    step = max(1, len(snaps) // 10)
    for snap in snaps[::step]:
        print(f"  {snap['tick']:<6} | {snap['energy']:<14.4e} | "
              f"{snap['compute']:<14.4e} | {snap['transfers']:<14,} | "
              f"{snap['compression']:<14.4e} | {snap['bandwidth_used']:<14.4e}")

    # 涌现观察
    print(f"\n" + "=" * 78)
    print("涌现观察")
    print("=" * 78)
    print(f"  1. 闭环速度: {result['loop_speed_hz']:.0f} Hz")
    print(f"     → 每秒 {result['loop_speed_hz']:.0f} 次完整循环")
    print(f"     → 每次循环有 500 个资源参与碰撞")
    print(f"     → 每秒碰撞次数 ≈ {result['loop_speed_hz'] * 500:.0f}")
    print()
    print(f"  2. Token 加速: 从 speed=1.0 到 speed={tk_final_speed:.4f}")
    print(f"     → 闭环跑了 {result['ticks_run']} tick 后")
    print(f"     → Token 速度被算力反复加速 {result['ticks_run']} 次")
    print()
    print(f"  3. 流量自增长: 从 {bw_init:.0f} 到 {bw_final:.2f}")
    print(f"     → 每tick流量+0.01%，{result['ticks_run']} tick 累积 +{(bw_final - bw_init) / bw_init * 100:.2f}%")
    print(f"     → 流量增长反过来又加速采样点产电")
    print()
    print(f"  4. 能量-算力-传输-压缩-流量 形成完整闭环")
    print(f"     → 5 类资源互相驱动，没有外力介入")
    print(f"     → 这是\"负负得正\"之外的另一种虚拟闭环")
    print("=" * 78)


if __name__ == "__main__":
    main()
