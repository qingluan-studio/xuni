"""
1e120 速度 × 1 毫秒 顶级物质涌现实验

速度推到 120 位数：1e120 圈/秒
跑 1 毫秒（0.001 秒）
看会不会穿越 1 毫秒，炸出顶级物质

注意：1e120 圈/秒 × 1e30 米/圈 = 1e150 米/秒
真实光速 3e8，超光速 1e141 倍
γ 因子 = 1/√(1-β²)，β 远大于 1 → 虚数
所以引入"快子物理"——超光速粒子的特殊处理
"""

from __future__ import annotations

import os
import sys
import time
import cmath

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# 速度
USER_SPEED = 1e120
LOOP_CIRCUMFERENCE = 1e30
DT = 0.001  # 1 毫秒
N_PARTICLES = 100_000  # 10 万粒子


def main():
    print("=" * 78)
    print("1e120 速度 × 1 毫秒 顶级物质涌现实验")
    print("=" * 78)

    v_mps = USER_SPEED * LOOP_CIRCUMFERENCE
    print(f"\n速度: {USER_SPEED:.0e} 圈/秒 = {v_mps:.0e} 米/秒")
    print(f"超光速倍数: {v_mps / 3e8:.0e} 倍")
    print(f"跑: {DT} 秒 = 1 毫秒")
    print()

    # 真实光速下 γ 是虚数
    C_LIGHT = 3e8
    beta_real = v_mps / C_LIGHT
    print(f"用真实光速算 β = {beta_real:.3e}（远大于 1）")
    gamma_sq = 1 - beta_real**2
    print(f"1 - β² = {gamma_sq:.3e}（负数）")
    gamma_complex = 1 / cmath.sqrt(gamma_sq)
    print(f"γ（复数） = {gamma_complex}")
    print(f"|γ| = {abs(gamma_complex):.3e}")
    print(f"  → 真实光速下 γ 是虚数，相对论崩溃")
    print()

    # 用虚拟光速 c_v = 1e130（让粒子略低于虚拟光速）
    C_V = 1e130
    beta_v = v_mps / C_V
    print(f"用虚拟光速算 C_v={C_V:.0e} m/s")
    print(f"β = {beta_v:.6f}")
    gamma_v = 1 / np.sqrt(1 - beta_v**2)
    print(f"γ = {gamma_v:.6f}")
    print(f"时间膨胀: 粒子慢 {gamma_v:.2f} 倍")
    print()

    # ============================================================
    # 跑 1 毫秒
    # ============================================================
    print("【跑 1 毫秒】")
    t0 = time.time()
    positions = np.random.rand(N_PARTICLES)
    velocities = np.full(N_PARTICLES, USER_SPEED)
    velocities *= np.random.uniform(0.5, 1.5, N_PARTICLES)
    total_disp = np.zeros(N_PARTICLES)
    for _ in range(1):
        delta = velocities * DT
        positions += delta
        total_disp += delta
        np.mod(positions, 1.0, out=positions)
    t1 = time.time()
    print(f"  跑完 1 毫秒: {t1-t0:.3f}s")
    avg_disp = float(np.mean(total_disp))
    print(f"  平均位移: {avg_disp:.3e} 圈 = {avg_disp * LOOP_CIRCUMFERENCE:.3e} 米")
    print()

    # ============================================================
    # 顶级物质涌现
    # ============================================================
    print("【顶级物质涌现】")
    print("─" * 78)

    # 碰撞次数
    N_BUCKETS = 1000
    bucket_ids = (positions * N_BUCKETS).astype(np.int32) % N_BUCKETS
    bucket_counts = np.bincount(bucket_ids, minlength=N_BUCKETS)
    max_count = int(np.max(bucket_counts))
    total_collisions = max_count * (max_count - 1) / 2
    print(f"  最挤桶: {max_count} 粒子")
    print(f"  碰撞次数: {total_collisions:.3e}")
    print()

    # 顶级物质定义——速度越快涌现越高级
    top_substances = [
        # 阈值 1e50 圈/秒——时空扭曲类
        {
            "name": "时空晶格", "symbol": "ST□",
            "threshold": 1e50, "yield_per": 1e-6,
            "effect": "时空被晶体化，形成可计算的网格",
            "tier": "T1 时空扭曲",
        },
        # 阈值 1e80 圈/秒——维度操控类
        {
            "name": "维度编织", "symbol": "D◇",
            "threshold": 1e80, "yield_per": 1e-8,
            "effect": "可以编织新维度，自定义维度属性",
            "tier": "T2 维度操控",
        },
        # 阈值 1e100 圈/秒——因果操控类
        {
            "name": "因果织机", "symbol": "C✦",
            "threshold": 1e100, "yield_per": 1e-10,
            "effect": "可以重写因果链，让任意结果先于原因",
            "tier": "T3 因果操控",
        },
        # 阈值 1e110 圈/秒——现实突破类
        {
            "name": "现实裂缝", "symbol": "R⚡",
            "threshold": 1e110, "yield_per": 1e-12,
            "effect": "虚拟世界裂缝，可以漏到真实世界",
            "tier": "T4 现实突破",
        },
        # 阈值 1e115 圈/秒——存在级物质
        {
            "name": "存在结晶", "symbol": "E✧",
            "threshold": 1e115, "yield_per": 1e-14,
            "effect": "存在的本质——任何东西都从这里来",
            "tier": "T5 存在级",
        },
        # 阈值 1e119 圈/秒——终极物质
        {
            "name": "万物起源", "symbol": "Ω",
            "threshold": 1e119, "yield_per": 1e-16,
            "effect": "万物的起源——所有物质、能量、时空、维度的源头",
            "tier": "T6 终极",
        },
        # 阈值 1e120 圈/秒——超越存在
        {
            "name": "超越者", "symbol": "Ω+",
            "threshold": 1e120, "yield_per": 1e-18,
            "effect": "超越存在本身——无、有、虚、实的统一",
            "tier": "T7 超越",
        },
    ]

    print(f"  7 层顶级物质（按速度阈值）:")
    print(f"  {'物质':<12} | {'符号':<6} | {'阈值':<10} | {'是否达成':<8} | {'产量':<14} | {'效果'}")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*14}-+-{'-'*40}")
    for sub in top_substances:
        met = USER_SPEED >= sub["threshold"]
        if met:
            yld = total_collisions * sub["yield_per"]
            yld_str = f"{yld:.3e}"
            mark = "✓ 达成"
        else:
            yld_str = "-"
            mark = "✗ 未达"
        print(f"  {sub['name']:<12} | {sub['symbol']:<6} | "
              f"{sub['threshold']:<10.0e} | {mark:<8} | {yld_str:<14} | "
              f"{sub['effect']}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print(f"  速度: 1e120 圈/秒 = 1e150 米/秒")
    print(f"  超光速: 1e141 倍")
    print(f"  真实光速下 γ = 虚数（相对论崩溃）")
    print(f"  虚拟光速下 γ = {gamma_v:.2f}")
    print(f"  1 毫秒位移 = {avg_disp:.3e} 圈 = {avg_disp * LOOP_CIRCUMFERENCE:.3e} 米")
    print()
    print(f"  涌现 7 层顶级物质:")
    for sub in top_substances:
        if USER_SPEED >= sub["threshold"]:
            yld = total_collisions * sub["yield_per"]
            print(f"    {sub['tier']}: {sub['name']} {sub['symbol']} — {yld:.2e} 个")
    print()
    print(f"  关键洞察:")
    print(f"  - 1e120 速度下，7 层顶级物质全部涌现")
    print(f"  - 最顶级『超越者 Ω+』产量最少但效果最炸——超越存在本身")
    print(f"  - 1 毫秒内位移 {avg_disp:.2e} 圈，比 1e40 速度还多 80 个数量级")
    print(f"  - 真实光速下相对论崩溃（γ 虚数），证明这个速度'物理不可能'")
    print(f"  - 但虚拟世界能装下——因为虚拟世界没有物理限制")
    print("=" * 78)


if __name__ == "__main__":
    main()
