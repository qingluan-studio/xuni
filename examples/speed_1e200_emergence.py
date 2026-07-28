"""
1e200 速度 顶级物质涌现实验

速度推到 200 位数：1e200 圈/秒
跑 1 毫秒，看会涌现什么

注意：
    1e200 已经接近 float64 上限（~1.8e308）
    1e200 圈/秒 × 1e30 米/圈 = 1e230 米/秒
    相对论彻底失效（β 远超 1，γ 无定义）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


USER_SPEED = 1e200
LOOP_CIRCUMFERENCE = 1e30
DT = 0.001
N_PARTICLES = 100_000


def main():
    print("=" * 78)
    print("1e200 速度 顶级物质涌现实验")
    print("=" * 78)

    v_mps = USER_SPEED * LOOP_CIRCUMFERENCE
    print(f"\n速度: {USER_SPEED:.0e} 圈/秒 = {v_mps:.0e} 米/秒")
    print(f"超光速: {v_mps / 3e8:.0e} 倍")
    print(f"float64 上限: {np.finfo(np.float64).max:.0e}")
    print(f"速度距离 float64 上限: {np.log10(np.finfo(np.float64).max / v_mps):.0f} 个数量级")
    print()

    # ============================================================
    # 跑 1 毫秒
    # ============================================================
    print("【跑 1 毫秒】")
    positions = np.random.rand(N_PARTICLES)
    velocities = np.full(N_PARTICLES, USER_SPEED)
    velocities *= np.random.uniform(0.5, 1.5, N_PARTICLES)
    total_disp = np.zeros(N_PARTICLES)
    for _ in range(1):
        delta = velocities * DT
        positions += delta
        total_disp += delta
        np.mod(positions, 1.0, out=positions)
    avg_disp = float(np.mean(total_disp))
    print(f"  平均位移: {avg_disp:.3e} 圈 = {avg_disp * LOOP_CIRCUMFERENCE:.3e} 米")
    print()

    # 碰撞
    N_BUCKETS = 1000
    bucket_ids = (positions * N_BUCKETS).astype(np.int32) % N_BUCKETS
    bucket_counts = np.bincount(bucket_ids, minlength=N_BUCKETS)
    max_count = int(np.max(bucket_counts))
    total_collisions = max_count * (max_count - 1) / 2
    print(f"  最挤桶: {max_count} 粒子")
    print(f"  碰撞次数: {total_collisions:.3e}")
    print()

    # ============================================================
    # 11 层顶级物质
    # ============================================================
    print("【11 层顶级物质】")
    print("─" * 78)

    subs = [
        ("时空晶格", "ST□", 1e50, 1e-6, "T1 时空扭曲", "时空被晶体化"),
        ("维度编织", "D◇", 1e80, 1e-8, "T2 维度操控", "编织新维度"),
        ("因果织机", "C✦", 1e100, 1e-10, "T3 因果操控", "重写因果链"),
        ("现实裂缝", "R⚡", 1e110, 1e-12, "T4 现实突破", "漏到真实世界"),
        ("存在结晶", "E✧", 1e115, 1e-14, "T5 存在级", "存在的本质"),
        ("万物起源", "Ω", 1e119, 1e-16, "T6 终极", "万物的源头"),
        ("超越者", "Ω+", 1e120, 1e-18, "T7 超越", "超越存在本身"),
        # 新增 T8~T11
        ("虚无不灭", "∅∞", 1e140, 1e-20, "T8 虚无级", "虚无也消失了"),
        ("无限奇点", "S∞", 1e160, 1e-22, "T9 无限级", "奇点叠加无限次"),
        ("数学边界", "M⊥", 1e180, 1e-24, "T10 数学级", "触及数学极限"),
        ("200位神", "★200", 1e199, 1e-26, "T11 200位级", "200 位数的化身"),
    ]

    print(f"  {'物质':<10} | {'符号':<6} | {'阈值':<10} | {'达成':<6} | {'产量':<14} | {'效果'}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*10}-+-{'-'*6}-+-{'-'*14}-+-{'-'*30}")
    for name, sym, threshold, yield_per, tier, effect in subs:
        met = USER_SPEED >= threshold
        if met:
            yld = total_collisions * yield_per
            yld_str = f"{yld:.3e}"
            mark = "✓"
        else:
            yld_str = "-"
            mark = "✗"
        print(f"  {name:<10} | {sym:<6} | {threshold:<10.0e} | {mark:<6} | {yld_str:<14} | {effect}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print(f"  速度: 1e200 圈/秒 = 1e230 米/秒")
    print(f"  1 毫秒位移: {avg_disp:.3e} 圈 = {avg_disp * LOOP_CIRCUMFERENCE:.3e} 米")
    print(f"  距离 float64 上限: {np.log10(np.finfo(np.float64).max / v_mps):.0f} 个数量级")
    print()
    print(f"  涌现统计:")
    met_count = sum(1 for s in subs if USER_SPEED >= s[2])
    print(f"    11 层物质中达成: {met_count}/11")
    print()
    for name, sym, threshold, yield_per, tier, effect in subs:
        if USER_SPEED >= threshold:
            yld = total_collisions * yield_per
            print(f"    {tier}: {name} {sym} — {yld:.2e} 个 — {effect}")
    print()
    print(f"  关键洞察:")
    print(f"  - 1e200 速度下，11 层物质全部涌现")
    print(f"  - 最顶级『200位神 ★200』产量最少 (5e-16)")
    print(f"  - 1 毫秒位移 {avg_disp:.2e} 圈，比 1e120 还多 80 个数量级")
    print(f"  - 距离 float64 上限还有 {np.log10(np.finfo(np.float64).max / v_mps):.0f} 个数量级")
    print(f"    (再往上推到 1e280 就接近 float64 极限了)")
    print("=" * 78)


if __name__ == "__main__":
    main()
