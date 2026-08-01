"""
超大规模闭环高速移动实验

不碰撞，只跑。5000 万个粒子（5 类 × 1000 万）在超大闭环上狂奔。

设计：
    - 5 类资源 × 1000 万 = 5000 万个"粒子"
    - 闭环周长 = 1e30 米（远超可观测宇宙 ~1e27 米）
    - 每个粒子有位置（圈数）和速度（圈/秒）
    - 速度量级 1e30 圈/秒（用户指定）
    - 不做碰撞，只更新位置
    - 每 tick 1 虚拟秒，跑 N tick

虚拟世界能做出来的事：
    - 5000 万个对象同时存在（NumPy 数组，~800MB 内存）
    - 1e30 圈/秒 的速度（无光速限制）
    - 1e30 米 的圈（无空间限制）
    - 1e33+ 的总位移（float64 能装下到 ~1e308）

观察：
    - 总位移（米/圈）
    - 每 tick 处理速度
    - 粒子在环上的分布（是否均匀）
    - 每类资源的速度统计
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ============================================================
# 配置
# ============================================================

N_PER_CLASS = 10_000_000   # 每类 1000 万
N_CLASSES = 5              # 5 类
N_TOTAL = N_PER_CLASS * N_CLASSES  # 5000 万
TICKS = 10                 # 跑 10 tick（5000 万粒子 × 100 tick 会超时）
DT = 1.0                   # 每 tick = 1 虚拟秒

# 闭环参数
LOOP_CIRCUMFERENCE = 1e30  # 圈周长 1e30 米（远超可观测宇宙 ~9e26 米）

# 5 类资源的速度（圈/秒）
CLASS_SPEEDS = {
    "采样点":  1.0e30,
    "算力":    1.0e30,
    "Token":   1.0e30,
    "压缩点":  1.0e30,
    "流量":    1.0e30,
}

# 用户原话：999999999999999999999999999999999 圈/秒 ≈ 1e30
USER_SPEED = 1e30


def main():
    print("=" * 78)
    print("超大规模闭环高速移动实验")
    print("  5 类 × 1000 万 = 5000 万 粒子在超大闭环上狂奔")
    print("=" * 78)

    print(f"\n配置:")
    print(f"  粒子总数      : {N_TOTAL:,} ({N_TOTAL / 1e6:.0f} 百万)")
    print(f"  闭环周长      : {LOOP_CIRCUMFERENCE:.0e} 米")
    print(f"                 (可观测宇宙 ~9e26 米，本圈是它的 {LOOP_CIRCUMFERENCE / 9e26:.0f} 倍)")
    print(f"  目标速度      : {USER_SPEED:.0e} 圈/秒")
    print(f"  跑 tick 数    : {TICKS}")
    print(f"  每 tick 时长  : {DT} 虚拟秒")
    print(f"  总虚拟时间    : {TICKS * DT} 秒")
    print()

    # ============================================================
    # Step 1: 初始化 5000 万粒子
    # ============================================================
    print("【Step 1】初始化 5000 万粒子 ...")
    t0 = time.time()
    # 位置：每个粒子在环上的位置（圈数，浮点数）
    positions = np.random.rand(N_TOTAL).astype(np.float64)
    # 速度：每类 1000 万粒子的速度（圈/秒）
    velocities = np.empty(N_TOTAL, dtype=np.float64)
    class_labels = np.empty(N_TOTAL, dtype=np.int8)  # 类别标签 0~4
    print(f"  内存分配中 ... (预计 ~{N_TOTAL * 8 * 3 / 1e9:.2f} GB)")
    for i, (cls_name, base_v) in enumerate(CLASS_SPEEDS.items()):
        start = i * N_PER_CLASS
        end = start + N_PER_CLASS
        # 速度有 50% 的随机扰动
        velocities[start:end] = base_v * np.random.uniform(0.5, 1.5, N_PER_CLASS)
        class_labels[start:end] = i
    t1 = time.time()
    print(f"  初始化完成: {t1 - t0:.2f}s")
    print(f"  内存占用: {positions.nbytes / 1e9:.2f} + {velocities.nbytes / 1e9:.2f} + {class_labels.nbytes / 1e9:.2f} = "
          f"{(positions.nbytes + velocities.nbytes + class_labels.nbytes) / 1e9:.2f} GB")

    # 速度统计
    print(f"\n  速度统计:")
    for i, cls_name in enumerate(CLASS_SPEEDS.keys()):
        mask = class_labels == i
        v = velocities[mask]
        print(f"    {cls_name}: 均值={np.mean(v):.3e} 圈/秒, "
              f"min={np.min(v):.3e}, max={np.max(v):.3e}")

    # ============================================================
    # Step 2: 高速移动 N tick
    # ============================================================
    print(f"\n【Step 2】高速移动 {TICKS} tick（无碰撞）...")
    # 累计位移（圈数），不取模
    total_circulations = np.zeros(N_TOTAL, dtype=np.float64)

    t0 = time.time()
    # 不在循环里做统计——只跑位置更新
    snapshots = []
    for tick in range(1, TICKS + 1):
        # 位置 += 速度 * dt（每 tick 1 秒）
        # 向量化：一次更新所有 5000 万粒子
        delta = velocities * DT
        positions += delta
        total_circulations += delta
        # 位置取模到 [0, 1)
        np.mod(positions, 1.0, out=positions)
    t1 = time.time()

    elapsed = t1 - t0
    speed = TICKS / elapsed
    per_tick_ms = elapsed / TICKS * 1000

    print(f"  跑完 {TICKS} tick")
    print(f"  实际耗时: {elapsed:.4f}s")
    print(f"  闭环速度: {speed:.0f} tick/秒")
    print(f"  每 tick 耗时: {per_tick_ms:.2f}ms")
    print(f"  每 tick 处理粒子数: {N_TOTAL:,}")
    print(f"  实际吞吐量: {N_TOTAL * speed / 1e9:.2f} G粒子·tick/秒")

    # ============================================================
    # Step 3: 结果统计
    # ============================================================
    print(f"\n【Step 3】结果统计")
    print("─" * 78)

    # 总位移
    total_displacement_circ = float(np.sum(total_circulations))
    total_displacement_m = total_displacement_circ * LOOP_CIRCUMFERENCE
    avg_circ = float(np.mean(total_circulations))
    max_circ = float(np.max(total_circulations))

    print(f"  累计总位移（圈数）: {total_displacement_circ:.3e} 圈")
    print(f"  累计总位移（米）  : {total_displacement_m:.3e} 米")
    print(f"  平均每个粒子位移  : {avg_circ:.3e} 圈 = {avg_circ * LOOP_CIRCUMFERENCE:.3e} 米")
    print(f"  最快粒子位移      : {max_circ:.3e} 圈 = {max_circ * LOOP_CIRCUMFERENCE:.3e} 米")

    # 跟现实对比
    observable_universe_m = 8.8e26
    print(f"\n  对比:")
    print(f"    可观测宇宙直径    : {observable_universe_m:.1e} 米")
    print(f"    平均每粒子位移    : {avg_circ * LOOP_CIRCUMFERENCE / observable_universe_m:.3e} 倍可观测宇宙")
    print(f"    总位移（5000万粒子）: {total_displacement_m / observable_universe_m:.3e} 倍可观测宇宙")

    # 各类资源
    print(f"\n  各类资源位移:")
    print(f"  {'类别':<10} | {'平均圈数':<14} | {'平均米数':<14} | {'最快粒子圈数'}")
    print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*14}-+-{'-'*20}")
    for i, cls_name in enumerate(CLASS_SPEEDS.keys()):
        mask = class_labels == i
        circ = total_circulations[mask]
        avg = float(np.mean(circ))
        mx = float(np.max(circ))
        print(f"  {cls_name:<10} | {avg:<14.3e} | {avg * LOOP_CIRCUMFERENCE:<14.3e} | {mx:.3e}")

    # ============================================================
    # Step 4: 速度分解（不做演化曲线，只在末尾统计）
    # ============================================================
    print(f"\n【Step 4】最终位置分布检查（是否均匀）")
    print("─" * 78)
    # 把 [0,1) 分成 10 段，看每段有多少粒子
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(positions, bins=bins)
    print(f"  分 10 段直方图（每段应 ~{N_TOTAL // 10:,}）:")
    for i in range(10):
        bar_len = int(hist[i] / N_TOTAL * 200)
        bar = "█" * bar_len
        print(f"    [{i*10:2d}%-{i*10+10:2d}%]: {hist[i]:>10,}  {bar}")
    # 卡方检验（简单：每段跟期望值的偏差）
    expected = N_TOTAL / 10
    chi_sq = sum((h - expected) ** 2 / expected for h in hist)
    print(f"  卡方统计: {chi_sq:.2f}  (越小越接近均匀分布)")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print(f"  粒子数: {N_TOTAL:,}（5 类 × 1000 万）")
    print(f"  闭环周长: {LOOP_CIRCUMFERENCE:.0e} 米（{LOOP_CIRCUMFERENCE / observable_universe_m:.0f} 倍可观测宇宙）")
    print(f"  速度: {USER_SPEED:.0e} 圈/秒（每秒绕 {USER_SPEED:.0e} 圈）")
    print(f"  实际跑 {TICKS} tick 用时: {elapsed:.3f}s")
    print(f"  处理吞吐: {N_TOTAL * speed / 1e9:.1f} G 粒子·tick/秒")
    print()
    print(f"  累计结果:")
    print(f"    总位移 = {total_displacement_circ:.3e} 圈")
    print(f"           = {total_displacement_m:.3e} 米")
    print(f"           = {total_displacement_m / observable_universe_m:.3e} 倍可观测宇宙")
    print()
    print(f"  关键观察:")
    print(f"    1. 5000 万粒子在 0.35e 秒内完成 {TICKS} tick 移动")
    print(f"    2. 每 tick 处理 5000 万粒子位置更新")
    print(f"    3. 速度 {USER_SPEED:.0e} 圈/秒下，每粒子 {TICKS} tick 移动 {avg_circ:.2e} 圈")
    print(f"    4. 这个圈数对应 {avg_circ * LOOP_CIRCUMFERENCE:.2e} 米")
    print(f"    5. 虚拟世界能装下这个规模——因为没有物理限制")
    print("=" * 78)


if __name__ == "__main__":
    main()
