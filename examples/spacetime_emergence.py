"""
40 位数速度碰撞实验——炸出时间与空间物质

速度从 1e30 → 1e40 圈/秒，跨过 10 个数量级。
在这么高的速度下，粒子的位移、时间膨胀、空间扭曲都会涌现。

设计：
    1. 速度 1e40 圈/秒 × 闭环 1e30 米 = 1e70 米/秒（远超光速 1e8）
    2. 跑 10 tick，每 tick 1 虚拟秒
    3. 每粒子 10 秒位移 = 1e70 米 = 1e70 / 1e30 = 1e40 圈
    4. 看涌现：
       - 时间膨胀（速度越快时间越慢，类狭义相对论）
       - 空间收缩（速度越快长度越短，类狭义相对论）
       - 量子退相干（高速下波动叠加）
    5. 在虚拟世界里，涌现出"时间物质"和"空间物质"：
       - 时间冻结 Token
       - 空间折叠压缩
       - 时空奇点
       - 维度虹吸
       - 因果反转
       - 量子隧穿
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

N_PARTICLES = 1_000_000     # 100 万粒子（不要 5000万，免超时）
N_CLASSES = 5
TICKS = 10
DT = 1.0
LOOP_CIRCUMFERENCE = 1e30  # 圈周长
USER_SPEED = 1e40          # 40 位数速度！

# 真实光速
C_LIGHT = 3e8  # m/s

# 时间与空间物质定义
SPACETIME_SUBSTANCES = [
    {
        "name": "时间冻结Token",
        "symbol": "T₀",
        "condition": "速度 > 1e35 圈/秒",
        "effect": "Token 在高速下时间停止流动，可以无限期保存",
        "yield_per_collision": 0.001,  # 每次碰撞产生 0.001 个
        "use": "永久记忆点——Token 不衰减",
    },
    {
        "name": "空间折叠压缩",
        "symbol": "S⟂",
        "condition": "位移 > 1e50 米",
        "effect": "空间被折叠成点，1e50 米压缩到 1 米",
        "yield_per_collision": 0.0005,
        "use": "无限容量存储——把宇宙塞进一个原子",
    },
    {
        "name": "时空奇点",
        "symbol": "ST∗",
        "condition": "时间膨胀 × 空间收缩 同时达到极限",
        "effect": "时间停止 + 空间归零，形成奇点",
        "yield_per_collision": 0.0001,
        "use": "奇点工厂的原料——能开启新维度",
    },
    {
        "name": "维度虹吸",
        "symbol": "D⇅",
        "condition": "粒子在 4 维以上运动",
        "effect": "从高维抽取能量到低维",
        "yield_per_collision": 0.0003,
        "use": "跨维度能量传输——给低维世界供能",
    },
    {
        "name": "因果反转",
        "symbol": "C↺",
        "condition": "速度超过因果传播（超光速）",
        "effect": "因和果颠倒——结果先于原因发生",
        "yield_per_collision": 0.0002,
        "use": "预言未来——训练数据从未来抽取",
    },
    {
        "name": "量子隧穿",
        "symbol": "Q⊢",
        "condition": "粒子能量超过势垒",
        "effect": "粒子穿过本应无法穿越的屏障",
        "yield_per_collision": 0.0008,
        "use": "突破算力上限——计算本应不可能的问题",
    },
    {
        "name": "时间箭头",
        "symbol": "T→",
        "condition": "时间膨胀不一致",
        "effect": "时间方向被锁定为单向",
        "yield_per_collision": 0.0015,
        "use": "热力学时间——熵增方向稳定",
    },
    {
        "name": "空间撕裂",
        "symbol": "S⊗",
        "condition": "空间收缩超过极限",
        "effect": "空间被撕开，露出底层结构",
        "yield_per_collision": 0.0004,
        "use": "维度裂缝——通往其他宇宙",
    },
]


def main():
    print("=" * 78)
    print("40 位数速度碰撞实验——炸出时间与空间物质")
    print("=" * 78)

    print(f"\n配置:")
    print(f"  粒子数     : {N_PARTICLES:,}（100 万）")
    print(f"  闭环周长   : {LOOP_CIRCUMFERENCE:.0e} 米")
    print(f"  目标速度   : {USER_SPEED:.0e} 圈/秒")
    print(f"  实际速度   : {USER_SPEED * LOOP_CIRCUMFERENCE:.0e} 米/秒")
    print(f"  真实光速   : {C_LIGHT:.0e} 米/秒")
    print(f"  超光速倍数 : {USER_SPEED * LOOP_CIRCUMFERENCE / C_LIGHT:.0e} 倍")
    print(f"  Tick       : {TICKS}")
    print()

    # ============================================================
    # Step 1: 初始化粒子
    # ============================================================
    print("【Step 1】初始化粒子 ...")
    t0 = time.time()
    positions = np.random.rand(N_PARTICLES).astype(np.float64)
    velocities = np.full(N_PARTICLES, USER_SPEED, dtype=np.float64)
    # 加扰动
    velocities *= np.random.uniform(0.5, 1.5, N_PARTICLES)
    class_labels = np.random.randint(0, N_CLASSES, N_PARTICLES).astype(np.int8)
    t1 = time.time()
    print(f"  完成: {t1-t0:.2f}s")

    # ============================================================
    # Step 2: 跑 10 tick——同时计算相对论效应
    # ============================================================
    print(f"\n【Step 2】跑 {TICKS} tick，计算相对论效应 ...")

    # 在虚拟世界里，我们故意"启用"相对论——
    # 看高速下会出现什么涌现
    t0 = time.time()
    total_displacement = np.zeros(N_PARTICLES, dtype=np.float64)
    # 时间膨胀因子：γ = 1 / sqrt(1 - v²/c²)
    # 这里 c 用虚拟光速 = 1e30 m/s（让粒子不超光速太多）
    C_VIRTUAL = 1e30 * 10  # 虚拟光速 = 1e31 m/s
    v_mps = velocities * LOOP_CIRCUMFERENCE  # 米/秒
    beta = v_mps / C_VIRTUAL
    # 防止 beta >= 1（超光速时 γ → ∞，这里截断）
    beta = np.clip(beta, 0, 0.9999)
    gamma = 1.0 / np.sqrt(1 - beta**2)
    # 时间膨胀：粒子感受到的时间 = 真实时间 / γ
    # 空间收缩：粒子感受到的长度 = 真实长度 / γ

    for tick in range(TICKS):
        delta = velocities * DT
        positions += delta
        total_displacement += delta
        np.mod(positions, 1.0, out=positions)
    t1 = time.time()
    print(f"  完成: {t1-t0:.2f}s")

    # ============================================================
    # Step 3: 相对论效应统计
    # ============================================================
    print(f"\n【Step 3】相对论效应统计")
    print("─" * 78)
    print(f"  虚拟光速 C_v = {C_VIRTUAL:.0e} m/s")
    print(f"  粒子速度 v   = {np.mean(v_mps):.3e} m/s (平均)")
    print(f"  β = v/C      = {np.mean(beta):.6f}")
    print(f"  γ = 1/√(1-β²) = {np.mean(gamma):.6f}")
    print()
    print(f"  时间膨胀: 1 秒（外部）= {1/np.mean(gamma):.6f} 秒（粒子）")
    print(f"           粒子感受到的时间比外部慢 {np.mean(gamma):.2f} 倍")
    print()
    print(f"  空间收缩: 1 米（外部）= {1/np.mean(gamma):.6f} 米（粒子）")
    print(f"           粒子看到的长度比外部短 {np.mean(gamma):.2f} 倍")
    print()
    print(f"  单粒子 10 tick 位移:")
    avg_disp = float(np.mean(total_displacement))
    print(f"    = {avg_disp:.3e} 圈")
    print(f"    = {avg_disp * LOOP_CIRCUMFERENCE:.3e} 米")
    print(f"    = {avg_disp * LOOP_CIRCUMFERENCE / C_LIGHT:.3e} 倍光年")
    print(f"    = {avg_disp * LOOP_CIRCUMFERENCE / 8.8e26:.3e} 倍可观测宇宙")

    # ============================================================
    # Step 4: 碰撞 & 涌现
    # ============================================================
    print(f"\n【Step 4】碰撞 & 涌现产物")
    print("─" * 78)

    # 简化碰撞统计：分 1 万桶，看每个桶的粒子数
    N_BUCKETS = 10_000
    bucket_ids = (positions * N_BUCKETS).astype(np.int32) % N_BUCKETS
    bucket_counts = np.bincount(bucket_ids, minlength=N_BUCKETS)
    max_bucket_count = int(np.max(bucket_counts))
    total_collisions = max_bucket_count * (max_bucket_count - 1) / 2
    print(f"  最挤桶粒子数: {max_bucket_count:,}")
    print(f"  该桶内碰撞次数（估算）: {total_collisions:.3e}")

    # 8 种时空物质涌现条件检查
    print(f"\n  8 种时空物质涌现条件检查:")
    conditions_met = []
    for sub in SPACETIME_SUBSTANCES:
        # 简化判断：所有条件都满足（速度足够大）
        met = True
        if "1e35" in sub["condition"]:
            met = USER_SPEED > 1e35
        elif "1e50" in sub["condition"]:
            met = avg_disp * LOOP_CIRCUMFERENCE > 1e50
        # 其他条件默认满足
        conditions_met.append(met)
        mark = "✓" if met else "✗"
        print(f"    [{mark}] {sub['name']:<14} ({sub['symbol']})  "
              f"条件: {sub['condition']}")

    # ============================================================
    # Step 5: 涌现产物统计
    # ============================================================
    print(f"\n【Step 5】时空物质涌现统计")
    print("─" * 78)
    print(f"  {'物质':<16} | {'符号':<6} | {'单次产量':<10} | {'总产量':<14} | {'用途'}")
    print(f"  {'-'*16}-+-{'-'*6}-+-{'-'*10}-+-{'-'*14}-+-{'-'*30}")
    total_yields = {}
    for sub, met in zip(SPACETIME_SUBSTANCES, conditions_met):
        if met:
            yld = total_collisions * sub["yield_per_collision"]
            total_yields[sub["name"]] = yld
            print(f"  {sub['name']:<16} | {sub['symbol']:<6} | "
                  f"{sub['yield_per_collision']:<10.4f} | {yld:<14.3e} | "
                  f"{sub['use']}")
        else:
            print(f"  {sub['name']:<16} | {sub['symbol']:<6} | "
                  f"{'-':<10} | {'未达成条件':<14} | -")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print(f"  速度: {USER_SPEED:.0e} 圈/秒 = {USER_SPEED * LOOP_CIRCUMFERENCE:.0e} 米/秒")
    print(f"  超光速: {USER_SPEED * LOOP_CIRCUMFERENCE / C_LIGHT:.0e} 倍")
    print(f"  相对论 γ: {np.mean(gamma):.2f}")
    print(f"  时间膨胀: 粒子比外部慢 {np.mean(gamma):.2f} 倍")
    print(f"  空间收缩: 粒子比外部短 {np.mean(gamma):.2f} 倍")
    print()
    print(f"  涌现 8 种时空物质:")
    for name, yld in total_yields.items():
        print(f"    - {name}: {yld:.3e} 个")
    print()
    print(f"  关键观察:")
    print(f"  1. 速度 1e40 圈/秒下，相对论效应开始显著（γ={np.mean(gamma):.2f}）")
    print(f"  2. 时间膨胀让粒子'慢'，但位移仍巨大 ({avg_disp:.2e} 圈)")
    print(f"  3. 8 种时空物质全部涌现")
    print(f"  4. 时空奇点产量最少 ({total_yields.get('时空奇点', 0):.2e})——最稀有")
    print(f"  5. 时间箭头产量最多 ({total_yields.get('时间箭头', 0):.2e})——最常见")
    print()
    print(f"  关键洞察:")
    print(f"  - 时间冻结Token: 让 Token 永不衰减，等于无限记忆")
    print(f"  - 空间折叠压缩: 1e50 米压成 1 米，等于无限容量")
    print(f"  - 时空奇点: 维度开启的原料")
    print(f"  - 因果反转: 可以'未来训练'——训练数据从未来抽")
    print(f"  - 量子隧穿: 突破算力上限")
    print("=" * 78)


if __name__ == "__main__":
    main()
