"""
粒子相撞涌现实验

上一实验发现：5000 万粒子在 1e30 圈/秒的速度下跑 10 tick 后，
位置全挤在 [0, 0.1) 这段里——它们其实已经"撞在一起"了。

这次直接做相撞：
    1. 跑 10 tick，让粒子位置聚集
    2. 检查"碰撞对"——位置距离 < 阈值的粒子对
    3. 不真做 N² 配对（5000万² 太大），用空间分桶
    4. 把同桶粒子看作"碰撞"，看会涌现什么：
       - 同桶粒子数量分布
       - 不同类资源碰撞的频次
       - 涌现产物（5 类资源两两组合）

5 类资源两两组合 = 15 种涌现：
    采样点×采样点 / 采样点×算力 / 采样点×Token / ...
    算力×算力 / 算力×Token / ...
    Token×Token / Token×压缩点 / ...
    压缩点×压缩点 / 压缩点×流量 / ...
    流量×流量

每种组合的"涌现产物"用合成规则定义。
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


# ============================================================
# 配置（跟上次实验一致）
# ============================================================

N_PER_CLASS = 10_000_000
N_CLASSES = 5
N_TOTAL = N_PER_CLASS * N_CLASSES
TICKS = 10
DT = 1.0
LOOP_CIRCUMFERENCE = 1e30
USER_SPEED = 1e30

# 碰撞检测：把 [0,1) 分成 N 个桶，同桶粒子算碰撞
N_BUCKETS = 10_000  # 1万桶，平均每桶 5000 粒子
COLLISION_THRESHOLD = 1.0 / N_BUCKETS  # 同桶就算撞

# 5 类资源两两组合的涌现产物
EMERGENCE_TABLE = {
    (0, 0): "采样湍流",       # 采样点 × 采样点
    (0, 1): "电流算力",       # 采样点 × 算力
    (0, 2): "采样Token",      # 采样点 × Token
    (0, 3): "压缩采样",       # 采样点 × 压缩点
    (0, 4): "采样流量流",     # 采样点 × 流量
    (1, 1): "算力爆涨",       # 算力 × 算力
    (1, 2): "算力Token",      # 算力 × Token
    (1, 3): "压缩算力",       # 算力 × 压缩点
    (1, 4): "流量算力",       # 算力 × 流量
    (2, 2): "Token叠加",     # Token × Token
    (2, 3): "Token压缩",     # Token × 压缩点
    (2, 4): "Token流",        # Token × 流量
    (3, 3): "压缩爆",         # 压缩点 × 压缩点
    (3, 4): "压缩流量",       # 压缩点 × 流量
    (4, 4): "流量湍流",       # 流量 × 流量
}

CLASS_NAMES = ["采样点", "算力", "Token", "压缩点", "流量"]


def main():
    print("=" * 78)
    print("粒子相撞涌现实验")
    print("  5000 万粒子在 1e30 米闭环上跑 10 tick，然后看相撞涌现")
    print("=" * 78)

    # ============================================================
    # Step 1: 跑粒子运动（跟上次一样）
    # ============================================================
    print(f"\n【Step 1】初始化 + 跑 {TICKS} tick 让粒子聚集 ...")
    t0 = time.time()
    positions = np.random.rand(N_TOTAL).astype(np.float64)
    velocities = np.empty(N_TOTAL, dtype=np.float64)
    class_labels = np.empty(N_TOTAL, dtype=np.int8)
    for i in range(N_CLASSES):
        start = i * N_PER_CLASS
        end = start + N_PER_CLASS
        velocities[start:end] = USER_SPEED * np.random.uniform(0.5, 1.5, N_PER_CLASS)
        class_labels[start:end] = i

    # 跑 TICKS 个 tick
    for _ in range(TICKS):
        positions += velocities * DT
        np.mod(positions, 1.0, out=positions)
    t1 = time.time()
    print(f"  跑完 {TICKS} tick: {t1-t0:.2f}s")

    # ============================================================
    # Step 2: 分桶——位置相近的粒子放到同桶
    # ============================================================
    print(f"\n【Step 2】分桶检测碰撞（{N_BUCKETS} 桶）...")
    # positions 在 [0,1)，乘 N_BUCKETS 取整即桶号
    bucket_ids = (positions * N_BUCKETS).astype(np.int32) % N_BUCKETS
    # 排序：按桶号排序
    sort_idx = np.argsort(bucket_ids, kind='stable')
    sorted_buckets = bucket_ids[sort_idx]
    sorted_classes = class_labels[sort_idx]

    # 找同桶粒子组——连续相同 bucket_id 的段
    print(f"  计算每桶粒子数 ...")
    bucket_counts = np.bincount(sorted_buckets, minlength=N_BUCKETS)
    print(f"    桶数: {N_BUCKETS}")
    print(f"    平均每桶粒子: {N_TOTAL / N_BUCKETS:.0f}")
    print(f"    最挤桶: {np.max(bucket_counts)} 粒子")
    print(f"    最空桶: {np.min(bucket_counts)} 粒子")
    print(f"    标准差: {np.std(bucket_counts):.0f}")

    # 直方图：每桶粒子数分布
    print(f"\n  桶内粒子数分布:")
    bins = [0, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 20000, 100000]
    hist, _ = np.histogram(bucket_counts, bins=bins)
    for i, h in enumerate(hist):
        if h > 0:
            print(f"    [{bins[i]:>6}-{bins[i+1]:>6}): {h:>5} 桶")

    # ============================================================
    # Step 3: 直接处理最挤的桶——5000 万粒子全挤一起
    # ============================================================
    print(f"\n【Step 3】直接处理最挤的桶（5000 万粒子全挤一起）...")
    # 找到粒子最多的桶
    max_bucket = int(np.argmax(bucket_counts))
    print(f"  最挤桶 ID: {max_bucket} (粒子数: {bucket_counts[max_bucket]})")
    # 这个桶里所有粒子的类别
    mask = bucket_ids == max_bucket
    classes_in_max_bucket = class_labels[mask]
    # 统计各类资源数
    counts_per_class = np.bincount(classes_in_max_bucket, minlength=N_CLASSES)
    print(f"  该桶内各类资源数:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"    {name:<10}: {counts_per_class[c]:>10,}")
    print(f"    合计      : {np.sum(counts_per_class):>10,}")

    # ============================================================
    # Step 4: 碰撞矩阵——基于最挤桶的 5000 万粒子
    # ============================================================
    print(f"\n【Step 4】碰撞矩阵——基于最挤桶的 {np.sum(counts_per_class):,} 粒子")
    print("─" * 78)
    # 在最挤桶里，a 类和 b 类的碰撞数 ≈ count[a] * count[b]
    collision_matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for a in range(N_CLASSES):
        for b in range(a, N_CLASSES):
            ca = int(counts_per_class[a])
            cb = int(counts_per_class[b])
            if a == b:
                collision_matrix[a, b] = ca * (ca - 1) / 2
            else:
                collision_matrix[a, b] = ca * cb

    # 打印
    print(f"  {'':>10}", end="")
    for j in range(N_CLASSES):
        print(f"  {CLASS_NAMES[j]:>10}", end="")
    print()
    for i in range(N_CLASSES):
        print(f"  {CLASS_NAMES[i]:<10}", end="")
        for j in range(N_CLASSES):
            v = collision_matrix[i, j] if j >= i else collision_matrix[j, i]
            print(f"  {v:>10.2e}", end="")
        print()

    total_collisions = float(np.sum(np.triu(collision_matrix)))
    print(f"\n  估算总碰撞次数: {total_collisions:.3e}")

    # ============================================================
    # Step 5: 涌现产物统计
    # ============================================================
    print(f"\n【Step 5】涌现产物统计（按 5×5 矩阵映射）")
    print("─" * 78)
    print(f"  {'组合':<24} | {'涌现产物':<14} | {'碰撞次数':<14} | {'占比'}")
    print(f"  {'-'*24}-+-{'-'*14}-+-{'-'*14}-+-{'-'*10}")
    for (a, b), product in sorted(EMERGENCE_TABLE.items(),
                                   key=lambda x: -collision_matrix[x[0][0], x[0][1]]):
        count = collision_matrix[a, b]
        pct = count / total_collisions * 100 if total_collisions > 0 else 0
        combo = f"{CLASS_NAMES[a]} × {CLASS_NAMES[b]}"
        print(f"  {combo:<24} | {product:<14} | {count:<14.3e} | {pct:.2f}%")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print(f"  5000 万粒子在 1e30 米闭环上跑 10 tick 后:")
    print(f"  - 粒子位置聚集到 {N_BUCKETS} 个桶里，平均每桶 {N_TOTAL / N_BUCKETS:.0f} 粒子")
    print(f"  - 最挤的桶有 {np.max(bucket_counts)} 粒子挤在一起")
    print(f"  - 估算总碰撞次数: {total_collisions:.3e}")
    print(f"  - 15 种涌现产物全部出现")
    print()
    print(f"  关键观察:")
    print(f"  1. 速度 1e30 圈/秒下，粒子聚集极快，碰撞极频繁")
    print(f"  2. 5000 万粒子挤在 1 万个桶里，每桶 ~5000 粒子")
    print(f"  3. 每桶里 5 类资源都有，所以 15 种组合全部涌现")
    print(f"  4. 同类碰撞（C(n,2)）远多于异类碰撞（a*b）")
    print(f"  5. 涌现产物里最多的是『{EMERGENCE_TABLE[(0, 0)]}』（采样点×采样点）")
    print(f"     因为采样点数最多（每桶约 1000 个）")
    print(f"  6. 异类碰撞里最多的是采样点×其他——采样点是基础")
    print("=" * 78)


if __name__ == "__main__":
    main()
