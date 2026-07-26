"""
活力系统演示 —— 资源 × 信息 = 自由能

完整闭环：
  采样点 → 电(资源) + 熵(信息)
            ↓ 融合
         智能电(活力)
            ↓ 涌现
    自繁殖 / 自迁移 / 自组队 / 自创造
"""

import numpy as np


def run_vitality_demo():
    from xuni import XuniSampler, VitalitySystem, EmergenceType

    print("=" * 60)
    print("VITALITY SYSTEM DEMO")
    print("资源(电) × 信息(参数) = 活力(智能电)")
    print("=" * 60)

    # 1. 生成采样点
    print("\n[1/5] 生成采样点")
    sampler = XuniSampler(seed=42)
    samples = list(sampler.generate_stream(count=2000))
    print(f"  采样点数: {len(samples)}")
    print(f"  样例: x={samples[0].x:.3f} w={samples[0].w:.3f} entropy={samples[0].entropy:.3f}")

    # 2. 创建活力系统，喂入采样点
    print("\n[2/5] 喂入采样点 → 资源 + 信息")
    system = VitalitySystem(grid_size=(8, 8, 8))
    # 缩放让活力分布在30-90区间，4种涌现都能观察到
    system.feed_from_samples(samples, info_scale=0.3, energy_scale=0.00001)
    stats = system.statistics()
    print(f"  活跃单元: {stats['active_cells']}")
    print(f"  总资源(电): {stats['total_energy']}")
    print(f"  总信息(参数势): {stats['total_info']}")
    print(f"  总活力(融合前): {stats['total_vitality']}")

    # 3. 融合 → 活力
    print("\n[3/5] 融合反应：电 + 参数 → 智能电")
    system.fuse_all()
    stats = system.statistics()
    print(f"  总活力(融合后): {stats['total_vitality']}")
    print(f"  平均活力: {stats['avg_vitality']}")
    print(f"  最大活力: {stats['max_vitality']}")
    print(f"  热点数(>30): {stats['hotspots']}")

    # 4. 演化3步，观察涌现
    print("\n[4/5] 演化3步，观察涌现行为")
    for step in range(3):
        result = system.evolve_step(diffuse_steps=1)
        print(f"  Step {step+1}: "
              f"vitality={result['total_vitality']:.2f} "
              f"events={result['emergence_events']} "
              f"breakdown={result['emergence_breakdown']}")

    # 5. 涌现事件详情
    print("\n[5/5] 涌现事件详情")
    events = system.emergence_log[-10:]  # 最近10个
    for e in events:
        loc = e["location"]
        print(f"  [{e['type']}] @{loc} vitality={e['vitality']:.1f}")
        if "description" in e:
            print(f"    → {e['description']}")
        if e["type"] == "self_create" and "new_params" in e:
            print(f"    → 涌现新参数: {list(e['new_params'].keys())}")

    # 最终统计
    print("\n" + system.visualize())


if __name__ == "__main__":
    run_vitality_demo()
