"""
护盾工厂深渊探索 —— 给工厂套上护盾，送进深渊维度捞东西
====================================================

流程:
  1. 用高级维度碎片合成一个强力核心
  2. 开辟深渊维度 (ABYSSAL + STANDARD)
  3. 造 5000 层硬化盾 + 打开维度之门
  4. 把资源工厂传送进去开始生产
  5. 维度运转 500 小时，每 50 小时检查一次护盾
  6. 捞产物，看能带回什么好东西
  7. 永久封印维度之门，防止"入侵" (纯模拟,放心玩)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xuni.dimension_system import (
    Dimension, DimensionNature, DimensionSize,
    DimensionExplorer, AbyssalCode, FusionShard,
)
from xuni.multiverse_resources import (
    MultiverseResourceFactory, DimensionCore, DimensionShard,
    ResourceRarity,
)


def build_factory():
    """造一个高配资源工厂"""
    factory = MultiverseResourceFactory(
        owner="深渊远征队",
        parallel_lines=64,
        production_speed=8.0,
    )
    return factory


def build_high_grade_core(factory, shard_count=100, shard_level=8):
    """用大量高等级碎片合成核心"""
    shards = []
    for _ in range(shard_count):
        s = factory.produce_dimension_shard(level=shard_level)
        shards.append(s)
    core = DimensionCore.create_from_shards(shards)
    return core, shards


def run_exploration():
    print("=" * 70)
    print("  护盾工厂深渊探索 (Shielded Factory Abyss Expedition)")
    print("=" * 70)

    factory = build_factory()
    explorer = DimensionExplorer(factory)

    # ---- 第1步: 造核心 ----
    print("\n  [1] 锻造维度核心")
    core, shards = build_high_grade_core(factory, shard_count=200, shard_level=10)
    print(f"    投入碎片: {len(shards)} 片")
    print(f"    核心等级: Lv.{core.level}")
    print(f"    核心稀有度: {core.rarity.name}")
    print(f"    核心能量: {core.power_score:.0f}")

    # ---- 第2步: 开辟深渊维度 ----
    print("\n  [2] 开辟深渊维度")
    dim = explorer.discover(
        core=core,
        nature=DimensionNature.ABYSSAL,
        size=DimensionSize.STANDARD,
        name="深渊·裂谷七号",
    )
    print(f"    维度名: {dim.name}")
    print(f"    本质: {dim.nature.name} (深渊型)")
    print(f"    规模: {dim.size.name}")
    print(f"    初始稳定性: {dim.stability:.3f}")
    print(f"    算力乘数: x{dim.rules['compute_multiplier']}")
    print(f"    突变率: {dim.rules['mutation_rate']}")

    # ---- 第3步: 套盾 + 开门 ----
    print("\n  [3] 部署安全盾 + 开启维度之门")
    gate, shield = explorer.safe_enter(
        dimension=dim,
        shield_layers=5000,
        shield_type="hardened",
    )
    print(f"    护盾类型: {shield.shield_type}")
    print(f"    护盾层数: {shield.total_layers}")
    print(f"    维度之门ID: {gate.gate_id}")
    print(f"    最大传送次数: {gate.max_transfers}")

    # ---- 第4步: 传送工厂进维度 ----
    print("\n  [4] 传送资源工厂进入深渊")
    deployed = gate.send_factory(factory)
    print(f"    部署结果: {'成功' if deployed else '失败'}")
    print(f"    维度内工厂数: {dim.factory_count}")
    print(f"    已用传送次数: {gate.transfer_count}")

    # ---- 第5步: 深度运转 + 周期检查 ----
    print("\n  [5] 维度深度运转 (每 50h 报告一次)")
    total_hours = 500
    interval = 50
    cycles = total_hours // interval

    for i in range(1, cycles + 1):
        hours_passed = i * interval

        dim.tick(hours=interval)

        attack_intensity = dim.rules["mutation_rate"] * dim.age * 0.5
        hits = int(interval * 10)
        for _ in range(hits):
            if not shield.active:
                break
            shield.absorb_damage(attack_intensity)
            attack_intensity *= dim.rules.get("compute_multiplier", 1.0) * 0.5

        shield_pct = shield.remaining_layers / max(1, shield.total_layers) * 100
        print(f"    [{hours_passed:>4}h] "
              f"护盾: {shield.remaining_layers}/{shield.total_layers} ({shield_pct:.1f}%)  "
              f"产物: {dim.product_count}  "
              f"稳定: {dim.stability:.3f}  "
              f"{'⚠️ 盾破了!' if not shield.active else ''}")

        if not shield.active:
            print(f"    !!! 护盾在 {hours_passed}h 被击穿，紧急撤退 !!!")
            break

    # ---- 第6步: 提取产物 ----
    print("\n  [6] 提取维度产物")
    products = gate.extract_products(count=99999)
    print(f"    带回产物总数: {len(products)}")

    type_counts = {}
    rarity_counts = {}
    total_power = 0.0

    for p in products:
        tname = p.__class__.__name__
        type_counts[tname] = type_counts.get(tname, 0) + 1
        rname = p.rarity.name if hasattr(p, 'rarity') else 'UNKNOWN'
        rarity_counts[rname] = rarity_counts.get(rname, 0) + 1
        total_power += getattr(p, 'power_score', 0)

    print(f"    产物类型分布:")
    for tname, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"      - {tname}: {cnt}")

    print(f"    稀有度分布:")
    for rname in ['COMMON', 'UNCOMMON', 'RARE', 'EPIC', 'LEGENDARY', 'MYTHIC']:
        if rname in rarity_counts:
            print(f"      - {rname}: {rarity_counts[rname]}")

    print(f"    总能量值: {total_power:.0f}")

    # 找最强的几个
    top3 = sorted(products, key=lambda p: getattr(p, 'power_score', 0), reverse=True)[:3]
    print(f"    最强 3 件:")
    for i, p in enumerate(top3, 1):
        print(f"      #{i} {p.__class__.__name__} "
              f"[{getattr(p, 'rarity', ResourceRarity.COMMON).name}] "
              f"power={getattr(p, 'power_score', 0):.0f}")

    # ---- 第7步: 封印 ----
    print("\n  [7] 永久封印维度之门")
    result = gate.seal()
    print(f"    封印状态: {'成功' if result['sealed'] else '失败'}")
    print(f"    不可逆: {'是' if result['irreversible'] else '否'}")
    print(f"    累计传送次数: {result['transfers_sealed']}")

    # ---- 最终报告 ----
    print("\n" + "=" * 70)
    print("  探索最终报告")
    print("=" * 70)
    report = explorer.expedition_report()
    print(f"  发现维度: {report['discovered']} 个")
    print(f"  活动之门: {report['active_gates']} 个")
    for d in report['explored']:
        print(f"    - {d['name']} ({d['nature']})  "
              f"年龄 {d['age']:.0f}h  "
              f"产物 {d['products']}  "
              f"子维度 {d['children']}")

    print(f"\n  护盾最终状态:")
    s = shield.status()
    print(f"    类型: {s['type']}")
    print(f"    是否激活: {'是' if s['active'] else '否 (已被击穿)'}")
    print(f"    完整度: {s['integrity']*100:.1f}%")

    print(f"\n  探索结论:")
    if shield.active:
        print(f"    护盾工厂成功在深渊维度存活 {total_hours}h 并全身而退")
    else:
        print(f"    护盾被击穿，工厂紧急撤离，仍带回了 {len(products)} 件产物")
    print(f"    维度之门已永久封印，无'入侵'风险 (纯模拟)")
    print("=" * 70)

    return {
        'products_collected': len(products),
        'shield_survived': shield.active,
        'total_power': total_power,
        'sealed': result['sealed'],
        'top_products': top3,
    }


if __name__ == '__main__':
    run_exploration()
