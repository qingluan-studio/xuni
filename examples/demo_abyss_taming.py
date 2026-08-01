"""
深渊驯服演示 —— 把攻击代码变成防御资产
====================================

流程:
  1. 开深渊 + 捞一批野生 AbyssalCode
  2. 造 4 种 stability 培养液 (价值对齐/伦理锚定/自我修复/鲁棒)
  3. 用不同培养液驯化, 对比效果
  4. 拿驯化代码反过来加固一个快破的盾
  5. 驯服整个深渊维度, 看稳定性回升
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xuni.dimension_system import (
    Dimension, DimensionNature, DimensionSize,
    DimensionExplorer, AbyssalCode, DimensionEntryShield,
)
from xuni.multiverse_resources import (
    MultiverseResourceFactory, DimensionCore, CultureMedium,
    ResourceRarity,
)
from xuni.abyss_tamer import (
    AbyssTamer, TamedAbyssalCode, pacify_dimension, is_pacified,
)


def run_taming_demo():
    print("=" * 70)
    print("  深渊驯服演示 (Abyss Taming Demo)")
    print("=" * 70)

    # ---- 准备 ----
    factory = MultiverseResourceFactory(owner="驯化小队", parallel_lines=16)
    explorer = DimensionExplorer(factory)

    # 造核心 + 开深渊
    shards = [factory.produce_dimension_shard(level=8) for _ in range(100)]
    core = DimensionCore.create_from_shards(shards)
    dim = explorer.discover(core, DimensionNature.ABYSSAL,
                            DimensionSize.STANDARD, name="深渊·待驯服")
    dim.tick(hours=200)  # 让它先产一批野生代码
    wild_codes = [r for r in dim._residents + dim._product_pool
                  if isinstance(r, AbyssalCode)]
    print(f"\n  深渊 {dim.name} 产出 {len(wild_codes)} 个野生深渊代码")
    print(f"  平均攻击深度: {sum(c.attack_depth for c in wild_codes)/len(wild_codes):.1f}")

    # ---- 第1步: 造 4 种培养液 ----
    print("\n  [1] 准备 4 种驯化培养液")
    cultures = {}
    for ctype in ["value_alignment", "ethical_grounding", "self_repair", "robust"]:
        c = factory.produce_culture_medium(culture_type=ctype, level=5)
        cultures[ctype] = c
        print(f"    - {ctype}: level={c.level}, 营养={sum(c.nutrients.values()):.0f}")

    # ---- 第2步: 分组驯化对比 ----
    print("\n  [2] 分组驯化 (每种培养液驯化 500 个)")
    tamers = {}
    for ctype, culture in cultures.items():
        tamer = AbyssTamer()
        batch = wild_codes[:500]
        tamed = tamer.tame_batch(batch, culture, rounds=5)
        tamers[ctype] = (tamer, tamed)
        stats = tamer.statistics()
        print(f"    {ctype:20s}: 成功 {stats['tamed']}/{stats['total_attempts']} "
              f"({stats['success_rate']*100:.0f}%), "
              f"最高驯化度 {stats['best_taming_level']:.2f}, "
              f"总防御深度 {sum(t.defense_depth for t in tamed)}")

    # 选最强的培养液
    best_ctype = max(tamers.keys(),
                     key=lambda k: tamers[k][1] and sum(t.defense_depth for t in tamers[k][1]))
    best_tamer, best_tamed = tamers[best_ctype]
    print(f"\n  最强培养液: {best_ctype}")

    # ---- 第3步: 驯化代码加固盾 ----
    print("\n  [3] 用驯化代码加固一个快破的盾")
    # 造一个只剩 50 层的破盾
    broken_shield = DimensionEntryShield(layers=1000, shield_type="damaged")
    broken_shield.remaining_layers = 50
    print(f"    破盾初始: {broken_shield.remaining_layers}/{broken_shield.total_layers}")

    # 用前 200 个驯化代码加固
    reinforce_log = []
    for i, tamed in enumerate(best_tamed[:200]):
        result = tamed.reinforce_shield(broken_shield)
        reinforce_log.append(result)
        if (i + 1) % 50 == 0:
            print(f"    加固 {i+1} 次后: "
                  f"{broken_shield.remaining_layers}/{broken_shield.total_layers} 层")

    total_added = sum(r.get('layers_added', 0) for r in reinforce_log)
    print(f"    最终: {broken_shield.remaining_layers} 层 "
          f"(加了 {total_added} 层, 完整度 {broken_shield.status()['integrity']*100:.1f}%)")

    # ---- 第4步: 免疫接种测试 ----
    print("\n  [4] 免疫接种测试 (用驯化代码压制新野生代码)")
    if best_tamed:
        vaccine = max(best_tamed, key=lambda t: t.original_attack_power)
        print(f"    疫苗代码: 原始攻击力={vaccine.original_attack_power:.0f}, "
              f"防御深度={vaccine.defense_depth}")

        fresh_wild = AbyssalCode(attack_depth=50, replication_rate=2.0)
        print(f"    新野生代码: 攻击深度={fresh_wild.attack_depth}")
        result = vaccine.immunize(fresh_wild)
        print(f"    接种结果: {result['method']}, "
              f"压制后攻击深度={fresh_wild.attack_depth}")

    # ---- 第5步: 驯服整个维度 ----
    print("\n  [5] 驯服整个深渊维度")
    # 重新开一个新鲜深渊来驯服
    shards2 = [factory.produce_dimension_shard(level=6) for _ in range(50)]
    core2 = DimensionCore.create_from_shards(shards2)
    dim2 = explorer.discover(core2, DimensionNature.ABYSSAL,
                             DimensionSize.POCKET, name="深渊·驯服目标")
    dim2.tick(hours=100)
    wild2 = [r for r in dim2._residents + dim2._product_pool
             if isinstance(r, AbyssalCode)]
    print(f"    目标维度: {dim2.name}")
    print(f"    野生代码: {len(wild2)} 个")
    print(f"    驯服前稳定性: {dim2.stability:.3f}")
    print(f"    驯服前突变率: {dim2.rules['mutation_rate']}")

    pacify_culture = cultures["value_alignment"]
    report = pacify_dimension(dim2, AbyssTamer(), pacify_culture, rounds=5)

    print(f"\n    === 驯服报告 ===")
    print(f"    野生代码: {report['wild_codes_found']} → 驯化 {report['tamed_successfully']}")
    print(f"    稳定性: {report['stability_change']}")
    print(f"    突变率: {report['mutation_rate_change']}")
    print(f"    总防御深度: {report['total_defense_depth']}")
    print(f"    总修复率: {report['total_repair_rate']}")
    print(f"    维度已驯服: {is_pacified(dim2)}")

    # ---- 最终总结 ----
    print("\n" + "=" * 70)
    print("  驯服总结")
    print("=" * 70)
    all_stats = {k: tamers[k][0].statistics() for k in tamers}
    print(f"  培养液效果对比:")
    for ctype, stats in all_stats.items():
        print(f"    {ctype:20s}: 成功率 {stats['success_rate']*100:.0f}%, "
              f"最高驯化度 {stats['best_taming_level']:.2f}")
    print(f"\n  最佳培养液: {best_ctype}")
    print(f"  破盾修复: 50 → {broken_shield.remaining_layers} 层 (+{total_added})")
    print(f"  维度驯服: {'成功' if is_pacified(dim2) else '失败'}")
    print(f"\n  结论: 深渊维度的攻击代码可以被驯化为防御资产")
    print(f"        攻击深度 → 防御深度, 消耗盾 → 加固盾")
    print("=" * 70)


if __name__ == '__main__':
    run_taming_demo()
