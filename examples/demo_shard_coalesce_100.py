"""
碎片聚合成功率测试 —— 100 次尝试,看工厂碎片能组合成完整维度不
================================================================

每次尝试:
  1. 工厂生产一批 DimensionShard (数量随机 10~100)
  2. 用碎片合成 DimensionCore
  3. 尝试聚合成 Dimension
  4. 判断是否"成功":
     - 核心等级 >= 3
     - 碎片数量 >= 15
     - 聚合后稳定性 >= 0.2
     - 维度能正常 tick 10h 不崩
  5. 统计成功率,看 100 次里有没有达到 10%

同时测试:不同维度本质的聚合成功率差异
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xuni.dimension_system import (
    Dimension, DimensionNature, DimensionSize,
    DimensionExplorer,
)
from xuni.multiverse_resources import (
    MultiverseResourceFactory, DimensionCore, DimensionShard,
    ResourceRarity,
)


def test_single_attempt(factory, nature, size):
    """单次尝试: 碎片 → 核心 → 维度, 看成不成"""

    # 1. 生产碎片 (较高等级 + 足够数量)
    shard_count = max(20, int(os.urandom(1)[0] % 81 + 20))
    shard_level = max(10, int(os.urandom(1)[0] % 20 + 10))

    shards = []
    for _ in range(shard_count):
        s = factory.produce_dimension_shard(level=shard_level)
        shards.append(s)

    if len(shards) < 15:
        return {"success": False, "reason": f"碎片不足: 只有 {len(shards)} 片"}

    # 2. 合成核心
    try:
        core = DimensionCore.create_from_shards(shards)
    except Exception as e:
        return {"success": False, "reason": f"核心合成失败: {e}"}

    if core.level < 3:
        return {"success": False, "reason": f"核心等级太低: Lv.{core.level}"}

    # 3. 聚合成维度
    try:
        dim = Dimension.coalesce(shards, nature, size,
                                  name=f"碎片聚合-{nature.name}")
    except Exception as e:
        return {"success": False, "reason": f"维度聚合失败: {e}"}

    # 4. 注入培养液提升稳定性 (模拟工厂的"启动扶持")
    try:
        culture = factory.produce_culture_medium(culture_type="robust", level=10)
        dim.inject_culture_boost(culture)
    except Exception:
        pass

    # 5. 检查初始稳定性
    if dim.stability < 0.2:
        return {
            "success": False,
            "reason": f"稳定性太低: {dim.stability:.3f}",
            "stability": dim.stability,
            "core_level": core.level,
            "shard_count": len(shards),
        }

    # 6. 尝试 tick 10h 看会不会崩
    try:
        dim.tick(hours=10)
    except Exception as e:
        return {"success": False, "reason": f"维度运转崩溃: {e}"}

    if not (0 < dim.stability <= 1.0):
        return {
            "success": False,
            "reason": f"运转后稳定性异常: {dim.stability:.3f}",
            "stability": dim.stability,
        }

    # 成功!
    return {
        "success": True,
        "nature": nature.name,
        "shard_count": len(shards),
        "core_level": core.level,
        "stability": round(dim.stability, 3),
        "dimension_name": dim.name,
        "product_count": dim.product_count,
    }


def run_100_trial():
    print("=" * 70)
    print("  碎片聚合成功率测试 (100 次)")
    print("=" * 70)

    factory = MultiverseResourceFactory(owner="聚合测试", parallel_lines=32)

    natures = list(DimensionNature)
    sizes = [DimensionSize.POCKET, DimensionSize.MINI, DimensionSize.STANDARD]

    total_attempts = 100
    successes = []
    failures = []
    nature_stats = {}

    for i in range(total_attempts):
        nature = natures[i % len(natures)]
        size = sizes[i % len(sizes)]

        result = test_single_attempt(factory, nature, size)

        if result["success"]:
            successes.append(result)
            nature_key = nature.name
            if nature_key not in nature_stats:
                nature_stats[nature_key] = {"success": 0, "fail": 0}
            nature_stats[nature_key]["success"] += 1
        else:
            failures.append(result)
            nature_key = nature.name
            if nature_key not in nature_stats:
                nature_stats[nature_key] = {"success": 0, "fail": 0}
            nature_stats[nature_key]["fail"] += 1

        if (i + 1) % 10 == 0:
            rate = len(successes) / (i + 1) * 100
            print(f"    [{i+1:>3}/100] 成功 {len(successes)} 个, 累计成功率 {rate:.1f}%")

    success_rate = len(successes) / total_attempts * 100

    print(f"\n{'='*70}")
    print(f"  最终结果")
    print(f"{'='*70}")
    print(f"  总尝试: {total_attempts}")
    print(f"  成功: {len(successes)}")
    print(f"  失败: {len(failures)}")
    print(f"  成功率: {success_rate:.1f}%")
    print(f"  达到 10% 目标: {'是! 🎉' if success_rate >= 10 else '否... 继续优化'}")

    print(f"\n  各维度本质成功率:")
    for nature_key in sorted(nature_stats.keys(),
                             key=lambda k: nature_stats[k]["success"],
                             reverse=True):
        s = nature_stats[nature_key]
        total = s["success"] + s["fail"]
        rate = s["success"] / max(1, total) * 100
        print(f"    {nature_key:20s}: {s['success']}/{total} ({rate:.0f}%)")

    if successes:
        print(f"\n  成功案例详情:")
        for s in successes[:5]:
            print(f"    - {s['dimension_name']} ({s['nature']}) "
                  f"碎片={s['shard_count']} 核心Lv={s['core_level']} "
                  f"稳定={s['stability']} 产物={s['product_count']}")

    if failures:
        # 统计失败原因
        reason_counts = {}
        for f in failures:
            r = f.get("reason", "unknown")
            # 简化原因
            if "碎片不足" in r:
                r = "碎片数量不足"
            elif "核心等级太低" in r:
                r = "核心等级太低"
            elif "稳定性太低" in r:
                r = "初始稳定性太低"
            elif "运转" in r and "崩溃" in r:
                r = "维度运转崩溃"
            elif "稳定性异常" in r:
                r = "运转后稳定性异常"
            reason_counts[r] = reason_counts.get(r, 0) + 1

        print(f"\n  失败原因分布:")
        for reason, count in sorted(reason_counts.items(),
                                     key=lambda x: -x[1]):
            print(f"    {reason}: {count} 次")

    return {
        "total": total_attempts,
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": success_rate,
        "achieved_10_percent": success_rate >= 10,
        "nature_stats": nature_stats,
        "success_details": successes[:10],
        "failure_details": failures[:10],
    }


if __name__ == '__main__':
    run_100_trial()
