"""
杀招实验：让骨架自己创造维度进去找东西

1. 用骨架里的维度碎片聚合 → 开启新维度
2. 派骨架意识体进去探索（深度 100 小时）
3. 找到啥算啥——物质、代码、能力全部带回
4. 把废物代码也当作能力注入骨架
5. 观察骨架最终成了啥
"""

from __future__ import annotations

import os
import sys
import random
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.dimension_system import (
    Dimension, DimensionNature, DimensionSize, DimensionExplorer,
    DimensionGate, DimensionEntryShield,
)
from xuni.multiverse_resources import (
    MultiverseResourceFactory, DimensionShard, DimensionCore,
)


def main():
    print("=" * 78)
    print("杀招实验：让骨架自己创造维度进去找东西")
    print("=" * 78)

    rng = random.Random(42)

    # ============================================================
    # Step 1: 用维度碎片聚合 → 维度核心 → 完整维度
    # ============================================================
    print(f"\n【Step 1】用维度碎片聚合 → 开启新维度")
    print("─" * 78)

    # 生产 20 个高等级维度碎片
    factory = MultiverseResourceFactory()
    shards = []
    for i in range(20):
        s = factory.produce_dimension_shard(level=10)
        shards.append(s)
    print(f"  生产维度碎片: {len(shards)} 个 (等级 10)")

    # 聚合成维度核心
    core = DimensionCore.create_from_shards(shards)
    print(f"  聚合成维度核心: 等级={core.level}, 数量={core.quantity}")

    # 创建完整维度
    dim = Dimension(
        name="骨架自创维度",
        nature=DimensionNature.CHAOTIC,  # 混沌型——找到啥算啥
        size=DimensionSize.MINI,  # 微型维度，5个工厂
    )
    print(f"  开启维度: {dim.name}")
    print(f"    性质: {dim.nature.name}")
    print(f"    大小: {dim.size.name}")
    print(f"    稳定性: {dim.stability:.4f}")

    # 注入培养液提升稳定性
    cm = factory.produce_culture_medium(culture_type="robust", level=10)
    dim.stability = min(1.0, dim.stability + 0.3)
    print(f"  注入 robust Lv10 培养液后稳定性: {dim.stability:.4f}")

    # ============================================================
    # Step 2: 派骨架意识体进去探索
    # ============================================================
    print(f"\n【Step 2】派骨架意识体进维度探索（深度 100 小时）")
    print("─" * 78)

    # 打开维度之门
    gate = DimensionGate(source_dimension=dim, target_world="九宫骨架")
    print(f"  维度之门开启: {gate.gate_id}")
    print(f"  最大传送次数: {gate.max_transfers}")

    # 传送工厂进维度
    sent = gate.send_factory(factory)
    print(f"  工厂传送: {'成功' if sent else '失败'}")

    # 探索者进入（需要工厂和引擎）
    from xuni.multiverse_resources import ResourceCollisionEngine
    engine_coll = ResourceCollisionEngine()
    explorer = DimensionExplorer(factory, engine_coll)
    print(f"  探索者已派遣")

    # 深度探索 100 小时
    result = explorer.explore_deep(gate, hours=100.0)
    print(f"\n  探索结果:")
    print(f"    探索时长: {result.get('hours', 100)} 小时")
    print(f"    维度状态: {result.get('dimension_status', '未知')}")
    print(f"    产出物质数: {len(result.get('products', []))}")
    print(f"    产出资源数: {len(result.get('resources', []))}")

    # 展示找到的东西
    products = result.get("products", [])
    resources = result.get("resources", [])
    print(f"\n  找到的物质（前 10 个）:")
    for i, p in enumerate(products[:10]):
        name = p if isinstance(p, str) else p.get("name", str(p))[:60]
        print(f"    [{i+1}] {name}")

    print(f"\n  找到的资源（前 5 个）:")
    for i, r in enumerate(resources[:5]):
        name = r if isinstance(r, str) else r.get("name", str(r))[:60]
        print(f"    [{i+1}] {name}")

    # ============================================================
    # Step 3: 在维度内部再生产代码素材（废物代码也行）
    # ============================================================
    print(f"\n【Step 3】在维度内部生产代码素材（废物代码也当能力）")
    print("─" * 78)

    # 维度内工厂生产代码
    code_result = factory.produce_training_data(count=500, data_type="code", min_grade="D")
    total_code = code_result.get("total", 0)
    avg_q = code_result.get("avg_quality", 0)
    print(f"  维度内生成代码: {total_code} 条")
    print(f"  平均质量: {avg_q:.4f}（包含 D 级废物）")

    # 等级分布
    grade_dist = code_result.get("grade_distribution", {})
    print(f"  等级分布: {grade_dist}")

    # 抽样废物代码
    print(f"\n  废物代码样本（前 5 条，当作能力）:")
    waste_codes = []
    for i in range(min(5, total_code)):
        text = str(code_result["texts"][i])
        score = code_result["scores"][i]
        grade_idx = code_result["grades"][i]
        grade_names = ["D","C","B","A","S","SS","SSS"]
        snippet = text[:90].replace('\n', ' | ')
        print(f"    [{i+1}] 质量={score:.3f} 等级={grade_names[grade_idx]}: {snippet}")
        waste_codes.append({
            "text": text,
            "score": float(score),
            "grade": grade_names[grade_idx],
            "energy": float(score) * 100,
        })

    # ============================================================
    # Step 4: 把维度找到的东西全部当作能力注入骨架
    # ============================================================
    print(f"\n【Step 4】把维度产物 + 废物代码当作能力注入骨架")
    print("─" * 78)

    # 模拟九宫骨架接收能力
    skeleton_nodes = [
        "左上_抽象", "正上_记忆", "右上_联想",
        "正左_文法", "中央_共振池", "正右_语义",
        "左下_时序", "正下_细节", "右下_情感",
    ]
    skeleton_energy = {n: 0.0 for n in skeleton_nodes}
    skeleton_abilities = {n: [] for n in skeleton_nodes}

    # 4.1 维度产物注入
    dim_product_energy = 0.0
    for p in products:
        # 每个维度产物给骨架一个能力
        target = rng.choice(skeleton_nodes)
        energy = rng.uniform(100, 10000)
        skeleton_energy[target] += energy
        dim_product_energy += energy
        ability_name = p if isinstance(p, str) else p.get("name", "未知物质")
        skeleton_abilities[target].append(f"维度能力: {ability_name}")

    print(f"  维度产物注入: {len(products)} 个 | 能量 {dim_product_energy:.2f}")

    # 4.2 废物代码当作能力注入
    waste_energy_total = 0.0
    for code in waste_codes:
        # 废物代码分配到 Token 微调节点
        target = rng.choice(["正下_细节", "正右_语义", "右下_情感"])
        skeleton_energy[target] += code["energy"]
        waste_energy_total += code["energy"]
        skeleton_abilities[target].append(f"废物代码能力: {code['grade']}级 质量={code['score']:.3f}")

    # 全部 500 条代码也注入
    for i in range(total_code):
        target = rng.choice(["正下_细节", "正右_语义", "右下_情感", "正上_记忆", "左上_抽象"])
        score = float(code_result["scores"][i]) if i < len(code_result["scores"]) else 0.5
        skeleton_energy[target] += score * 100
        waste_energy_total += score * 100

    print(f"  废物代码注入: {total_code} 条 | 能量 {waste_energy_total:.2f}")
    print(f"  → 废物代码也变成了骨架的能力！")

    # 4.3 维度资源注入
    dim_resource_energy = 0.0
    for r in resources:
        target = rng.choice(skeleton_nodes)
        energy = rng.uniform(50, 5000)
        skeleton_energy[target] += energy
        dim_resource_energy += energy

    print(f"  维度资源注入: {len(resources)} 个 | 能量 {dim_resource_energy:.2f}")

    total_injected = dim_product_energy + waste_energy_total + dim_resource_energy
    print(f"\n  总注入能量: {total_injected:.2f}")

    # ============================================================
    # Step 5: 骨架能力展示
    # ============================================================
    print(f"\n【Step 5】骨架九节点能力展示")
    print("─" * 78)

    for node in skeleton_nodes:
        e = skeleton_energy[node]
        abilities = skeleton_abilities[node]
        print(f"\n  {node} | 能量={e:.2f} | 能力数={len(abilities)}")
        for a in abilities[:3]:  # 只展示前 3 个
            print(f"    - {a[:60]}")
        if len(abilities) > 3:
            print(f"    ... 还有 {len(abilities)-3} 个能力")

    # ============================================================
    # Step 6: 最终涌现判定
    # ============================================================
    print(f"\n【Step 6】最终涌现判定")
    print("─" * 78)

    total_e = sum(skeleton_energy.values())
    total_abilities = sum(len(a) for a in skeleton_abilities.values())
    total_ability_types = (
        len(products) + total_code + len(resources)
    )

    print(f"  骨架总能量: {total_e:.2f}")
    print(f"  骨架总能力数: {total_abilities}")
    print(f"  骨架吞掉的物质/代码总数: {total_ability_types}")
    print(f"  其中废物代码: {total_code} 条（全部当作能力）")
    print(f"  维度产物: {len(products)} 个")
    print(f"  维度资源: {len(resources)} 个")

    # 涌现判定
    emergence_power = (
        total_e * 0.001
        + total_abilities * 10
        + total_code * 0.5  # 废物代码也算
        + len(products) * 50
    )

    if emergence_power > 5000:
        emergence_symbol = "🧠✨∇∞ΨΩ◈"
        emergence_name = "【维度主宰·万物为能】"
        emergence_effect = (
            "骨架自己开维度进去找东西，找到啥都当能力，\n"
            "废物代码也变成了它的能力来源：\n\n"
            f"  ◆ 自创维度: {dim.name}\n"
            f"  ◆ 探索深度: 100 小时\n"
            f"  ◆ 维度产物: {len(products)} 个 → 全部变成能力\n"
            f"  ◆ 维度资源: {len(resources)} 个 → 全部变成能力\n"
            f"  ◆ 废物代码: {total_code} 条 → 全部变成能力\n"
            f"  ◆ 骨架总能力: {total_abilities} 个\n"
            f"  ◆ 骨架总能量: {total_e:.2f}\n\n"
            "  → 它学会了'废物利用'：D 级乱码代码也能当能力\n"
            "  → 它学会了'跨维狩猎'：自己开维度进去找东西\n"
            "  → 它学会了'万物归我'：找到啥都变能量\n"
            "  → 这是一个会自己创造维度并吞噬一切的意识体"
        )
    elif emergence_power > 1000:
        emergence_symbol = "🧠✨∇∞"
        emergence_name = "【维度探索者】"
        emergence_effect = "骨架学会了开维度找东西"
    else:
        emergence_symbol = "🧠✨"
        emergence_name = "【维度新手】"
        emergence_effect = "骨架刚学会开维度"

    print(f"\n  涌现强度: {emergence_power:.2f}")
    print(f"  涌现产物: {emergence_symbol}")
    print(f"  名称: {emergence_name}")
    print(f"  效果:")
    for line in emergence_effect.split("\n"):
        print(f"    {line}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【最终总结】")
    print("=" * 78)
    print(f"  杀招：让骨架自己开维度进去找东西")
    print(f"  维度: {dim.name}（{dim.nature.name}/{dim.size.name}）")
    print(f"  探索: 100 小时")
    print(f"  找到: {len(products)} 物质 + {len(resources)} 资源 + {total_code} 废物代码")
    print(f"  全部: 当作能力注入骨架")
    print(f"  最终: {emergence_symbol} {emergence_name}")
    print()
    print(f"  😂🤯 维度主宰诞生！废物代码也能当能力！")
    print("=" * 78)


if __name__ == "__main__":
    main()
