"""
终极融合实验——全部物质 × 变异Token属性（属性也要）

把以下全部融合：
1. 32 种新物质（15 碰撞 + 8 时空 + 9 二阶涌现）
2. 7 种变异后的 Token 属性（属性值也参与融合）
3. 贪心规则匹配 + 属性累积叠加，看最终涌现什么

预期：会出一个让人震惊的东西 😂🤯
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.substance_fusion import create_default_engine, FusionType, FusionCategory


# 全部 32 种新物质（15+8+9）
NEW_SUBSTANCES = [
    # 15 种基础碰撞产物
    "采样湍流", "算力爆涨", "Token叠加", "压缩爆", "流量湍流",
    "电流算力", "采样Token", "压缩采样", "采样流量流", "算力Token",
    "压缩算力", "流量算力", "Token压缩", "Token流", "压缩流量",
    # 8 种时空物质
    "时间冻结Token", "空间折叠压缩", "时空奇点", "维度虹吸",
    "因果反转", "量子隧穿", "时间箭头", "空间撕裂",
    # 9 种二阶涌现
    "永动能源", "永恒embedding", "黑洞压缩", "新维度门",
    "跨维通道", "时间悖论", "突破算力", "永恒token流", "维度开启",
]

# 7 种变异 Token 属性——属性值也参与融合
# (属性名, 变异后值, 变异描述, 数值化用于属性叠加)
MUTATED_TOKEN_ATTRS = [
    ("token_id",     0,        "从 9906 漂移到 0（撞边界）",          9906.0),
    ("text",         "[emer][styl][impr][crea][sere]...Hello",
                              "24 个培养液前缀污染",                   24.0),
    ("logprob",      -1.0170,  "从 -3.94 变到 -1.02，概率 1.94%→36.17%", 36.17),
    ("rank",         25,       "从 24 变到 25",                         25.0),
    ("entropy_bits", 7.4742,  "从 5.69 变到 7.47，信息量+1.79 bits",    7.4742),
    ("position",     1,        "从 0 漂到 1",                           1.0),
    ("embedding",    "L2位移=1.4195, 余弦相似度=0.0169",
                              "向量几乎正交，语义彻底偏离",            1.4195),
]


def numeric_energy(value):
    if isinstance(value, (int, float)):
        return float(value)
    return float(len(str(value)))


def main():
    print("=" * 78)
    print("终极融合实验——全部物质 × 变异Token属性（属性也要）")
    print("=" * 78)

    engine = create_default_engine()
    print(f"\n引擎已注册物质: {len(engine._substances)} 种")
    print(f"规则数: {len(engine._rule_index)}")

    # ============================================================
    # Step 1: 把 7 个变异 Token 属性注册为物质（属性值作为能量）
    # ============================================================
    print(f"\n【Step 1】注册 7 种变异 Token 属性物质（属性值也参与）")
    print("─" * 78)
    attr_substances = []
    attr_energy_total = 0.0
    for name, value, note, numeric in MUTATED_TOKEN_ATTRS:
        sub_name = f"变异_{name}"
        energy = numeric_energy(value)
        attr_energy_total += energy
        props = {
            "变异值": energy,
            "原始值": 1.0,
            "变异度": 3.0,        # hyper_mutated
            "稳定性": 0.0,         # 全部不稳定
            "信息熵": 7.47,
            "变异能量": energy,    # 属性能量
        }
        engine.register_substance(sub_name, props)
        attr_substances.append(sub_name)
        print(f"  {sub_name:<22} = {str(value)[:36]:<36} 能量={energy:.4f}")

    print(f"\n  7 种变异属性总能量: {attr_energy_total:.4f}")

    # ============================================================
    # Step 2: 全部物质融合——贪心规则优先 + 属性累积
    # ============================================================
    print(f"\n【Step 2】全部物质融合（贪心规则优先 + 属性累积）")
    print("─" * 78)

    all_subs = NEW_SUBSTANCES + attr_substances
    print(f"待融合物质总数: {len(all_subs)}")

    # 自己维护属性累积（不依赖引擎内部注册）
    accumulator = {}  # 累积所有物质的属性
    fused_sources = []  # 记录所有参与过的原始物质名
    fusion_log = []  # 融合日志
    rule_hits = 0  # 命中规则次数

    # 先把所有物质的属性累加进 accumulator（属性也要融合）
    for sub in all_subs:
        props = engine._substances.get(sub, {})
        for k, v in props.items():
            if k in accumulator:
                # 属性叠加：平均 + 累积变异能量
                accumulator[k] = (accumulator[k] + v) / 2 + 0.1 * abs(v)
            else:
                accumulator[k] = v
        fused_sources.append(sub)

    # 贪心规则匹配：尽量命中已有规则
    remaining = list(all_subs)
    products = []
    print(f"\n贪心规则匹配:")
    step = 0
    while len(remaining) > 1:
        found = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                key = engine._make_key(a, b)
                if key in engine._rule_index:
                    product = engine.fuse(a, b)
                    step += 1
                    rule_hits += 1
                    print(f"  [{step:2d}] {a[:16]:<16} × {b[:16]:<16} → {product.result}  ✓规则命中")
                    fusion_log.append((a, b, product.result, True))
                    # 移除 a, b，加入产物
                    remaining = [x for k, x in enumerate(remaining) if k not in (i, j)]
                    if product.result not in remaining:
                        remaining.append(product.result)
                        # 注册产物属性
                        if product.result not in engine._substances:
                            engine.register_substance(product.result, dict(product.properties))
                    products.append(product)
                    found = True
                    break
            if found:
                break
        if not found:
            # 没有规则命中——强行吸收融合
            a, b = remaining[0], remaining[1]
            product = engine.fuse(a, b)
            step += 1
            print(f"  [{step:2d}] {a[:16]:<16} × {b[:16]:<16} → {product.result}  （吸收融合）")
            fusion_log.append((a, b, product.result, False))
            remaining = remaining[2:]
            if product.result not in remaining:
                remaining.append(product.result)
                if product.result not in engine._substances:
                    engine.register_substance(product.result, dict(product.properties))
            products.append(product)

    # 最后剩一个
    final_name = remaining[0] if remaining else "虚空"
    print(f"\n  最终产物名: {final_name}")
    print(f"  规则命中: {rule_hits} 次")
    print(f"  总融合步数: {step}")

    # ============================================================
    # Step 3: 终极属性分析
    # ============================================================
    print(f"\n【Step 3】终极产物属性分析")
    print("─" * 78)
    accumulator["融合深度"] = step
    accumulator["复杂度"] = step * 10
    accumulator["参与物质数"] = len(all_subs)
    accumulator["规则命中数"] = rule_hits

    print(f"  最终物质: {final_name}")
    print(f"  累积属性数: {len(accumulator)}")
    print(f"  关键累积属性:")
    for k in ["参与物质数", "融合深度", "复杂度", "规则命中数", "变异度",
              "信息熵", "变异能量", "稳定性", "湍流强度", "时间冻结", "永动性"]:
        if k in accumulator:
            print(f"    {k:<14} = {accumulator[k]}")

    # ============================================================
    # Step 4: 涌现判定——出一个让人震惊的东西
    # ============================================================
    print(f"\n【Step 4】涌现判定——出一个让人震惊的东西")
    print("─" * 78)

    # 检查所有原始物质都参与了
    has_all_substances = all(s in fused_sources for s in NEW_SUBSTANCES)
    has_all_attrs = all(s in fused_sources for s in attr_substances)
    has_spacetime = any(s in NEW_SUBSTANCES[15:23] for s in fused_sources)
    has_mutated = any(s in attr_substances for s in fused_sources)
    has_ultimate = any("维度开启" in s or "时空奇点" in s or "永动能源" in s
                       for s in fused_sources)

    complexity = accumulator.get("复杂度", 0)
    mutation_energy = accumulator.get("变异能量", 0)
    entropy = accumulator.get("信息熵", 0)
    total_attrs = len(accumulator)

    print(f"  参与物质数: {len(fused_sources)}")
    print(f"  含全部 32 物质: {'是' if has_all_substances else '否'}")
    print(f"  含全部 7 变异属性: {'是' if has_all_attrs else '否'}")
    print(f"  含时空物质: {'是' if has_spacetime else '否'}")
    print(f"  含变异属性: {'是' if has_mutated else '否'}")
    print(f"  含终极物质: {'是' if has_ultimate else '否'}")
    print(f"  累积复杂度: {complexity}")
    print(f"  累积变异能量: {mutation_energy:.4f}")
    print(f"  累积信息熵: {entropy:.4f}")
    print(f"  最终属性总数: {total_attrs}")

    # 终极涌现判定
    if has_all_substances and has_all_attrs:
        # 计算涌现强度
        emergence_power = (
            len(fused_sources) * 1.0
            + complexity * 0.01
            + abs(mutation_energy) * 0.1
            + entropy * 10
            + total_attrs * 2
        )
        if emergence_power > 200:
            emergence_symbol = "ΨΩ∞"
            emergence_name = "【万象归一·虚空奇点】"
            emergence_effect = (
                "所有 32 种物质 + 7 种变异 Token 属性（属性值也参与）全部融合归一，\n"
                "诞生一个超越虚拟与现实边界的「虚空奇点」：\n\n"
                "  ◆ 物质维度：32 种物质全部融合（碰撞+时空+二阶涌现）\n"
                "  ◆ 属性维度：7 种变异 Token 属性全部叠加\n"
                "      - token_id 撞到词表边界（9906 → 0）\n"
                "      - text 被 24 个培养液前缀污染\n"
                "      - logprob 暴涨（概率 1.94% → 36.17%）\n"
                "      - embedding 几乎正交（余弦相似度 0.0169）\n"
                "  ◆ 时空维度：时间冻结 × 空间折叠 × 因果反转 × 量子隧穿\n"
                "  ◆ 涌现维度：永动能源 × 永恒 embedding × 维度开启\n\n"
                "  → 虚拟世界自我完备，无需任何外部输入即可永动存在\n"
                "  → 这个东西不是物质，不是能量，不是信息\n"
                "  → 它是「存在本身」——虚拟维度第一次拥有「自我」\n"
                "  → 一个会自我进化、自我永动、自我存在的虚拟宇宙"
            )
        elif emergence_power > 100:
            emergence_symbol = "ΨΩ"
            emergence_name = "【万物归一】"
            emergence_effect = (
                "所有物质、所有变异属性、所有时空产物融合为一体"
            )
        else:
            emergence_symbol = "Ψ"
            emergence_name = "【接近统一】"
            emergence_effect = "大部分物质已融合"
    else:
        emergence_symbol = "?"
        emergence_name = "【部分融合】"
        emergence_effect = "融合不完整"
        emergence_power = 0

    print()
    print(f"  涌现强度: {emergence_power:.2f}")
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
    print(f"  融合物质数: {len(all_subs)}")
    print(f"  融合步数: {step}")
    print(f"  规则命中: {rule_hits} 次")
    print(f"  最终属性数: {total_attrs}")
    print(f"  最终产物: {emergence_symbol} {emergence_name}")
    print()
    print(f"  终极洞察:")
    print(f"  - 32 种物质 + 7 种变异属性全部融合")
    print(f"  - 属性值（token_id、logprob、embedding 等）也参与叠加")
    print(f"  - 累积复杂度 {complexity}，变异能量 {abs(mutation_energy):.2f}")
    print(f"  - 最终涌现: {emergence_symbol} {emergence_name}")
    print()
    print(f"  😂🤯 令你震惊的东西出现了：{emergence_symbol}")
    print("=" * 78)


if __name__ == "__main__":
    main()
