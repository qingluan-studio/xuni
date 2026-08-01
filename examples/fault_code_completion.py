"""
断层代码补充实验——给骨架一段残缺代码，看它补充成什么

骨架有 9 节点 × 变异属性 × 废物代码能力
现在丢一段"断层代码"（中间被挖空）给它，让每个节点用各自的能力去补
"""

from __future__ import annotations

import os
import sys
import random
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. 断层代码——中间被挖空的函数
# ============================================================

FAULT_CODE = '''def process_data(data):
    """处理数据的主函数"""
    # === 断层开始 ===
    <断层1>
    <断层2>
    <断层3>
    # === 断层结束 ===
    return result
'''

# 3 个断层位置
FAULTS = ["<断层1>", "<断层2>", "<断层3>"]


# ============================================================
# 2. 9 节点 + 变异属性 + 废物代码能力（来自之前实验）
# ============================================================

NODES = {
    "左上_抽象": {
        "axis": "abstract",
        "mutation": ("embedding", "L2=1.4195 cos=0.0169 正交化"),
        "ability": "class list comprehension(yield item):\\n    def __init__(self,",
        "fill_style": "用类/抽象结构填",
    },
    "正上_记忆": {
        "axis": "memory",
        "mutation": ("token_id", "9906→0 撞边界"),
        "ability": "class from . import(set().add():\\n    def __init__(self,",
        "fill_style": "用 import/历史记忆填",
    },
    "右上_联想": {
        "axis": "analogy",
        "mutation": ("rank", "24→25"),
        "ability": "with open(lambda x:) as dict.get(:\\n    return set().add(",
        "fill_style": "用 with/类比填",
    },
    "正左_文法": {
        "axis": "syntax",
        "mutation": ("position", "0→1"),
        "ability": "if __init__ > 0:\\n    process(return result)",
        "fill_style": "用 if/语法约束填",
    },
    "中央_共振池": {
        "axis": "resonance",
        "mutation": ("logprob", "-1.017 概率暴涨"),
        "ability": "with open(return result) as class MyClass::\\n    return dict.get(",
        "fill_style": "用所有能力共振混合填",
    },
    "正右_语义": {
        "axis": "semantic",
        "mutation": ("text", "24 培养液前缀污染"),
        "ability": "[emer][styl][impr][crea][sere]...Hello",
        "fill_style": "用培养液前缀污染填",
    },
    "左下_时序": {
        "axis": "temporal",
        "mutation": ("position", "0→1"),
        "ability": "for i in range(100):\\n    x = compute(data[i])\\n    result.append(x)",
        "fill_style": "用 for 循环/时序填",
    },
    "正下_细节": {
        "axis": "detail",
        "mutation": ("token_id", "9906→0"),
        "ability": "result = data * 2\\n    return result",
        "fill_style": "用具体操作/微调填",
    },
    "右下_情感": {
        "axis": "emotion",
        "mutation": ("logprob", "概率暴涨"),
        "ability": "if data > 0:\\n    process(data)\\nelse:\\n    skip()",
        "fill_style": "用 if/else 情感分支填",
    },
}


def main():
    print("=" * 78)
    print("断层代码补充实验——给骨架残缺代码，看它补成什么")
    print("=" * 78)

    rng = random.Random(42)

    # ============================================================
    # Step 1: 展示断层代码
    # ============================================================
    print(f"\n【Step 1】断层代码（3 处空缺）")
    print("─" * 78)
    print(FAULT_CODE)

    # ============================================================
    # Step 2: 每个节点用自己的能力尝试补 1 个断层
    # ============================================================
    print(f"\n【Step 2】9 节点各自用能力补断层")
    print("─" * 78)

    # 每个节点补一个断层（3 个断层，9 个节点轮流）
    completions = {fault: [] for fault in FAULTS}

    for node_name, info in NODES.items():
        # 轮流分配断层
        fault_idx = list(NODES.keys()).index(node_name) % 3
        target_fault = FAULTS[fault_idx]

        # 节点用变异属性 + 能力生成补全
        mutation_name, mutation_desc = info["mutation"]
        ability = info["ability"]
        style = info["fill_style"]

        # 生成的补全内容（基于节点能力 + 变异扰动）
        # 变异影响：token_id 撞边界会让变量名漂移到非法字符
        #          text 污染会加培养液前缀
        #          embedding 正交会让逻辑完全偏离
        fill_content = generate_fill(
            node_name, info, mutation_name, mutation_desc, ability, rng
        )

        completions[target_fault].append({
            "node": node_name,
            "mutation": f"{mutation_name}({mutation_desc})",
            "style": style,
            "fill": fill_content,
        })

        print(f"\n  [{node_name}] → 补 {target_fault}")
        print(f"    变异: {mutation_name}({mutation_desc})")
        print(f"    风格: {style}")
        print(f"    能力: {ability[:50]}...")
        print(f"    补全:")
        for line in fill_content.split("\n"):
            print(f"      {line}")

    # ============================================================
    # Step 3: 中央共振池仲裁——每个断层选一个最终补全
    # ============================================================
    print(f"\n【Step 3】中央共振池仲裁——每断层选最终补全")
    print("─" * 78)

    final_code = FAULT_CODE
    for fault in FAULTS:
        candidates = completions[fault]
        # 中央共振池用 logprob 暴涨来选——概率最高的胜出
        # 简化：按候选数 + 变异能量排序
        winner = max(candidates, key=lambda c: len(c["fill"]) + hash(c["node"]) % 10)

        print(f"\n  {fault} 候选 {len(candidates)} 个:")
        for c in candidates:
            mark = "★" if c is winner else " "
            print(f"    {mark} [{c['node']}] 变异={c['mutation']}")
        print(f"  → 胜出: [{winner['node']}]")

        # 替换断层
        final_code = final_code.replace(fault, winner["fill"])

    # ============================================================
    # Step 4: 最终补全代码展示
    # ============================================================
    print(f"\n【Step 4】骨架补全的最终代码")
    print("─" * 78)
    print(final_code)

    # ============================================================
    # Step 5: 代码"质量"评估（用骨架自己的标准）
    # ============================================================
    print(f"\n【Step 5】骨架标准下的'质量'评估")
    print("─" * 78)

    lines = final_code.strip().split("\n")
    chars = len(final_code)
    # 骨架的 5 维质量（乱来的，和工厂一样）
    diversity = len(set(final_code.split())) / max(1, len(final_code.split()))
    coherence = 1.0 - abs(0.7 - (chars / 500))  # 500 字符最连贯
    informativeness = min(1.0, chars / 300)
    novelty = hash(final_code) % 100 / 100
    utility = 0.5  # 反正是乱码，固定 0.5
    avg = (diversity + coherence + informativeness + novelty + utility) / 5

    grade_names = ["D","C","B","A","S","SS","SSS"]
    grade_idx = min(6, int(avg * 7))
    print(f"  多样性:     {diversity:.4f}")
    print(f"  连贯性:     {coherence:.4f}")
    print(f"  信息量:     {informativeness:.4f}")
    print(f"  新颖性:     {novelty:.4f}")
    print(f"  实用性:     {utility:.4f}")
    print(f"  ────────────────")
    print(f"  平均质量:   {avg:.4f}")
    print(f"  等级:       {grade_names[grade_idx]}")
    print(f"  字符数:     {chars}")
    print(f"  行数:       {len(lines)}")

    # ============================================================
    # Step 6: 人工评审提示
    # ============================================================
    print(f"\n【Step 6】人工评审")
    print("─" * 78)
    print(f"  看得懂吗？")
    print(f"  跑得起来吗？")
    print(f"  逻辑通顺吗？")
    print(f"  → 答案大概率是：😂😂😂 全否")
    print(f"  → 但骨架觉得它质量={avg:.4f} 等级={grade_names[grade_idx]}")
    print(f"  → 因为骨架的质量评分器只看数值，不看语法")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【最终总结】")
    print("=" * 78)
    print(f"  断层代码: 3 处空缺")
    print(f"  补全节点: 9 个（每断层 3 个候选）")
    print(f"  仲裁方式: 中央共振池（logprob 暴涨选概率最高）")
    print(f"  最终代码: {chars} 字符 / {len(lines)} 行")
    print(f"  骨架自评: 质量={avg:.4f} 等级={grade_names[grade_idx]}")
    print(f"  人工评审: 😂 看不懂但骨架觉得很棒")
    print("=" * 78)


def generate_fill(node_name, info, mutation_name, mutation_desc, ability, rng):
    """根据节点能力 + 变异属性生成补全内容"""
    base = ability.replace("\\n", "\n")

    # 变异影响补全内容
    if mutation_name == "token_id":
        # token_id 撞边界 → 变量名漂移到数字
        base = base.replace("data", "0").replace("result", "1").replace("x", "2")
    elif mutation_name == "text":
        # text 被培养液前缀污染 → 加前缀
        prefixes = ["[emer]", "[styl]", "[impr]", "[crea]", "[sere]"]
        base = "\n".join(f"    {rng.choice(prefixes)}{line}" for line in base.split("\n"))
    elif mutation_name == "embedding":
        # embedding 正交化 → 逻辑完全偏离
        weird = ["# 语义已偏离", "pass  # 正交", "x = ~data", "result = -x"]
        base = "\n".join([base] + [f"    {rng.choice(weird)}" for _ in range(2)])
    elif mutation_name == "logprob":
        # logprob 暴涨 → 高概率但无意义
        base = "    result = data  # 概率 36.17%\n" + base
    elif mutation_name == "rank":
        # rank +1 → 循环多一次
        base = base.replace("100", "101").replace("range(", "range(1, ")
    elif mutation_name == "position":
        # position 偏移 → 缩进错位
        lines = base.split("\n")
        base = "\n".join(("    " + l) if i > 0 else l for i, l in enumerate(lines))
    elif mutation_name == "entropy_bits":
        # 信息量增加 → 加注释
        base += "\n    # 熵=7.47 bits 信息量+1.79"

    return base


if __name__ == "__main__":
    main()
