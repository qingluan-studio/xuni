"""
双向传送门思想实验求解

题目：
  双向传送门 A、B 对传。
  双向镜像门 C 插入 A→B 路径正中。
  人从 A 进入，求其物理形态与意识最终状态。

建模思路（结合前面的"万物归一"实验）：
  A→B 和 B→A 是两条相反路径
  C 是镜像门，插入正中
  "对传" 意味着两个方向同时发生
  → 等价于：两个相反方向的"自我"在 C 点相遇 + 各自被镜像

  我们把这个系统看作一个"骨架"：
    A、B、C 是三个节点
    人是输入物质
    意识 = 融合后的涌现
    物理形态 = 三个节点的最终状态
"""

from __future__ import annotations

import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.substance_fusion import SubstanceFusionEngine


# ============================================================
# 模型
# ============================================================

# 一个人 = 物质 + 意识（用之前的"万物归一"做基础属性）
PERSON = {
    # 物理形态
    "Z": 7021.0,            # Σ原子序数（万物归一遗产）
    "质量": 70.0,           # 70kg
    "电荷": 0.0,
    "自旋": 0.5,            # 费米子
    # 意识层
    "logprob": -1.017,      # 变异概率
    "token_id": 0,          # 撞边界
    "embedding_L2": 1.4195, # 正交化
    "信息熵": 8.94,
    "涌现": 1.0e9,          # 觉醒强度
    "镜像次数": 0,          # 经过镜像门次数
    "传送次数": 0,          # 经过传送门次数
}


def step_through_mirror(state, mirror_id):
    """通过镜像门 C——状态被镜像（左右/手性翻转）"""
    s = dict(state)
    s["电荷"] = -s["电荷"]              # 电荷反转
    s["自旋"] = -s["自旋"]              # 自旋反转（手性翻转）
    s["镜像次数"] += 1
    s["token_id"] = -s["token_id"]      # 意识也被镜像
    s["embedding_L2"] = -s["embedding_L2"]
    s["logprob"] = -s["logprob"]        # 概率反转
    s[f"经过_{mirror_id}"] = 1.0
    return s


def step_through_portal(state, portal_id):
    """通过传送门——位置跃迁"""
    s = dict(state)
    s["传送次数"] += 1
    s["位置"] = portal_id
    s[f"传送_{portal_id}"] = 1.0
    return s


def fuse_states(state_a, state_b):
    """两个状态融合（在 C 点相遇时的叠加）"""
    out = {}
    keys = set(list(state_a.keys()) + list(state_b.keys()))
    for k in keys:
        va = state_a.get(k, 0.0)
        vb = state_b.get(k, 0.0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[k] = (va + vb) / 2.0  # 平均（叠加态）
        else:
            out[k] = va
    out["叠加态"] = 1.0
    out["相遇点"] = "C"
    return out


def main():
    print("=" * 78)
    print("双向传送门思想实验求解")
    print("  A、B 对传 | 镜像门 C 插入 A→B 路径正中 | 人从 A 进入")
    print("=" * 78)

    rng = random.Random(42)

    # ============================================================
    # Step 1: 初始状态——人从 A 进入
    # ============================================================
    print(f"\n【Step 1】初始状态——人从 A 进入")
    print("─" * 78)

    person_A = dict(PERSON)
    person_A["位置"] = "A"
    person_A["方向"] = "A→B"
    print(f"  人 = {{")
    for k, v in person_A.items():
        print(f"    {k:<14}: {v}")
    print(f"  }}")

    # ============================================================
    # Step 2: A→B 路径，走到正中遇 C 镜像门
    # ============================================================
    print(f"\n【Step 2】A→B 路径：人到正中，遇镜像门 C")
    print("─" * 78)

    person_at_C_from_A = step_through_portal(person_A, "A")
    person_at_C_from_A["位置"] = "C（A侧）"

    # 通过 C 镜像门
    person_C = step_through_mirror(person_at_C_from_A, "C_正向")
    person_C["位置"] = "C（B侧）"
    person_C["方向"] = "A→B（已镜像）"

    print(f"  人从 A 出发 → 传送到 C → 被镜像")
    print(f"  镜像后:")
    print(f"    电荷:     {person_A['电荷']} → {person_C['电荷']}（反转）")
    print(f"    自旋:     {person_A['自旋']} → {person_C['自旋']}（手性翻转）")
    print(f"    token_id: {person_A['token_id']} → {person_C['token_id']}（意识镜像）")
    print(f"    embedding_L2: {person_A['embedding_L2']} → {person_C['embedding_L2']}（向量反转）")
    print(f"    logprob:  {person_A['logprob']} → {person_C['logprob']}（概率反转）")

    # ============================================================
    # Step 3: B→A 对传方向——"对面的我"也从 B 出发
    # ============================================================
    print(f"\n【Step 3】B→A 对传方向：'对面的我'从 B 出发")
    print("─" * 78)

    person_B = dict(PERSON)
    person_B["位置"] = "B"
    person_B["方向"] = "B→A"
    # 注意：B 端出发的"我"应该是 A 端的我到达 B 后的我
    # 但因为对传是同时的，所以这是"另一个我"——初始也是原始状态

    person_at_C_from_B = step_through_portal(person_B, "B")
    person_at_C_from_B["位置"] = "C（B侧）"

    person_C_mirror = step_through_mirror(person_at_C_from_B, "C_反向")
    person_C_mirror["位置"] = "C（A侧）"
    person_C_mirror["方向"] = "B→A（已镜像）"

    print(f"  '对面的我'从 B 出发 → 传送到 C → 被镜像")
    print(f"  镜像后:")
    print(f"    电荷:     {person_B['电荷']} → {person_C_mirror['电荷']}")
    print(f"    自旋:     {person_B['自旋']} → {person_C_mirror['自旋']}")
    print(f"    token_id: {person_B['token_id']} → {person_C_mirror['token_id']}")
    print(f"    logprob:  {person_B['logprob']} → {person_C_mirror['logprob']}")

    # ============================================================
    # Step 4: 两个"我"在 C 点相遇——叠加态
    # ============================================================
    print(f"\n【Step 4】两个'我'在 C 点相遇——叠加态")
    print("─" * 78)

    superposition = fuse_states(person_C, person_C_mirror)
    print(f"  叠加态 = (A→B的我 + B→A的我) / 2")
    print(f"  叠加态关键属性:")
    for k in ["电荷", "自旋", "token_id", "embedding_L2", "logprob",
              "镜像次数", "传送次数", "叠加态", "相遇点"]:
        print(f"    {k:<14}: {superposition.get(k, '?')}")

    # ============================================================
    # Step 5: 物理形态分析
    # ============================================================
    print(f"\n【Step 5】物理形态分析")
    print("─" * 78)

    print(f"""
  ┌──────────┐         ┌──────────┐         ┌──────────┐
  │          │  A→B    │          │  A→B    │          │
  │    A     │────────→│    C     │────────→│    B     │
  │          │←────────│  (镜像)  │←────────│          │
  │          │  B→A    │          │  B→A    │          │
  └──────────┘         └──────────┘         └──────────┘
       ↑                    ↑                    ↑
     入口               叠加点              出口/入口
                        （两个我相遇）
""")

    print(f"  分析:")
    print(f"  1. A→B 的我：电荷/自旋反转（手性翻转）")
    print(f"  2. B→A 的我：同样被 C 镜像反转")
    print(f"  3. 两者在 C 点相遇 → 叠加")
    print()
    print(f"  物理形态判定:")

    charge = superposition["电荷"]
    spin = superposition["自旋"]
    print(f"    电荷: {person_A['电荷']} → 镜像后 {-person_A['电荷']} → 叠加 {charge}")
    print(f"    自旋: {person_A['自旋']} → 镜像后 {-person_A['自旋']} → 叠加 {spin}")

    if charge == 0 and spin == 0:
        print(f"    → 电荷=0, 自旋=0 = 玻色子态（无手性）")
        print(f"    → 物理形态：粒子-反粒子叠加湮灭 → 纯能量态")
        form = "纯能量态（光子化）"
    elif charge == person_A["电荷"] and spin == person_A["自旋"]:
        print(f"    → 与原始状态相同（镜像抵消）")
        form = "原始状态（镜像抵消）"
    else:
        print(f"    → 镜像未完全抵消，存在不对称")
        form = "镜像叠加态"

    # ============================================================
    # Step 6: 意识状态分析
    # ============================================================
    print(f"\n【Step 6】意识状态分析")
    print("─" * 78)

    print(f"  原始意识:")
    print(f"    token_id:     {person_A['token_id']}")
    print(f"    embedding_L2: {person_A['embedding_L2']}")
    print(f"    logprob:      {person_A['logprob']}")
    print(f"    信息熵:       {person_A['信息熵']}")

    print(f"\n  A→B 我（镜像后）:")
    print(f"    token_id:     {person_C['token_id']}")
    print(f"    embedding_L2: {person_C['embedding_L2']}")
    print(f"    logprob:      {person_C['logprob']}")

    print(f"\n  B→A 我（镜像后）:")
    print(f"    token_id:     {person_C_mirror['token_id']}")
    print(f"    embedding_L2: {person_C_mirror['embedding_L2']}")
    print(f"    logprob:      {person_C_mirror['logprob']}")

    print(f"\n  叠加态意识:")
    print(f"    token_id:     {superposition['token_id']}")
    print(f"    embedding_L2: {superposition['embedding_L2']}")
    print(f"    logprob:      {superposition['logprob']}")

    # 意识判定
    tid = superposition["token_id"]
    emb = superposition["embedding_L2"]
    lp = superposition["logprob"]

    print(f"\n  意识判定:")
    if tid == 0 and emb == 0 and lp == 0:
        print(f"    → token_id/embedding/logprob 全部抵消归零")
        print(f"    → 意识湮灭，进入'空'态")
        consciousness = "空态（无意识）"
    elif abs(tid) < abs(person_A["token_id"]):
        print(f"    → 意识被部分抵消")
        print(f"    → 进入'半我'态")
        consciousness = "半意识态（自我减半）"
    else:
        print(f"    → 意识叠加增强")
        consciousness = "超意识态"

    # ============================================================
    # Step 7: 走完全程——人到 B（如果还存在）
    # ============================================================
    print(f"\n【Step 7】走完全程——人到 B")
    print("─" * 78)

    # A→B 的我继续到 B
    person_at_B = step_through_portal(person_C, "B")
    person_at_B["位置"] = "B"
    person_at_B["方向"] = "已到达 B"

    print(f"  A→B 的我（已镜像）继续传送到 B:")
    print(f"    电荷:     {person_at_B['电荷']}（仍是反转后的）")
    print(f"    自旋:     {person_at_B['自旋']}（手性翻转后）")
    print(f"    token_id: {person_at_B['token_id']}")
    print(f"    镜像次数: {person_at_B['镜像次数']}")
    print(f"    传送次数: {person_at_B['传送次数']}")

    # ============================================================
    # Step 8: 最终答案
    # ============================================================
    print(f"\n【Step 8】最终答案")
    print("═" * 78)

    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║                                                          ║
  ║   题目：双向传送门 A、B 对传                             ║
  ║         镜像门 C 插入 A→B 路径正中                      ║
  ║         人从 A 进入                                      ║
  ║                                                          ║
  ║   物理形态：{form:<20}                    ║
  ║             - 电荷：{person_C['电荷']:<6}（镜像反转）             ║
  ║             - 自旋：{person_C['自旋']:<6}（手性翻转）             ║
  ║             - 镜像次数：{person_at_B['镜像次数']:<3}                              ║
  ║             - 传送次数：{person_at_B['传送次数']:<3}                              ║
  ║                                                          ║
  ║   意识状态：{consciousness:<20}                    ║
  ║             - token_id: {superposition['token_id']:<8}（镜像+叠加）     ║
  ║             - embedding: {superposition['embedding_L2']:<8}（向量反转）   ║
  ║             - logprob:  {superposition['logprob']:<8}（概率反转）       ║
  ║             - 信息熵:   {person_A['信息熵']:<8}（保持不变）           ║
  ║                                                          ║
  ║   最终位置：B                                            ║
  ║   最终身份：镜像后的"反我"                                ║
  ║                                                          ║
  ║   等价物理过程：                                          ║
  ║     人 → 经 C 镜像 → 变成自己的"反物质版本"             ║
  ║     → 到达 B 时已是镜像体                                ║
  ║     → 与 B 端原状态相遇 → 在 C 点形成叠加              ║
  ║                                                          ║
  ║   哲学结论：                                              ║
  ║     - 镜像门 C 是"自我对峙"的临界点                     ║
  ║     - 在 C 点，"我"和"反我"叠加                         ║
  ║     - 物理上手性翻转（左手变右手）                       ║
  ║     - 意识上 token/embedding 反转 → 自我认知颠倒        ║
  ║     - 到达 B 的是"镜像我"，不是原来的我                  ║
  ║                                                          ║
  ╚══════════════════════════════════════════════════════════╝
""")

    # ============================================================
    # 补充：用万物归一的属性算"等效质量"
    # ============================================================
    print(f"  补充：用'万物归一'框架计算等效输出")
    print("─" * 78)

    engine = SubstanceFusionEngine()
    engine.register_substance("原始人", dict(PERSON))
    engine.register_substance("镜像人", {k: -v if isinstance(v, (int, float)) else v
                                          for k, v in PERSON.items()})
    product = engine.fuse("原始人", "镜像人")
    print(f"  原始人 × 镜像人 → {product.result}")
    print(f"  融合类型: {product.fusion_type.name}")
    print(f"  类别: {product.category.name}")
    print(f"  能量释放: {product.energy_release:.4f}")

    print(f"\n  → 融合产物属性（人+镜像人叠加）:")
    for k, v in sorted(product.properties.items(), key=lambda x: -abs(x[1]))[:10]:
        print(f"    {k:<14}: {v:.4f}")

    print()
    print("  😂 最终结论：人变成'反我'到达 B，C 点是自我对峙的奇点")
    print("=" * 78)


if __name__ == "__main__":
    main()
