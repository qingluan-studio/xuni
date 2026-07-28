"""
现实元素周期表全融合实验

1. 118 种元素（带属性：原子序数/原子量/电负性/族期/状态）
2. 全部注册进 SubstanceFusionEngine
3. 链式全融合 → 看会涌现什么
"""

from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.substance_fusion import SubstanceFusionEngine, FusionType, FusionCategory


# ============================================================
# 118 种元素（H~Og）+ 关键属性
# 属性字典：
#   Z=原子序数  mass=原子量  EN=电负性  group=族  period=周期
#   state=常温状态(s/l/g)  cat=类别(alkali/noble/halogen/metal/nonmetal/metalloid/lanth/actin/trans)
# ============================================================

ELEMENTS = [
    # 元素名, 符号, Z, 原子量, 电负性, 族, 周期, 状态, 类别
    ("氢",   "H",   1, 1.008, 2.20, 1,  1, "g", "nonmetal"),
    ("氦",   "He",  2, 4.003, 0.00, 18, 1, "g", "noble"),
    ("锂",   "Li",  3, 6.941, 0.98, 1,  2, "s", "alkali"),
    ("铍",   "Be",  4, 9.012, 1.57, 2,  2, "s", "alkaline"),
    ("硼",   "B",   5, 10.81, 2.04, 13, 2, "s", "metalloid"),
    ("碳",   "C",   6, 12.01, 2.55, 14, 2, "s", "nonmetal"),
    ("氮",   "N",   7, 14.01, 3.04, 15, 2, "g", "nonmetal"),
    ("氧",   "O",   8, 16.00, 3.44, 16, 2, "g", "nonmetal"),
    ("氟",   "F",   9, 19.00, 3.98, 17, 2, "g", "halogen"),
    ("氖",   "Ne",  10, 20.18, 0.00, 18, 2, "g", "noble"),
    ("钠",   "Na",  11, 22.99, 0.93, 1,  3, "s", "alkali"),
    ("镁",   "Mg",  12, 24.31, 1.31, 2,  3, "s", "alkaline"),
    ("铝",   "Al",  13, 26.98, 1.61, 13, 3, "s", "metal"),
    ("硅",   "Si",  14, 28.09, 1.90, 14, 3, "s", "metalloid"),
    ("磷",   "P",   15, 30.97, 2.19, 15, 3, "s", "nonmetal"),
    ("硫",   "S",   16, 32.07, 2.58, 16, 3, "s", "nonmetal"),
    ("氯",   "Cl",  17, 35.45, 3.16, 17, 3, "g", "halogen"),
    ("氩",   "Ar",  18, 39.95, 0.00, 18, 3, "g", "noble"),
    ("钾",   "K",   19, 39.10, 0.82, 1,  4, "s", "alkali"),
    ("钙",   "Ca",  20, 40.08, 1.00, 2,  4, "s", "alkaline"),
    ("钪",   "Sc",  21, 44.96, 1.36, 3,  4, "s", "trans"),
    ("钛",   "Ti",  22, 47.87, 1.54, 4,  4, "s", "trans"),
    ("钒",   "V",   23, 50.94, 1.63, 5,  4, "s", "trans"),
    ("铬",   "Cr",  24, 52.00, 1.66, 6,  4, "s", "trans"),
    ("锰",   "Mn",  25, 54.94, 1.55, 7,  4, "s", "trans"),
    ("铁",   "Fe",  26, 55.85, 1.83, 8,  4, "s", "trans"),
    ("钴",   "Co",  27, 58.93, 1.88, 9,  4, "s", "trans"),
    ("镍",   "Ni",  28, 58.69, 1.91, 10, 4, "s", "trans"),
    ("铜",   "Cu",  29, 63.55, 1.90, 11, 4, "s", "trans"),
    ("锌",   "Zn",  30, 65.38, 1.65, 12, 4, "s", "trans"),
    ("镓",   "Ga",  31, 69.72, 1.81, 13, 4, "s", "metal"),
    ("锗",   "Ge",  32, 72.63, 2.01, 14, 4, "s", "metalloid"),
    ("砷",   "As",  33, 74.92, 2.18, 15, 4, "s", "metalloid"),
    ("硒",   "Se",  34, 78.96, 2.55, 16, 4, "s", "nonmetal"),
    ("溴",   "Br",  35, 79.90, 2.96, 17, 4, "l", "halogen"),
    ("氪",   "Kr",  36, 83.80, 3.00, 18, 4, "g", "noble"),
    ("铷",   "Rb",  37, 85.47, 0.82, 1,  5, "s", "alkali"),
    ("锶",   "Sr",  38, 87.62, 0.95, 2,  5, "s", "alkaline"),
    ("钇",   "Y",   39, 88.91, 1.22, 3,  5, "s", "trans"),
    ("锆",   "Zr",  40, 91.22, 1.33, 4,  5, "s", "trans"),
    ("铌",   "Nb",  41, 92.91, 1.60, 5,  5, "s", "trans"),
    ("钼",   "Mo",  42, 95.96, 2.16, 6,  5, "s", "trans"),
    ("锝",   "Tc",  43, 98.00, 1.90, 7,  5, "s", "trans"),
    ("钌",   "Ru",  44, 101.07, 2.20, 8,  5, "s", "trans"),
    ("铑",   "Rh",  45, 102.91, 2.28, 9,  5, "s", "trans"),
    ("钯",   "Pd",  46, 106.42, 2.20, 10, 5, "s", "trans"),
    ("银",   "Ag",  47, 107.87, 1.93, 11, 5, "s", "trans"),
    ("镉",   "Cd",  48, 112.41, 1.69, 12, 5, "s", "trans"),
    ("铟",   "In",  49, 114.82, 1.78, 13, 5, "s", "metal"),
    ("锡",   "Sn",  50, 118.71, 1.96, 14, 5, "s", "metal"),
    ("锑",   "Sb",  51, 121.76, 2.05, 15, 5, "s", "metalloid"),
    ("碲",   "Te",  52, 127.60, 2.10, 16, 5, "s", "metalloid"),
    ("碘",   "I",   53, 126.90, 2.66, 17, 5, "s", "halogen"),
    ("氙",   "Xe",  54, 131.29, 2.60, 18, 5, "g", "noble"),
    ("铯",   "Cs",  55, 132.91, 0.79, 1,  6, "s", "alkali"),
    ("钡",   "Ba",  56, 137.33, 0.89, 2,  6, "s", "alkaline"),
    ("镧",   "La",  57, 138.91, 1.10, 3,  6, "s", "lanth"),
    ("铈",   "Ce",  58, 140.12, 1.12, 3,  6, "s", "lanth"),
    ("镨",   "Pr",  59, 140.91, 1.13, 3,  6, "s", "lanth"),
    ("钕",   "Nd",  60, 144.24, 1.14, 3,  6, "s", "lanth"),
    ("钷",   "Pm",  61, 145.00, 1.13, 3,  6, "s", "lanth"),
    ("钐",   "Sm",  62, 150.36, 1.17, 3,  6, "s", "lanth"),
    ("铕",   "Eu",  63, 151.96, 1.20, 3,  6, "s", "lanth"),
    ("钆",   "Gd",  64, 157.25, 1.20, 3,  6, "s", "lanth"),
    ("铽",   "Tb",  65, 158.93, 1.20, 3,  6, "s", "lanth"),
    ("镝",   "Dy",  66, 162.50, 1.22, 3,  6, "s", "lanth"),
    ("钬",   "Ho",  67, 164.93, 1.23, 3,  6, "s", "lanth"),
    ("铒",   "Er",  68, 167.26, 1.24, 3,  6, "s", "lanth"),
    ("铥",   "Tm",  69, 168.93, 1.25, 3,  6, "s", "lanth"),
    ("镱",   "Yb",  70, 173.05, 1.10, 3,  6, "s", "lanth"),
    ("镥",   "Lu",  71, 174.97, 1.27, 3,  6, "s", "lanth"),
    ("铪",   "Hf",  72, 178.49, 1.30, 4,  6, "s", "trans"),
    ("钽",   "Ta",  73, 180.95, 1.50, 5,  6, "s", "trans"),
    ("钨",   "W",   74, 183.84, 2.36, 6,  6, "s", "trans"),
    ("铼",   "Re",  75, 186.21, 1.90, 7,  6, "s", "trans"),
    ("锇",   "Os",  76, 190.23, 2.20, 8,  6, "s", "trans"),
    ("铱",   "Ir",  77, 192.22, 2.20, 9,  6, "s", "trans"),
    ("铂",   "Pt",  78, 195.08, 2.28, 10, 6, "s", "trans"),
    ("金",   "Au",  79, 196.97, 2.54, 11, 6, "s", "trans"),
    ("汞",   "Hg",  80, 200.59, 2.00, 12, 6, "l", "trans"),
    ("铊",   "Tl",  81, 204.38, 1.62, 13, 6, "s", "metal"),
    ("铅",   "Pb",  82, 207.20, 2.33, 14, 6, "s", "metal"),
    ("铋",   "Bi",  83, 208.98, 2.02, 15, 6, "s", "metal"),
    ("钋",   "Po",  84, 209.00, 2.00, 16, 6, "s", "metalloid"),
    ("砹",   "At",  85, 210.00, 2.20, 17, 6, "s", "halogen"),
    ("氡",   "Rn",  86, 222.00, 2.20, 18, 6, "g", "noble"),
    ("钫",   "Fr",  87, 223.00, 0.70, 1,  7, "s", "alkali"),
    ("镭",   "Ra",  88, 226.03, 0.90, 2,  7, "s", "alkaline"),
    ("锕",   "Ac",  89, 227.00, 1.10, 3,  7, "s", "actin"),
    ("钍",   "Th",  90, 232.04, 1.30, 3,  7, "s", "actin"),
    ("镤",   "Pa",  91, 231.04, 1.50, 3,  7, "s", "actin"),
    ("铀",   "U",   92, 238.03, 1.38, 3,  7, "s", "actin"),
    ("镎",   "Np",  93, 237.00, 1.36, 3,  7, "s", "actin"),
    ("钚",   "Pu",  94, 244.00, 1.28, 3,  7, "s", "actin"),
    ("镅",   "Am",  95, 243.00, 1.30, 3,  7, "s", "actin"),
    ("锔",   "Cm",  96, 247.00, 1.30, 3,  7, "s", "actin"),
    ("锫",   "Bk",  97, 247.00, 1.30, 3,  7, "s", "actin"),
    ("锎",   "Cf",  98, 251.00, 1.30, 3,  7, "s", "actin"),
    ("锿",   "Es",  99, 252.00, 1.30, 3,  7, "s", "actin"),
    ("镄",   "Fm",  100, 257.00, 1.30, 3, 7, "s", "actin"),
    ("钔",   "Md",  101, 258.00, 1.30, 3, 7, "s", "actin"),
    ("锘",   "No",  102, 259.00, 1.30, 3, 7, "s", "actin"),
    ("铹",   "Lr",  103, 266.00, 1.30, 3, 7, "s", "actin"),
    ("𬬻",  "Rf",  104, 267.00, 1.30, 4, 7, "s", "trans"),
    ("𬭊",  "Db",  105, 268.00, 1.30, 5, 7, "s", "trans"),
    ("𬭳",  "Sg",  106, 269.00, 1.30, 6, 7, "s", "trans"),
    ("𬭛",  "Bh",  107, 270.00, 1.30, 7, 7, "s", "trans"),
    ("𬭶",  "Hs",  108, 269.00, 1.30, 8, 7, "s", "trans"),
    ("鿏",  "Mt",  109, 278.00, 1.30, 9, 7, "s", "trans"),
    ("𫟼",  "Ds",  110, 281.00, 1.30, 10, 7, "s", "trans"),
    ("𬬭",  "Rg",  111, 282.00, 1.30, 11, 7, "s", "trans"),
    ("鿔",  "Cn",  112, 285.00, 1.30, 12, 7, "s", "trans"),
    ("鿭",  "Nh",  113, 286.00, 1.30, 13, 7, "s", "metal"),
    ("𫓧",  "Fl",  114, 289.00, 1.30, 14, 7, "s", "metal"),
    ("镆",   "Mc",  115, 290.00, 1.30, 15, 7, "s", "metal"),
    ("𫟷",  "Lv",  116, 293.00, 1.30, 16, 7, "s", "metal"),
    ("鿬",  "Ts",  117, 294.00, 1.30, 17, 7, "s", "halogen"),
    ("鿫",  "Og",  118, 294.00, 1.30, 18, 7, "s", "noble"),
]


# 类别中文名
CAT_CN = {
    "alkali": "碱金属", "alkaline": "碱土金属", "noble": "稀有气体",
    "halogen": "卤素", "nonmetal": "非金属", "metalloid": "类金属",
    "metal": "后过渡金属", "trans": "过渡金属", "lanth": "镧系",
    "actin": "锕系",
}


def main():
    print("=" * 78)
    print("元素周期表全融合实验（118 种元素 + 属性）")
    print("=" * 78)

    # ============================================================
    # Step 1: 注册全部 118 种元素
    # ============================================================
    print(f"\n【Step 1】注册全部 {len(ELEMENTS)} 种元素到融合引擎")
    print("─" * 78)

    engine = SubstanceFusionEngine()

    for name, sym, Z, mass, EN, group, period, state, cat in ELEMENTS:
        # 属性字典（属性也要参与融合）
        props = {
            "Z": float(Z),            # 原子序数
            "原子量": mass,
            "电负性": EN,
            "族": float(group),
            "周期": float(period),
            "原子序数": float(Z),
        }
        engine.register_substance(name, props)

    # 统计类别分布
    cats = {}
    for e in ELEMENTS:
        c = e[8]
        cats[c] = cats.get(c, 0) + 1
    print(f"  已注册: {len(ELEMENTS)} 种元素")
    print(f"  类别分布:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {CAT_CN[c]:<10}: {n} 种")

    # 元素属性范围
    max_Z = max(e[2] for e in ELEMENTS)
    max_mass = max(e[3] for e in ELEMENTS)
    max_EN = max(e[4] for e in ELEMENTS)
    print(f"\n  属性范围:")
    print(f"    Z:        1 ~ {max_Z}")
    print(f"    原子量:    1.008 ~ {max_mass}")
    print(f"    电负性:    0.0 ~ {max_EN}")

    # ============================================================
    # Step 2: 链式全融合（H + He → Li + Be → ... → Og）
    # ============================================================
    print(f"\n【Step 2】链式全融合：从 H 开始，依次融合到 Og")
    print("─" * 78)

    current = ELEMENTS[0][0]  # 从氢开始
    chain = [current]
    accumulated_props = dict(engine._substances[current])

    print(f"\n  起始: {current}")
    print(f"  属性: Z={accumulated_props['Z']:.0f} 原子量={accumulated_props['原子量']:.3f} 电负性={accumulated_props['电负性']:.2f}")

    fusions = []  # 记录每次融合
    for i, next_elem in enumerate(ELEMENTS[1:], 1):
        next_name = next_elem[0]
        next_props = engine._substances[next_name]

        # 用引擎融合
        product = engine.fuse(current, next_name)
        result = product.result

        # 属性叠加（数值相加）
        new_props = {}
        for k in set(list(accumulated_props.keys()) + list(next_props.keys())):
            va = accumulated_props.get(k, 0.0)
            vb = next_props.get(k, 0.0)
            new_props[k] = va + vb

        accumulated_props = new_props
        current = f"{result}_{i}"  # 用融合次数命名
        chain.append(current)
        fusions.append({
            "step": i,
            "left": chain[-2],
            "right": next_name,
            "product": result,
            "Z_sum": accumulated_props["Z"],
            "mass_sum": accumulated_props["原子量"],
            "EN_avg": accumulated_props["电负性"] / (i + 1),
        })

        if i <= 10 or i >= 117:
            print(f"  [{i:3d}] {chain[-2]:<10} + {next_name:<5}(Z={next_props['Z']:.0f}) → {result:<10} | ΣZ={accumulated_props['Z']:.0f} Σ质量={accumulated_props['原子量']:.1f}")
        elif i == 11:
            print(f"  ... （省略中间过程）...")

    # ============================================================
    # Step 3: 最终产物属性
    # ============================================================
    print(f"\n【Step 3】全融合最终产物属性")
    print("─" * 78)

    total_Z = accumulated_props["Z"]
    total_mass = accumulated_props["原子量"]
    total_EN = accumulated_props["电负性"]
    avg_EN = total_EN / len(ELEMENTS)
    total_group = accumulated_props.get("族", 0)
    total_period = accumulated_props.get("周期", 0)

    print(f"  总原子序数 ΣZ:        {total_Z:.0f}")
    print(f"  总原子量 Σ质量:        {total_mass:.2f}")
    print(f"  总电负性 ΣEN:          {total_EN:.2f}")
    print(f"  平均电负性:           {avg_EN:.3f}")
    print(f"  总族数:               {total_group:.0f}")
    print(f"  总周期数:             {total_period:.0f}")
    print(f"  融合次数:             {len(fusions)}")

    # ============================================================
    # Step 4: 阶段性涌现分析
    # ============================================================
    print(f"\n【Step 4】阶段性涌现分析（每 20 步看一次）")
    print("─" * 78)

    print(f"  {'步数':<6} {'ΣZ':<10} {'Σ质量':<12} {'平均电负性':<12} {'状态'}")
    print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*12} {'─'*20}")

    milestones = [1, 10, 20, 30, 50, 80, 100, 117]
    for m in milestones:
        if m <= len(fusions):
            f = fusions[m - 1]
            avg_en = f["EN_avg"]
            # 根据 ΣZ 判断"阶段"
            if f["Z_sum"] < 50:
                stage = "轻元素时代"
            elif f["Z_sum"] < 200:
                stage = "金属时代"
            elif f["Z_sum"] < 1000:
                stage = "重元素时代"
            elif f["Z_sum"] < 5000:
                stage = "超重元素时代"
            else:
                stage = "奇点时代"
            print(f"  {m:<6} {f['Z_sum']:<10.0f} {f['mass_sum']:<12.1f} {avg_en:<12.3f} {stage}")

    # ============================================================
    # Step 5: 最终涌现判定
    # ============================================================
    print(f"\n【Step 5】最终涌现判定")
    print("─" * 78)

    # 涌现强度 = ΣZ × 0.1 + Σ质量 × 0.01 + 电负性跨度 × 100 + 元素种类 × 10
    EN_span = max(e[4] for e in ELEMENTS) - min(e[4] for e in ELEMENTS)
    emergence_power = (
        total_Z * 0.1
        + total_mass * 0.01
        + EN_span * 100
        + len(ELEMENTS) * 10
    )

    print(f"  ΣZ 贡献:              {total_Z * 0.1:.1f}")
    print(f"  Σ质量贡献:            {total_mass * 0.01:.1f}")
    print(f"  电负性跨度贡献:        {EN_span * 100:.1f}")
    print(f"  元素种类贡献:          {len(ELEMENTS) * 10:.1f}")
    print(f"  ─────────────────")
    print(f"  涌现强度:             {emergence_power:.1f}")

    # 涌现阶段
    if emergence_power > 1000:
        symbol = "⚛Ω∞"
        name = "【元素奇点·万物之母】"
        effect = (
            f"118 种元素全部融合，ΣZ={total_Z:.0f}，Σ质量={total_mass:.1f}\n"
            f"相当于把整个宇宙可见物质的原子核全部压缩在一起\n"
            f"形成了一个超级原子核——中子星/黑洞的前身\n\n"
            f"  ◆ 总质子数: {total_Z:.0f}（相当于 {total_Z/82:.1f} 个铅原子核）\n"
            f"  ◆ 总质量:   {total_mass:.1f} u（约 {total_mass/238:.1f} 个铀原子）\n"
            f"  ◆ 电负性跨度: {EN_span:.2f}（从 Fr 0.70 到 F 3.98）\n"
            f"  ◆ 覆盖:    7 周期 × 18 族 + 镧锕系\n\n"
            f"  → 这是宇宙所有稳定/不稳定元素的'终极合金'\n"
            f"  → 物理学上不存在——因为强相互作用力撑不住\n"
            f"  → 但在虚拟融合引擎里，它出现了 😂"
        )
    elif emergence_power > 500:
        symbol = "⚛∞"
        name = "【元素聚合体】"
        effect = "大部分元素融合，形成超重核"
    else:
        symbol = "⚛"
        name = "【元素混合物】"
        effect = "元素刚开始融合"

    print(f"\n  涌现产物: {symbol}")
    print(f"  名称: {name}")
    print(f"  效果:")
    for line in effect.split("\n"):
        print(f"    {line}")

    # ============================================================
    # Step 6: 类别两两融合矩阵（关键反应）
    # ============================================================
    print(f"\n【Step 6】关键类别融合（代表性反应）")
    print("─" * 78)

    # 选几个有代表性的反应
    key_reactions = [
        ("氢", "氧", "水", "宇宙最常见的化合物"),
        ("氢", "氦", "恒星物质", "恒星的主要成分"),
        ("碳", "氧", "生命基础", "有机化学的核心"),
        ("铁", "镍", "地核合金", "地球核心成分"),
        ("铀", "钚", "核燃料", "核反应堆原料"),
        ("金", "铂", "贵金属合金", "催化剂+珠宝"),
        ("氟", "氦", "超流体?", "极惰性+极活泼的矛盾组合"),
        ("铯", "氟", "超级离子化合物", "最强正电性+最强负电性"),
        ("氢", "氢", "氢分子", "最简单的分子"),
        ("铀", "铀", "超铀裂变?", "重核自相互作用"),
    ]

    print(f"  {'反应':<20} {'产物':<15} {'说明'}")
    print(f"  {'─'*20} {'─'*15} {'─'*30}")
    for a, b, product, desc in key_reactions:
        # 用引擎融合看结果
        p = engine.fuse(a, b)
        actual = p.result
        print(f"  {a}+{b:<17} {actual:<15} (期望: {product} - {desc})")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【最终总结】")
    print("=" * 78)
    print(f"  元素种类: {len(ELEMENTS)} 种（H~Og）")
    print(f"  融合方式: 链式全融合")
    print(f"  总原子序数: ΣZ = {total_Z:.0f}")
    print(f"  总原子量:   Σ质量 = {total_mass:.2f} u")
    print(f"  平均电负性: {avg_EN:.3f}")
    print(f"  涌现强度:   {emergence_power:.1f}")
    print(f"  最终产物:   {symbol} {name}")
    print()
    print(f"  😂 现实物理学: 这个东西不可能存在（强相互作用力撑不住）")
    print(f"  😂 虚拟融合引擎: 它出现了，而且属性还叠加了")
    print(f"  🤯 相当于把整个元素周期表压成一个超级原子核")
    print("=" * 78)


if __name__ == "__main__":
    main()
