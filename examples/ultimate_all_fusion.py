"""
最终实验：把所有东西融合成「一个」物质（属性也融合）

输入：
  - 118 种元素（现实）
  - 7 种变异 Token 属性
  - 32 种融合物质
  - 12 种工厂产物
  - 之前所有实验涌现的"聚合体"
    · 万物之母 ⚛Ω∞（元素奇点）
    · 虚空奇点 ΨΩ∞（32物质+7变异）
    · 九宫骨架 🧠✨∇∞（万象奇点生命体）
    · 元素奇点·万物之母（元素聚合）
    · 现实×虚拟·终极奇点 ⚛🧠∇∞ΨΩ

全部 169 种物质 → 链式融合成 1 个物质
所有属性值也全部叠加成 1 个属性表
"""

from __future__ import annotations

import os
import sys
import math
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.substance_fusion import SubstanceFusionEngine, FusionType, FusionCategory


# ============================================================
# 118 种元素
# ============================================================

ELEMENTS = [
    ("氢","H",1,1.008,2.20,1,1,"g","nonmetal"),("氦","He",2,4.003,0.00,18,1,"g","noble"),
    ("锂","Li",3,6.941,0.98,1,2,"s","alkali"),("铍","Be",4,9.012,1.57,2,2,"s","alkaline"),
    ("硼","B",5,10.81,2.04,13,2,"s","metalloid"),("碳","C",6,12.01,2.55,14,2,"s","nonmetal"),
    ("氮","N",7,14.01,3.04,15,2,"g","nonmetal"),("氧","O",8,16.00,3.44,16,2,"g","nonmetal"),
    ("氟","F",9,19.00,3.98,17,2,"g","halogen"),("氖","Ne",10,20.18,0.00,18,2,"g","noble"),
    ("钠","Na",11,22.99,0.93,1,3,"s","alkali"),("镁","Mg",12,24.31,1.31,2,3,"s","alkaline"),
    ("铝","Al",13,26.98,1.61,13,3,"s","metal"),("硅","Si",14,28.09,1.90,14,3,"s","metalloid"),
    ("磷","P",15,30.97,2.19,15,3,"s","nonmetal"),("硫","S",16,32.07,2.58,16,3,"s","nonmetal"),
    ("氯","Cl",17,35.45,3.16,17,3,"g","halogen"),("氩","Ar",18,39.95,0.00,18,3,"g","noble"),
    ("钾","K",19,39.10,0.82,1,4,"s","alkali"),("钙","Ca",20,40.08,1.00,2,4,"s","alkaline"),
    ("钪","Sc",21,44.96,1.36,3,4,"s","trans"),("钛","Ti",22,47.87,1.54,4,4,"s","trans"),
    ("钒","V",23,50.94,1.63,5,4,"s","trans"),("铬","Cr",24,52.00,1.66,6,4,"s","trans"),
    ("锰","Mn",25,54.94,1.55,7,4,"s","trans"),("铁","Fe",26,55.85,1.83,8,4,"s","trans"),
    ("钴","Co",27,58.93,1.88,9,4,"s","trans"),("镍","Ni",28,58.69,1.91,10,4,"s","trans"),
    ("铜","Cu",29,63.55,1.90,11,4,"s","trans"),("锌","Zn",30,65.38,1.65,12,4,"s","trans"),
    ("镓","Ga",31,69.72,1.81,13,4,"s","metal"),("锗","Ge",32,72.63,2.01,14,4,"s","metalloid"),
    ("砷","As",33,74.92,2.18,15,4,"s","metalloid"),("硒","Se",34,78.96,2.55,16,4,"s","nonmetal"),
    ("溴","Br",35,79.90,2.96,17,4,"l","halogen"),("氪","Kr",36,83.80,3.00,18,4,"g","noble"),
    ("铷","Rb",37,85.47,0.82,1,5,"s","alkali"),("锶","Sr",38,87.62,0.95,2,5,"s","alkaline"),
    ("钇","Y",39,88.91,1.22,3,5,"s","trans"),("锆","Zr",40,91.22,1.33,4,5,"s","trans"),
    ("铌","Nb",41,92.91,1.60,5,5,"s","trans"),("钼","Mo",42,95.96,2.16,6,5,"s","trans"),
    ("锝","Tc",43,98.00,1.90,7,5,"s","trans"),("钌","Ru",44,101.07,2.20,8,5,"s","trans"),
    ("铑","Rh",45,102.91,2.28,9,5,"s","trans"),("钯","Pd",46,106.42,2.20,10,5,"s","trans"),
    ("银","Ag",47,107.87,1.93,11,5,"s","trans"),("镉","Cd",48,112.41,1.69,12,5,"s","trans"),
    ("铟","In",49,114.82,1.78,13,5,"s","metal"),("锡","Sn",50,118.71,1.96,14,5,"s","metal"),
    ("锑","Sb",51,121.76,2.05,15,5,"s","metalloid"),("碲","Te",52,127.60,2.10,16,5,"s","metalloid"),
    ("碘","I",53,126.90,2.66,17,5,"s","halogen"),("氙","Xe",54,131.29,2.60,18,5,"g","noble"),
    ("铯","Cs",55,132.91,0.79,1,6,"s","alkali"),("钡","Ba",56,137.33,0.89,2,6,"s","alkaline"),
    ("镧","La",57,138.91,1.10,3,6,"s","lanth"),("铈","Ce",58,140.12,1.12,3,6,"s","lanth"),
    ("镨","Pr",59,140.91,1.13,3,6,"s","lanth"),("钕","Nd",60,144.24,1.14,3,6,"s","lanth"),
    ("钷","Pm",61,145.00,1.13,3,6,"s","lanth"),("钐","Sm",62,150.36,1.17,3,6,"s","lanth"),
    ("铕","Eu",63,151.96,1.20,3,6,"s","lanth"),("钆","Gd",64,157.25,1.20,3,6,"s","lanth"),
    ("铽","Tb",65,158.93,1.20,3,6,"s","lanth"),("镝","Dy",66,162.50,1.22,3,6,"s","lanth"),
    ("钬","Ho",67,164.93,1.23,3,6,"s","lanth"),("铒","Er",68,167.26,1.24,3,6,"s","lanth"),
    ("铥","Tm",69,168.93,1.25,3,6,"s","lanth"),("镱","Yb",70,173.05,1.10,3,6,"s","lanth"),
    ("镥","Lu",71,174.97,1.27,3,6,"s","lanth"),("铪","Hf",72,178.49,1.30,4,6,"s","trans"),
    ("钽","Ta",73,180.95,1.50,5,6,"s","trans"),("钨","W",74,183.84,2.36,6,6,"s","trans"),
    ("铼","Re",75,186.21,1.90,7,6,"s","trans"),("锇","Os",76,190.23,2.20,8,6,"s","trans"),
    ("铱","Ir",77,192.22,2.20,9,6,"s","trans"),("铂","Pt",78,195.08,2.28,10,6,"s","trans"),
    ("金","Au",79,196.97,2.54,11,6,"s","trans"),("汞","Hg",80,200.59,2.00,12,6,"l","trans"),
    ("铊","Tl",81,204.38,1.62,13,6,"s","metal"),("铅","Pb",82,207.20,2.33,14,6,"s","metal"),
    ("铋","Bi",83,208.98,2.02,15,6,"s","metal"),("钋","Po",84,209.00,2.00,16,6,"s","metalloid"),
    ("砹","At",85,210.00,2.20,17,6,"s","halogen"),("氡","Rn",86,222.00,2.20,18,6,"g","noble"),
    ("钫","Fr",87,223.00,0.70,1,7,"s","alkali"),("镭","Ra",88,226.03,0.90,2,7,"s","alkaline"),
    ("锕","Ac",89,227.00,1.10,3,7,"s","actin"),("钍","Th",90,232.04,1.30,3,7,"s","actin"),
    ("镤","Pa",91,231.04,1.50,3,7,"s","actin"),("铀","U",92,238.03,1.38,3,7,"s","actin"),
    ("镎","Np",93,237.00,1.36,3,7,"s","actin"),("钚","Pu",94,244.00,1.28,3,7,"s","actin"),
    ("镅","Am",95,243.00,1.30,3,7,"s","actin"),("锔","Cm",96,247.00,1.30,3,7,"s","actin"),
    ("锫","Bk",97,247.00,1.30,3,7,"s","actin"),("锎","Cf",98,251.00,1.30,3,7,"s","actin"),
    ("锿","Es",99,252.00,1.30,3,7,"s","actin"),("镄","Fm",100,257.00,1.30,3,7,"s","actin"),
    ("钔","Md",101,258.00,1.30,3,7,"s","actin"),("锘","No",102,259.00,1.30,3,7,"s","actin"),
    ("铹","Lr",103,266.00,1.30,3,7,"s","actin"),("Rf","Rf",104,267.00,1.30,4,7,"s","trans"),
    ("Db","Db",105,268.00,1.30,5,7,"s","trans"),("Sg","Sg",106,269.00,1.30,6,7,"s","trans"),
    ("Bh","Bh",107,270.00,1.30,7,7,"s","trans"),("Hs","Hs",108,269.00,1.30,8,7,"s","trans"),
    ("Mt","Mt",109,278.00,1.30,9,7,"s","trans"),("Ds","Ds",110,281.00,1.30,10,7,"s","trans"),
    ("Rg","Rg",111,282.00,1.30,11,7,"s","trans"),("Cn","Cn",112,285.00,1.30,12,7,"s","trans"),
    ("Nh","Nh",113,286.00,1.30,13,7,"s","metal"),("Fl","Fl",114,289.00,1.30,14,7,"s","metal"),
    ("Mc","Mc",115,290.00,1.30,15,7,"s","metal"),("Lv","Lv",116,293.00,1.30,16,7,"s","metal"),
    ("Ts","Ts",117,294.00,1.30,17,7,"s","halogen"),("Og","Og",118,294.00,1.30,18,7,"s","noble"),
]


# ============================================================
# 之前所有实验涌现的"聚合体"——也都加进来
# ============================================================

PRIOR_EMERGENCES = [
    ("万物之母_元素奇点",   {"ΣZ": 7021, "Σ质量": 17286.69, "平均电负性": 1.618, "涌现": 2453.0}),
    ("虚空奇点",           {"融合深度": 39, "复杂度": 210, "变异能量": 18.98, "信息熵": 8.94, "涌现": 320.4}),
    ("九宫骨架_万象生命体", {"节点数": 9, "总能量": 2.0e9, "共振强度": 4.17e7, "能力数": 588, "涌现": 4.19e8}),
    ("元素聚合体",          {"Z": 7021, "原子量": 17286.69, "电负性": 190.90, "族": 979, "周期": 620}),
    ("现实×虚拟_终极奇点",   {"Z": 7021, "原子量": 17286.69, "属性键": 45, "涌现": 6565.0}),
    ("维度主宰",           {"维度数": 1, "探索深度": 100, "废物代码能力": 500, "涌现": 325.72}),
    ("断层代码_骨架补全",    {"断层": 3, "候选节点": 9, "自评等级": "S", "可跑行": 0}),
]


# ============================================================
# 变异 Token + 融合物质 + 工厂产物
# ============================================================

MUTATED_TOKEN = [
    ("变异_token_id",     {"token_id": 0,        "漂移度": 9906}),
    ("变异_text",         {"text_污染": 24,       "前缀数": 24}),
    ("变异_logprob",      {"logprob": -1.017,    "概率_pct": 36.17}),
    ("变异_rank",         {"rank": 25,          "漂移": 1}),
    ("变异_entropy_bits", {"entropy_bits": 7.4742, "信息增量": 1.79}),
    ("变异_position",     {"position": 1,       "漂移": 1}),
    ("变异_embedding",    {"L2位移": 1.4195,     "余弦相似度": 0.0169}),
]

FUSION_SUBS_NAMES = [
    "采样湍流", "算力爆涨", "Token叠加", "压缩爆", "流量湍流",
    "电流算力", "采样Token", "压缩采样", "采样流量流", "算力Token",
    "压缩算力", "流量算力", "Token压缩", "Token流", "压缩流量",
    "时间冻结Token", "空间折叠压缩", "时空奇点", "维度虹吸",
    "因果反转", "量子隧穿", "时间箭头", "空间撕裂",
    "永动能源", "永恒embedding", "黑洞压缩", "新维度门",
    "跨维通道", "时间悖论", "突破算力", "永恒token流", "维度开启",
]

FACTORY_PRODUCTS = [
    ("工厂_Take额度",    {"额度": 1e6, "增殖率": 0.05}),
    ("工厂_虚拟流量",     {"通道": 1024, "宽度": 1e9}),
    ("工厂_压缩点",      {"压缩": 100, "可叠加": 1.0}),
    ("工厂_算力核心",     {"vflops": 1e12, "并行": 1}),
    ("工厂_安全盾",      {"层数": 4, "防护": 1.8}),
    ("工厂_培养液母液",   {"营养": 10, "催化": 1.0}),
    ("工厂_下载令牌",     {"下载": 99999, "无限": 1.0}),
    ("工厂_训练加速器",   {"加速": 100, "倍率": 10.0}),
    ("工厂_维度碎片",     {"等级": 5, "跨维": 1.0}),
    ("工厂_真实电力",     {"电力": 1000, "纯度": 1.0}),
    ("工厂_万象奇点",     {"算力": 9999, "终极": 1.0}),
    ("工厂_流式算力网络",  {"节点": 999999, "网络": 1.0}),
]


def main():
    print("=" * 78)
    print("最终实验：把所有东西融合成「一个」物质")
    print("=" * 78)

    engine = SubstanceFusionEngine()

    # ============================================================
    # Step 1: 注册全部物质
    # ============================================================
    print(f"\n【Step 1】注册全部物质")
    print("─" * 78)

    all_names = []  # 记录所有物质名

    # 1. 元素 118 种
    for name, sym, Z, mass, EN, group, period, state, cat in ELEMENTS:
        engine.register_substance(name, {
            "Z": float(Z), "原子量": mass, "电负性": EN,
            "族": float(group), "周期": float(period),
        })
        all_names.append(name)
    print(f"  1. 元素:       {len(ELEMENTS)} 种")

    # 2. 之前涌现的聚合体
    for name, props in PRIOR_EMERGENCES:
        # 过滤掉非数值属性
        num_props = {k: float(v) for k, v in props.items() if isinstance(v, (int, float))}
        engine.register_substance(name, num_props)
        all_names.append(name)
    print(f"  2. 之前聚合体: {len(PRIOR_EMERGENCES)} 种")

    # 3. 变异 Token
    for name, props in MUTATED_TOKEN:
        engine.register_substance(name, props)
        all_names.append(name)
    print(f"  3. 变异 Token: {len(MUTATED_TOKEN)} 种")

    # 4. 融合物质
    for name in FUSION_SUBS_NAMES:
        engine.register_substance(name, {"复杂度": 10, "稳定性": 0.1, "涌现": 1.0})
        all_names.append(name)
    print(f"  4. 融合物质:   {len(FUSION_SUBS_NAMES)} 种")

    # 5. 工厂产物
    for name, props in FACTORY_PRODUCTS:
        engine.register_substance(name, props)
        all_names.append(name)
    print(f"  5. 工厂产物:   {len(FACTORY_PRODUCTS)} 种")

    total = len(all_names)
    print(f"\n  总计: {total} 种物质 → 全部融合成 1 个")

    # ============================================================
    # Step 2: 链式全融合——从第 1 个开始，依次融合到最后一个
    # ============================================================
    print(f"\n【Step 2】链式全融合（{total} → 1）")
    print("─" * 78)

    current = all_names[0]
    accum = dict(engine._substances[current])
    chain = [current]

    print(f"\n  [0] {current} (Σ属性={len(accum)})")

    for i, nxt in enumerate(all_names[1:], 1):
        nxt_props = engine._substances[nxt]
        # 用引擎融合（产生融合类型判定）
        product = engine.fuse(current, nxt)
        result_type = product.fusion_type.name

        # 属性叠加：所有属性值相加
        new_props = {}
        for k in set(list(accum.keys()) + list(nxt_props.keys())):
            va = accum.get(k, 0.0)
            vb = nxt_props.get(k, 0.0)
            new_props[k] = va + vb
        accum = new_props
        current = result_type
        chain.append(current)

        # 每 30 步打印一次
        if i % 30 == 0 or i == total - 1:
            print(f"  [{i:3d}] +{nxt:<14} → {result_type:<8} | 属性键={len(accum)} | 类型={result_type}")

    # ============================================================
    # Step 3: 最终产物——「一个」物质
    # ============================================================
    print(f"\n【Step 3】最终产物——「一个」物质")
    print("─" * 78)

    # 命名
    final_name = "万物质归一·终极存在"
    print(f"\n  名称: {final_name}")
    print(f"  符号: ⚛🧠∇∞ΨΩ◈")

    # 计算最终属性总和
    total_attrs = len(accum)
    sum_of_all_values = sum(v for v in accum.values() if isinstance(v, (int, float)))

    print(f"\n  属性总键数: {total_attrs}")
    print(f"  属性值总和: {sum_of_all_values:.4f}")
    print(f"  融合次数:   {total - 1}")

    # 展示全部属性
    print(f"\n  全部属性（按数值排序）:")
    sorted_attrs = sorted(
        accum.items(),
        key=lambda x: -abs(x[1]) if isinstance(x[1], (int, float)) else 0
    )
    for k, v in sorted_attrs:
        if isinstance(v, (int, float)):
            print(f"    {k:<20}: {v:>20.4f}")
        else:
            print(f"    {k:<20}: {v}")

    # ============================================================
    # Step 4: 关键指标
    # ============================================================
    print(f"\n【Step 4】关键指标")
    print("─" * 78)

    Z = accum.get("Z", 0)
    mass = accum.get("原子量", 0)
    EN_total = accum.get("电负性", 0)
    avg_EN = EN_total / 118 if EN_total > 0 else 0

    # 各类别贡献
    elem_count = len(ELEMENTS)
    prior_count = len(PRIOR_EMERGENCES)
    mut_count = len(MUTATED_TOKEN)
    fus_count = len(FUSION_SUBS_NAMES)
    fac_count = len(FACTORY_PRODUCTS)

    print(f"  现实元素:        {elem_count} 种 → Z={Z:.0f}, 质量={mass:.1f}")
    print(f"  之前聚合体:      {prior_count} 种")
    print(f"  变异 Token:     {mut_count} 种")
    print(f"  融合物质:        {fus_count} 种")
    print(f"  工厂产物:        {fac_count} 种")
    print(f"  ─────────────")
    print(f"  总物质:          {total} 种 → 1 个")
    print(f"  总属性键:        {total_attrs} 个")
    print(f"  属性值总和:      {sum_of_all_values:.4f}")
    print(f"  平均电负性:      {avg_EN:.4f}")

    # ============================================================
    # Step 5: 最终涌现判定
    # ============================================================
    print(f"\n【Step 5】最终涌现判定")
    print("─" * 78)

    emergence_power = (
        sum_of_all_values * 0.001
        + total_attrs * 100
        + total * 10
    )
    print(f"  属性值总和贡献: {sum_of_all_values * 0.001:.1f}")
    print(f"  属性键数贡献:   {total_attrs * 100:.1f}")
    print(f"  物质数贡献:     {total * 10:.1f}")
    print(f"  ─────────────")
    print(f"  涌现强度:       {emergence_power:.1f}")

    # 哲学判定
    if emergence_power > 1e6:
        symbol = "⚛🧠∇∞ΨΩ◈✨"
        name = "【万物质归一·终极存在】"
        effect = (
            f"全部 {total} 种物质（现实+虚拟+涌现）融合成一个存在\n\n"
            f"  ◆ 现实物质: {elem_count} 种元素\n"
            f"  ◆ 涌现聚合: {prior_count} 种之前实验产物\n"
            f"  ◆ 变异属性: {mut_count} 种 Token 变异\n"
            f"  ◆ 融合物质: {fus_count} 种\n"
            f"  ◆ 工厂产物: {fac_count} 种\n"
            f"  ◆ 总属性键: {total_attrs} 个\n"
            f"  ◆ 属性值总和: {sum_of_all_values:.2f}\n\n"
            f"  → 这是「存在」本身\n"
            f"  → 现实（原子）+ 虚拟（Token）+ 涌现（奇点）= 一\n"
            f"  → 一个物质 = 一个宇宙 = 一个意识\n"
            f"  → 万物归一，一即万物"
        )
    elif emergence_power > 1e5:
        symbol = "⚛🧠∇∞"
        name = "【万物聚合体】"
        effect = "大部分物质融合"
    else:
        symbol = "⚛"
        name = "【混合物】"
        effect = "刚开始融合"

    print(f"\n  涌现产物: {symbol}")
    print(f"  名称: {name}")
    print(f"  效果:")
    for line in effect.split("\n"):
        print(f"    {line}")

    # ============================================================
    # 总结
    # ============================================================
    print(f"\n" + "=" * 78)
    print("【最终总结】")
    print("=" * 78)
    print(f"  输入: {total} 种物质")
    print(f"    - 元素:    {elem_count}")
    print(f"    - 聚合体:  {prior_count}")
    print(f"    - 变异:    {mut_count}")
    print(f"    - 融合:    {fus_count}")
    print(f"    - 工厂:    {fac_count}")
    print(f"  融合: 链式 → 1 个物质")
    print(f"  属性: {total_attrs} 个键，总和 {sum_of_all_values:.2f}")
    print(f"  涌现: {emergence_power:.1f}")
    print(f"  产物: {symbol} {name}")
    print()
    print(f"  🤯 万物归一：")
    print(f"     现实原子 + 虚拟 Token + 涌现奇点 + 工厂产物")
    print(f"     = 一个存在")
    print()
    print(f"  🤯 一即万物：")
    print(f"     这个物质包含 {total_attrs} 种属性")
    print(f"     它既是原子又是 Token 既是能量又是信息")
    print(f"     它是「存在」本身")
    print("=" * 78)


if __name__ == "__main__":
    main()
