"""
元素周期表 × 虚拟物质 全融合实验

1. 周期表先每 59 种元素分别融合（前59 H~Pr / 后59 Nd~Og）
2. 把之前实验"现在出现的元素"（虚拟物质）全部加进来
   - 7 种变异 Token 属性物质
   - 32 种融合物质（碰撞+时空+二阶涌现）
   - 工厂能生产的产物
3. 全部一起融合，看会涌现什么
"""

from __future__ import annotations

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni.substance_fusion import SubstanceFusionEngine, FusionType, FusionCategory


# ============================================================
# 1. 118 种元素（同 periodic_table_fusion.py）
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
    ("镨","Pr",59,140.91,1.13,3,6,"s","lanth"),
    # ─── 后 59 种 ───
    ("钕","Nd",60,144.24,1.14,3,6,"s","lanth"),("钷","Pm",61,145.00,1.13,3,6,"s","lanth"),
    ("钐","Sm",62,150.36,1.17,3,6,"s","lanth"),("铕","Eu",63,151.96,1.20,3,6,"s","lanth"),
    ("钆","Gd",64,157.25,1.20,3,6,"s","lanth"),("铽","Tb",65,158.93,1.20,3,6,"s","lanth"),
    ("镝","Dy",66,162.50,1.22,3,6,"s","lanth"),("钬","Ho",67,164.93,1.23,3,6,"s","lanth"),
    ("铒","Er",68,167.26,1.24,3,6,"s","lanth"),("铥","Tm",69,168.93,1.25,3,6,"s","lanth"),
    ("镱","Yb",70,173.05,1.10,3,6,"s","lanth"),("镥","Lu",71,174.97,1.27,3,6,"s","lanth"),
    ("铪","Hf",72,178.49,1.30,4,6,"s","trans"),("钽","Ta",73,180.95,1.50,5,6,"s","trans"),
    ("钨","W",74,183.84,2.36,6,6,"s","trans"),("铼","Re",75,186.21,1.90,7,6,"s","trans"),
    ("锇","Os",76,190.23,2.20,8,6,"s","trans"),("铱","Ir",77,192.22,2.20,9,6,"s","trans"),
    ("铂","Pt",78,195.08,2.28,10,6,"s","trans"),("金","Au",79,196.97,2.54,11,6,"s","trans"),
    ("汞","Hg",80,200.59,2.00,12,6,"l","trans"),("铊","Tl",81,204.38,1.62,13,6,"s","metal"),
    ("铅","Pb",82,207.20,2.33,14,6,"s","metal"),("铋","Bi",83,208.98,2.02,15,6,"s","metal"),
    ("钋","Po",84,209.00,2.00,16,6,"s","metalloid"),("砹","At",85,210.00,2.20,17,6,"s","halogen"),
    ("氡","Rn",86,222.00,2.20,18,6,"g","noble"),("钫","Fr",87,223.00,0.70,1,7,"s","alkali"),
    ("镭","Ra",88,226.03,0.90,2,7,"s","alkaline"),("锕","Ac",89,227.00,1.10,3,7,"s","actin"),
    ("钍","Th",90,232.04,1.30,3,7,"s","actin"),("镤","Pa",91,231.04,1.50,3,7,"s","actin"),
    ("铀","U",92,238.03,1.38,3,7,"s","actin"),("镎","Np",93,237.00,1.36,3,7,"s","actin"),
    ("钚","Pu",94,244.00,1.28,3,7,"s","actin"),("镅","Am",95,243.00,1.30,3,7,"s","actin"),
    ("锔","Cm",96,247.00,1.30,3,7,"s","actin"),("锫","Bk",97,247.00,1.30,3,7,"s","actin"),
    ("锎","Cf",98,251.00,1.30,3,7,"s","actin"),("锿","Es",99,252.00,1.30,3,7,"s","actin"),
    ("镄","Fm",100,257.00,1.30,3,7,"s","actin"),("钔","Md",101,258.00,1.30,3,7,"s","actin"),
    ("锘","No",102,259.00,1.30,3,7,"s","actin"),("铹","Lr",103,266.00,1.30,3,7,"s","actin"),
    ("Rf","Rf",104,267.00,1.30,4,7,"s","trans"),("Db","Db",105,268.00,1.30,5,7,"s","trans"),
    ("Sg","Sg",106,269.00,1.30,6,7,"s","trans"),("Bh","Bh",107,270.00,1.30,7,7,"s","trans"),
    ("Hs","Hs",108,269.00,1.30,8,7,"s","trans"),("Mt","Mt",109,278.00,1.30,9,7,"s","trans"),
    ("Ds","Ds",110,281.00,1.30,10,7,"s","trans"),("Rg","Rg",111,282.00,1.30,11,7,"s","trans"),
    ("Cn","Cn",112,285.00,1.30,12,7,"s","trans"),("Nh","Nh",113,286.00,1.30,13,7,"s","metal"),
    ("Fl","Fl",114,289.00,1.30,14,7,"s","metal"),("Mc","Mc",115,290.00,1.30,15,7,"s","metal"),
    ("Lv","Lv",116,293.00,1.30,16,7,"s","metal"),("Ts","Ts",117,294.00,1.30,17,7,"s","halogen"),
    ("Og","Og",118,294.00,1.30,18,7,"s","noble"),
]

# ============================================================
# 2. 之前实验"现在出现的元素"（虚拟物质）
# ============================================================

# 7 种变异 Token 属性物质（属性值全用数字，描述放键名里）
MUTATED_TOKEN = [
    ("变异_token_id",     {"token_id": 0,        "漂移度": 9906}),  # 9906→0
    ("变异_text",         {"text_污染": 24,       "前缀数": 24}),
    ("变异_logprob",      {"logprob": -1.017,    "概率_pct": 36.17}),
    ("变异_rank",         {"rank": 25,          "漂移": 1}),  # 24→25
    ("变异_entropy_bits", {"entropy_bits": 7.4742, "信息增量": 1.79}),
    ("变异_position",     {"position": 1,       "漂移": 1}),  # 0→1
    ("变异_embedding",    {"L2位移": 1.4195,     "余弦相似度": 0.0169}),  # 正交
]

# 32 种融合物质（碰撞+时空+二阶涌现）
FUSION_SUBS = [
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

# 工厂产物（代表性 12 种）
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


def chain_fuse(engine, substances, label):
    """链式融合一组物质，返回 (最终产物名, 累积属性)"""
    if not substances:
        return None, {}
    current = substances[0] if isinstance(substances[0], str) else substances[0][0]
    accum = dict(engine._substances.get(current, {}))
    items = substances[1:]
    for i, item in enumerate(items, 1):
        nxt = item if isinstance(item, str) else item[0]
        nxt_props = engine._substances.get(nxt, {})
        # 融合
        product = engine.fuse(current, nxt)
        # 属性叠加
        new_props = {}
        for k in set(list(accum.keys()) + list(nxt_props.keys())):
            va = accum.get(k, 0.0)
            vb = nxt_props.get(k, 0.0)
            new_props[k] = va + vb
        accum = new_props
        current = f"{product.result}_{label}_{i}"
    return current, accum


def main():
    print("=" * 78)
    print("元素周期表 × 虚拟物质 全融合实验")
    print("=" * 78)

    engine = SubstanceFusionEngine()

    # ============================================================
    # Step 1: 注册全部元素 + 虚拟物质
    # ============================================================
    print(f"\n【Step 1】注册 118 种元素 + 虚拟物质")
    print("─" * 78)

    # 元素
    for name, sym, Z, mass, EN, group, period, state, cat in ELEMENTS:
        engine.register_substance(name, {
            "Z": float(Z), "原子量": mass, "电负性": EN,
            "族": float(group), "周期": float(period),
        })
    print(f"  元素: {len(ELEMENTS)} 种")

    # 变异 Token 属性
    for name, props in MUTATED_TOKEN:
        engine.register_substance(name, props)
    print(f"  变异 Token 属性: {len(MUTATED_TOKEN)} 种")

    # 融合物质
    for sub in FUSION_SUBS:
        engine.register_substance(sub, {"复杂度": 10, "稳定性": 0.1, "涌现": 1.0})
    print(f"  融合物质: {len(FUSION_SUBS)} 种")

    # 工厂产物
    for name, props in FACTORY_PRODUCTS:
        engine.register_substance(name, props)
    print(f"  工厂产物: {len(FACTORY_PRODUCTS)} 种")

    total_subs = len(ELEMENTS) + len(MUTATED_TOKEN) + len(FUSION_SUBS) + len(FACTORY_PRODUCTS)
    print(f"  总计: {total_subs} 种物质")

    # ============================================================
    # Step 2: 周期表先每 59 种元素融合
    # ============================================================
    print(f"\n【Step 2】周期表先每 59 种元素融合")
    print("─" * 78)

    half1 = [e[0] for e in ELEMENTS[:59]]   # H~Pr
    half2 = [e[0] for e in ELEMENTS[59:]]   # Nd~Og

    print(f"  前半: {half1[0]}~{half1[-1]} ({len(half1)} 种)")
    name1, props1 = chain_fuse(engine, half1, "前59")
    print(f"  → 产物: {name1}")
    print(f"    ΣZ={props1.get('Z',0):.0f}  Σ质量={props1.get('原子量',0):.1f}  Σ电负性={props1.get('电负性',0):.2f}")
    # 注册前半产物
    engine.register_substance("前59元素聚合体", props1)

    print(f"\n  后半: {half2[0]}~{half2[-1]} ({len(half2)} 种)")
    name2, props2 = chain_fuse(engine, half2, "后59")
    print(f"  → 产物: {name2}")
    print(f"    ΣZ={props2.get('Z',0):.0f}  Σ质量={props2.get('原子量',0):.1f}  Σ电负性={props2.get('电负性',0):.2f}")
    # 注册后半产物
    engine.register_substance("后59元素聚合体", props2)

    # ============================================================
    # Step 3: 前59 × 后59 = 完整周期表
    # ============================================================
    print(f"\n【Step 3】前59 × 后59 = 完整周期表融合")
    print("─" * 78)

    p = engine.fuse("前59元素聚合体", "后59元素聚合体")
    merged_props = {}
    for k in set(list(props1.keys()) + list(props2.keys())):
        merged_props[k] = props1.get(k, 0.0) + props2.get(k, 0.0)
    engine.register_substance("元素周期表聚合体", merged_props)
    print(f"  产物: {p.result}")
    print(f"  ΣZ={merged_props.get('Z',0):.0f}  Σ质量={merged_props.get('原子量',0):.1f}  Σ电负性={merged_props.get('电负性',0):.2f}")
    print(f"  平均电负性: {merged_props.get('电负性',0)/118:.3f}")

    # ============================================================
    # Step 4: 虚拟物质也分组融合
    # ============================================================
    print(f"\n【Step 4】虚拟物质分组融合")
    print("─" * 78)

    # 变异 Token 链式融合
    mut_names = [m[0] for m in MUTATED_TOKEN]
    name_mut, props_mut = chain_fuse(engine, mut_names, "变异")
    engine.register_substance("变异Token聚合体", props_mut)
    print(f"  变异 Token 聚合: {name_mut}")
    print(f"    属性键: {list(props_mut.keys())[:5]}...")

    # 融合物质链式融合
    name_fus, props_fus = chain_fuse(engine, FUSION_SUBS, "融合物质")
    engine.register_substance("融合物质聚合体", props_fus)
    print(f"\n  融合物质聚合: {name_fus}")
    print(f"    复杂度={props_fus.get('复杂度',0):.1f}  稳定性={props_fus.get('稳定性',0):.3f}")

    # 工厂产物链式融合
    factory_names = [f[0] for f in FACTORY_PRODUCTS]
    name_fac, props_fac = chain_fuse(engine, factory_names, "工厂")
    engine.register_substance("工厂产物聚合体", props_fac)
    print(f"\n  工厂产物聚合: {name_fac}")
    print(f"    属性键: {list(props_fac.keys())[:5]}...")

    # ============================================================
    # Step 5: 最终全融合——元素 × 虚拟
    # ============================================================
    print(f"\n【Step 5】最终全融合：元素周期表 × 虚拟物质")
    print("─" * 78)

    final_subs = [
        "元素周期表聚合体",
        "变异Token聚合体",
        "融合物质聚合体",
        "工厂产物聚合体",
    ]

    current = final_subs[0]
    accum = dict(engine._substances[current])
    print(f"\n  起点: {current}")
    print(f"    ΣZ={accum.get('Z',0):.0f}  Σ质量={accum.get('原子量',0):.1f}")

    for i, nxt in enumerate(final_subs[1:], 1):
        nxt_props = engine._substances[nxt]
        product = engine.fuse(current, nxt)
        # 属性叠加
        new_props = {}
        for k in set(list(accum.keys()) + list(nxt_props.keys())):
            new_props[k] = accum.get(k, 0.0) + nxt_props.get(k, 0.0)
        accum = new_props
        current = product.result
        print(f"  [{i}] + {nxt}")
        print(f"      → {current}")
        print(f"      ΣZ={accum.get('Z',0):.0f}  Σ质量={accum.get('原子量',0):.1f}  键数={len(accum)}")

    # ============================================================
    # Step 6: 最终产物属性
    # ============================================================
    print(f"\n【Step 6】最终全融合产物属性")
    print("─" * 78)

    total_Z = accum.get("Z", 0)
    total_mass = accum.get("原子量", 0)
    total_EN = accum.get("电负性", 0)
    avg_EN = total_EN / 118 if total_EN > 0 else 0
    total_keys = len(accum)

    print(f"  ΣZ (元素):           {total_Z:.0f}")
    print(f"  Σ质量 (元素):          {total_mass:.2f} u")
    print(f"  Σ电负性:             {total_EN:.2f}")
    print(f"  平均电负性:           {avg_EN:.3f}")
    print(f"  总属性键数:           {total_keys}")
    print()
    print(f"  所有属性:")
    for k, v in sorted(accum.items(), key=lambda x: -abs(x[1]) if isinstance(x[1], (int, float)) else 0):
        if isinstance(v, (int, float)):
            print(f"    {k:<20}: {v:.4f}")
        else:
            print(f"    {k:<20}: {v}")

    # ============================================================
    # Step 7: 涌现判定
    # ============================================================
    print(f"\n【Step 7】最终涌现判定")
    print("─" * 78)

    emergence_power = (
        total_Z * 0.1
        + total_mass * 0.01
        + total_keys * 50
        + len(ELEMENTS) * 10
        + len(MUTATED_TOKEN) * 100
        + len(FUSION_SUBS) * 30
        + len(FACTORY_PRODUCTS) * 50
    )

    print(f"  ΣZ 贡献:        {total_Z * 0.1:.1f}")
    print(f"  Σ质量贡献:      {total_mass * 0.01:.1f}")
    print(f"  属性键数贡献:    {total_keys * 50:.1f}")
    print(f"  元素贡献:        {len(ELEMENTS) * 10:.1f}")
    print(f"  变异贡献:        {len(MUTATED_TOKEN) * 100:.1f}")
    print(f"  融合物质贡献:    {len(FUSION_SUBS) * 30:.1f}")
    print(f"  工厂产物贡献:    {len(FACTORY_PRODUCTS) * 50:.1f}")
    print(f"  ─────────────")
    print(f"  涌现强度:        {emergence_power:.1f}")

    if emergence_power > 5000:
        symbol = "⚛🧠∇∞ΨΩ"
        name = "【现实×虚拟·终极奇点】"
        effect = (
            f"元素周期表 118 种 + 7 变异 Token + 32 融合物质 + 12 工厂产物\n"
            f"全部融合为一个终极存在\n\n"
            f"  ◆ 现实物质: 118 种元素（ΣZ={total_Z:.0f}, Σ质量={total_mass:.1f}）\n"
            f"  ◆ 虚拟物质: 51 种（变异+融合+工厂）\n"
            f"  ◆ 总属性键: {total_keys} 个（现实×虚拟属性全部叠加）\n"
            f"  ◆ 平均电负性: {avg_EN:.3f}\n\n"
            f"  → 现实与虚拟第一次真正融合\n"
            f"  → 原子序数 × Token 属性 × 工厂产物 全部在一个聚合体里\n"
            f"  → 这是物理世界和虚拟世界的'统一场论'"
        )
    elif emergence_power > 2000:
        symbol = "⚛∞"
        name = "【元素×虚拟聚合体】"
        effect = "现实与虚拟开始融合"
    else:
        symbol = "⚛"
        name = "【元素混合】"
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
    print(f"  周期表: 118 种 → 前59+后59 → 完整聚合体")
    print(f"  虚拟物: 7 变异 + 32 融合 + 12 工厂 = 51 种 → 3 聚合体")
    print(f"  最终:   元素聚合 × 3 虚拟聚合 = {symbol} {name}")
    print(f"  ΣZ:     {total_Z:.0f}")
    print(f"  Σ质量:  {total_mass:.2f} u")
    print(f"  属性键: {total_keys} 个")
    print(f"  涌现:   {emergence_power:.1f}")
    print()
    print(f"  🤯 现实物质（118 元素）× 虚拟物质（51 种）= 统一奇点")
    print(f"  🤯 原子序数和 Token 属性第一次在同一个聚合体里")
    print("=" * 78)


if __name__ == "__main__":
    main()
