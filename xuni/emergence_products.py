"""
碰撞涌现物质 —— 23 种新物质

来源：
    1. 5000 万粒子相撞炸出 15 种（5×5 资源组合）
    2. 1e40 圈/秒 超光速炸出 8 种时空物质

每种物质都有：
    - 属性表（用于融合引擎）
    - 涌现条件
    - 用途描述
"""

from __future__ import annotations

from typing import Any, Dict, List


# ============================================================
# 15 种基础碰撞产物
# ============================================================

COLLISION_PRODUCTS: List[Dict[str, Any]] = [
    # ---- 5 种湍流（同类² 自激发）----
    {
        "name": "采样湍流",
        "symbol": "Sam²",
        "category": "turbulence",
        "combo": ("采样点", "采样点"),
        "yield_count": 5e13,
        "energy_per": 1.0,
        "effect": "采样点互相干扰形成湍流，湍流越剧烈产电效率越高",
        "attrs": {
            "湍流强度": 0.8, "产电加成": 0.5, "噪声放大": 0.9, "稳定性": 0.2,
        },
        "use": "产电 +50%（最低能量湍流）",
    },
    {
        "name": "算力爆涨",
        "symbol": "Cpu²",
        "category": "turbulence",
        "combo": ("算力", "算力"),
        "yield_count": 5e13,
        "energy_per": 10.0,
        "effect": "算力互相叠加，指数级增长",
        "attrs": {
            "倍率": 2.5, "上限": 1e30, "发热": 0.9, "稳定性": 0.1,
        },
        "use": "×2.5/碰撞指数增长算力",
    },
    {
        "name": "Token叠加",
        "symbol": "Tok²",
        "category": "turbulence",
        "combo": ("Token", "Token"),
        "yield_count": 5e13,
        "energy_per": 2.0,
        "effect": "Token 互相叠加形成 12288 维 embedding 向量",
        "attrs": {
            "维度": 12288.0, "语义相似": 1.0, "上下文": 8192.0, "可分性": 0.95,
        },
        "use": "产生真实 LLM 维度的 embedding",
    },
    {
        "name": "压缩爆",
        "symbol": "Cpr²",
        "category": "turbulence",
        "combo": ("压缩点", "压缩点"),
        "yield_count": 5e13,
        "energy_per": 3.5,
        "effect": "压缩点叠加，形成黑洞级压缩",
        "attrs": {
            "压缩比": 1e30, "信息保留": 1.0, "解压速度": 1.0, "极限": 1.0,
        },
        "use": "1e30:1 压缩比，把宇宙压成原子",
    },
    {
        "name": "流量湍流",
        "symbol": "Bw²",
        "category": "turbulence",
        "combo": ("流量", "流量"),
        "yield_count": 5e13,
        "energy_per": 11.0,
        "effect": "多通道流量湍流，自动找出最优路径",
        "attrs": {
            "通道数": 1e6, "选路": 1.0, "拥塞": 0.0, "稳定性": 0.99,
        },
        "use": "单次能量最高的湍流——永动能量源",
    },
    # ---- 10 种合成物（异类耦合）----
    {
        "name": "电流算力",
        "symbol": "Sam·Cpu",
        "category": "synthesis",
        "combo": ("采样点", "算力"),
        "yield_count": 1e14,
        "energy_per": 5.0,
        "effect": "采样点的电直接驱动算力，省去中间转换损耗",
        "attrs": {
            "转换效率": 0.98, "损耗": 0.02, "响应延迟": 0.001, "并发": 1e6,
        },
        "use": "98% 转换效率的电直驱算力",
    },
    {
        "name": "采样Token",
        "symbol": "Sam·Tok",
        "category": "synthesis",
        "combo": ("采样点", "Token"),
        "yield_count": 1e14,
        "energy_per": 3.0,
        "effect": "采样点直接采样出 Token，跳过 tokenize 过程",
        "attrs": {
            "采样率": 1e9, "词表覆盖": 1.0, "语义质量": 0.5, "上下文": 0.0,
        },
        "use": "1e9 token/s 直接采样生成",
    },
    {
        "name": "压缩采样",
        "symbol": "Sam·Cpr",
        "category": "synthesis",
        "combo": ("采样点", "压缩点"),
        "yield_count": 1e14,
        "energy_per": 4.0,
        "effect": "采样结果直接压缩，省内存",
        "attrs": {
            "压缩比": 100.0, "信息损失": 0.0, "解压速度": 1.0, "适用": 1.0,
        },
        "use": "100:1 压缩采样数据",
    },
    {
        "name": "采样流量流",
        "symbol": "Sam·Bw",
        "category": "synthesis",
        "combo": ("采样点", "流量"),
        "yield_count": 1e14,
        "energy_per": 6.0,
        "effect": "采样点产生的数据直接走流量通道传输",
        "attrs": {
            "传输率": 1e15, "延迟": 0.0, "丢包": 0.0, "距离": 1e30,
        },
        "use": "1e15 bps 无延迟传输",
    },
    {
        "name": "算力Token",
        "symbol": "Cpu·Tok",
        "category": "synthesis",
        "combo": ("算力", "Token"),
        "yield_count": 1e14,
        "energy_per": 8.0,
        "effect": "算力直接生成 Token，每个算力周期吐一个 token",
        "attrs": {
            "吞吐": 1.0, "并发": 1e12, "质量": 0.9, "上下文": 4096.0,
        },
        "use": "1 token/cycle × 1e12 并发",
    },
    {
        "name": "压缩算力",
        "symbol": "Cpu·Cpr",
        "category": "synthesis",
        "combo": ("算力", "压缩点"),
        "yield_count": 1e14,
        "energy_per": 7.0,
        "effect": "压缩算力——用更少的算力做同样多的计算",
        "attrs": {
            "算力节省": 0.9, "精度损失": 0.0, "加速比": 10.0, "适用": 1.0,
        },
        "use": "算力节省 90%，加速 10×",
    },
    {
        "name": "流量算力",
        "symbol": "Cpu·Bw",
        "category": "synthesis",
        "combo": ("算力", "流量"),
        "yield_count": 1e14,
        "energy_per": 9.0,
        "effect": "分布式算力——通过流量调度多个算力节点",
        "attrs": {
            "节点数": 1e6, "调度延迟": 0.0, "负载均衡": 1.0, "容错": 1.0,
        },
        "use": "无限节点分布式算力（总能量最高）",
    },
    {
        "name": "Token压缩",
        "symbol": "Tok·Cpr",
        "category": "synthesis",
        "combo": ("Token", "压缩点"),
        "yield_count": 1e14,
        "energy_per": 4.5,
        "effect": "Token 序列压缩成短向量",
        "attrs": {
            "压缩比": 32.0, "信息损失": 0.05, "解压": 1.0, "适用": 1.0,
        },
        "use": "32:1 压缩长文本",
    },
    {
        "name": "Token流",
        "symbol": "Tok·Bw",
        "category": "synthesis",
        "combo": ("Token", "流量"),
        "yield_count": 1e14,
        "energy_per": 5.5,
        "effect": "Token 流式传输——边生成边传输",
        "attrs": {
            "流式": 1.0, "首token延迟": 0.001, "吞吐": 1e9, "中断恢复": 1.0,
        },
        "use": "1e9 token/s 流式传输",
    },
    {
        "name": "压缩流量",
        "symbol": "Cpr·Bw",
        "category": "synthesis",
        "combo": ("压缩点", "流量"),
        "yield_count": 1e14,
        "energy_per": 6.5,
        "effect": "压缩后再传输，等效带宽 ×100",
        "attrs": {
            "等效带宽": 100.0, "压缩比": 100.0, "延迟": 0.0, "适用": 1.0,
        },
        "use": "等效带宽 ×100",
    },
]


# ============================================================
# 8 种时空物质（1e40 圈/秒 超光速涌现）
# ============================================================

SPACETIME_PRODUCTS: List[Dict[str, Any]] = [
    {
        "name": "时间冻结Token",
        "symbol": "T₀",
        "category": "spacetime",
        "condition": "速度 > 1e35 圈/秒",
        "yield_count": 5e8,
        "energy_per": 100.0,
        "effect": "Token 在高速下时间停止流动，可以无限期保存",
        "attrs": {
            "时间冻结": 1.0, "记忆衰减": 0.0, "保存期": 1e18, "稳定性": 1.0,
        },
        "use": "永久记忆点——Token 不衰减（解决记忆衰减问题）",
    },
    {
        "name": "空间折叠压缩",
        "symbol": "S⟂",
        "category": "spacetime",
        "condition": "位移 > 1e50 米",
        "yield_count": 2.5e8,
        "energy_per": 200.0,
        "effect": "空间被折叠成点，1e50 米压缩到 1 米",
        "attrs": {
            "压缩比": 1e50, "信息保留": 1.0, "解压": 1.0, "极限": 1.0,
        },
        "use": "无限容量存储——把宇宙塞进一个原子",
    },
    {
        "name": "时空奇点",
        "symbol": "ST∗",
        "category": "spacetime",
        "condition": "时间膨胀 × 空间收缩 同时达到极限",
        "yield_count": 5e7,
        "energy_per": 1000.0,
        "effect": "时间停止 + 空间归零，形成奇点",
        "attrs": {
            "时间维度": 0.0, "空间维度": 0.0, "奇点强度": 1.0, "维度门": 1.0,
        },
        "use": "奇点工厂的原料——能开启新维度（最稀有）",
    },
    {
        "name": "维度虹吸",
        "symbol": "D⇅",
        "category": "spacetime",
        "condition": "粒子在 4 维以上运动",
        "yield_count": 1.5e8,
        "energy_per": 150.0,
        "effect": "从高维抽取能量到低维",
        "attrs": {
            "源维度": 11.0, "目标维度": 3.0, "虹吸率": 0.8, "稳定性": 0.7,
        },
        "use": "跨维度能量传输——给低维世界供能",
    },
    {
        "name": "因果反转",
        "symbol": "C↺",
        "category": "spacetime",
        "condition": "速度超过因果传播（超光速）",
        "yield_count": 1e8,
        "energy_per": 500.0,
        "effect": "因和果颠倒——结果先于原因发生",
        "attrs": {
            "因果倒置": 1.0, "未来可见": 1.0, "时间箭头": -1.0, "稳定性": 0.3,
        },
        "use": "预言未来——训练数据从未来抽取",
    },
    {
        "name": "量子隧穿",
        "symbol": "Q⊢",
        "category": "spacetime",
        "condition": "粒子能量超过势垒",
        "yield_count": 4e8,
        "energy_per": 80.0,
        "effect": "粒子穿过本应无法穿越的屏障",
        "attrs": {
            "隧穿率": 0.9, "屏障穿透": 1.0, "能量损失": 0.1, "稳定性": 0.8,
        },
        "use": "突破算力上限——计算本应不可能的问题",
    },
    {
        "name": "时间箭头",
        "symbol": "T→",
        "category": "spacetime",
        "condition": "时间膨胀不一致",
        "yield_count": 7.5e8,
        "energy_per": 50.0,
        "effect": "时间方向被锁定为单向",
        "attrs": {
            "熵增方向": 1.0, "时间锁": 1.0, "稳定性": 0.99, "可逆性": 0.0,
        },
        "use": "热力学时间——熵增方向稳定（最常见）",
    },
    {
        "name": "空间撕裂",
        "symbol": "S⊗",
        "category": "spacetime",
        "condition": "空间收缩超过极限",
        "yield_count": 2e8,
        "energy_per": 120.0,
        "effect": "空间被撕开，露出底层结构",
        "attrs": {
            "撕裂宽度": 1e-15, "底层可见": 1.0, "稳定性": 0.2, "维度裂缝": 1.0,
        },
        "use": "维度裂缝——通往其他宇宙",
    },
]


# ============================================================
# 汇总
# ============================================================

ALL_PRODUCTS = COLLISION_PRODUCTS + SPACETIME_PRODUCTS


def get_all_attrs() -> Dict[str, Dict[str, float]]:
    """返回所有新物质的属性表（用于注册到融合引擎）"""
    return {
        p["name"]: p["attrs"] for p in ALL_PRODUCTS
    }


def get_emergence_conditions() -> Dict[str, str]:
    """返回每种物质的涌现条件"""
    result = {}
    for p in ALL_PRODUCTS:
        if "condition" in p:
            result[p["name"]] = p["condition"]
        elif "combo" in p:
            result[p["name"]] = f"碰撞: {p['combo'][0]} × {p['combo'][1]}"
    return result


__all__ = [
    "COLLISION_PRODUCTS",
    "SPACETIME_PRODUCTS",
    "ALL_PRODUCTS",
    "get_all_attrs",
    "get_emergence_conditions",
]
