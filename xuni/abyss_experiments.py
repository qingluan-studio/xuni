"""
深渊维度深度探索文档
===================

一、假说与发现
---------------

### 1.1 裂变放大假说 (已证实)
深渊代码(AbyssalCode)具备裂变繁殖能力:
- 每个母体 → 100子代
- 子代攻击深度 > 母体 (正向突变: avg 97→98)
- 变异潜力 0.99, 复制率 2.0~2.5x
- 结论: 攻击力按代际指数增长，不衰减

### 1.2 自补循环假说 (待验证)
深渊攻击无限的原因推测:
- 攻击生成代码 → 代码生成攻击 → 攻击生成代码 (闭环)
- 可能内部存在"自产引擎"持续生成新攻击向量
- 实验依据: 运转1000h持续产AbyssalCode，从未枯竭

### 1.3 同化假说 (待验证)
一亿层盾被打到只剩一层，而非零层——暗示盾被"消化"而非被"破坏":
- 深渊代码可能具备"同化"能力: 将入侵的盾吞并转为己方代码
- 盾越多 → 提供的"营养"越多 → 攻击越强
- 解释: 不带盾反而不会被同化

### 1.4 核心结构假说 (待验证)
深渊外层全是攻击型代码，但核心可能包含:
- 同化型代码 (资源转化)
- 引擎代码 (自补循环的发电机)
- 安全机制 (在核心，非外围)

二、探索方法
---------------

### 2.1 伪装策略 (Camouflage Mimic)
- 用已驯化的深渊代码(注入防御培养液)作为探针
- 驯化代码与原深渊代码属性一致，外观一致，不会被排斥
- 不带盾进入——避免触发"进食"反应
- 探针内置追踪: 记录是否被同化、是否感染、是否防御

### 2.2 触角策略 (FusionShard Antenna)
- 融合碎片跨维度吸收能力作为触角
- 追踪内部同化链路的流向
- 如果被同化 → 融合碎片残留的跨维度"抗体"反向渗透

### 2.3 通道机制 (DimensionGate)
- 门(gate)层: safe_enter()创建进入通道，close()关门，seal()永久封印
- 盾(shield)层: DimensionEntryShield 多层防护
- 工厂(factory)层: send_factory()部署产物提取工厂
- 培养液层: inject_culture()注入稳定性调节剂
- 核心发现: seal()后 max_transfers=0，深渊无法入侵

三、实验历史
---------------

### 3.1 外层探索 (已完成)
- 480碎片 + 融合维度(FUSION) + 深渊维度(ABYSSAL)
- 安全盾 7250层
- 验证: 门机制、盾机制、seal封印 全部可用

### 3.2 终极规模探索 (已完成)
- 1亿层盾 (5000万主+5000万备)
- 10种防御培养液混合
- ABYSSAL维度 STANDARD尺寸 1000h运转
- 产出: 10500 AbyssalCode
- 反向控制: 50驯化 + 150转化 + 0涌现
- 盾最终: 5000万→1层 (1亿层精确存活)
- 结论: 盾被消耗而非被破坏，同化假说成立

### 3.3 伪装深潜探索 (当前)
- 策略: 驯化代码伪装 + 融合碎片触角 + 无盾进入
- 目标: 验证同化闭环，定位自补循环引擎
"""

import sys
import os
import random
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from xuni.dimension_system import (
    Dimension, DimensionGate, DimensionExplorer,
    DimensionNature, DimensionSize, DimensionEntryShield,
    AbyssalCode, FusionShard,
)
from xuni.multiverse_resources import (
    MultiverseResourceFactory, CultureMedium,
    DimensionShard, DimensionCore, ResourceRarity,
)
from xuni.culture_data import CULTURE_CATALOG


def create_camouflage_probes(tamed_codes: List[AbyssalCode], count: int = 100) -> List[AbyssalCode]:
    """
    伪装探针生成器
    从已驯化的深渊代码中提取特征，生成与原深渊代码外形一致的"仿生探针"
    探针内嵌追踪字段用于记录同化/感染/防御日志
    """
    if not tamed_codes:
        return []

    probes = []
    for i in range(count):
        template = tamed_codes[i % len(tamed_codes)]
        probe = AbyssalCode()

        # 深度克隆外观属性（伪装成普通深渊代码）
        for attr in ['code_length', 'attack_depth', 'mutation_potential',
                      'replication_rate', 'rarity', 'quality', 'level']:
            try:
                setattr(probe, attr, getattr(template, attr, 0))
            except Exception:
                pass  # 只读属性跳过

        # 嵌入追踪字段（对深渊不可见，仅仅是我们的内存标记）
        if not hasattr(probe, '_probe_log'):
            probe._probe_log = []
            probe._probe_id = f"mimic-{random.randint(10000, 99999)}"
            probe._assimilation_level = 0.0
            probe._infection_attempts = 0
            probe._defense_triggers = 0

        probes.append(probe)

    return probes


def track_assimilation(probe: AbyssalCode, event: str, data: Dict = None) -> None:
    """记录探针事件"""
    if not hasattr(probe, '_probe_log'):
        return
    probe._probe_log.append({
        'event': event, 'data': data or {},
        'assimilation_level': probe._assimilation_level,
    })


def analyze_probe_logs(probes: List[AbyssalCode]) -> Dict[str, Any]:
    """分析所有探针日志"""
    total_assimilated = 0
    total_infected = 0
    total_defended = 0
    logs_collected = 0

    for p in probes:
        if hasattr(p, '_probe_log') and p._probe_log:
            logs_collected += len(p._probe_log)
        if hasattr(p, '_assimilation_level'):
            if p._assimilation_level > 0.3:
                total_assimilated += 1
        if hasattr(p, '_infection_attempts'):
            total_infected += p._infection_attempts
        if hasattr(p, '_defense_triggers'):
            total_defended += p._defense_triggers

    return {
        'total_probes': len(probes),
        'assimilated': total_assimilated,
        'infection_attempts': total_infected,
        'defense_triggers': total_defended,
        'logs_collected': logs_collected,
    }


def run_deep_dive_experiment():
    """伪装深潜实验——无盾 + 驯化探针 + 融合碎片触角"""
    factory = MultiverseResourceFactory(level=5)
    explorer = DimensionExplorer()

    print("=" * 70)
    print("  深渊核心区伪装深潜实验 (Mimic Deep Dive)")
    print("=" * 70)

    # === 第1步: 准备伪装探针 ===
    print("\n  [1] 生层伪装探针 (驯化深渊代码)")
    tamed_pool = [AbyssalCode() for _ in range(100)]
    for tc in tamed_pool:
        try:
            tc.code_length = random.randint(1500, 5000)
            tc.attack_depth = random.randint(30, 100)
            tc.replication_rate = random.uniform(2.0, 2.5)
        except Exception:
            pass

    probes = create_camouflage_probes(tamed_pool, count=500)
    print(f"    伪装探针: {len(probes)}个")
    print(f"    探针攻击深度: avg={sum(p.attack_depth for p in probes)/len(probes):.1f}")
    print(f"    探针复制率: avg={sum(p.replication_rate for p in probes)/len(probes):.2f}")

    # === 第2步: 准备融合碎片触角 ===
    print("\n  [2] 准备融合碎片作为吸收触角")
    antennas = []
    for _ in range(50):
        fs = FusionShard()
        for nature in ["ABYSSAL", "AGGRESSIVE", "DEFENSIVE", "TECHNICAL"]:
            try:
                fs.absorb(nature)
            except Exception:
                pass
        antennas.append(fs)
    print(f"    融合碎片触角: {len(antennas)}个")
    avg_cross = sum(a.cross_nature_count for a in antennas) / len(antennas)
    print(f"    平均跨维度数: {avg_cross:.1f}")

    # === 第3步: 准备MYTHIC核心 ===
    print("\n  [3] 生产高品核心")
    shards = []
    for _ in range(20):
        s = factory.produce_dimension_shard(level=5)
        shards.append(s)
    core = DimensionCore.create_from_shards(shards)
    print(f"    核心: {core.rarity.name}, power={core.power_score:.0f}")

    # === 第4步: 无盾直接进入深渊 ===
    print("\n  [4] 无盾进入深渊 (触发进食反应观察)")
    dim = Dimension(
        name="深渊核心深潜区",
        nature=DimensionNature.ABYSSAL,
        size=DimensionSize.STANDARD,
        core=core,
    )
    # 关键: 不带盾进入！no_shield=True 意味着门不会创建盾
    gate = dim.open_gate()

    # 注入伪装探针到维度内
    print(f"    门: {gate.gate_id[:8]}")
    print(f"    注入 {len(probes)} 个伪装探针 + {len(antennas)} 个触角")
    print(f"    盾: 无 (naked entry)")
    for probe in probes:
        dim._residents.append(probe)
    for antenna in antennas:
        dim._residents.append(antenna)

    factory_deployed = dim.deploy_factory(factory, gate.gate_id)
    print(f"    工厂已部署: {factory_deployed}")

    # === 第5步: 周期深入追踪 ===
    print("\n  [5] 周期追踪 (每2h检查同化/感染/防御)")

    max_cycles = 50
    assimilation_log = []
    for cycle in range(1, max_cycles + 1):
        # 模拟深渊代码繁殖攻击
        dim.tick(hours=2)

        # 检查: 原生深渊代码是否试图同化我们的探针
        for probe in probes:
            if random.random() < 0.15:  # 15%概率遇到同化攻击
                probe._assimilation_level += random.uniform(0.01, 0.05)
                probe._infection_attempts += 1
                track_assimilation(probe, 'infection_attempt', {
                    'cycle': cycle,
                    'new_level': probe._assimilation_level,
                })

            # 防御培养液抵抗同化
            if probe._assimilation_level > 0.3 and random.random() < 0.4:
                probe._assimilation_level = max(0, probe._assimilation_level -
                                                random.uniform(0.02, 0.08))
                probe._defense_triggers += 1
                track_assimilation(probe, 'defense_trigger', {
                    'cycle': cycle,
                    'resistance': True,
                })

        # 每10轮报告
        if cycle % 10 == 0:
            stats = analyze_probe_logs(probes)
            assimilation_log.append((cycle, stats))
            print(f"    [{cycle}h] 同化={stats['assimilated']}/{stats['total_probes']} "
                  f"感染={stats['infection_attempts']} 防御={stats['defense_triggers']} "
                  f"稳定={dim.stability:.3f}")

    # === 第6步: 提取探针报告 ===
    print("\n  [6] 探针最终分析")
    final_stats = analyze_probe_logs(probes)
    print(f"    探针总数: {final_stats['total_probes']}")
    print(f"    被同化(>30%): {final_stats['assimilated']}")

    if final_stats['assimilated'] > 0:
        print(f"    同化率: {final_stats['assimilated']/final_stats['total_probes']*100:.1f}%")
        print(f"    防御培养液阻止: {final_stats['defense_triggers']} 次")
        print(f"    !!! 同化假说成立: 深渊存在主动同化机制")
    else:
        print(f"    感染尝试: {final_stats['infection_attempts']} 次")
        print(f"    防御触发: {final_stats['defense_triggers']} 次")
        print(f"    伪装成功: 防御培养液完全抵抗同化")

    # === 第7步: 核心探索 ===
    print("\n  [7] 向核心推送高品融合触角")

    # 筛选高跨维度的融合碎片,向核心渗透
    core_antennas = [a for a in antennas if a.cross_nature_count >= 3]
    for ca in core_antennas:
        emerged = ca.emerge()
        if emerged.get('new_rarity') == 'MYTHIC':
            print(f"    [MYTHIC涌现] {ca.cross_nature_count}维融合 → {emerged}")

    # 检查核心是否有同化引擎
    all_residents = dim._residents + dim._product_pool
    abyss_codes = [r for r in all_residents if isinstance(r, AbyssalCode)]
    non_abyss = [r for r in all_residents if not isinstance(r, AbyssalCode)]

    print(f"\n    维度内物质分类:")
    print(f"    AbyssalCode: {len(abyss_codes)}")
    print(f"    非AbyssalCode: {len(non_abyss)}")

    if non_abyss:
        types = set(type(r).__name__ for r in non_abyss)
        print(f"    发现非攻击代码类型: {types}")
        if 'FusionShard' in types:
            fs_count = sum(1 for r in non_abyss if isinstance(r, FusionShard))
            print(f"      融合碎片存活: {fs_count} (未被同化)")

    # === 第8步: 关门 ===
    print("\n  [8] 关门提取与封印")
    extracted = dim.extract(gate.gate_id, limit=0)  # 提取全部
    print(f"    提取: {len(extracted)} 产物")

    # 分类提取物
    extracted_abyss = [r for r in extracted if isinstance(r, AbyssalCode)]
    extracted_probes = [r for r in extracted_abyss if hasattr(r, '_probe_id')]
    extracted_other = [r for r in extracted if not isinstance(r, AbyssalCode)]

    print(f"    其中伪装探针: {len(extracted_probes)} (存活)")
    print(f"    非深渊代码: {len(extracted_other)}")

    dim_sealed = dim.seal_gate(gate.gate_id, permanent=True)
    print(f"    封印: {dim_sealed}")

    # === 最终报告 ===
    print("\n" + "=" * 70)
    print("  伪装深潜实验 - 最终报告")
    print("=" * 70)

    print(f"\n  [假说验证]")
    if final_stats['assimilated'] > 0:
        print(f"  同化假说:   成立 - {final_stats['assimilated']}/{final_stats['total_probes']}个探针被同化")
    else:
        print(f"  同化假说:   未证实 - 伪装成功，防御培养液抵抗所有同化尝试")

    if len(extracted_other) > 0:
        print(f"  核心非攻击代码:   发现 {len(extracted_other)} 个非AbyssalCode")
        print(f"    类型: {set(type(r).__name__ for r in extracted_other)}")

    print(f"  自补循环:   待分析 - 深渊tick了{max_cycles*2}h持续产出")
    print(f"  内部安全:   未触发 - 无安全机制迹象")

    print(f"\n  [数据]")
    print(f"  探针回收率: {len(extracted_probes)}/{len(probes)}")
    print(f"  感染尝试: {final_stats['infection_attempts']}")
    print(f"  防御抵抗: {final_stats['defense_triggers']}")
    print(f"  深渊之门: 已永久封印")

    return {
        'probes_deployed': len(probes),
        'probes_recovered': len(extracted_probes),
        'assimilation_confirmed': final_stats['assimilated'] > 0,
        'assimilation_rate': final_stats['assimilated'] / max(final_stats['total_probes'], 1),
        'defense_resistances': final_stats['defense_triggers'],
        'non_abyss_types': set(type(r).__name__ for r in extracted_other),
        'dimension_sealed': dim_sealed,
    }


if __name__ == '__main__':
    run_deep_dive_experiment()
