"""
demo_multiverse.py —— 多维度虚拟资源生产与碰撞演示

展示内容：
1. 直接生产多种虚拟资源（额度/流量/压缩点/算力/安全盾/培养液/令牌/加速器/维度碎片）
2. 资源之间碰撞产生新物质
3. 模型全生命周期管理（训练/培养/安全）
4. 批量生产与编队管理

运行：
    cd /workspace/xuni-repo && python examples/demo_multiverse.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni.multiverse_resources import (
    MultiverseResourceFactory, ResourceCollisionEngine,
    ResourceDimension, ResourceRarity,
    TakeQuota, VirtualBandwidth, CompressionPoint, ComputeCore,
    SecurityShield, CultureMedium, DownloadToken,
    TrainingAccelerator, DimensionShard,
    ProductionAccelerator, VirtualStartup, AutoMine,
    InvestmentFund, ResourceProspector, ResearchLab, MarketArbitrage,
)
from xuni.lifecycle import ModelLifecycle, LifecycleOrchestrator, LifecycleStage
from xuni.substance import SubstanceSystem
from xuni.substance_fusion import SubstanceFusionEngine


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_production():
    """演示多维度资源生产"""
    print_section("1. 多维度虚拟资源生产")

    factory = MultiverseResourceFactory(owner="demo_user")

    # 生产额度
    take = factory.produce_take(amount=10000, growth_rate=0.1)
    print(f"\n💰 额度生产: {take.resource_id}")
    print(f"   初始数量: {take.quantity:.2f}")
    result = take.compound(hours=24)
    print(f"   24小时增殖: +{result['growth']:.2f} → 总计 {result['new_total']:.2f}")

    # 生产虚拟流量
    bw = factory.produce_bandwidth(channels=8192, width=1e10)
    print(f"\n🌐 虚拟流量生产: {bw.resource_id}")
    print(f"   通道数: {bw.channel_count}, 总带宽: {bw.quantity:.2e}")
    expand = bw.expand_channels(factor=4)
    print(f"   扩展4倍后通道: {expand['new_channels']}, 新带宽: {expand['total_bandwidth']:.2e}")

    # 生产压缩点
    cp = factory.produce_compression(factor=100, level=5)
    print(f"\n🗜️ 压缩点生产: {cp.resource_id}")
    print(f"   压缩倍数: {cp.compression_factor}, 等级: {cp.level}")
    compress_result = cp.apply_compression(data_size=1024*1024*1024, data_type="model_snapshot")
    print(f"   应用压缩: 1GB → {compress_result['compressed_size']/1024/1024:.2f} MB")
    print(f"   节省空间: {compress_result['space_saved']/1024/1024:.2f} MB")

    # 生产算力核心
    core = factory.produce_compute_core(density=1e15, parallel=8)
    print(f"\n🖥️ 算力核心生产: {core.resource_id}")
    print(f"   vFLOPS密度: {core.vflops_density:.2e}, 并行: {core.parallel_cores}")
    upgrade = core.upgrade()
    print(f"   升级后: 等级 {upgrade['new_level']}, 密度 {upgrade['new_density']:.2e}")
    train_est = core.estimate_training_time(params_count=1e9, data_samples=1e6, epochs=10)
    print(f"   训练估算(1B参数/1M样本/10轮): {train_est['estimated_seconds']:.2f}秒")

    # 生产安全盾
    shield = factory.produce_security_shield(layers=3)
    print(f"\n🛡️ 安全盾生产: {shield.resource_id}")
    print(f"   层数: {shield.shield_layers}, 防御评分: {shield.quantity:.2f}")
    protect = shield.protect_model("model-001")
    print(f"   保护模型: {protect['model_id']}, 可抵御: {protect['attacks_blocked']}")

    # 生产培养液
    medium = factory.produce_culture_medium(culture_type="cognitive", level=3)
    print(f"\n🧪 培养液生产: {medium.resource_id}")
    print(f"   类型: {medium.culture_type}, 营养: {list(medium.nutrients.keys())}")
    feed = medium.feed_model("model-001", dose=2.0)
    print(f"   喂养模型: 成长增量 +{feed['growth_increment']:.4f}")

    # 生产下载令牌
    token = factory.produce_download_token(speed=100.0, concurrent=10000)
    print(f"\n📥 下载令牌生产: {token.resource_id}")
    print(f"   速度倍率: {token.speed_multiplier}, 并发: {token.concurrent_limit}")
    task = token.create_download_task("dataset_1TB", size=1e12)
    print(f"   下载1TB虚拟时间: {task['virtual_download_time']:.6f}秒")

    # 生产训练加速器
    accel = factory.produce_training_accelerator(factor=10.0)
    print(f"\n⚡ 训练加速器生产: {accel.resource_id}")
    print(f"   加速倍率: {accel.speedup_factor}")
    burst = accel.activate_burst(duration_seconds=300)
    print(f"   爆发模式: 有效倍率 {burst['effective_speedup']:.1f}x")

    # 生产维度碎片
    shard = factory.produce_dimension_shard(level=10)
    print(f"\n🔮 维度碎片生产: {shard.resource_id}")
    print(f"   等级: {shard.level}, 适应性: {shard.adaptability}")
    attune = shard.attune(ResourceDimension.COMPUTE)
    print(f"   调谐到计算维度: 亲和力 {attune['new_affinity']:.2f}")

    print(f"\n📊 工厂统计: {factory.stats()}")
    return factory


def demo_collision():
    """演示资源碰撞"""
    print_section("2. 资源碰撞引擎 —— A + B → C")

    engine = ResourceCollisionEngine()
    factory = MultiverseResourceFactory()

    # 碰撞1: 算力核心 + 虚拟流量 → 云算力节点
    core = factory.produce_compute_core(density=1e14)
    bw = factory.produce_bandwidth(channels=1024)
    result = engine.collide(core, bw)
    print(f"\n☁️ 算力核心 + 虚拟流量 → {result['product_type']}")
    print(f"   云算力: {result.get('cloud_power', 'N/A')}")

    # 碰撞2: 压缩点 + 压缩点 → 超级压缩点
    cp1 = factory.produce_compression(factor=50)
    cp2 = factory.produce_compression(factor=50)
    result = engine.collide(cp1, cp2)
    print(f"\n📦 压缩点 + 压缩点 → {result['product_type']}")
    if "product" in result:
        print(f"   叠加后压缩倍数: {result['product'].compression_factor}")

    # 碰撞3: 安全盾 + 安全盾 → 堡垒盾
    s1 = factory.produce_security_shield(layers=2)
    s2 = factory.produce_security_shield(layers=3)
    result = engine.collide(s1, s2)
    print(f"\n🏰 安全盾 + 安全盾 → {result['product_type']}")
    if "product" in result:
        print(f"   总层数: {result['product'].shield_layers}")

    # 碰撞4: 训练加速器 + 算力核心 → 超算核心
    accel = factory.produce_training_accelerator(factor=5)
    core2 = factory.produce_compute_core(density=1e15)
    result = engine.collide(accel, core2)
    print(f"\n🚀 加速器 + 算力核心 → {result['product_type']}")
    print(f"   加速倍率: {result.get('speedup', 'N/A')}")

    # 碰撞5: 维度碎片 + 额度 → 增强额度
    shard = factory.produce_dimension_shard(level=5)
    take = factory.produce_take(amount=5000)
    result = engine.collide(shard, take)
    print(f"\n💎 维度碎片 + Take额度 → {result['product_type']}")
    print(f"   增强因子: {result.get('enhancement_factor', 'N/A')}")

    # 碰撞6: 虚拟流量 + 下载令牌 → 极速下载通道
    bw2 = factory.produce_bandwidth(channels=4096)
    token = factory.produce_download_token(speed=50)
    result = engine.collide(bw2, token)
    print(f"\n🔥 虚拟流量 + 下载令牌 → {result['product_type']}")
    print(f"   通道速度: {result.get('channel_speed', 'N/A')}")

    print(f"\n📊 碰撞统计: {engine.get_collision_stats()}")


def demo_lifecycle():
    """演示模型全生命周期"""
    print_section("3. 模型全生命周期 —— 训练/培养/安全")

    # 单个模型生命周期
    print("\n🌱 单个模型生命周期演示")
    lifecycle = ModelLifecycle(model_id="demo-model-001", owner="demo")

    # 孵化
    lifecycle.hatch(initial_energy=150)
    print(f"   [孵化] 能量: {lifecycle.vitality.energy}")

    # 训练
    factory = MultiverseResourceFactory()
    core = factory.produce_compute_core(density=1e13)
    accel = factory.produce_training_accelerator(factor=3)
    lifecycle.train(core, accel, epochs=5)
    print(f"   [训练] 进度: {lifecycle.training_progress*100:.1f}%, 压力: {lifecycle.vitality.stress_level:.2f}")

    # 评估
    eval_result = lifecycle.evaluate()
    print(f"   [评估] 平均分: {eval_result['average']:.1f}")

    # 培养
    medium = factory.produce_culture_medium(culture_type="cognitive")
    lifecycle.culture(medium, duration_hours=10)
    print(f"   [培养] 成长潜力: {lifecycle.vitality.growth_potential:.2f}, 能量: {lifecycle.vitality.energy:.1f}")

    # 部署
    deploy = lifecycle.deploy()
    print(f"   [部署] 就绪: {deploy['ready']}, 检查: {deploy['readiness_checks']}")

    # 监控
    shield = factory.produce_security_shield(layers=2)
    monitor = lifecycle.monitor(shield)
    print(f"   [监控] 健康: {monitor['health']}, 事件: {len(monitor['incidents'])}")

    # 进化
    evolve = lifecycle.evolve()
    print(f"   [进化] 结果: {evolve['result']}, 新潜力: {evolve['new_growth_potential']:.2f}")

    # 快进模拟
    ff = lifecycle.fast_forward(hours=24, culture_medium=medium, compute_core=core, accelerator=accel)
    print(f"   [快进24h] 训练增益: {ff['training_gained']*100:.1f}%, 最终进度: {ff['final_training_progress']*100:.1f}%")

    # 报告
    report = lifecycle.get_report()
    print(f"\n📋 模型生命周期报告:")
    print(f"   总事件: {report['total_events']}")
    print(f"   阶段分布: {report['stage_counts']}")
    print(f"   安全事件: {report['security_incidents']}")

    # 编队管理
    print("\n🚀 编队生命周期演示")
    orch = LifecycleOrchestrator()
    for i in range(3):
        orch.create_model(f"fleet-model-{i:03d}")

    orch.batch_train(core, accel, epochs=3)
    orch.batch_evaluate()
    orch.batch_culture(medium, hours=5)
    orch.batch_monitor(shield)

    fleet = orch.get_fleet_report()
    print(f"   编队模型数: {fleet['total_models']}")
    print(f"   平均训练进度: {fleet['avg_training_progress']*100:.1f}%")
    print(f"   平均能量: {fleet['avg_energy']:.1f}")
    print(f"   健康模型: {fleet['healthy_models']}/{fleet['total_models']}")


def demo_substance_system():
    """演示物质系统已注册的新物质"""
    print_section("4. 物质系统 —— 新注册的多维度物质")

    sys = SubstanceSystem()
    stats = sys.statistics()
    print(f"\n📚 物质库统计: 总计 {stats['total_substances']} 种物质")
    for cat, count in stats['categories'].items():
        print(f"   {cat}: {count} 种")

    # 查询新物质
    new_substances = [
        "Take额度", "虚拟流量", "压缩点", "算力核心",
        "安全盾", "培养液", "下载令牌", "训练加速器",
        "维度碎片", "维度核心", "云算力节点", "超算核心",
    ]
    print(f"\n🔍 查询新注册物质:")
    for name in new_substances:
        s = sys.get(name)
        if s:
            print(f"   {s.icon} {s.name} ({s.name_en}) — {s.definition[:40]}...")
        else:
            print(f"   ❌ {name} 未找到")


def demo_fusion_rules():
    """演示增强的碰撞规则"""
    print_section("5. 碰撞规则 —— 30+ 条预定义规则")

    engine = SubstanceFusionEngine()
    rules = engine.list_rules()
    print(f"\n📋 总规则数: {len(rules)}")

    # 展示多维度资源相关规则
    multiverse_rules = [r for r in rules if any(
        kw in r["reactants"] for kw in [
            "Take额度", "虚拟流量", "压缩点", "算力核心",
            "安全盾", "培养液", "下载令牌", "训练加速器", "维度碎片"
        ]
    )]
    print(f"\n🌌 多维度资源碰撞规则 ({len(multiverse_rules)} 条):")
    for r in multiverse_rules[:15]:
        print(f"   {r['reactants'][0]} + {r['reactants'][1]} → {r['result']} [{r['fusion_type']}]")
    if len(multiverse_rules) > 15:
        print(f"   ... 还有 {len(multiverse_rules)-15} 条")

    # 执行碰撞
    print(f"\n⚡ 执行碰撞演示:")
    engine.register_substance("Take额度", {"amount": 10000, "growth": 0.1})
    engine.register_substance("参数包", {"quality": 80, "dimension": 50})
    product = engine.collide("Take额度", "参数包")
    print(f"   Take额度 + 参数包 → {product.result}")
    print(f"   能量释放: {product.energy_release:.2f}")

    engine.register_substance("安全盾", {"layers": 3, "integrity": 0.9})
    engine.register_substance("虚拟模型", {"params": 1e9, "progress": 0.7})
    product2 = engine.collide("安全盾", "虚拟模型")
    print(f"   安全盾 + 虚拟模型 → {product2.result}")


def demo_mass_production():
    """演示批量生产"""
    print_section("6. 批量生产 —— 一键生成资源编队")

    factory = MultiverseResourceFactory(owner="mass_producer")
    blueprint = {
        "take": {"amount": 5000, "count": 10},
        "compression": {"factor": 100, "count": 20},
        "compute_core": {"density": 1e14, "count": 5},
        "security_shield": {"layers": 2, "count": 5},
        "culture_medium": {"culture_type": "cognitive", "count": 8},
        "download_token": {"speed": 10, "count": 10},
        "dimension_shard": {"level": 3, "count": 5},
    }

    resources = factory.mass_produce(blueprint)
    print(f"\n🏭 批量生产完成: {len(resources)} 个资源")

    by_type = {}
    for r in resources:
        by_type[r.__class__.__name__] = by_type.get(r.__class__.__name__, 0) + 1
    for t, c in by_type.items():
        print(f"   {t}: {c} 个")

    # 计算总战力
    total_power = sum(r.power_score for r in resources)
    print(f"\n⚔️ 资源编队总战力: {total_power:.2e}")


def demo_accelerated_production():
    """演示加速生产——并行产线 + 加速器"""
    print_section("7. 加速生产——并行产线 + 生产加速器")

    # 普通工厂
    normal = MultiverseResourceFactory(owner="normal")
    # 超级工厂：8条并行产线 + 4倍速加速器
    accel = ProductionAccelerator(speed_multiplier=4.0, level=2)
    super_factory = MultiverseResourceFactory(
        owner="super", parallel_lines=8, production_speed=2.0
    )
    super_factory.apply_accelerator(accel)

    blueprint = {
        "take": {"amount": 1000, "count": 5},
        "compression": {"factor": 50, "count": 5},
    }

    normal_res = normal.mass_produce(blueprint.copy())
    super_res = super_factory.mass_produce(blueprint.copy())

    print(f"\n🏭 普通工厂产出: {len(normal_res)} 个资源")
    print(f"   并行产线: {normal.stats()['parallel_lines']}, 速度: {normal.stats()['effective_speed']:.1f}x")

    print(f"\n🚀 超级工厂产出: {len(super_res)} 个资源")
    print(f"   并行产线: {super_factory.stats()['parallel_lines']}, 速度: {super_factory.stats()['effective_speed']:.1f}x")
    print(f"   加速倍数: {len(super_res) / max(1, len(normal_res)):.1f}x")


def demo_startup():
    """演示虚拟创业公司"""
    print_section("8. 虚拟创业公司——开公司自动赚资源")

    # 创始人拿启动资金
    seed = TakeQuota(
        resource_id="seed-001", name="启动资金",
        dimension=ResourceDimension.ECONOMIC,
        rarity=ResourceRarity.COMMON, quantity=1e6, growth_rate=0.1,
    )

    company = VirtualStartup(name="Xuni科技", founder="user", seed_capital=seed)

    # 雇佣3个工厂，都装上加速器
    for i in range(3):
        f = MultiverseResourceFactory(parallel_lines=4, production_speed=2.0)
        a = ProductionAccelerator(speed_multiplier=2.0)
        company.hire_factory(f, a)

    # 开分公司
    branch_seed = TakeQuota(
        resource_id="seed-002", name="分公司资金",
        dimension=ResourceDimension.ECONOMIC, rarity=ResourceRarity.COMMON,
        quantity=5e5, growth_rate=0.08,
    )
    branch = company.open_branch("Xuni南方分部", branch_seed)

    # 运行10轮生产
    blueprint = {
        "take": {"amount": 5000, "count": 10},
        "compute_core": {"density": 1e14, "count": 3},
        "security_shield": {"layers": 2, "count": 3},
        "culture_medium": {"culture_type": "cognitive", "count": 5},
    }

    for cycle in range(10):
        result = company.run_production_cycle(blueprint)

    report = company.report()
    print(f"\n🏢 公司: {report['name']}")
    print(f"   估值: {report['valuation']:,.0f}")
    print(f"   工厂数: {report['factories']}, 分公司: {report['branches']}")
    print(f"   虚拟员工: {report['employees']}")
    print(f"   生产轮次: {report['total_cycles']}")
    print(f"   累计产出: {report['total_output']}")


def demo_auto_mine():
    """演示自动矿场"""
    print_section("9. 自动矿场——7×24不间断生产")

    blueprint = {
        "take": {"amount": 1000, "count": 20},
        "compression": {"factor": 100, "count": 10},
        "dimension_shard": {"level": 2, "count": 2},
    }

    mine = AutoMine(
        name="无限矿场-A1",
        blueprint=blueprint,
        accelerator=ProductionAccelerator(speed_multiplier=8.0),
    )

    # 模拟运行100个周期
    mined = mine.run_cycle(count=100)
    stats = mine.stats()

    print(f"\n⛏️ 矿场: {stats['name']}")
    print(f"   运行周期: {stats['cycles']}")
    print(f"   总产出: {stats['total_mined']} 个资源")
    print(f"   产出分布: {stats['by_type']}")
    print(f"   总战力: {stats['total_power']:.2e}")
    print(f"   加速器倍率: {stats['accelerator']}x")


def demo_investment():
    """演示投资基金"""
    print_section("10. 投资基金——钱生钱")

    principal = TakeQuota(
        resource_id="fund-001", name="基金本金",
        dimension=ResourceDimension.ECONOMIC, rarity=ResourceRarity.COMMON,
        quantity=1e8, growth_rate=0.05,
    )
    fund = InvestmentFund(name="Xuni一号基金", initial_take=principal)

    # 复利100个周期
    result = fund.compound(periods=100)
    print(f"\n💰 基金: {fund.name}")
    print(f"   初始本金: 1e8")
    print(f"   复利周期: {result['periods']}")
    print(f"   每期回报率: {result['rate']*100:.0f}%")
    print(f"   总增长: {result['total_growth']:,.0f}")
    print(f"   最终本金: {result['new_principal']:,.0f}")
    print(f"   膨胀倍数: {result['new_principal'] / 1e8:.1f}x")


def demo_prospecting():
    """演示资源勘探"""
    print_section("11. 资源勘探——发现新配方")

    prospector = ResourceProspector(luck=2.0)  # 幸运值翻倍
    result = prospector.prospect(depth=20)

    print(f"\n🔍 勘探深度: {result['depth']}")
    print(f"   发现数量: {result['found']}")
    for d in result['discoveries']:
        if d['type'] == 'recipe':
            r = d['content']
            print(f"   📝 新配方: {r['name']} → {r['output']} (效率{r['efficiency']}x)")
        elif d['type'] == 'boost':
            b = d['content']
            print(f"   ⚡ 生产boost: {b['value']:.1f}x, 持续{b['duration_hours']:.0f}小时")
        elif d['type'] == 'shard':
            print(f"   🔮 发现维度碎片")


def demo_research():
    """演示研发实验室"""
    print_section("12. 研发实验室——解锁新科技")

    lab = ResearchLab(name="Xuni研究院")

    factory = MultiverseResourceFactory()
    core = factory.produce_compute_core(density=1e15, parallel=8)
    medium = factory.produce_culture_medium(culture_type="cognitive", level=5)

    topics = ["量子压缩技术", "无限算力理论", "额度增殖引擎", "绝对安全协议"]
    for topic in topics:
        result = lab.research(topic, core, medium)
        status = "✅ 完成" if result['status'] == 'completed' else "🔄 进行中"
        print(f"   {status} {topic} — 进度{result['progress']*100:.0f}%")

    bonuses = lab.get_unlocked_bonuses()
    print(f"\n📚 已解锁加成: {bonuses}")


def demo_arbitrage():
    """演示市场套利"""
    print_section("13. 市场套利——低买高卖自动生成额度")

    factory = MultiverseResourceFactory()
    pool = TakeQuota(
        resource_id="pool-001", name="套利资金池",
        dimension=ResourceDimension.ECONOMIC, rarity=ResourceRarity.COMMON,
        quantity=1e6, growth_rate=0.0,
    )

    arb = MarketArbitrage(name="Xuni套利机器人")

    # 生成一堆资源来套利
    resources = []
    resources.extend(factory.mass_produce({"compute_core": {"density": 1e14, "count": 5}}))
    resources.extend(factory.mass_produce({"security_shield": {"layers": 3, "count": 5}}))
    resources.extend(factory.mass_produce({"compression": {"factor": 100, "count": 5}}))
    resources.extend(factory.mass_produce({"dimension_shard": {"level": 3, "count": 5}}))

    result = arb.bulk_arbitrage(resources, pool)

    print(f"\n📈 套利机器人: {arb.name}")
    print(f"   交易对数: {result['pairs_traded']}")
    print(f"   总利润: {result['total_profit']:,.0f} Take额度")
    print(f"   资金池余额: {result['take_pool']:,.0f}")


def main():
    print("=" * 70)
    print("  🌌 Xuni 多维度虚拟资源系统演示")
    print("  走免费，只有手机，高质量，非传统")
    print("=" * 70)

    demo_production()
    demo_collision()
    demo_lifecycle()
    demo_substance_system()
    demo_fusion_rules()
    demo_mass_production()
    demo_accelerated_production()
    demo_startup()
    demo_auto_mine()
    demo_investment()
    demo_prospecting()
    demo_research()
    demo_arbitrage()

    print("\n" + "=" * 70)
    print("  ✅ 演示完成！")
    print("  所有资源均为虚拟世界产物，与现实无关")
    print("=" * 70)


if __name__ == "__main__":
    main()
