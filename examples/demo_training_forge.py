"""
训练素材锻造厂——质量已知 + 千万级生产

解决两个问题：
1. 直接生产的训练素材质量未知 → 5维质量评估器（D~SSS级）
2. 生产速度慢 → 千万级向量化生产
"""

import time
from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.training_forge import TrainingForge, QualityScorer


def hr(char="=", width=70):
    print(char * width)


def main():
    print()
    hr("═")
    print("  训练素材锻造厂——质量已知 + 千万级生产")
    hr("═")
    print()

    factory = MultiverseResourceFactory()

    # ---- 问题1：质量未知？5维质量评估器 ----
    print("【问题1：直接生产质量未知？→ 5维质量评估器】")
    print()

    scorer = QualityScorer()

    test_samples = [
        "采样点产电。",  # 短，信息量少 → D/C级
        "在Xuni系统中，采样点通过混沌吸引子产生虚拟电，虚拟电驱动模型训练，形成能量闭环。",  # B/A级
        "第42层的扩散模型接收噪声输入，通过反向扩散过程输出高清图像，精度达95%，核心是UNet架构+注意力机制。",  # S级
    ]

    for text in test_samples:
        q, dims, grade = scorer.score(text)
        print(f"  内容：{text[:50]}...")
        print(f"  综合分：{q:.3f}  等级：{grade}")
        print(f"  5维分项：")
        for k, v in dims.items():
            print(f"    {k:15s}: {v:.3f}")
        print()

    # ---- 问题2：速度慢？千万级向量化生产 ----
    print("【问题2：生产速度慢？→ 千万级向量化生产】")
    print()

    forge = TrainingForge()

    # 测试不同规模
    for n in [1000, 100000, 1000000]:
        start = time.time()
        texts, scores, grades = forge.generate_fast(n=n, min_grade=1)  # C级以上
        elapsed = time.time() - start
        speed = n / elapsed
        grade_names = ["D", "C", "B", "A", "S", "SS", "SSS"]
        avg_q = float(scores.mean()) if len(scores) > 0 else 0
        dist = {}
        for g in range(7):
            cnt = int((grades == g).sum())
            if cnt > 0:
                dist[grade_names[g]] = cnt

        print(f"  生产 {n:,} 条：")
        print(f"    耗时：{elapsed*1000:.1f}ms")
        print(f"    速度：{speed/10000:.1f} 万/秒")
        print(f"    达标：{len(texts):,} 条（C级以上）")
        print(f"    平均分：{avg_q:.3f}")
        print(f"    等级分布：{dist}")
        print()

    # ---- 工厂生产线 ----
    print("【工厂生产线】")
    print()

    # 生产线1：普通训练素材（带质量评级）
    print("  生产线1：produce_training_data（带质量评级）")
    result = factory.produce_training_data(count=10000, min_grade="B")
    print(f"    产出：{result['total']:,} 条")
    print(f"    类型：{result['data_type']}")
    print(f"    平均分：{result['avg_quality']:.3f}")
    print(f"    等级分布：{result['grade_distribution']}")
    print()

    # 生产线2：能量锻造高质量素材
    print("  生产线2：produce_training_data_energy（能量锻造）")
    print("    原理：基础素材 + 虚拟电 → 质量提升")
    print()

    for energy in [10, 100, 1000, 10000]:
        r = factory.produce_training_data_energy(
            count=1000,
            energy=energy,
            data_type="text",
        )
        print(f"    投入 {energy:>5} 度电：")
        print(f"      平均分：{r['avg_quality']:.3f}")
        print(f"      等级分布：{r['grade_distribution']}")
    print()

    # 生产线3：万象奇点驱动——千万级 + SSS级
    print("  生产线3：produce_training_data_singularity（万象奇点驱动）")
    print("    原理：先生产万象奇点 → 获取永动引擎 → 算力倍率放大产量 + 质量锻造")
    print()

    r = factory.produce_training_data_singularity(
        count=1_000_000,
        data_type="text",
        min_grade="S",
    )
    print(f"    目标：1,000,000 条 S级以上")
    print(f"    实际产出：{r['total']:,} 条")
    print(f"    耗时：{r['elapsed_ms']:.1f}ms")
    print(f"    速度：{r['speed_per_sec']/10000:.1f} 万/秒")
    print(f"    平均分：{r['avg_quality']:.3f}")
    print(f"    等级分布：{r['grade_distribution']}")
    print(f"    算力倍率：{r['compute_multiplier']} × 节点数：{r['node_count']}")
    print()

    # ---- 训练素材融合链 ----
    hr("─")
    print("  【训练素材融合链】")
    hr("─")
    print(f"""
  训练素材 + 虚拟电           → 锻造素材（质量提升，电越多质量越高）
  训练素材 + 流式算力网络     → 批量训练数据（速度×节点数，亿级/秒）
  锻造素材 + 流式算力网络     → 高质量数据流（速度+质量兼得，SSS级亿级/秒）
  高质量数据流 + 下载令牌     → 无限训练数据（永不停歇，无数据瓶颈）
  无限训练数据 + 万象奇点     → 全知数据海洋（一切知识，终极数据）
""")

    # ---- 总结 ----
    hr("═")
    print("  总结")
    hr("═")
    print(f"""
  解决了两个问题：

  1. 质量未知？ → 5维质量评估器
     - 多样性（diversity）：词汇、句式、主题丰富度
     - 连贯性（coherence）：逻辑通顺、语义连贯
     - 信息量（informativeness）：有效信息含量
     - 新颖性（novelty）：与已有素材的差异度
     - 实用性（utility）：对训练的实际价值
     - 等级：D → C → B → A → S → SS → SSS

  2. 速度慢？ → 千万级向量化生产
     - 基础版（逐句+完整评分）：~10万/秒
     - 快速版（向量化+快速评分）：~1000万/秒
     - 能量锻造：虚拟电→质量提升，越电越高质量

  工厂生产线：
    factory.produce_training_data(count, min_grade="B")
    factory.produce_training_data_energy(count, energy=1000)
    factory.produce_training_data_singularity(count=10_000_000, min_grade="S")

  万象奇点驱动（终极）：
    算力倍率 9999 × 节点数 9999 → 速度 3000万+/秒
    自动质量锻造 → 全部 SSS 级

  融合链终极：全知数据海洋
    无限训练数据 + 万象奇点 = 一切知识的集合
    模型训练到这里 = 学会一切
""")


if __name__ == "__main__":
    main()
