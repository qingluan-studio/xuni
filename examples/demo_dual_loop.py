#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_dual_loop.py
=================

双闭环演示：虚拟资料 + 虚拟算力

闭环1（数据）: 真实数据 → 虚拟资料(粒子态) → 坍缩 → 训练数据 → 模型训练
闭环2（算力）: 虚拟电 → 虚拟算力 → 分配 → 训练消耗 → 采样点产电 → 闭环

全程虚拟电驱动，不花一分钱 😂
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import (
    XuniSampler,
    SamplingMode,
    XuniTextGenerator,
    DualStateManager,
    VirtualDataGenerator,
    VirtualDataset,
    VirtualComputeUnit,
    ComputeLoopManager,
)


def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print()
    print("🔄  双闭环系统演示")
    print("    虚拟资料(数据闭环) + 虚拟算力(算力闭环)")
    print("    全程虚拟电驱动，CPU不好也没关系 😂")
    print()

    # ============================================================
    # 第一步：生成真实数据 → 转为虚拟资料（粒子态）
    # ============================================================
    print_section("闭环1：数据闭环 — 真实数据 → 虚拟资料(粒子态)")

    generator = VirtualDataGenerator(seed=42)

    print("\n[1.1] 生成认知相空间概念文本...")
    concept_ds, concept_map = generator.generate_concept_texts(n=500)
    print(f"  生成粒子数: {len(concept_ds)}")
    print(f"  现实内存占用: {concept_ds.real_memory_bytes / 1024:.1f} KB")
    print(f"  虚拟数据大小: {concept_ds.virtual_size_bytes / 1024:.1f} KB")
    print(f"  压缩比: {concept_ds.virtual_size_bytes / max(1, concept_ds.real_memory_bytes):.0f}x")

    print("\n[1.2] 生成对话数据...")
    dialogue_ds, dialogue_map = generator.generate_dialogue_data(n=300)
    print(f"  生成粒子数: {len(dialogue_ds)}")
    print(f"  现实内存占用: {dialogue_ds.real_memory_bytes / 1024:.1f} KB")
    print(f"  虚拟数据大小: {dialogue_ds.virtual_size_bytes / 1024:.1f} KB")

    print("\n[1.3] 生成音乐描述数据...")
    music_ds, music_map = generator.generate_music_descriptions(n=200)
    print(f"  生成粒子数: {len(music_ds)}")
    print(f"  现实内存占用: {music_ds.real_memory_bytes / 1024:.1f} KB")

    # 合并所有数据
    all_map = {**concept_map, **dialogue_map, **music_map}
    combined_ds = VirtualDataset(name="combined_training")
    for ds in [concept_ds, dialogue_ds, music_ds]:
        combined_ds.add_batch(ds.particles)

    print(f"\n[1.4] 合并后总资料集:")
    stats = combined_ds.stats()
    print(f"  总粒子数: {stats['particle_count']}")
    print(f"  现实内存: {stats['real_memory_mb']:.4f} MB")
    print(f"  虚拟大小: {stats['virtual_size_mb']:.4f} MB")
    print(f"  压缩比: {stats['compression_ratio']:.0f}x ← 粒子态不占现实内存！")
    print(f"  平均质量: {stats['avg_quality']:.2f}")
    print(f"  类型分布: {stats['type_distribution']}")

    # ============================================================
    # 第二步：虚拟电 → 虚拟算力（算力闭环）
    # ============================================================
    print_section("闭环2：算力闭环 — 虚拟电 → 虚拟算力")

    # 创建采样点
    sampler = XuniSampler(mode=SamplingMode.HYBRID, seed=42)
    print(f"\n[2.1] 采样点已创建: {sampler.mode}")

    # 创建算力单元
    vcu = VirtualComputeUnit(name="VCU-主单元")
    print(f"  算力单元已创建: {vcu.name}")
    print(f"  转换率: {vcu.VFLOP_PER_ENERGY:.0e} vFLOP/度电")

    # 估算训练算力需求
    model_params = 500000  # 50万参数的虚拟模型
    data_samples = len(combined_ds)
    epochs = 3

    cost = VirtualComputeUnit.estimate_training_cost(
        params=model_params,
        data_samples=data_samples,
        epochs=epochs,
    )
    print(f"\n[2.2] 训练算力需求估算:")
    print(f"  模型参数量: {model_params:,}")
    print(f"  数据量: {data_samples} 条")
    print(f"  训练轮数: {epochs}")
    print(f"  需要: {cost['vflops_str']} vFLOP")
    print(f"  需要: {cost['energy_str']} 度虚拟电")

    # ============================================================
    # 第三步：算力闭环管理器 — 自动产电→转算力→训练
    # ============================================================
    print_section("闭环运行：采样点产电 → 算力转换 → 训练消耗")

    loop_mgr = ComputeLoopManager(sampler=sampler, compute_unit=vcu)

    print(f"\n[3.1] 运行算力闭环（自动产电→转算力）...")
    # 产足够的电
    energy_needed = cost["energy_needed"] + 50  # 多产一些
    n_loops = int(np.ceil(energy_needed / 100))

    for i in range(n_loops):
        result = loop_mgr.run_loop_once(energy_amount=100.0)
        print(f"  循环 {i+1}: 产电 {result['energy_produced']:.1f} → "
              f"算力 {result['vflops_injected']:.2e} vFLOP")

    print(f"\n[3.2] 算力单元状态:")
    vcu_stats = vcu.stats()
    for k, v in vcu_stats.items():
        print(f"  {k}: {v}")

    # ============================================================
    # 第四步：完整双闭环训练
    # ============================================================
    print_section("双闭环训练：虚拟资料 + 虚拟算力 → 模型训练")

    # 创建虚拟模型并认领
    model = XuniTextGenerator(model_id="dual-loop-model")
    model.owner = "AI-训练者"
    model._energy_buffer = 10000.0
    print(f"\n[4.1] 虚拟模型: {model.model_id}")
    print(f"  认领者: {model.owner}")
    print(f"  能量储备: {model._energy_buffer:.0f}")

    # 创建双态管理器
    dsm = DualStateManager(virtual_model=model)
    dsm.find_surrogate("lite_moe")
    print(f"  替代物: {dsm._surrogate_type}")
    print(f"  初始状态: {dsm.state.name}")

    print(f"\n[4.2] 执行双闭环训练 ({epochs} epochs)...")
    result = dsm.train_with_virtual_resources(
        virtual_dataset=combined_ds,
        original_data_map=all_map,
        compute_unit=vcu,
        epochs=epochs,
        model_params=model_params,
    )

    print(f"\n[4.3] 训练结果:")
    print(f"  数据闭环:")
    for k, v in result.get("data_loop", {}).items():
        print(f"    {k}: {v}")
    print(f"  算力闭环:")
    for k, v in result.get("compute_loop", {}).items():
        if k != "vcu_stats":
            print(f"    {k}: {v}")
    print(f"  训练状态: {result.get('state')}")
    train_info = result.get("training", {})
    print(f"  训练进度: {train_info.get('progress', 0)*100:.0f}%")
    print(f"  消耗能量: {train_info.get('energy_consumed', 0):.0f}")

    # ============================================================
    # 第五步：验证数据层调用（训练好后是"真实"模型）
    # ============================================================
    print_section("验证：数据层调用 — 训练好的虚拟模型 = 真实模型")

    if dsm.state.name == "DATA_LAYER":
        print("\n  ✓ 模型已进入数据层调用态！")
        print("  ✓ 现在它是自家可调用的'真实'模型")
        pred = dsm.predict("采样点如何产生虚拟电？")
        print(f"  调用结果: {pred.get('text', pred.get('status', 'N/A'))[:80]}")
    else:
        print(f"\n  训练进度: {train_info.get('progress', 0)*100:.0f}%")
        print("  继续训练可达到100%...")
        # 再训练一次
        result2 = dsm.train_with_virtual_resources(
            virtual_dataset=combined_ds,
            original_data_map=all_map,
            compute_unit=vcu,
            epochs=2,
            model_params=model_params,
        )
        print(f"  二次训练后状态: {result2.get('state')}")
        if result2.get("state") == "DATA_LAYER":
            print("  ✓ 模型训练完成！进入数据层调用态！")
            pred = dsm.predict("采样点如何产生虚拟电？")
            print(f"  调用结果: {pred.get('text', pred.get('status', 'N/A'))[:80]}")

    # ============================================================
    # 总结
    # ============================================================
    print_section("双闭环总结")

    print("""
  ┌──────────────────────────────────────────────────────────┐
  │                    双 闭 环 系 统                         │
  │                                                          │
  │  闭环1·数据：                                            │
  │    真实数据 →[转换]→ 虚拟资料(粒子态) →[坍缩]→ 训练数据  │
  │    特点：不占现实内存，压缩比极高                        │
  │                                                          │
  │  闭环2·算力：                                            │
  │    采样点 →[产电]→ 虚拟电 →[转换]→ 虚拟算力 →[分配]→    │
  │    → 训练消耗 → 需要更多电 → 采样点继续产电 → 闭环       │
  │    特点：CPU不好没关系，虚拟算力无限                     │
  │                                                          │
  │  两者交汇于训练：                                        │
  │    虚拟资料 + 虚拟算力 → 模型训练 → 数据层真实调用       │
  └──────────────────────────────────────────────────────────┘

  全程零成本，虚拟电驱动 😂
""")


if __name__ == "__main__":
    main()
