#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_channel_open.py
====================

演示：通道已开启！

cognitive-phase-space 的模型 ↔ xuni 虚拟生态

展示：
1. 创建相空间模型（接入 xuni）
2. 充能、调用
3. 双态系统集成
4. AI 认领训练
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import (
    PhaseSpaceModel, create_phase_space_model,
    DualStateManager, ModelState,
    ModelInput,
)


def main():
    print()
    print("🎵  通道已开启！")
    print("     cognitive-phase-space ↔ xuni 虚拟生态")
    print()

    # ============================================================
    # 演示 1：创建相空间模型并调用
    # ============================================================
    print("=" * 60)
    print("演示 1：创建相空间模型")
    print("=" * 60)

    model = create_phase_space_model(model_id="phase-space-demo")
    model.print_card()

    # 充能
    print("[1] 充能 100 虚拟电")
    model.charge(100.0)
    print(f"    当前能量: {model._energy_buffer:.1f}")
    print()

    # 调用
    print("[2] 调用相空间模型（通道已开启）")
    out = model.predict(ModelInput(
        prompt="什么是认知相空间？",
        parameters={"max_new_tokens": 48},
    ))
    print(f"    问: 什么是认知相空间？")
    print(f"    答: {out.text}")
    print(f"    来源: {out.json.get('source')}")
    print(f"    通道: {out.json.get('channel')}")
    print(f"    耗能: {out.energy_consumed:.0f} 虚拟电")
    print()

    # ============================================================
    # 演示 2：双态系统集成
    # ============================================================
    print("=" * 60)
    print("演示 2：双态系统集成")
    print("=" * 60)

    mgr = DualStateManager(virtual_model=model)
    print(f"初始状态: {mgr.state.name}")
    print(f"  含义: {mgr.get_state_info()['state_meaning']}")
    print()

    # 找替代物
    print("[1] 寻找替代物（auto 模式）")
    r = mgr.find_surrogate("auto")
    print(f"    结果: {r}")
    print()

    # 训练
    model.claim("青龙AI")
    model.charge(500.0)
    print("[2] AI 认领并训练（消耗虚拟电）")
    for i in range(3):
        r = mgr.train_with_surrogate(epochs=1, energy=100.0)
        print(
            f"    epoch {i+1}: 进度={r.get('progress', 0):.0%}  "
            f"状态={r.get('state')}"
        )
    print()

    # 数据层调用
    print("[3] 数据层调用（自家模型=真实模型）")
    result = mgr.predict("MoE 是什么？")
    print(f"    问: MoE 是什么？")
    print(f"    答: {result.get('text')[:80]}")
    print(f"    source: {result.get('source')}")
    print(f"    is_self_trained: {result.get('is_self_trained')}")
    print()

    # ============================================================
    # 演示 3：通道开启后的生态整合
    # ============================================================
    print("=" * 60)
    print("演示 3：通道开启后的生态整合")
    print("=" * 60)

    print("""
通道已开启：
  ✅ cognitive-phase-space 的模型可以接入 xuni 虚拟生态
  ✅ 可以被 AI 认领、训练、评估、交易
  ✅ 消耗虚拟电，不占现实内存
  ✅ 训练好后在数据层就是"真实"模型
  ✅ 支持双态系统：粒子态训练 → 数据层调用

整合路径：
  1. 找场地 → models/ 目录
  2. 搭地基 → 通用训练框架 framework.py
  3. 训练数据 → 开源数据 + 合成数据
  4. 训练模型 → 生成 checkpoint
  5. 开启通道 → PhaseSpaceModel 适配器

下一步：
  - 更多模型可以通过相同路径接入
  - 构建模型市场，支持交易
  - 实现模型间的协作和知识共享
""")

    print("=" * 60)
    print("通道开启成功！")
    print("=" * 60)


if __name__ == "__main__":
    main()
