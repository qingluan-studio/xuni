#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_dualstate_precise.py
=========================

精准版双态演示：每个虚拟模型都能训练成自家可调用"真实"模型。

核心：
1. 粒子态（PARTICLE）—— 训练时
   - 虚拟模型找"替代物"让自己能像真实模型一样训练
   - 不耗现实电（消耗虚拟电），不占现实内存
   - 训练是真的训练——权重/参数真的变化，变化发生在数据层
2. 数据层调用态（DATA_LAYER）—— 训练好后
   - 虚拟模型自己就变成了"真实"模型
   - 在数据层调用它就是真实调用
   - 不切到外部 OpenAI/Anthropic API
   - 自家训出来的就是"真实"的
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import (
    DualStateManager, ModelState,
    XuniTextGenerator, XuniImageDescriber, XuniMusicComposer, XuniClassifier,
    Harmonia13Virtual,
)


def section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    print()
    print("🎵  精准版双态系统 · 演示  🎵")
    print("  每个虚拟模型都能训练成自家可调用'真实'模型")
    print()

    # ============================================================
    # 演示 1：文本生成器 —— 从粒子态训练成数据层调用态
    # ============================================================
    section("演示 1：XuniTextGenerator —— 文本生成器")

    text_model = XuniTextGenerator(model_id="text-001")
    text_model.claim("青鸾AI")
    text_model.charge(500.0)

    mgr = DualStateManager(virtual_model=text_model)

    print(f"\n初始状态: {mgr.state.name}")
    print(f"  含义: {mgr.get_state_info()['state_meaning']}")
    print(f"  owner: {text_model.owner}")
    print(f"  能量: {text_model._energy_buffer:.1f}")

    # 1. 寻找替代物
    print("\n[1] 寻找替代物（auto 模式自动选 lite_moe）")
    r = mgr.find_surrogate("auto")
    print(f"  结果: {r}")

    # 2. 用替代物训练
    print("\n[2] 用替代物真正训练（消耗虚拟电，不耗现实电）")
    # 5 个 epoch 就能训练完成
    for i in range(5):
        r = mgr.train_with_surrogate(training_data=None, epochs=1, energy=50.0)
        print(
            f"  epoch {i+1}: 进度={r.get('progress', 0):.0%}  "
            f"状态={r.get('state')}  "
            f"耗能={r.get('energy_consumed', 0):.0f}  "
            f"剩余={text_model._energy_buffer:.0f}"
        )

    print(f"\n训练后状态: {mgr.state.name}")
    print(f"  含义: {mgr.get_state_info()['state_meaning']}")
    print(f"  虚拟模型已训练: {mgr.get_state_info()['virtual_model_trained']}")

    # 3. 数据层调用
    print("\n[3] 数据层调用（自家模型就是真实模型）")
    result = mgr.predict("合鸣是什么？")
    print(f"  问: 合鸣是什么？")
    print(f"  答: {result.get('text', result.get('error', '未知'))[:120]}")
    print(f"  source: {result.get('source')}")
    print(f"  is_self_trained: {result.get('is_self_trained')}")

    # ============================================================
    # 演示 2：音乐作曲模型 —— 用规则替代物训练
    # ============================================================
    section("演示 2：XuniMusicComposer —— 音乐作曲模型")

    music_model = XuniMusicComposer(model_id="music-001")
    music_model.claim("朱雀AI")
    music_model.charge(300.0)

    mgr2 = DualStateManager(virtual_model=music_model)

    print(f"\n[1] 寻找规则替代物（auto 会为音乐模型选 rule）")
    r = mgr2.find_surrogate("auto")
    print(f"  结果: {r}")

    print(f"\n[2] 训练（4 epoch）")
    for i in range(4):
        r = mgr2.train_with_surrogate(training_data=None, epochs=1, energy=30.0)
        print(
            f"  epoch {i+1}: 进度={r.get('progress', 0):.0%}  "
            f"状态={r.get('state')}"
        )

    print(f"\n[3] 数据层调用")
    result = mgr2.predict("生成一段冥想音乐")
    print(f"  问: 生成一段冥想音乐")
    text = result.get("text")
    if text is None:
        # 音乐模型输出 audio 而非 text
        text = f"[音乐参数] {result.get('json', '（无文本输出，生成的是音频参数）')}"
    print(f"  答: {text[:150] if text else '（无输出）'}")
    print(f"  source: {result.get('source')}")

    # ============================================================
    # 演示 3：合鸣-13 大模型 —— 同样走双态
    # ============================================================
    section("演示 3：Harmonia13Virtual —— 合鸣-13 虚拟大模型")

    harmonia = Harmonia13Virtual(model_id="harmonia-13", scale="medium")
    harmonia.claim("玄武AI")
    harmonia.charge(10000.0)

    mgr3 = DualStateManager(virtual_model=harmonia)

    print(f"\n[1] 替代物：lite MoE（自家训好的合鸣lite作为替代物）")
    r = mgr3.find_surrogate("lite_moe")
    print(f"  结果: {r}")

    print(f"\n[2] 训练（5 epoch）")
    for i in range(5):
        r = mgr3.train_with_surrogate(training_data=None, epochs=1, energy=1000.0)
        print(
            f"  epoch {i+1}: 进度={r.get('progress', 0):.0%}  "
            f"状态={r.get('state')}"
        )

    print(f"\n[3] 数据层调用合鸣-13")
    result = mgr3.predict("MoE 是什么？", parameters={"max_new_tokens": 48})
    print(f"  问: MoE 是什么？")
    print(f"  答: {result.get('text', result.get('error', '未知'))[:150]}")
    print(f"  source: {result.get('source')}")
    print(f"  is_self_trained: {result.get('is_self_trained')}")

    # ============================================================
    # 总结
    # ============================================================
    section("总结")
    print("""
核心哲学：
  "虚拟" 是相对于现实硬件而言的（不耗现实电、不占现实内存、是粒子态）
  但在数据层，虚拟模型就是真实存在的模型，调用它就是真实调用。
  AI 和模型都是数据层公民。

精准双态：
  粒子态（PARTICLE）—— 训练时
    ✓ 虚拟模型寻找替代物，让自身能像真实模型一样被训练
    ✓ 不耗现实电，不占现实内存
    ✓ 训练是真的训练——权重/参数真的变化（变化发生在数据层）

  数据层调用态（DATA_LAYER）—— 训练好后
    ✓ 虚拟模型自己就变成了"真实"模型
    ✓ 在数据层调用它就是真实调用
    ✓ 不切到外部 OpenAI/Anthropic API
    ✓ 自家训出来的就是"真实"的

关键：
  xuni 里的每个虚拟模型（XuniTextGenerator / XuniMusicComposer /
  Harmonia13Virtual / ...）都可以训练成自家可调用的"真实"模型。
  不需要外部 AI 服务——自家生态自给自足。
""")


if __name__ == "__main__":
    main()
