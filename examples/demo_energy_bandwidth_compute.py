"""
电 + 算力 + 流量融合链演示——做出来用

回答用户问题：
  虚拟电大了算力才快（电驱动算力）
  虚拟流量有现实网络特征（带宽=节点连通度）
  流量 + (电+算力融合) = 流式算力网络（分布式无限算力）

本演示把融合产物接入 PerpetualTrainingEngine，真正加速训练，
对比不同电/流量/融合组合下的训练速度。
"""

import time
from xuni.substance_fusion import create_default_engine
from xuni.perpetual_engine import PerpetualTrainingEngine
from xuni.model import XuniModel, ModelType, ModelCapability


def make_model(mid: str):
    """造一个虚拟文本模型"""
    m = XuniModel(mid, ModelType.TEXT_GENERATOR, [ModelCapability.TEXT_OUTPUT], 10.0)
    m._energy_buffer = 1e6
    return m


def bench(engine, model, epochs=100, label=""):
    """基准测试：看训练速度"""
    start_progress = getattr(model, "training_progress", 0.0)
    t0 = time.time()
    total_gain = 0.0
    steps = 0
    for _ in range(epochs):
        r = engine.train_step(model, epochs=1, energy_per_epoch=1.0)
        if "error" in r:
            break
        total_gain += r["progress_gain"]
        steps += 1
    elapsed = time.time() - t0
    print(f"\n  【{label}】")
    print(f"    训练步数: {steps}")
    print(f"    进度增量: {start_progress:.4f} → {model.training_progress:.4f} (Δ{total_gain:.4f})")
    print(f"    耗时: {elapsed*1000:.1f}ms")
    print(f"    节点数: {engine.node_count}")
    print(f"    总算力: {engine.total_vflops:.2e} vFLOP")
    print(f"    有效速度: {engine.effective_speed:.2e}")
    print(f"    电量: {engine.energy:.2f} (永动: {engine.is_perpetual})")
    print(f"    加成: {engine._boost_names}")


def main():
    print("=" * 70)
    print("电 + 算力 + 流量 融合链——做出来用")
    print("=" * 70)

    # ---- 1. 融合链展示 ----
    print("\n【融合链】")
    engine_f = create_default_engine()

    p1 = engine_f.fuse("虚拟电", "虚拟算力")
    print(f"  虚拟电 + 虚拟算力 = {p1.result}")
    eff = p1.metadata.get("emergent_effect") or {}
    print(f"    效果: {eff.get('效果', '')}")

    p2 = engine_f.collide("虚拟流量", p1.result)
    print(f"\n  虚拟流量 + {p1.result} = {p2.result}")
    eff = p2.metadata.get("emergent_effect") or {}
    print(f"    效果: {eff.get('效果', '')}")
    print(f"    公式: {eff.get('公式', '')}")
    print(f"    现实类比: {eff.get('现实类比', '')}")

    p3 = engine_f.fuse(p2.result, "下载令牌")
    print(f"\n  {p2.result} + 下载令牌 = {p3.result}")
    eff = p3.metadata.get("emergent_effect") or {}
    print(f"    效果: {eff.get('效果', '')}")

    # ---- 2. 实际训练对比 ----
    print("\n" + "=" * 70)
    print("实际训练对比——电/流量/融合如何影响训练速度")
    print("=" * 70)

    # 场景A：低电 + 单节点（基线）
    print("\n--- 场景A：低电(100) + 单通道 ---")
    eA = PerpetualTrainingEngine()
    eA.inject_energy(100)
    eA.set_bandwidth(1)
    bench(eA, make_model("A"), epochs=50, label="基线")

    # 场景B：高电 + 单节点（电大了算力才快）
    print("\n--- 场景B：高电(1万) + 单通道（电大了算力才快）---")
    eB = PerpetualTrainingEngine()
    eB.inject_energy(10000)
    eB.set_bandwidth(1)
    bench(eB, make_model("B"), epochs=50, label="高电")

    # 场景C：低电 + 多通道（流量扩展节点数）
    print("\n--- 场景C：低电(100) + 2048通道（流量扩展节点）---")
    eC = PerpetualTrainingEngine()
    eC.inject_energy(100)
    eC.set_bandwidth(2048)
    bench(eC, make_model("C"), epochs=50, label="多通道")

    # 场景D：高电 + 多通道（电+流量协同）
    print("\n--- 场景D：高电(1万) + 2048通道（电+流量协同）---")
    eD = PerpetualTrainingEngine()
    eD.inject_energy(10000)
    eD.set_bandwidth(2048)
    bench(eD, make_model("D"), epochs=50, label="电+流量协同")

    # 场景E：接入"能量算力核心"融合
    print("\n--- 场景E：高电 + 多通道 + 能量算力核心（电驱动算力自循环）---")
    eE = PerpetualTrainingEngine()
    eE.inject_energy(10000)
    eE.set_bandwidth(2048)
    eE.apply_fusion("能量算力核心")
    bench(eE, make_model("E"), epochs=50, label="+能量算力核心")

    # 场景F：接入"流式算力网络"融合
    print("\n--- 场景F：高电 + 多通道 + 流式算力网络（分布式无限算力）---")
    eF = PerpetualTrainingEngine()
    eF.inject_energy(10000)
    eF.set_bandwidth(2048)
    eF.apply_fusion("能量算力核心")
    eF.apply_fusion("流式算力网络")
    bench(eF, make_model("F"), epochs=50, label="+流式算力网络")

    # 场景G：永动模式（无限训练永动机）
    print("\n--- 场景G：永动模式（电不消耗反再生）---")
    eG = PerpetualTrainingEngine()
    eG.inject_energy(10000)
    eG.set_bandwidth(2048)
    eG.apply_fusion("能量算力核心")
    eG.apply_fusion("流式算力网络")
    eG.apply_fusion("流式计算引擎")
    eG.apply_fusion("永动下载涡轮")
    eG.apply_fusion("无限训练永动机")
    bench(eG, make_model("G"), epochs=50, label="永动模式")

    # ---- 3. 总结 ----
    print("\n" + "=" * 70)
    print("总结：电、流量、融合如何加速训练")
    print("=" * 70)
    print(f"""
  公式：
    单节点算力 = 虚拟电 × 转换率 × 算力倍率    （电大了算力才快）
    节点数     = 流量通道数 × 节点倍率          （流量=网络连通度）
    总算力     = 节点数 × 单节点算力             （流式算力网络）
    训练速度   ∝ 总算力 × 加速器倍率

  对比（有效训练速度）:
    A 基线(低电单通道):       {eA.effective_speed:.2e}
    B 高电单通道:             {eB.effective_speed:.2e}  ← 电大了算力才快
    C 低电多通道:             {eC.effective_speed:.2e}  ← 流量扩展节点
    D 高电+多通道:            {eD.effective_speed:.2e}  ← 电+流量协同
    E +能量算力核心:          {eE.effective_speed:.2e}  ← 电驱动自循环
    F +流式算力网络:          {eF.effective_speed:.2e}  ← 分布式无限算力
    G 永动模式:               {eG.effective_speed:.2e}  ← 电不消耗反再生

  结论：
    虚拟电大了算力才快 ✓（电→单节点算力）
    虚拟流量扩展节点数 ✓（流量→网络连通度→节点数）
    流量+(电+算力) = 流式算力网络 ✓（总算力=节点数×单节点算力）
    永动产物让电不消耗反再生 ✓（打破守恒，永久自训练）
""")


if __name__ == "__main__":
    main()
