"""
参数包 + 流式算力网络 = 参数流式训练场（指数级训练）

回答用户问题：
  工厂原来就能生产参数包，但不知道有什么用、效果未知。
  参数 + 流式算力网络 = 参数流式训练场 → 线性变指数级训练。

对比：
  原始参数训练（ParameterTrainer）：增量 = 质量 × 0.005（线性，慢）
  参数流式训练场：增量 = 质量 × 0.005 × 节点数（乘法扩展，快）
  永动参数引擎：增量 = 质量 × 0.005 × 节点 × 加速 × 算力（指数级，一步完成）
"""

import time
import numpy as np
from xuni.parameter import ParameterPack, ParameterTrainer
from xuni.perpetual_engine import PerpetualTrainingEngine
from xuni.model import XuniModel, ModelType, ModelCapability
from xuni.sampler import XuniSampler, SamplingMode


def make_model(mid: str):
    """造一个虚拟文本模型"""
    m = XuniModel(mid, ModelType.TEXT_GENERATOR, [ModelCapability.TEXT_OUTPUT], 10.0)
    m._energy_buffer = 1e8
    m.owner = "test"
    return m


def make_param_pack(quality: float = 50.0, n: int = 20):
    """造一个参数包"""
    return ParameterPack(
        pack_id=f"pack-{quality:.0f}-{n}",
        source="sampler",
        params={f"p{i}": float(np.random.random()) for i in range(n)},
        quality=quality,
    )


def main():
    print("=" * 70)
    print("参数包 + 流式算力网络 = 参数流式训练场（指数级训练）")
    print("=" * 70)

    # 造一批参数包（工厂原来就能生产）
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    print("\n用采样器生产参数包...")
    packs = []
    batch = sampler.generate_batch(1000)
    for i in range(0, 1000, 50):
        row = batch[i]
        quality = float(np.clip(abs(row[0]) * 100, 10, 100))
        params = {
            "charge": float(row[4]),
            "entropy": float(row[5]),
            "x": float(row[0]),
            "y": float(row[1]),
            "z": float(row[2]),
            "w": float(row[3]),
        }
        packs.append(ParameterPack(
            pack_id=f"sampler-{i:04d}",
            source="sampler",
            params=params,
            quality=quality,
        ))
    print(f"  生产了 {len(packs)} 个参数包")
    print(f"  质量范围: [{min(p.quality for p in packs):.1f}, {max(p.quality for p in packs):.1f}]")

    # ---- 对比 1：原始参数训练（线性）----
    print("\n" + "=" * 70)
    print("对比 1：原始参数训练（ParameterTrainer，线性）")
    print("=" * 70)

    model1 = make_model("linear")
    # 用低质量小包，让线性效果可见
    small_pack = make_param_pack(quality=10.0, n=3)
    start = time.time()
    increments_1 = []
    for i in range(20):
        r = ParameterTrainer.train_with_params(model1, small_pack)
        if r.get("error"):
            break
        increments_1.append(r["increment"])
    elapsed = time.time() - start
    print(f"  20次注入（质量10的小包），进度: {model1.training_progress:.4f}")
    print(f"  单包增量: {increments_1[0]:.6f}（恒定，线性）")
    print(f"  耗时: {elapsed*1000:.1f}ms")

    # ---- 对比 2：参数流式训练场（乘法扩展）----
    print("\n" + "=" * 70)
    print("对比 2：参数流式训练场（参数 + 流式算力网络，乘法扩展）")
    print("=" * 70)

    model2 = make_model("stream")
    engine2 = PerpetualTrainingEngine()
    engine2.inject_energy(10000)
    engine2.set_bandwidth(2048)
    engine2.apply_fusion("能量算力核心")
    engine2.apply_fusion("流式算力网络")
    engine2.apply_fusion("参数流式训练场")

    start = time.time()
    r = engine2.train_with_params(model2, small_pack)
    elapsed = time.time() - start
    print(f"  1次注入（同质量10的小包），进度: {model2.training_progress:.4f}")
    print(f"  节点数: {engine2.node_count}")
    print(f"  基础增量: {r['base_increment']:.6f}")
    print(f"  节点放大: {r['node_boost']:.0f}x")
    print(f"  总增量: {r['total_increment']:.6f}")
    print(f"  放大倍率: {r['total_increment']/r['base_increment']:.0f}x")
    print(f"  耗时: {elapsed*1000:.1f}ms")

    # ---- 对比 3：永动参数引擎（指数级，一步完成）----
    print("\n" + "=" * 70)
    print("对比 3：永动参数引擎（超频+全网永动算力，指数级）")
    print("=" * 70)

    model3 = make_model("perpetual")
    engine3 = PerpetualTrainingEngine()
    engine3.inject_energy(10000)
    engine3.set_bandwidth(2048)
    engine3.apply_fusion("能量算力核心")
    engine3.apply_fusion("流式算力网络")
    engine3.apply_fusion("参数流式训练场")
    engine3.apply_fusion("超频参数训练")
    engine3.apply_fusion("全网永动算力")
    engine3.apply_fusion("永动参数引擎")

    start = time.time()
    r = engine3.train_with_params(model3, small_pack)
    elapsed = time.time() - start
    print(f"  1次注入（同质量10的小包），进度: {model3.training_progress:.4f}")
    print(f"  基础增量: {r['base_increment']:.6f}")
    print(f"  节点放大: {r['node_boost']:.0f}x")
    print(f"  加速放大: {r['accel_boost']:.0f}x")
    print(f"  算力放大: {r['compute_boost']:.0f}x")
    print(f"  能量质量放大: {r['energy_quality_boost']:.1f}x")
    print(f"  总增量: {r['total_increment']:.6f}")
    print(f"  放大倍率: {r['total_increment']/r['base_increment']:.0f}x")
    print(f"  永动: {r['perpetual']}")
    print(f"  耗时: {elapsed*1000:.1f}ms")

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("总结：参数包有什么用？融合后效果如何？")
    print("=" * 70)
    inc1 = increments_1[0] if increments_1 else 0
    print(f"""
  参数包是模型权重的"原料"，工厂原来就能生产但效果未知。

  原始效果（ParameterTrainer，线性）：
    单包增量 = {inc1:.6f}（恒定，20次才到 {model1.training_progress:.4f}）

  参数 + 流式算力网络 = 参数流式训练场：
    单包增量 = {inc1:.6f} × {engine2.node_count}节点（乘法扩展）
    1次就到 {model2.training_progress:.4f}（放大 {engine2.node_count}x）

  永动参数引擎（终极）：
    1次注入 → 进度 {model3.training_progress:.4f}（一步完成训练！）
    放大 = 节点×加速×算力 = {r['node_boost']:.0f}×{r['accel_boost']:.0f}×{r['compute_boost']:.0f}

  结论：
    参数包原来"不知道有什么用" → 现在知道了：它是训练原料
    参数 + 流式算力网络 → 线性变乘法（N节点并行注入）
    永动参数引擎 → 一步完成训练（打破训练速度守恒）
""")


if __name__ == "__main__":
    main()
