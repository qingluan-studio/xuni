"""
分层模型系统扩展演示

演示：
1. 8层模型（音乐/扩散/对话/文本/分类/图像/预测/自编码），每层5个=40个模型
2. 26个 AI 从名称池自动认领所有模型
3. 协作训练（已训练模型给同层加成）
4. 集成预测（分类层投票、预测层平均）
5. 层间数据流动
"""

import numpy as np


def run_extended_demo():
    from xuni import (
        XuniSampler, SamplingMode, XuniField,
        LayeredModelSystem, LayerType, LayerConfig, AI_NAME_POOL,
        ModelInput, TrainingState,
    )

    print("=" * 70)
    print("XUNI LAYERED MODEL SYSTEM - EXTENDED DEMO")
    print("=" * 70)

    # ===== 1. 创建8层模型系统 =====
    print("\n[1/7] 创建8层模型系统（每层5个=40个模型）...")
    system = LayeredModelSystem()
    system.setup_default_layers(models_per_layer=5)
    
    stats = system.statistics()
    print(f"  总层数: {stats['total_layers']}")
    print(f"  总模型数: {stats['total_models']}")
    for layer_stat in stats["layers"]:
        print(f"    Layer {layer_stat['level']}: {layer_stat['layer_name']} "
              f"({layer_stat['total_models']}个)")

    # ===== 2. AI 名称池批量认领 =====
    print(f"\n[2/7] 从 AI 名称池（{len(AI_NAME_POOL)}个AI）批量认领所有模型...")
    assignments = system.auto_assign_from_pool()
    
    total_assigned = 0
    for layer_id, layer_assignments in assignments.items():
        layer = system.get_layer(layer_id)
        print(f"  {layer.config.layer_name}: {len(layer_assignments)} 个模型被认领")
        total_assigned += len(layer_assignments)
    
    print(f"  总计认领: {total_assigned} 个模型")

    # ===== 3. 协作训练（逐步可视化）=====
    print("\n[3/7] 协作训练（已训练模型给同层加成）...")
    
    # 先让所有已认领的模型开始训练
    for layer in system.get_layers_ordered():
        for model in layer.models.values():
            if model.training_state == TrainingState.CLAIMED:
                model.start_training()
    
    # 逐步训练
    training_result = system.train_until_complete(step_progress=0.3, max_steps=10)
    print(f"  训练步数: {training_result['total_steps']}")
    for h in training_result["history"]:
        print(f"    Step {h['step']}: 已训练 {h['trained']}/{h['claimed']}")
    print(f"  最终已训练: {training_result['final_trained']}")

    # ===== 4. 采样点能量充能 =====
    print("\n[4/7] 生成采样点能量充能...")
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    samples = list(sampler.generate_stream(count=5000))
    samples_array = np.array([s.to_array() for s in samples])
    
    field = XuniField(grid_size=(32, 32, 32))
    field.ingest_batch(samples_array)
    field.compute_field()
    total_energy = field.get_total_energy()
    print(f"  采样点: {len(samples)} 个, 场能量: {total_energy:.4f}")
    
    for layer in system.get_layers_ordered():
        # 按每层模型的能量需求充能（至少给3倍，保证能多次调用）
        for model in layer.models.values():
            if model.training_state == TrainingState.TRAINED:
                model.charge(model.energy_requirement * 3)
        energy = sum(m._energy_buffer for m in layer.models.values())
        print(f"  Layer {layer.config.level} ({layer.config.layer_name}): 充能 {energy:.2f}")

    # ===== 5. 集成预测 =====
    print("\n[5/7] 集成预测（每层所有已训练模型协作）...")
    test_input = ModelInput(prompt="预测未来的能量趋势")
    
    ensemble_results = system.ensemble_all_layers(test_input)
    for layer_id, result in ensemble_results.items():
        print(f"  Layer {result['level']} ({result['layer_name']}):")
        if result.get("classification"):
            print(f"    集成分类: {result['classification']}")
        if result.get("prediction") is not None:
            print(f"    集成预测: {result['prediction']:.2f}")
        if result.get("text"):
            text = result["text"][:60] + "..." if len(result["text"]) > 60 else result["text"]
            print(f"    输出: {text}")
        if result.get("json"):
            print(f"    数据: {result['json']}")

    # ===== 6. 层间数据流动 =====
    print("\n[6/7] 层间数据流动（音乐层 → 扩散层 → ... → 自编码层）...")
    flow_input = ModelInput(prompt="生成一段宁静的夜晚音乐")
    flow_results = system.flow_through_all(flow_input)
    
    for result in flow_results:
        output_count = len(result["outputs"])
        print(f"  Layer {result['level']} ({result['layer_name']}): {output_count} 个模型输出")

    # ===== 7. 可视化 =====
    print("\n[7/7] 可视化整个分层系统...")
    print(system.visualize())

    # ===== 全局统计 =====
    final_stats = system.statistics()
    print(f"全局统计:")
    print(f"  总层数: {final_stats['total_layers']}")
    print(f"  总模型: {final_stats['total_models']}")
    print(f"  已认领: {final_stats['total_claimed']}")
    print(f"  已训练: {final_stats['total_trained']}")
    print(f"  总调用: {final_stats['total_calls']}")
    print(f"  总能量消耗: {final_stats['total_energy_consumed']}")
    print(f"  参与AI数: {final_stats['unique_owners']}")
    print(f"  参与AI: {final_stats['owners']}")

    print("\n" + "=" * 70)
    print("EXTENDED LAYERED DEMO COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    run_extended_demo()
