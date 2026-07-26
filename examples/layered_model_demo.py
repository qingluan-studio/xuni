"""
分层模型系统演示

演示：
1. 创建5层模型（音乐/扩散/对话/文本/分类），每层5个模型
2. 5个 AI 认领第1层音乐模型进行训练
3. 训练完成后充能
4. 层间数据流动：音乐层输出 → 扩散层 → 对话层...
5. 可视化整个分层系统
"""

import numpy as np


def run_layered_demo():
    from xuni import (
        XuniSampler, SamplingMode, XuniField,
        LayeredModelSystem, LayerType, LayerConfig,
        ModelInput, TrainingState,
    )

    print("=" * 70)
    print("XUNI LAYERED MODEL SYSTEM DEMO")
    print("=" * 70)

    # ===== 1. 创建分层系统 =====
    print("\n[1/6] 创建5层模型系统（每层5个模型）...")
    system = LayeredModelSystem()
    system.setup_default_layers(models_per_layer=5)
    
    stats = system.statistics()
    print(f"  总层数: {stats['total_layers']}")
    print(f"  总模型数: {stats['total_models']}")
    for layer_stat in stats["layers"]:
        print(f"    Layer {layer_stat['level']}: {layer_stat['layer_name']} "
              f"({layer_stat['total_models']}个模型)")

    # ===== 2. AI 认领第1层音乐模型 =====
    print("\n[2/6] 5个 AI 认领第1层音乐模型...")
    ai_names = ["Aria", "Bolt", "Coda", "Dusk", "Echo"]
    
    music_layer = system.get_layer_by_level(1)
    assignments = music_layer.auto_assign(ai_names)
    
    for ai_name, model_id in assignments.items():
        print(f"  {ai_name} → 认领 {model_id}")
    
    unclaimed = music_layer.get_unclaimed()
    print(f"  未认领: {len(unclaimed)} 个")

    # ===== 3. AI 认领其他层 =====
    print("\n[3/6] AI 认领其他层模型...")
    # 第2层扩散
    diffusion_layer = system.get_layer_by_level(2)
    diff_assignments = diffusion_layer.auto_assign(["Aria", "Bolt", "Coda"])
    for ai, mid in diff_assignments.items():
        print(f"  {ai} → 认领扩散模型 {mid}")
    
    # 第3层对话
    chat_layer = system.get_layer_by_level(3)
    chat_assignments = chat_layer.auto_assign(["Dusk", "Echo", "Aria"])
    for ai, mid in chat_assignments.items():
        print(f"  {ai} → 认领对话模型 {mid}")

    # ===== 4. 训练所有已认领的模型 =====
    print("\n[4/6] 训练所有已认领的模型...")
    
    # 批量训练（逐步增加进度）
    for step in range(5):
        progress = (step + 1) * 0.2
        for layer in system.get_layers_ordered():
            for model in layer.models.values():
                if model.training_state == TrainingState.CLAIMED:
                    model.start_training()
                if model.training_state == TrainingState.TRAINING:
                    model.update_training(progress)
    
    # 检查训练结果
    for layer in system.get_layers_ordered():
        trained = layer.get_trained()
        claimed = layer.get_claimed()
        print(f"  Layer {layer.config.level} ({layer.config.layer_name}): "
              f"已认领 {len(claimed)}, 已训练 {len(trained)}")

    # ===== 5. 生成采样点能量充能 =====
    print("\n[5/6] 生成采样点能量并充能...")
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    samples = list(sampler.generate_stream(count=3000))
    samples_array = np.array([s.to_array() for s in samples])
    
    field = XuniField(grid_size=(32, 32, 32))
    field.ingest_batch(samples_array)
    field.compute_field()
    total_energy = field.get_total_energy()
    print(f"  采样点: {len(samples)} 个")
    print(f"  场能量: {total_energy:.4f}")
    
    # 只给已训练的模型充能
    for layer in system.get_layers_ordered():
        energy = layer.charge_trained_only(total_energy / 10)
        print(f"  Layer {layer.config.level} 充能: {energy:.2f}")

    # ===== 6. 层间数据流动 =====
    print("\n[6/6] 层间数据流动（音乐层 → 扩散层 → 对话层）...")
    initial_input = ModelInput(prompt="生成一段宁静的夜晚音乐")
    
    flow_results = system.flow_through_all(initial_input)
    
    for result in flow_results:
        level = result["level"]
        layer_name = result["layer_name"]
        output_count = len(result["outputs"])
        
        print(f"\n  Layer {level} ({layer_name}): {output_count} 个模型输出")
        for model_id, output in list(result["outputs"].items())[:2]:
            if output.get("text"):
                text = output["text"][:80] + "..." if len(output.get("text", "")) > 80 else output.get("text", "")
                print(f"    {model_id}: {text}")
            elif output.get("json"):
                print(f"    {model_id}: {output['json']}")

    # ===== 可视化 =====
    print("\n" + system.visualize())

    # ===== 全局统计 =====
    final_stats = system.statistics()
    print(f"\n全局统计:")
    print(f"  总模型: {final_stats['total_models']}")
    print(f"  已认领: {final_stats['total_claimed']}")
    print(f"  已训练: {final_stats['total_trained']}")
    print(f"  总调用: {final_stats['total_calls']}")
    print(f"  总能量消耗: {final_stats['total_energy_consumed']}")
    print(f"  参与AI: {final_stats['owners']}")

    print("\n" + "=" * 70)
    print("LAYERED MODEL DEMO COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    run_layered_demo()
