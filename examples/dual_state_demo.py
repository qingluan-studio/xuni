"""
双态切换系统完整示例

演示你的核心思路：
1. 虚拟模型（训练时）- 不占内存，消耗虚拟电
2. 真实模型（调用时）- 接入真实AI服务
3. 双态切换 - 用户看到的始终是虚拟状态
4. 数据层转换 - 真实模型的数据供虚拟模型训练

重点演示：
- 虚拟模式调用
- 混合模式调用（训练虚拟，调用真实）
- 真实模式调用
- 凭证执行验证
- 物质产出链查询
"""

import time
import json


def run_dual_state_demo():
    print("=" * 70)
    print("XUNI DUAL STATE SYSTEM DEMO")
    print("=" * 70)
    
    from xuni import (
        XuniSampler, SamplingMode, XuniField,
        XuniCredential, CredentialType,
        XuniModelRegistry, XuniChatBot,
        DualStateManager, ModelState, LocalModelAdapter,
        SubstanceSystem, SubstanceCategory,
    )

    print("\n[1/6] 生成采样点并计算场能量...")
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    samples = list(sampler.generate_stream(count=5000))
    samples_array = np.array([s.to_array() for s in samples])
    
    field = XuniField(grid_size=(32, 32, 32))
    field.ingest_batch(samples_array)
    field.compute_field()
    total_energy = field.get_total_energy()
    print(f"  采样点: {len(samples)} 个")
    print(f"  场能量: {total_energy:.4f}")

    print("\n[2/6] 铸造24位虚拟凭证...")
    credential = XuniCredential(energy_conversion_rate=100.0)
    
    model_token = credential.mint(
        field_energy=total_energy,
        token_type=CredentialType.MODEL_TOKEN,
        max_calls=10,
    )
    print(f"  模型令牌: {model_token.token_id}")
    print(f"  令牌长度: {len(model_token.token_id)} 位")
    print(f"  能量值: {model_token.energy_value:.2f}")

    print("\n[3/6] 创建双态模型（虚拟+真实）...")
    virtual_model = XuniChatBot("chatbot-dual-001", "friendly")
    real_adapter = LocalModelAdapter(model_path="./local-model")
    
    manager = DualStateManager(virtual_model, real_adapter)
    print(f"  当前状态: {manager.state.name}")

    print("\n[4/6] 虚拟模式调用...")
    manager.switch_to_virtual()
    print(f"  切换到: {manager.state.name}")
    
    virtual_result = manager.predict("What is virtual electricity?")
    print(f"  来源: {virtual_result['source']}")
    print(f"  回复: {virtual_result['text']}")
    
    credential.record_execution(model_token.token_id, True, "predict", virtual_result)

    print("\n[5/6] 混合模式调用（训练虚拟，调用真实）...")
    switch_ok = manager.switch_to_hybrid()
    if switch_ok:
        print(f"  切换到: {manager.state.name}")
        
        hybrid_result = manager.predict("Explain dual-state switching")
        print(f"  来源: {hybrid_result['source']}")
        print(f"  回复: {hybrid_result['text']}")
        
        credential.record_execution(model_token.token_id, True, "predict", hybrid_result)

        print("\n  [数据层转换] 从真实模型获取训练数据...")
        training_result = manager.train_virtual_from_real(max_samples=500, energy=total_energy)
        print(f"    训练状态: {training_result['status']}")
        print(f"    数据形状: {training_result['data_shape']}")
        print(f"    数据哈希: {training_result['data_hash'][:16]}...")
        
        credential.record_execution(model_token.token_id, True, "train", training_result)
    else:
        print("  无法切换到混合模式（真实适配器未连接）")

    print("\n[6/6] 验证凭证执行能力...")
    verification = credential.verify_execution(model_token.token_id)
    print(f"  凭证有效: {verification['valid']}")
    print(f"  可执行: {verification['can_execute']}")
    print(f"  执行历史: {verification['execution_history']} 次")
    print(f"  成功率: {verification['success_rate']}%")

    print("\n[7/6] 物质产出链查询...")
    substance_system = SubstanceSystem()
    chain = substance_system.get_production_chain("生成文本")
    print(f"  生成文本的产出链:")
    for i, item in enumerate(chain):
        sub = substance_system.get(item)
        if sub:
            print(f"    {i+1}. {sub.icon} {item}")

    print("\n  所有物质分类:")
    stats = substance_system.statistics()
    for cat, count in stats["categories"].items():
        if count > 0:
            print(f"    {cat}: {count} 种")

    print("\n" + "=" * 70)
    print("DUAL STATE DEMO COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    import numpy as np
    run_dual_state_demo()