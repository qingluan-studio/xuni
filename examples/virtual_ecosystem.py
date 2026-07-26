"""
虚拟生态系统完整示例

演示完整闭环：
1. 采样点生成（超混沌采样）
2. 场能量计算（虚拟电场）
3. 凭证铸造（场能量→虚拟凭证）
4. API调用（凭证认证+模型路由）
5. 模型输出（文本/图像描述/音乐/分类/聊天）
"""

import numpy as np
import time
import json


def run_virtual_ecosystem():
    print("=" * 70)
    print("XUNI VIRTUAL ECOSYSTEM DEMO")
    print("=" * 70)
    
    from xuni import (
        XuniSampler, SamplingMode, XuniField,
        XuniCredential, CredentialType,
        XuniModelRegistry,
        XuniGateway, APIEndpoint, APIRequest,
    )

    print("\n[1/5] 生成采样点...")
    sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
    samples = list(sampler.generate_stream(count=10000))
    print(f"  生成采样点: {len(samples)} 个")
    print(f"  示例采样点: [{samples[0].x:.3f}, {samples[0].y:.3f}, {samples[0].z:.3f}]")

    print("\n[2/5] 计算场能量...")
    field = XuniField(grid_size=(64, 64, 64))
    samples_array = np.array([s.to_array() for s in samples])
    field.ingest_batch(samples_array)
    field.compute_field()
    total_energy = field.get_total_energy()
    dominant_direction = field.get_dominant_vector()
    print(f"  总场能量: {total_energy:.4f}")
    print(f"  主导场方向: [{dominant_direction[0]:.3f}, {dominant_direction[1]:.3f}, {dominant_direction[2]:.3f}]")
    print(f"  总采样点数: {field._total_samples}")

    print("\n[3/5] 铸造虚拟凭证...")
    credential = XuniCredential(energy_conversion_rate=100.0)
    
    access_token = credential.mint(
        field_energy=total_energy * 0.3,
        token_type=CredentialType.ACCESS_TOKEN,
        duration_hours=24.0,
        max_calls=100,
        scope=["read", "list"],
    )
    print(f"  Access Token: {access_token.token_id[:12]}...")
    print(f"  能量值: {access_token.energy_value:.2f}")
    
    model_token = credential.mint(
        field_energy=total_energy * 0.5,
        token_type=CredentialType.MODEL_TOKEN,
        duration_hours=24.0,
        max_calls=50,
        scope=["predict", "generate"],
    )
    print(f"  Model Token: {model_token.token_id[:12]}...")
    print(f"  能量值: {model_token.energy_value:.2f}")
    
    premium_token = credential.mint(
        field_energy=total_energy * 0.2,
        token_type=CredentialType.PREMIUM_TOKEN,
        duration_hours=72.0,
        max_calls=0,
        scope=["admin", "stats", "mint"],
    )
    print(f"  Premium Token: {premium_token.token_id[:12]}...")
    print(f"  能量值: {premium_token.energy_value:.2f}")

    print("\n[4/5] 初始化虚拟模型...")
    model_registry = XuniModelRegistry()
    model_registry.register_default_models()
    model_stats = model_registry.statistics()
    print(f"  已注册模型数: {model_stats['total_models']}")
    for model_type, count in model_stats['models_by_type'].items():
        if count > 0:
            print(f"    - {model_type}: {count}")

    print("\n[5/5] 通过虚拟 API 调用模型...")
    gateway = XuniGateway(credential_manager=credential, model_registry=model_registry)

    print("\n  ├─ 列出所有模型 (使用 Access Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_LIST,
        token_id=access_token.token_id,
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 成功获取 {len(resp.data['models'])} 个模型")

    print("\n  ├─ 获取模型信息 (使用 Access Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_INFO,
        token_id=access_token.token_id,
        model_id="text-gen-001",
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 模型: {resp.data['model_id']}")
        print(f"      类型: {resp.data['model_type']}")
        print(f"      能量需求: {resp.data['energy_requirement']}")

    print("\n  ├─ 调用文本生成模型 (使用 Model Token)")
    model_registry.charge_all(total_energy)
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_PREDICT,
        token_id=model_token.token_id,
        model_id="text-gen-001",
        prompt="The universe is a beautiful place where",
        parameters={"max_length": 100},
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 生成文本: {resp.data['text'][:80]}...")
        print(f"      延迟: {resp.data['latency_ms']:.2f}ms")

    print("\n  ├─ 调用图像描述模型 (使用 Model Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_PREDICT,
        token_id=model_token.token_id,
        model_id="image-desc-001",
        prompt="A mysterious landscape",
        parameters={"detail": True},
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 图像描述: {resp.data['text']}")

    print("\n  ├─ 调用音乐作曲模型 (使用 Model Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_PREDICT,
        token_id=model_token.token_id,
        model_id="music-comp-001",
        prompt="Create ambient music",
    )
    resp = gateway.handle_request(req)
    if resp.success:
        music_params = resp.data['json']
        print(f"    ✓ 音乐参数:")
        print(f"      风格: {music_params['genre']}")
        print(f"      调性: {music_params['scale']}")
        print(f"      BPM: {music_params['tempo']}")
        print(f"      乐器: {music_params['instrument']}")
        print(f"      情绪: {music_params['mood']}")

    print("\n  ├─ 调用分类模型 (使用 Model Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_PREDICT,
        token_id=model_token.token_id,
        model_id="sentiment-001",
        prompt="I'm so happy today!",
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 分类结果: {resp.data['classification']}")
        print(f"      概率分布: {json.dumps(resp.data['json']['probabilities'], indent=6)}")

    print("\n  ├─ 调用聊天机器人 (使用 Model Token)")
    req = APIRequest(
        endpoint=APIEndpoint.MODELS_PREDICT,
        token_id=model_token.token_id,
        model_id="chatbot-creative-001",
        prompt="What is the meaning of life?",
    )
    resp = gateway.handle_request(req)
    if resp.success:
        print(f"    ✓ 聊天回复: {resp.data['text']}")

    print("\n  └─ 获取系统统计 (使用 Premium Token)")
    req = APIRequest(
        endpoint=APIEndpoint.SYSTEM_STATISTICS,
        token_id=premium_token.token_id,
    )
    resp = gateway.handle_request(req)
    if resp.success:
        stats = resp.data
        print(f"    ✓ 总请求数: {stats['total_requests']}")
        print(f"    ✓ 模型统计:")
        print(f"      总调用: {stats['models']['total_calls']}")
        print(f"      消耗能量: {stats['models']['total_energy_consumed']}")
        print(f"    ✓ 凭证统计:")
        print(f"      总凭证数: {stats['credentials']['total_tokens']}")
        print(f"      总能量值: {stats['credentials']['total_energy_value']}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETED - Virtual Ecosystem is working!")
    print("=" * 70)
    
    return {
        "samples": len(samples),
        "field_energy": total_energy,
        "tokens_created": credential.statistics()["total_tokens"],
        "total_requests": gateway._request_counter,
        "models_called": model_registry.statistics()["total_calls"],
    }


if __name__ == "__main__":
    results = run_virtual_ecosystem()
    
    import os
    os.makedirs("output", exist_ok=True)
    
    with open("output/virtual_ecosystem_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 output/virtual_ecosystem_results.json")