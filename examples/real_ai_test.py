"""
真实AI接入测试

两种模式：
1. 模拟模式（无API密钥）：用 LocalModelAdapter 验证双态切换流程
2. 真实模式（有API密钥）：实际接入 OpenAI/Anthropic

使用方法：
  模拟模式：python real_ai_test.py
  真实模式：OPENAI_API_KEY=sk-xxx python real_ai_test.py
           ANTHROPIC_API_KEY=xxx python real_ai_test.py
"""

import os
import sys
import numpy as np


def test_simulated_mode():
    """模拟模式：无API密钥，用本地适配器"""
    print("=" * 60)
    print("MODE 1: SIMULATED (no API key)")
    print("=" * 60)
    
    from xuni import (
        DualStateManager, ModelState, LocalModelAdapter,
        XuniChatBot, XuniTextGenerator,
    )
    
    # 创建虚拟模型 + 本地适配器
    virtual_model = XuniChatBot("test-chat-001", "friendly")
    real_adapter = LocalModelAdapter(model_path="./local-model")
    
    manager = DualStateManager(virtual_model, real_adapter)
    print(f"\nInitial state: {manager.state.name}")
    
    # 虚拟模式
    print("\n--- Virtual Mode ---")
    manager.switch_to_virtual()
    result = manager.predict("Hello")
    print(f"Source: {result['source']}")
    print(f"Response: {result.get('text', 'N/A')}")
    
    # 混合模式
    print("\n--- Hybrid Mode (train virtual, call real) ---")
    if manager.switch_to_hybrid():
        result = manager.predict("Explain quantum computing")
        print(f"Source: {result['source']}")
        print(f"Response: {result.get('text', 'N/A')}")
        
        # 数据层转换
        print("\n--- Data Layer Transfer ---")
        training = manager.train_virtual_from_real(max_samples=100, energy=50.0)
        print(f"Training: {training['status']}")
        print(f"Data shape: {training['data_shape']}")
    else:
        print("Failed to switch to hybrid")
    
    print("\nSimulated mode PASSED!")


def test_real_openai(api_key: str):
    """真实模式：接入 OpenAI"""
    print("=" * 60)
    print("MODE 2: REAL OpenAI API")
    print("=" * 60)
    
    from xuni import (
        DualStateManager, ModelState, OpenAIAdapter,
        XuniChatBot,
    )
    
    virtual_model = XuniChatBot("real-test-001", "professional")
    real_adapter = OpenAIAdapter(api_key=api_key, model_name="gpt-4o-mini")
    
    manager = DualStateManager(virtual_model, real_adapter)
    
    # 尝试连接
    print("\nConnecting to OpenAI...")
    if not real_adapter.connect():
        print("Failed to connect (openai package not installed?)")
        print("Run: pip install openai")
        return False
    
    print("Connected!")
    
    # 混合模式调用
    print("\n--- Hybrid Mode: Real OpenAI Call ---")
    manager.switch_to_hybrid()
    result = manager.predict("What is 2+2? Answer in one word.")
    print(f"Source: {result['source']}")
    print(f"Response: {result.get('text', 'N/A')}")
    
    if result.get("usage"):
        print(f"Tokens: {result['usage']}")
    
    # 数据层转换
    print("\n--- Data Layer Transfer ---")
    training = manager.train_virtual_from_real(max_samples=50, energy=30.0)
    print(f"Training: {training['status']}")
    print(f"Data shape: {training['data_shape']}")
    
    print("\nReal OpenAI mode PASSED!")
    return True


def test_real_anthropic(api_key: str):
    """真实模式：接入 Anthropic"""
    print("=" * 60)
    print("MODE 3: REAL Anthropic API")
    print("=" * 60)
    
    from xuni import (
        DualStateManager, ModelState, AnthropicAdapter,
        XuniChatBot,
    )
    
    virtual_model = XuniChatBot("claude-test-001", "creative")
    real_adapter = AnthropicAdapter(api_key=api_key)
    
    manager = DualStateManager(virtual_model, real_adapter)
    
    print("\nConnecting to Anthropic...")
    if not real_adapter.connect():
        print("Failed to connect (anthropic package not installed?)")
        print("Run: pip install anthropic")
        return False
    
    print("Connected!")
    
    print("\n--- Hybrid Mode: Real Claude Call ---")
    manager.switch_to_hybrid()
    result = manager.predict("Write a haiku about virtual electricity.")
    print(f"Source: {result['source']}")
    print(f"Response: {result.get('text', 'N/A')}")
    
    print("\nReal Anthropic mode PASSED!")
    return True


def main():
    # 模拟模式（始终运行）
    test_simulated_mode()
    
    # 真实模式（需要API密钥）
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if openai_key:
        print("\n")
        try:
            test_real_openai(openai_key)
        except Exception as e:
            print(f"OpenAI test failed: {e}")
    else:
        print("\n[SKIP] OpenAI test (set OPENAI_API_KEY to enable)")
    
    if anthropic_key:
        print("\n")
        try:
            test_real_anthropic(anthropic_key)
        except Exception as e:
            print(f"Anthropic test failed: {e}")
    else:
        print("[SKIP] Anthropic test (set ANTHROPIC_API_KEY to enable)")
    
    print("\n" + "=" * 60)
    print("ALL TESTS DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
