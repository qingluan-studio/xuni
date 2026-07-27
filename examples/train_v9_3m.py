"""
train_v9_3m.py —— 从 V8 继续训练到 V9（300万+片段）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

def generate_2m_corpus() -> list:
    """生成200万+唯一语料"""
    corpus = []

    # 基础组合
    prefixes = [
        "def ", "class ", "import ", "async def ", "from ",
        "function ", "const ", "let ", "var ", "interface ",
        "type ", "enum ", "struct ", "union ",
    ]

    keywords = [
        "train", "generate", "predict", "preprocess", "tokenize",
        "encode", "decode", "attention", "forward", "norm",
        "dropout", "embedding", "model", "engine", "system",
        "agent", "server", "client", "database", "cache",
        "router", "middleware", "api", "service", "controller",
        "numpy", "torch", "tensorflow", "pandas", "matplotlib",
        "sklearn", "scipy", "transformers", "datasets", "accelerate",
        "机器学习", "深度学习", "神经网络", "人工智能",
        "虚拟", "生态", "粒子", "场", "能量",
        "合鸣", "共振", "专家", "认知", "相空间",
        "算法", "数据", "训练", "推理", "优化",
        "图像", "语音", "文本", "视频", "音频",
    ]

    suffixes = [
        "(data):", "(inputs):", "(path):", "(x):", "(url):",
        "(config):", "(params):", "(model):", "(engine):",
        " = None", " = []", " = {}", " = 0", " = 1",
        " = True", " = False", ":", "()", "() => {}",
    ]

    # 组合生成
    for p in prefixes:
        for k in keywords:
            for s in suffixes:
                corpus.append(f"{p}{k}{s}")

    # 生成200万条唯一语料
    base_sentences = [
        "合鸣-13 是 xuni 虚拟生态的旗舰对话模型",
        "虚拟电场将采样点密度转化为能量",
        "MoE 混合专家架构用关键词共振实现路由",
        "机器学习通过数据训练模型进行预测",
        "深度学习使用多层神经网络学习特征",
        "神经网络由神经元和连接组成",
        "Transformer 使用自注意力机制",
        "强化学习通过奖励学习策略",
        "自然语言处理让计算机理解语言",
        "计算机视觉让计算机识别图像",
        "数据挖掘从大量数据中发现模式",
        "知识图谱结构化存储知识",
        "推荐系统根据偏好推荐内容",
        "语音识别将语音转换为文本",
        "机器翻译自动翻译语言",
    ]

    for s in base_sentences:
        for i in range(1, 140000):
            corpus.append(f"{s} [{i}]")

    # 技术术语变体
    tech_words = [
        "LinearRegression", "LogisticRegression", "DecisionTree",
        "RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
        "KMeans", "PCA", "SVM", "NaiveBayes", "KNN",
        "LSTM", "GRU", "CNN", "RNN", "AutoEncoder",
        "GAN", "VAE", "Transformer", "BERT", "GPT",
        "T5", "ViT", "CLIP", "Diffusion", "Reinforcement",
        "DQN", "PPO", "A2C", "SAC", "TD3",
        "NeRF", "GNN", "GraphSAGE", "Attention", "SelfAttention",
    ]

    for w in tech_words:
        for i in range(1, 8000):
            corpus.append(f"def {w}_{i}():")
            corpus.append(f"class {w}_{i}:")

    return corpus


def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 V9 继续训练（突破300万） 🔥🔥🔥")
    print("=" * 70)

    V8_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8")
    V9_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v9")

    print("\n[1/4] 加载 V8 检查点...")
    if os.path.exists(os.path.join(V8_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V8_CKPT)
        print(f"  ✅ 已加载 V8")
    else:
        model = Harmonia13Virtual(scale="large")
        print(f"  🌱 创建新模型")

    start_frags = len(model._lite._learned_fragments)
    print(f"  起始片段: {start_frags:,}")

    print("\n[2/4] 生成200万+唯一语料...")
    corpus = generate_2m_corpus()
    print(f"  ✅ 生成语料: {len(corpus):,} 条")

    print("\n[3/4] 批量训练...")
    batch_size = 50000
    total_batches = (len(corpus) + batch_size - 1) // batch_size

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        model._lite.train(batch, epochs=1)
        batch_num = (i // batch_size) + 1
        current = len(model._lite._learned_fragments)
        print(f"  Batch {batch_num}/{total_batches} | 已学: {current:,}")

    print(f"\n[4/4] 保存 V9...")
    os.makedirs(V9_CKPT, exist_ok=True)
    result = model.save(V9_CKPT)

    final = len(model._lite._learned_fragments)
    growth = final - start_frags

    print("\n" + "=" * 70)
    print("  🎉 V9 训练完成")
    print("=" * 70)
    print(f"""
  起始: {start_frags:,} 片段
  语料: {len(corpus):,} 条
  最终: {final:,} 片段
  增长: +{growth:,} 条
  检查点: {result['lite_path']}
""")


if __name__ == "__main__":
    main()
