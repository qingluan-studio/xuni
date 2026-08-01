"""
train_v10_stream.py —— 流式训练到 V10（突破 1000万片段）

策略：流式生成语料，边生成边训练，不占用大量内存
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

def stream_corpus(num_batches: int, batch_size: int = 50000):
    """流式生成语料"""
    prefixes = [
        "def ", "class ", "import ", "async def ", "from ",
        "function ", "const ", "let ", "var ", "interface ",
    ]

    keywords = [
        "train", "generate", "predict", "preprocess", "tokenize",
        "encode", "decode", "attention", "forward", "norm",
        "dropout", "embedding", "model", "engine", "system",
        "agent", "server", "client", "database", "cache",
        "numpy", "torch", "tensorflow", "pandas", "matplotlib",
        "机器学习", "深度学习", "神经网络", "人工智能",
        "虚拟", "生态", "粒子", "场", "能量",
        "合鸣", "共振", "专家", "认知", "相空间",
    ]

    suffixes = [
        "(data):", "(inputs):", "(path):", "(x):", "(url):",
        " = None", " = []", " = {}", " = 0", ":", "()",
    ]

    base_sentences = [
        "合鸣-13 是 xuni 虚拟生态的旗舰对话模型",
        "虚拟电场将采样点密度转化为能量",
        "MoE 混合专家架构用关键词共振实现路由",
        "机器学习通过数据训练模型进行预测",
        "深度学习使用多层神经网络学习特征",
        "Transformer 使用自注意力机制",
        "强化学习通过奖励学习策略",
        "自然语言处理让计算机理解语言",
        "计算机视觉让计算机识别图像",
        "数据挖掘从大量数据中发现模式",
    ]

    tech_words = [
        "LinearRegression", "LogisticRegression", "DecisionTree",
        "RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
        "KMeans", "PCA", "SVM", "LSTM", "GRU", "CNN",
        "GAN", "VAE", "Transformer", "BERT", "GPT",
        "T5", "ViT", "CLIP", "Diffusion", "ResNet",
    ]

    batch = []
    counter = 0

    for batch_idx in range(num_batches):
        batch = []
        while len(batch) < batch_size:
            # 组合语料
            p = prefixes[counter % len(prefixes)]
            k = keywords[(counter // len(prefixes)) % len(keywords)]
            s = suffixes[(counter // (len(prefixes) * len(keywords))) % len(suffixes)]
            batch.append(f"{p}{k}{s}")

            # 句子变体
            sentence = base_sentences[(counter // 100) % len(base_sentences)]
            batch.append(f"{sentence} [{counter}]")

            # 技术术语
            tw = tech_words[(counter // 1000) % len(tech_words)]
            batch.append(f"def {tw}_{counter}():")
            batch.append(f"class {tw}_{counter}:")

            counter += 1
            if len(batch) >= batch_size:
                break

        yield batch


def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 V10 流式训练（突破1000万） 🔥🔥🔥")
    print("=" * 70)

    V8_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8")
    V10_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v10")

    print("\n[1/4] 加载 V8 检查点...")
    if os.path.exists(os.path.join(V8_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V8_CKPT)
        print(f"  ✅ 已加载 V8")
    else:
        model = Harmonia13Virtual(scale="large")
        print(f"  🌱 创建新模型")

    start_frags = len(model._lite._learned_fragments)
    print(f"  起始片段: {start_frags:,}")

    print("\n[2/4] 流式训练...")
    total_batches = 100
    batch_size = 50000

    for i, batch in enumerate(stream_corpus(total_batches, batch_size), 1):
        model._lite.train(batch, epochs=1)
        current = len(model._lite._learned_fragments)
        print(f"  Batch {i}/{total_batches} | 已学: {current:,}")

    print(f"\n[3/4] 保存 V10...")
    os.makedirs(V10_CKPT, exist_ok=True)
    result = model.save(V10_CKPT)

    final = len(model._lite._learned_fragments)
    growth = final - start_frags

    print("\n" + "=" * 70)
    print("  🎉 V10 流式训练完成")
    print("=" * 70)
    print(f"""
  起始: {start_frags:,} 片段
  语料: {total_batches * batch_size:,} 条
  最终: {final:,} 片段
  增长: +{growth:,} 条
  检查点: {result['lite_path']}
""")


if __name__ == "__main__":
    main()
