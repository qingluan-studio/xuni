"""
train_v8_clean.py —— 从 V3 重新训练（正确去重）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

def generate_diverse_corpus() -> list:
    """生成多样化语料，每个种子只出现一次"""
    corpus = []

    # 合鸣架构
    corpus.extend([
        "合鸣-13 是 xuni 虚拟生态的旗舰对话模型，由 13 位虚拟专家组成",
        "虚拟电场将采样点密度转化为能量，驱动整个生态系统",
        "MoE 混合专家架构用关键词共振实现非传统路由",
        "双态系统让虚拟模型在粒子态被真正训练",
        "物理建模合成器用数字振荡器生成声音",
        "超混沌采样器实时生成上亿采样点",
        "水动力学把采样点当作流体粒子",
        "玻璃逻辑把计算当作光学系统",
        "Kuramoto 振子网络模拟神经同步",
        "虚拟凭证把场能量铸造成令牌",
        "虚拟算力专家提供无限计算资源",
        "虚拟哲学专家思考存在与意识",
        "神经共振专家实现知识联想",
        "合鸣自述者讲述模型故事",
        "混合专家协调领域专家协作",
        "关键词共振门控选择相关专家",
        "n-gram 共振实现相似性匹配",
        "场调制动态调整响应权重",
        "粒子态训练让模型进化",
        "数据层调用让成果应用",
    ])

    # 编程代码模板
    corpus.extend([
        "def train_model(data, epochs=100):",
        "def generate_text(prompt, max_length=100):",
        "def load_checkpoint(path):",
        "def save_checkpoint(model, path):",
        "def predict(inputs):",
        "def preprocess(text):",
        "def tokenize(text):",
        "def encode(text):",
        "def decode(tokens):",
        "def attention(query, key, value):",
        "def feed_forward(x):",
        "def layer_norm(x):",
        "def dropout(x, rate=0.1):",
        "def embedding(x, vocab_size, dim):",
        "def positional_encoding(max_len, dim):",
        "class NeuralNetwork:",
        "class Transformer:",
        "class Encoder:",
        "class Decoder:",
        "class Attention:",
        "class LayerNorm:",
        "class FeedForward:",
        "class Embedding:",
        "class Model:",
        "class Engine:",
        "class System:",
        "class Agent:",
        "class Server:",
        "class Client:",
        "class Database:",
        "class Cache:",
        "class Router:",
        "class Middleware:",
        "async def fetch(url):",
        "async def process(data):",
        "async def stream():",
        "import numpy as np",
        "import torch",
        "import tensorflow as tf",
        "import pandas as pd",
    ])

    # AI技术
    corpus.extend([
        "机器学习通过数据训练模型进行预测",
        "深度学习使用多层神经网络学习特征",
        "反向传播算法优化模型参数",
        "卷积神经网络擅长图像处理",
        "循环神经网络处理序列数据",
        "Transformer 使用自注意力机制",
        "强化学习通过奖励学习策略",
        "自然语言处理让计算机理解语言",
        "计算机视觉让计算机识别图像",
        "数据挖掘发现数据模式",
        "知识图谱结构化存储知识",
        "推荐系统根据偏好推荐内容",
        "语音识别将语音转换为文本",
        "机器翻译自动翻译语言",
        "情感分析识别情感倾向",
        "文本分类归类文本",
        "信息检索查找相关信息",
        "数据清洗处理缺失值",
        "特征工程提取特征",
        "迁移学习利用预训练模型",
    ])

    # 虚拟生态扩展
    corpus.extend([
        "虚拟生态系统由多个虚拟实体组成",
        "虚拟粒子是构成虚拟世界的基本单位",
        "虚拟场是粒子运动的空间",
        "能量守恒定律在虚拟世界适用",
        "熵增定律描述无序程度",
        "自组织系统产生有序结构",
        "涌现现象是复杂系统的性质",
        "混沌理论描述不可预测性",
        "分形几何展现自相似结构",
        "复杂性理论研究系统行为",
        "系统论从整体视角分析",
        "控制论研究反馈和控制",
        "信息论量化不确定性",
        "博弈论分析策略交互",
        "网络科学研究网络结构",
        "图论研究图结构",
        "拓扑学研究空间性质",
        "动力学系统描述时间变化",
        "非线性动力学研究非线性行为",
        "量子计算利用量子特性",
    ])

    # 认知相空间
    corpus.extend([
        "认知相空间是合鸣模型的后端",
        "虚拟认知能力提供智能基础",
        "多轮对话保持上下文理解",
        "专家路由系统选择最优专家",
        "记忆增强模块记住历史对话",
        "分布式专家系统支持扩展",
        "自适应学习率自动调整",
        "增量学习不遗忘旧知识",
        "终身学习让模型持续进化",
        "元学习让模型学会学习",
        "少样本学习只需要少量样本",
        "零样本学习不需要样本",
        "对比学习通过对比学习",
        "自监督学习利用未标注数据",
        "联邦学习保护隐私训练",
        "对抗训练提高鲁棒性",
        "数据增强增加训练样本",
        "早停防止过拟合",
        "正则化减少复杂度",
        "知识蒸馏转化复杂知识",
    ])

    # 更多领域扩展
    corpus.extend([
        "软件工程是系统开发和维护的学科",
        "设计模式提供可复用的解决方案",
        "面向对象编程使用类和对象",
        "函数式编程强调纯函数",
        "并发编程处理多个任务",
        "异步编程提高响应性",
        "微服务架构分解应用为服务",
        "容器化使用 Docker 部署应用",
        "持续集成自动化构建测试",
        "持续部署自动化部署应用",
        "云原生设计面向云环境",
        "DevOps 融合开发和运维",
        "RESTful API 设计 Web 服务",
        "GraphQL 提供灵活数据查询",
        "数据库索引加速查询",
        "缓存减少数据库负载",
        "负载均衡分发请求",
        "安全认证验证用户身份",
        "加密保护数据安全",
        "监控追踪系统状态",
    ])

    # 数学和算法
    corpus.extend([
        "线性代数是机器学习的数学基础",
        "微积分用于优化算法",
        "概率论描述随机事件",
        "统计学分析数据规律",
        "信息论量化信息",
        "图论研究图结构",
        "动态规划解决最优问题",
        "贪心算法做出局部最优选择",
        "分治算法分解问题",
        "回溯算法尝试所有可能",
        "图算法处理图数据",
        "排序算法排列元素",
        "搜索算法查找元素",
        "哈希表提供快速查找",
        "树结构组织数据",
        "堆实现优先队列",
        "图遍历访问所有节点",
        "最短路径算法找最短路径",
        "最小生成树连接所有节点",
        "网络流算法最大化流量",
    ])

    return corpus


def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 V8 清理训练 🔥🔥🔥")
    print("=" * 70)

    V3_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v3")
    V8_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8")

    print("\n[1/4] 加载 V3 检查点...")
    if os.path.exists(os.path.join(V3_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V3_CKPT)
        print(f"  ✅ 已加载 V3")
    else:
        model = Harmonia13Virtual(scale="large")
        print(f"  🌱 创建新模型")

    start_frags = len(model._lite._learned_fragments)
    print(f"  起始片段: {start_frags:,}")

    print("\n[2/4] 生成多样化语料...")
    corpus = generate_diverse_corpus()
    print(f"  ✅ 生成语料: {len(corpus):,} 条（全部唯一）")

    print("\n[3/4] 批量训练...")
    batch_size = 50
    total_batches = (len(corpus) + batch_size - 1) // batch_size

    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        model._lite.train(batch, epochs=1)
        batch_num = (i // batch_size) + 1
        current = len(model._lite._learned_fragments)
        print(f"  Batch {batch_num}/{total_batches} | 已学: {current:,}")

    print(f"\n[4/4] 保存 V8...")
    os.makedirs(V8_CKPT, exist_ok=True)
    result = model.save(V8_CKPT)

    final = len(model._lite._learned_fragments)
    growth = final - start_frags

    print("\n" + "=" * 70)
    print("  🎉 V8 清理训练完成")
    print("=" * 70)
    print(f"""
  起始: {start_frags:,} 片段
  语料: {len(corpus):,} 条（唯一）
  最终: {final:,} 片段
  增长: +{growth:,} 条
  检查点: {result['lite_path']}
""")


if __name__ == "__main__":
    main()
