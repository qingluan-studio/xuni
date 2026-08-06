"""
种子架构：把 5 张手绘架构图解析成基因组，作为进化初始种群的一部分。
每张图的核心拓扑特征被映射为 DAG 节点 + 跳连。
"""
from __future__ import annotations
from typing import List
from .genome import Genome, Node


def _g(nodes_spec) -> Genome:
    """快速构造：nodes_spec = [(op, act, expand, heads, [inputs]), ...]"""
    nodes = []
    for op, act, exp, heads, inputs in nodes_spec:
        nodes.append(Node(op=op, act=act, expand=exp, heads=heads, inputs=list(inputs)))
    return Genome(nodes)


def seed_architectures() -> List[Genome]:
    """返回 5 个种子基因组，对应 5 张手绘图。"""
    seeds = []

    # ---- 图1：沙漏瓶颈 + 密集跳连 (U-Net 风格) ----
    # 输入→编码→中心注意力瓶颈→解码(带对称跳连)→输出融合
    seeds.append(_g([
        ("conv3", "relu", 1.0, 4, [-1]),        # 0: 编码层1
        ("conv5", "relu", 1.0, 4, [0]),          # 1: 编码层2(下采样)
        ("attn",  "gelu", 2.0, 4, [1, -1]),      # 2: 中心瓶颈(stem跳连)
        ("conv5", "gelu", 1.0, 4, [2, 1]),       # 3: 解码层1(跳连回node1)
        ("conv3", "silu", 1.0, 4, [3, 0]),       # 4: 解码层2(跳连回node0)
        ("ffn",   "gelu", 4.0, 8, [4, 2]),       # 5: 输出融合(跳连回瓶颈)
    ]))

    # ---- 图2：六边形交叉连接 (双分支并行+交叉跳连) ----
    # 两条并行分支→中心融合→交叉跳连到对侧→输出合并
    seeds.append(_g([
        ("conv3", "relu", 1.0, 4, [-1]),         # 0: 左分支
        ("conv5", "silu", 1.0, 4, [-1]),         # 1: 右分支(并行)
        ("attn",  "gelu", 2.0, 4, [0, 1]),       # 2: 中心融合
        ("conv5", "gelu", 1.0, 4, [2, 1]),       # 3: 右扩展(交叉跳连自node1)
        ("conv3", "silu", 1.0, 4, [2, 0]),       # 4: 左扩展(交叉跳连自node0)
        ("ffn",   "relu", 4.0, 8, [3, 4]),       # 5: 输出合并
    ]))

    # ---- 图3：半圆弧形路径 (顺序流+长程跳连) ----
    # 沿弧顺序传播，弧两端有长程跳连回起点
    seeds.append(_g([
        ("conv3", "relu", 1.0, 4, [-1]),         # 0: 弧左端
        ("conv5", "gelu", 1.0, 4, [0]),          # 1: 弧上行
        ("attn",  "gelu", 2.0, 8, [1]),          # 2: 弧顶(注意力)
        ("conv5", "silu", 1.0, 4, [2]),          # 3: 弧下行
        ("ffn",   "relu", 4.0, 8, [3, 0, -1]),   # 4: 弧右端(长程跳连回node0和stem)
    ]))

    # ---- 图4：圆形+菱形门控循环 (LSTM式双时间尺度) ----
    # 自循环注意力→双支路(卷积/FFN)→合并→反馈跳连(模拟循环)
    seeds.append(_g([
        ("attn",  "gelu", 2.0, 4, [-1]),         # 0: 顶部(自循环→残差注意力)
        ("conv3", "silu", 1.0, 4, [0]),          # 1: 左支路
        ("ffn",   "gelu", 4.0, 8, [0]),          # 2: 中心(FFN门控)
        ("conv5", "relu", 1.0, 4, [1, 2]),       # 3: 右支路(合并)
        ("conv3", "silu", 1.0, 4, [3, 0]),       # 4: 底部(反馈跳连回顶部)
    ]))

    # ---- 图5：对称菱形中心瓶颈 (编码-瓶颈-解码) ----
    # 双输入并行→中心注意力瓶颈→交叉解码→合并
    seeds.append(_g([
        ("conv3", "relu", 1.0, 4, [-1]),         # 0: 左上
        ("conv5", "silu", 1.0, 4, [-1]),         # 1: 右上(并行)
        ("attn",  "gelu", 2.0, 4, [0, 1]),       # 2: 中心瓶颈
        ("conv5", "gelu", 1.0, 4, [2, 1]),       # 3: 右下(交叉跳连)
        ("ffn",   "relu", 4.0, 8, [2, 0]),       # 4: 左下(交叉跳连)
        ("conv3", "silu", 1.0, 4, [3, 4]),       # 5: 底部合并
    ]))

    return seeds
