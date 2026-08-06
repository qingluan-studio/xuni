"""
基因组 (Genome) = 一个有向无环图 (DAG)。
- Node = 一个算子基因 (op 类型 + 超参数)，输入来自更早的节点(-1 表示 stem)。
- 所有中间张量保持常量宽度 C 与空间尺寸，因此节点输入可直接"求和"实现残差融合。
- 这种表示让"融合"(拓扑/跳连) 与 "创造"(算子变异) 都很容易保持合法性。
"""
from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- 算子基因库
OP_TYPES = ["conv3", "conv5", "conv1", "ffn", "attn", "identity", "norm_act"]
ACT_TYPES = ["relu", "gelu", "silu"]
EXPAND_CHOICES = [1.0, 2.0, 4.0]
HEAD_CHOICES = [2, 4, 8]

ACT_MAP = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


@dataclass
class Node:
    """一个计算节点：对“其所有输入之和”施加 op，输出 C 通道。"""
    op: str = "conv3"
    act: str = "relu"
    expand: float = 1.0    # ffn 的扩展倍率
    heads: int = 4         # attn 的头数
    inputs: List[int] = field(default_factory=list)   # -1 = stem；否则为节点索引

    def copy(self) -> "Node":
        return Node(self.op, self.act, self.expand, self.heads, list(self.inputs))

    def signature(self) -> tuple:
        return (self.op, self.act, self.expand, self.heads)


@dataclass
class Genome:
    nodes: List[Node] = field(default_factory=list)

    def copy(self) -> "Genome":
        return Genome([n.copy() for n in self.nodes])

    def __len__(self) -> int:
        return len(self.nodes)

    def topological_ok(self) -> bool:
        for i, n in enumerate(self.nodes):
            for j in n.inputs:
                if j != -1 and not (-1 < j < i):
                    return False
        return True

    def terminal_nodes(self) -> List[int]:
        """不被任何后续节点引用的节点（输出融合点）。"""
        used = set()
        for n in self.nodes:
            used.update(j for j in n.inputs if j != -1)
        return [i for i in range(len(self.nodes)) if i not in used]


# ---------------------------------------------------------------- 模型构建
def _act(name: str) -> nn.Module:
    return ACT_MAP[name]()


class AttentionOp(nn.Module):
    """轻量自注意力：在 (B,C,H,W) 上做 token=空间位置的 MHA + 残差。"""
    def __init__(self, c: int, heads: int):
        super().__init__()
        assert c % heads == 0, f"channels {c} 不能被 heads {heads} 整除"
        self.norm = nn.GroupNorm(1, c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.heads = heads
        self.scale = (c // heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)     # (B, N, C)
        qkv = self.qkv(h).reshape(B, -1, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # (3, B, heads, N, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class FFNOp(nn.Module):
    """点式前馈：1x1 扩展 + act + 1x1 收缩，带残差。"""
    def __init__(self, c: int, expand: float, act: str):
        super().__init__()
        hidden = max(c, int(c * expand))
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
        self.act = _act(act)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class ConvOp(nn.Module):
    """深度可分离卷积 + BN + act，带残差。"""
    def __init__(self, c: int, k: int, act: str):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = _act(act)

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


def build_op(node: Node, c: int) -> nn.Module:
    op = node.op
    if op in ("conv3", "conv5", "conv1"):
        return ConvOp(c, {"conv3": 3, "conv5": 5, "conv1": 1}[op], node.act)
    if op == "ffn":
        return FFNOp(c, node.expand, node.act)
    if op == "attn":
        return AttentionOp(c, node.heads)
    if op == "identity":
        return nn.Identity()
    if op == "norm_act":
        return nn.Sequential(nn.BatchNorm2d(c), _act(node.act))
    raise ValueError(f"未知算子: {op}")


class GenomeModel(nn.Module):
    """把 Genome 编译成可训练模型：stem -> 节点DAG -> 全局池化 -> 分类头。"""
    def __init__(self, genome: Genome, in_channels: int, num_classes: int, c: int, stem_target: int = 7):
        super().__init__()
        self.genome = genome
        self.c = c
        # stem：两次 stride2 下采样到 stem_target x stem_target
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.nodes = nn.ModuleList([build_op(n, c) for n in genome.nodes])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        s = self.stem(x)
        outs: Dict[int, torch.Tensor] = {-1: s}
        for i, node in enumerate(self.genome.nodes):
            if node.inputs:
                inp = outs[node.inputs[0]]
                for j in node.inputs[1:]:
                    inp = inp + outs[j]
            else:
                inp = s
            outs[i] = self.nodes[i](inp)
        terminals = self.genome.terminal_nodes()
        if not terminals:
            feat = outs[len(self.genome.nodes) - 1] if len(self.genome) else s
        else:
            feat = outs[terminals[0]]
            for j in terminals[1:]:
                feat = feat + outs[j]
        feat = self.pool(feat).flatten(1)
        return self.head(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- 随机生成
def random_genome(cfg, rng: random.Random) -> Genome:
    n = rng.randint(cfg.min_nodes, cfg.max_nodes)
    nodes: List[Node] = []
    for i in range(n):
        op = rng.choice(OP_TYPES)
        node = Node(
            op=op,
            act=rng.choice(ACT_TYPES),
            expand=rng.choice(EXPAND_CHOICES),
            heads=rng.choice(HEAD_CHOICES),
        )
        # 默认顺序连接到上一个节点
        if i == 0:
            base = [-1]
        else:
            base = [i - 1]
            # 额外跳连到更早节点或 stem
            for j in range(-1, i - 1):
                if rng.random() < cfg.skip_prob:
                    base.append(j)
        node.inputs = base
        nodes.append(node)
    # 保证末节点是 terminal 之一（默认就是）
    return Genome(nodes)


def repair(genome: Genome, cfg) -> Genome:
    """修复非法拓扑、限制节点数。"""
    # 删除越界输入
    for i, n in enumerate(genome.nodes):
        n.inputs = [j for j in n.inputs if j == -1 or (-1 < j < i)]
        if not n.inputs:
            n.inputs = [-1] if i == 0 else [i - 1]
        # 去重保序
        seen = set()
        uniq = []
        for j in n.inputs:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        n.inputs = uniq
    # 限制最大节点数：删掉末尾多余节点并重连
    while len(genome.nodes) > cfg.max_nodes:
        genome.nodes.pop()
    # 若被删空导致没有 terminal，强制最后一个为 terminal
    return genome
