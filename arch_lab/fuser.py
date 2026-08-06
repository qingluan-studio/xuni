"""
多亲本融合器：把 N 个基因组(来自手绘图)融合成一个新架构。
三种策略：
  - chain:  顺序串联，前一个的输出接后一个的入口
  - parallel: 并联分支，所有分支共享 stem，输出汇聚到一个融合节点
  - graft:  渐进嫁接，逐对 crossover 叠加所有亲本的特征
融合后的架构再经 repair 保证合法，并可裁剪到 max_nodes。
"""
from __future__ import annotations
import random
from typing import List
from .genome import Genome, Node, repair
from .config import Config


def fuse_chain(genomes: List[Genome], cfg: Config) -> Genome:
    """顺序串联：g0 → g1 → g2 → ... 每段的 stem 入口接上一段末节点。"""
    if not genomes:
        return Genome([])
    result = genomes[0].copy()
    for k in range(1, len(genomes)):
        g = genomes[k]
        offset = len(result.nodes)
        last_prev = offset - 1
        for n in g.nodes:
            m = n.copy()
            # 偏移内部输入
            m.inputs = [j + offset if j >= 0 else j for j in n.inputs]
            # stem(-1) 入口改接上一段末节点
            m.inputs = [last_prev if j == -1 else j for j in m.inputs]
            result.nodes.append(m)
    return repair(result, cfg)


def fuse_parallel(genomes: List[Genome], cfg: Config, rng: random.Random) -> Genome:
    """并联分支：所有分支从 stem(-1) 出发，各自独立计算，末尾用一个融合节点汇聚。"""
    if not genomes:
        return Genome([])
    result = Genome([])
    branch_terminals: List[int] = []   # 每个分支的末节点索引
    for g in genomes:
        offset = len(result.nodes)
        for n in g.nodes:
            m = n.copy()
            m.inputs = [j + offset if j >= 0 else -1 for j in n.inputs]
            result.nodes.append(m)
        branch_terminals.append(len(result.nodes) - 1)
    # 融合节点：汇聚所有分支末节点 + stem
    # 用一个 attn 做跨分支注意力融合（呼应"涌现"——多分支信息融合）
    if len(result.nodes) < cfg.max_nodes:
        merge_node = Node(op="attn", act="gelu", expand=2.0,
                          heads=next((h for h in [4, 8, 2] if cfg.channels % h == 0), 4),
                          inputs=list(branch_terminals) + [-1])
        result.nodes.append(merge_node)
    else:
        # 若超限，把最后一个节点改成汇聚所有分支
        last = result.nodes[-1]
        for t in branch_terminals:
            if t != len(result.nodes) - 1 and t not in last.inputs:
                last.inputs.append(t)
    return repair(result, cfg)


def fuse_graft(genomes: List[Genome], cfg: Config, rng: random.Random) -> Genome:
    """渐进嫁接：从第一个基因组开始，逐个把后续基因组的核心子图嫁接进来。"""
    from .operations import crossover
    if not genomes:
        return Genome([])
    result = genomes[0].copy()
    for k in range(1, len(genomes)):
        # 每次取第 k 个基因组的一段子图嫁接进当前结果
        result = crossover(result, genomes[k], cfg, rng)
        # 偶尔再串一小段（增加多样性）
        if rng.random() < 0.3 and k < len(genomes) - 1:
            result = crossover(result, genomes[k + 1], cfg, rng)
    return repair(result, cfg)


def fuse_all(genomes: List[Genome], cfg: Config, rng: random.Random) -> List[Genome]:
    """返回三种融合策略各产出一个，供进化选择。"""
    return [
        fuse_chain(genomes, cfg),
        fuse_parallel(genomes, cfg, rng),
        fuse_graft(genomes, cfg, rng),
    ]


def genome_signature(g: Genome) -> str:
    """简短可读签名。"""
    return "→".join(n.op for n in g.nodes)
