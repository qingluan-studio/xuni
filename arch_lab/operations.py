"""
融合 (Fusion) 与 创造 (Creation) 操作。
- 创造 = 变异 (mutation)：单个基因组内部做局部改造（换算子/改超参/增删节点/增删跳连）。
- 融合 = 交叉 (crossover) / 嫁接 (graft)：两个基因组杂交，取一片子图拼接进另一个。
所有操作都返回新对象，并通过 repair() 保证拓扑合法。
"""
from __future__ import annotations
import random
from typing import List
from .genome import (
    Genome, Node, OP_TYPES, ACT_TYPES, EXPAND_CHOICES, HEAD_CHOICES, repair,
)


# -------------------------------------------------------------------- 创造(变异)
def mutate(genome: Genome, cfg, rng: random.Random) -> Genome:
    g = genome.copy()
    if not g.nodes:
        return repair(g, cfg)
    # 随机选 1~2 个节点施加变异
    for _ in range(rng.randint(1, 2)):
        kind = rng.choice(["op", "act", "param", "add_node", "del_node", "add_edge", "del_edge"])
        i = rng.randrange(len(g.nodes))
        n = g.nodes[i]
        if kind == "op":
            new = rng.choice([o for o in OP_TYPES if o != n.op] or [n.op])
            n.op = new
            if new == "attn" and cfg.channels % n.heads != 0:
                n.heads = next((h for h in HEAD_CHOICES if cfg.channels % h == 0), n.heads)
        elif kind == "act":
            n.act = rng.choice([a for a in ACT_TYPES if a != n.act] or [n.act])
        elif kind == "param":
            if rng.random() < 0.5:
                n.expand = rng.choice(EXPAND_CHOICES)
            else:
                n.heads = rng.choice([h for h in HEAD_CHOICES if cfg.channels % h == 0] or [n.heads])
        elif kind == "add_node":
            if len(g.nodes) < cfg.max_nodes:
                src = rng.choice(range(-1, i + 1))   # -1..i，插入后这些索引不变
                new_node = Node(op=rng.choice(OP_TYPES), act=rng.choice(ACT_TYPES),
                                expand=rng.choice(EXPAND_CHOICES),
                                heads=rng.choice(HEAD_CHOICES), inputs=[src])
                # 插入位置 i+1 之后的所有节点右移，其指向 >i 的输入也要 +1
                for k in range(i + 1, len(g.nodes)):
                    g.nodes[k].inputs = [j + 1 if (j != -1 and j > i) else j
                                         for j in g.nodes[k].inputs]
                g.nodes.insert(i + 1, new_node)
        elif kind == "del_node":
            if len(g.nodes) > cfg.min_nodes:
                # 把引用 i 的后续节点改指到 i 的第一个输入
                rew = n.inputs[0] if n.inputs else -1
                g.nodes.pop(i)
                for m in g.nodes:
                    m.inputs = [rew if j == i else (j if j < i else j - 1) for j in m.inputs]
        elif kind == "add_edge":
            cand = [j for j in range(-1, i) if j not in n.inputs]
            if cand:
                n.inputs.append(rng.choice(cand))
        elif kind == "del_edge":
            if len(n.inputs) > 1:
                n.inputs.pop(rng.randrange(len(n.inputs)))
    return repair(g, cfg)


# -------------------------------------------------------------------- 融合(交叉)
def crossover(parent_a: Genome, parent_b: Genome, cfg, rng: random.Random) -> Genome:
    """从 B 取一段连续子图，嫁接进 A 的某个位置，重连输入。"""
    a = [n.copy() for n in parent_a.nodes]
    if not parent_b.nodes:
        return repair(Genome(a), cfg)
    # B 的子图 [bi, bj)
    bi = rng.randrange(len(parent_b.nodes))
    bj = rng.randint(bi + 1, len(parent_b.nodes))
    b_sub = [n.copy() for n in parent_b.nodes[bi:bj]]
    insert_at = rng.randint(0, len(a))            # 0..len
    graft_src = (insert_at - 1) if insert_at > 0 else -1   # 嫁接入口(A 中 insert_at-1 或 stem)
    # 第一步：b_sub 内部输入转为 local 索引[0,len)；外部输入用哨兵 -999 标记
    for n in b_sub:
        new_inputs = []
        for j in n.inputs:
            local = j - bi
            if 0 <= local < len(b_sub):
                new_inputs.append(local)
            else:
                new_inputs.append(-999)           # 外部，稍后接 graft_src
        n.inputs = new_inputs
    # 第二步：A 中 insert_at 及之后的节点右移 len(b_sub)；指向 <insert_at 或 stem 的不变
    for m in a[insert_at:]:
        m.inputs = [j if (j < 0 or j < insert_at) else j + len(b_sub) for j in m.inputs]
    # 第三步：拼装；并把 b_sub 的 local 索引 +insert_at，哨兵替换为 graft_src
    combined = a[:insert_at] + b_sub + a[insert_at:]
    for n in b_sub:
        n.inputs = [j + insert_at if j >= 0 else graft_src for j in n.inputs]
    return repair(Genome(combined), cfg)


def fusion(parent_a: Genome, parent_b: Genome, cfg, rng: random.Random) -> Genome:
    """并行融合：A 与 B 串联后，在接口处额外加一条跨支路(来自 B 的一个节点)。"""
    g = parent_a.copy()
    offset = len(g.nodes)
    # 串接 B（B 内部输入偏移 offset，B 的 stem 入口接 A 的末节点）
    last_a = offset - 1
    for n in parent_b.nodes:
        m = n.copy()
        m.inputs = [j if j < 0 else j + offset for j in n.inputs]
        # B 第一个节点的 stem(-1) 入口改接 A 末节点
        m.inputs = [last_a if j == -1 else j for j in m.inputs]
        g.nodes.append(m)
    # 跨支路：从 A 某节点连到 B 某节点（制造跨分支的残差融合）
    if parent_a.nodes and parent_b.nodes:
        ai = rng.randrange(len(parent_a.nodes))
        bi = rng.randrange(len(parent_b.nodes)) + offset
        node_b = g.nodes[bi]
        if ai not in node_b.inputs:
            node_b.inputs.append(ai)
    return repair(g, cfg)
