"""
涌现追踪 v2：从单纯的"相邻算子对"升级为**图级 motif 挖掘**。
新增 motif 类别：
  - 子图模式: fork(一进多出), merge(多进一出), chain(线性链), bottleneck(宽→窄→宽)
  - 路径模式: 最长路径长度, stem直达输出, 多跳残差链
  - 扇入扇出: 高扇入节点(多输入融合), 高扇出节点(特征分发)
  - 跳连分类: stem_skip, long_skip(≥2跳), cross_skip(跨分支), dense(≥3输入)
  - 算子组合: 邻接对 + 三元组(如 conv→attn→ffn)
  - 结构指标: 分支数, 并行度, 跳连密度
"""
from __future__ import annotations
from collections import defaultdict
from typing import List, Dict, Set, Tuple
from .genome import Genome


def _fanin(g: Genome, i: int) -> int:
    """节点 i 的输入数（不含 stem）。"""
    return sum(1 for j in g.nodes[i].inputs if j != -1)


def _fanout(g: Genome, i: int) -> int:
    """节点 i 被多少后续节点引用。"""
    return sum(1 for n in g.nodes for j in n.inputs if j == i)


def _longest_path(g: Genome) -> int:
    """DAG 最长路径（边数），-1 视为起点。"""
    memo: Dict[int, int] = {}

    def dp(i: int) -> int:
        if i in memo:
            return memo[i]
        best = 0
        for j in g.nodes[i].inputs:
            if j == -1:
                best = max(best, 1)
            else:
                best = max(best, dp(j) + 1)
        memo[i] = best
        return best

    if not g.nodes:
        return 0
    return max(dp(i) for i in range(len(g.nodes)))


def _has_path(g: Genome, src: int, dst: int) -> bool:
    """是否存在 src→...→dst 的路径。"""
    if src == dst:
        return True
    visited = set()
    stack = [src]
    while stack:
        i = stack.pop()
        if i in visited:
            continue
        visited.add(i)
        for j in range(i + 1, len(g.nodes)):
            if i in g.nodes[j].inputs:
                if j == dst:
                    return True
                stack.append(j)
    return False


def genome_motifs(g: Genome) -> Set[str]:
    """提取一个基因组的完整 motif 集合（v2 图级挖掘）。"""
    s: Set[str] = set()
    if not g.nodes:
        return s
    n = len(g.nodes)

    # ---- 1. 算子级 ----
    for i, node in enumerate(g.nodes):
        s.add(f"op:{node.op}")
        s.add(f"act:{node.act}")

    # ---- 2. 邻接对 + 三元组 ----
    for i in range(n):
        for j in g.nodes[i].inputs:
            if j == -1:
                continue
            s.add(f"pair:{g.nodes[j].op}->{g.nodes[i].op}")
            # 三元组: k→j→i
            for k in g.nodes[j].inputs:
                if k != -1:
                    s.add(f"tri:{g.nodes[k].op}->{g.nodes[j].op}->{g.nodes[i].op}")

    # ---- 3. 子图模式 ----
    for i in range(n):
        fi, fo = _fanin(g, i), _fanout(g, i)
        if fi >= 2:
            s.add("subgraph:merge")           # 多输入融合
            s.add(f"merge_fanin:{fi}")
        if fo >= 2:
            s.add("subgraph:fork")            # 一进多出(分发)
            s.add(f"fork_fanout:{fo}")
        if fi == 1 and fo == 1:
            s.add("subgraph:chain")           # 纯线性链
        if fi >= 2 and fo >= 2:
            s.add("subgraph:hub")             # 高扇入高扇出枢纽

    # ---- 4. 跳连分类 ----
    for i, node in enumerate(g.nodes):
        for j in node.inputs:
            if j == -1 and i > 0:
                s.add("skip:stem_skip")       # 直接引用 stem(绕过前面所有节点)
            elif j != -1 and j <= i - 2:
                gap = i - j
                s.add(f"skip:long_skip")      # 跨≥2跳
                if gap >= 3:
                    s.add("skip:very_long")   # 跨≥3跳
        # 密集连接: 单节点≥3个输入
        real_inputs = sum(1 for j in node.inputs if j != -1)
        if real_inputs >= 3:
            s.add("skip:dense_input")

    # ---- 5. 瓶颈模式 (宽→窄→宽) ----
    for i in range(1, n - 1):
        prev_fanin = _fanin(g, i - 1) + 1  # +1 for stem or its input
        curr_fanin = _fanin(g, i) + 1
        next_fanin = _fanin(g, i + 1) + 1
        if curr_fanin < prev_fanin and curr_fanin < next_fanin:
            s.add("pattern:bottleneck")
        if g.nodes[i].op == "attn" and _fanin(g, i) >= 2:
            s.add("pattern:attn_merge")     # 注意力做融合点

    # ---- 6. 并行分支 (两个节点都从同一源出发) ----
    for i in range(n):
        for j in range(i + 1, n):
            common = set(g.nodes[i].inputs) & set(g.nodes[j].inputs)
            if len(common) >= 1 and any(c != -1 for c in common):
                s.add("pattern:parallel_branch")
                break

    # ---- 7. 结构指标 ----
    lp = _longest_path(g)
    s.add(f"metric:longest_path:{lp}")
    s.add(f"metric:depth_bucket:{n // 2}")
    total_skips = sum(1 for node in g.nodes for j in node.inputs if j != -1 and j < (g.nodes.index(node) - 1) if g.nodes.index(node) > 0)
    # 简化：跳连密度 = 非相邻边数 / 总边数
    total_edges = sum(len(node.inputs) for node in g.nodes)
    non_adj = sum(1 for idx, node in enumerate(g.nodes) for j in node.inputs if j != -1 and j < idx - 1)
    if total_edges > 0:
        density = non_adj / total_edges
        s.add(f"metric:skip_density:{density:.1f}")
    s.add(f"metric:avg_fanin:{sum(_fanin(g, i) for i in range(n)) / n:.1f}")

    # ---- 8. 经典组合检测 ----
    ops_seq = [node.op for node in g.nodes]
    # conv→attn→ffn (CNN-Transformer 混合)
    for a, b, c in zip(ops_seq, ops_seq[1:], ops_seq[2:]):
        if b == "attn" and a in ("conv3", "conv5") and c == "ffn":
            s.add("combo:conv_attn_ffn")
        if a == "attn" and c == "attn":
            s.add("combo:attn_x_attn")    # 双注意力
    # 注意力+FFN 邻接
    for a, b in zip(ops_seq, ops_seq[1:]):
        if {a, b} == {"attn", "ffn"}:
            s.add("combo:attn_ffn_adj")

    return s


class EmergenceTracker:
    """追踪 motif 频率随代数的演变，检测涌现。"""

    def __init__(self, threshold: float = 0.5, initial_rare: float = 0.3):
        self.threshold = threshold
        self.initial_rare = initial_rare
        self.history: List[Dict[str, float]] = []
        self.initial: Dict[str, float] = {}

    def _freq(self, genomes: List[Genome]) -> Dict[str, float]:
        counts = defaultdict(int)
        n = max(1, len(genomes))
        for g in genomes:
            for m in genome_motifs(g):
                counts[m] += 1
        return {k: v / n for k, v in counts.items()}

    def record(self, elites: List[Genome]) -> Dict[str, float]:
        freq = self._freq(elites)
        self.history.append(freq)
        if len(self.history) == 1:
            self.initial = dict(freq)
        return freq

    def emerged(self) -> List[Dict]:
        """返回"涌现"的 motif：初始罕见→末期普及并越过阈值。"""
        if len(self.history) < 2:
            return []
        last = self.history[-1]
        out = []
        for m, f in last.items():
            f0 = self.initial.get(m, 0.0)
            if f0 < self.initial_rare and f >= self.threshold:
                out.append({"motif": m, "initial_freq": round(f0, 2),
                            "final_freq": round(f, 2), "growth": round(f - f0, 2)})
        out.sort(key=lambda d: d["growth"], reverse=True)
        return out

    def freq_series(self, motif: str) -> List[float]:
        return [h.get(motif, 0.0) for h in self.history]

    def top_motifs(self, k: int = 10) -> List[Tuple[str, float]]:
        """末期频率最高的 k 个 motif。"""
        if not self.history:
            return []
        last = self.history[-1]
        return sorted(last.items(), key=lambda x: x[1], reverse=True)[:k]
