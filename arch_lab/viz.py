"""
可视化：进化曲线、motif 频率演变、最优架构 DAG、Pareto 前沿。
中文标签会尝试使用 Noto/WenQuanYi 字体，缺失则回退英文，避免方框。
"""
from __future__ import annotations
import os
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .genome import Genome
from .emergence import EmergenceTracker
from .evolution import GenerationLog

_CN_FONTS = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Noto Sans CJK JP",
             "Source Han Sans SC", "Microsoft YaHei", "PingFang SC", "SimHei", "DejaVu Sans"]


def _set_font():
    try:
        from matplotlib.font_manager import findfont, FontProperties
        plt.rcParams["font.sans-serif"] = _CN_FONTS
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def plot_evolution(history: List[GenerationLog], path: str):
    _set_font()
    gens = [h.gen for h in history]
    best = [h.best_fitness for h in history]
    mean = [h.mean_fitness for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(gens, best, "o-", label="最优适应度")
    ax[0].plot(gens, mean, "s--", label="平均适应度")
    ax[0].set_xlabel("代"); ax[0].set_ylabel("适应度"); ax[0].set_title("进化曲线")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    accs = [h.best_acc for h in history]
    ax[1].plot(gens, [a if a is not None else float("nan") for a in accs], "o-", color="tab:green")
    ax[1].set_xlabel("代"); ax[1].set_ylabel("验证准确率"); ax[1].set_title("精英真实准确率")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_motif_frequency(tracker: EmergenceTracker, path: str, topk: int = 8):
    _set_font()
    if not tracker.history:
        return
    # 选末期频率最高的 motif
    last = tracker.history[-1]
    motifs = sorted(last, key=last.get, reverse=True)[:topk]
    gens = list(range(len(tracker.history)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in motifs:
        ax.plot(gens, tracker.freq_series(m), "o-", label=m)
    ax.set_xlabel("代"); ax.set_ylabel("在精英中的出现频率")
    ax.set_title("架构 motif 频率演变（涌现追踪）")
    ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


_OP_COLORS = {
    "conv3": "#4C72B0", "conv5": "#55A868", "conv1": "#8172B2",
    "ffn": "#C44E52", "attn": "#DD8452", "identity": "#999999", "norm_act": "#937860",
}


def draw_architecture(genome: Genome, path: str, title: str = "最优架构"):
    _set_font()
    n = len(genome.nodes)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2 + 2), 5))
    # 布局：stem 在左，节点顺序向右
    pos = {}
    pos[-1] = (0, 0.5)
    for i in range(n):
        pos[i] = (i + 1, 0.5 + 0.18 * ((i % 3) - 1))
    # 画边
    for i, node in enumerate(genome.nodes):
        for j in node.inputs:
            x0, y0 = pos[j]; x1, y1 = pos[i]
            style = "-" if j == i - 1 or j == -1 else "--"
            ax.plot([x0, x1], [y0, y1], style, color="gray", alpha=0.6, zorder=1)
    # 画 stem
    ax.scatter([pos[-1][0]], [pos[-1][1]], s=900, c="#222", zorder=3)
    ax.text(pos[-1][0], pos[-1][1], "stem", color="white", ha="center", va="center", fontsize=9, zorder=4)
    # 画节点
    for i, node in enumerate(genome.nodes):
        x, y = pos[i]
        c = _OP_COLORS.get(node.op, "#333")
        ax.scatter([x], [y], s=1100, c=c, zorder=3, edgecolors="black")
        label = node.op if node.op not in ("ffn", "attn") else f"{node.op}\n{node.heads}h" if node.op == "attn" else f"ffn\nx{int(node.expand)}"
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=7.5, zorder=4)
    # 输出
    ox = n + 1.5
    ax.scatter([ox], [0.5], s=1000, c="#222", marker="s", zorder=3)
    ax.text(ox, 0.5, "pool\n→head", color="white", ha="center", va="center", fontsize=8, zorder=4)
    for t in genome.terminal_nodes():
        ax.plot([pos[t][0], ox], [pos[t][1], 0.5], "-", color="black", alpha=0.7, zorder=1)
    ax.set_title(f"{title}  (节点={n})")
    ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pareto(front_points, path: str):
    """绘制 Pareto 前沿散点图(准确率 vs 参数量)。"""
    _set_font()
    if not front_points:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    params = [p.params / 1000 for p in front_points]    # K 参数
    accs = [p.acc for p in front_points]
    ax.plot(params, accs, "o--", color="tab:red", markersize=9, label="Pareto 前沿", zorder=3)
    for p, pa in zip(front_points, params):
        ax.annotate(f"{p.acc:.3f}", (pa, p.acc), textcoords="offset points",
                    xytext=(6, 6), fontsize=7)
    ax.set_xlabel("参数量 (K)")
    ax.set_ylabel("验证准确率")
    ax.set_title("Pareto 前沿: 准确率 vs 参数量")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
