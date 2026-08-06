"""4-Agent协作实验 — 可视化图表生成"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 中文字体
for name in ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "DejaVu Sans"]:
    try:
        fp = fm.FontProperties(family=name)
        if fp.get_name():
            plt.rcParams["font.family"] = name
            break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/arch_lab/runs/four_agent"
with open(os.path.join(OUT, "four_agent_results.json")) as f:
    data = json.load(f)

cls = data["cls_results"]
add = data["add_results"]

# 颜色: Solo用冷色系, 协作用暖色系
COLORS_SOLO = {"Solo-Agent": "#4A90D9", "Solo-TRAE": "#5B5BD6", "Solo-DeepSeek": "#7B68EE", "Solo-Chrome": "#6495ED"}
COLORS_COLL = {"4-Way-Vote": "#E67E22", "4-Way-Router": "#E74C3C", "Hierarchical": "#F39C12", "Feature-Stack": "#C0392B"}
ALL_COLORS = {**COLORS_SOLO, **COLORS_COLL}

# ============================================================
# 图1: 分类准确率排名 (水平条形图)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
items = sorted(cls.items(), key=lambda x: x[1]["acc"])
names = [n for n, _ in items]
accs = [d["acc"] * 100 for _, d in items]
colors = [ALL_COLORS.get(n, "#888") for n in names]
bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("准确率 (%)", fontsize=12)
ax.set_title("MNIST 分类 — 4-Agent协作 vs 单兵作战", fontsize=14, fontweight="bold")
ax.set_xlim(88, 97)
ax.axvline(10, color="gray", ls="--", alpha=0.3, label="随机基线 10%")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_cls_ranking.png"), dpi=150)
plt.close()
print("✓ chart_cls_ranking.png")

# ============================================================
# 图2: 推理准确率排名 (水平条形图)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
items = sorted(add.items(), key=lambda x: x[1]["acc"])
names = [n for n, _ in items]
accs = [d["acc"] * 100 for _, d in items]
colors = [ALL_COLORS.get(n, "#888") for n in names]
bars = ax.barh(names, accs, color=colors, edgecolor="white", height=0.6)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("准确率 (%)", fontsize=12)
ax.set_title("MNIST 加法推理 — 4-Agent协作 vs 单兵作战", fontsize=14, fontweight="bold")
ax.set_xlim(20, 42)
ax.axvline(5.3, color="gray", ls="--", alpha=0.3, label="随机基线 5.3%")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_add_ranking.png"), dpi=150)
plt.close()
print("✓ chart_add_ranking.png")

# ============================================================
# 图3: 参数效率 Pareto 前沿 (散点图)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, results, title in [(axes[0], cls, "MNIST 分类"), (axes[1], add, "MNIST 加法推理")]:
    for name, d in results.items():
        x = d["params"] / 1000
        y = d["acc"] * 100
        c = ALL_COLORS.get(name, "#888")
        marker = "o" if "Solo" in name else "s"
        size = 120 if "Solo" not in name else 80
        ax.scatter(x, y, c=c, s=size, marker=marker, edgecolors="white", linewidth=1.5, zorder=3)
        offset_x = 3 if name != "Solo-DeepSeek" else 3
        offset_y = 0.3 if "Solo" in name else -0.6
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(offset_x, offset_y),
                    fontsize=8, fontweight="bold", color=c)
    ax.set_xlabel("参数量 (K)", fontsize=11)
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
fig.suptitle("参数效率 Pareto — ○单兵 □协作", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_pareto.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ chart_pareto.png")

# ============================================================
# 图4: 协作增益热力图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
best_solo_cls = max(v["acc"] for k, v in cls.items() if "Solo" in k)
best_solo_add = max(v["acc"] for k, v in add.items() if "Solo" in k)

collab_names = ["4-Way-Vote", "4-Way-Router", "Hierarchical", "Feature-Stack"]
matrix = []
for name in collab_names:
    gain_cls = (cls[name]["acc"] - best_solo_cls) * 100
    gain_add = (add[name]["acc"] - best_solo_add) * 100
    matrix.append([gain_cls, gain_add])
matrix = np.array(matrix)

im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=-8, vmax=8)
ax.set_xticks([0, 1])
ax.set_xticklabels(["分类增益", "推理增益"], fontsize=12)
ax.set_yticks(range(len(collab_names)))
ax.set_yticklabels(collab_names, fontsize=11)
for i in range(len(collab_names)):
    for j in range(2):
        val = matrix[i, j]
        color = "white" if abs(val) > 5 else "black"
        ax.text(j, i, f"{val:+.1f}%", ha="center", va="center", fontsize=13, fontweight="bold", color=color)
ax.set_title("协作增益 (vs 最佳单兵Chrome)", fontsize=14, fontweight="bold")
plt.colorbar(im, ax=ax, label="增益 (%)", shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_gain_heatmap.png"), dpi=150)
plt.close()
print("✓ chart_gain_heatmap.png")

# ============================================================
# 图5: 训练曲线对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, results, title in [(axes[0], cls, "分类"), (axes[1], add, "推理")]:
    for name, d in results.items():
        hist = d["hist"]
        epochs = [h["epoch"] + 1 for h in hist]
        accs = [h["acc"] * 100 for h in hist]
        c = ALL_COLORS.get(name, "#888")
        ls = "--" if "Solo" in name else "-"
        lw = 1.5 if "Solo" in name else 2.5
        ax.plot(epochs, accs, ls, color=c, label=name, linewidth=lw, marker="o" if "Solo" not in name else None, markersize=4)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("准确率 (%)", fontsize=11)
    ax.set_title(f"训练曲线 — {title}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
fig.suptitle("4-Agent协作训练曲线 (实线=协作, 虚线=单兵)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ chart_training_curves.png")

# ============================================================
# 图6: 协作策略雷达图 (多维度对比)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
dims = ["分类准确率", "推理准确率", "参数效率\n(1/params)", "训练速度\n(1/time)", "协作简单度"]
# 归一化到0-1
max_cls = max(v["acc"] for v in cls.values())
max_add = max(v["acc"] for v in add.values())
max_params = max(v["params"] for v in cls.values())
max_time = max(v["elapsed"] for v in cls.values())
# 简单度: Vote=1.0, Router=0.8, Hier=0.6, FeatStack=0.4
simplicity = {"4-Way-Vote": 1.0, "4-Way-Router": 0.8, "Hierarchical": 0.6, "Feature-Stack": 0.4}

angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]

for name in collab_names:
    vals = [
        cls[name]["acc"] / max_cls,
        add[name]["acc"] / max_add,
        1 - cls[name]["params"] / max_params,
        1 - cls[name]["elapsed"] / max_time,
        simplicity[name],
    ]
    vals += vals[:1]
    c = ALL_COLORS.get(name, "#888")
    ax.plot(angles, vals, "-", color=c, label=name, linewidth=2)
    ax.fill(angles, vals, alpha=0.08, color=c)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dims, fontsize=10)
ax.set_ylim(0, 1.1)
ax.set_title("4种协作策略 — 多维度雷达对比", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_radar.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ chart_radar.png")

print(f"\n全部图表已保存到: {OUT}/")
