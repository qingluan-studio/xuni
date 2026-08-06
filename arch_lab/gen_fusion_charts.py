"""归元×TuiLi融合实验图表生成"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/arch_lab/runs/guiyuan_tuili"

with open(os.path.join(OUT, "guiyuan_tuili_results.json")) as f:
    data = json.load(f)

# ============================================================
# 图1: 训练曲线对比
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {"TuiLi (冠军)": "#4A90D9", "归元推理 (融合)": "#FF6B35"}

for ax, task_key, task_title, ylim in [
    (axes[0], "cls_results", "MNIST 分类任务", (0.7, 1.0)),
    (axes[1], "add_results", "MNIST 加法推理任务", (0.1, 0.5)),
]:
    for name, color in colors.items():
        hist = data[task_key][name]["hist"]
        epochs = [h["epoch"] + 1 for h in hist]
        accs = [h["acc"] for h in hist]
        ax.plot(epochs, accs, "o-", color=color, linewidth=2.5, markersize=8, label=name)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("准确率", fontsize=12)
    ax.set_title(task_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 9))

fig.suptitle("归元 × TuiLi 融合实验 — 训练曲线对比", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fusion_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ fusion_training_curves.png")

# ============================================================
# 图2: 融合增益柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

tasks = ["分类准确率", "推理准确率"]
tuili_vals = [data["cls_results"]["TuiLi (冠军)"]["acc"] * 100,
              data["add_results"]["TuiLi (冠军)"]["acc"] * 100]
fused_vals = [data["cls_results"]["归元推理 (融合)"]["acc"] * 100,
              data["add_results"]["归元推理 (融合)"]["acc"] * 100]

x = np.arange(len(tasks))
w = 0.32
bars1 = ax.bar(x - w/2, tuili_vals, w, color="#4A90D9", label="TuiLi (冠军)", edgecolor="white")
bars2 = ax.bar(x + w/2, fused_vals, w, color="#FF6B35", label="归元推理 (融合)", edgecolor="white")

for bar, val in zip(bars1, tuili_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#4A90D9")
for bar, val in zip(bars2, fused_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold", color="#FF6B35")

# 标注增益
for i in range(len(tasks)):
    gain = fused_vals[i] - tuili_vals[i]
    ax.annotate(f"+{gain:.1f}%", xy=(x[i] + w/2, fused_vals[i]),
               xytext=(x[i] + w/2 + 0.15, fused_vals[i] + 3),
               fontsize=13, fontweight="bold", color="#2ECC71",
               arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=2))

ax.set_ylabel("准确率 (%)", fontsize=13)
ax.set_title("归元 × TuiLi 融合增益", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.set_ylim(0, max(fused_vals) * 1.25)
ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fusion_gain_bars.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ fusion_gain_bars.png")

# ============================================================
# 图3: 参数效率对比
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

models = [
    ("TuiLi (冠军)", data["cls_results"]["TuiLi (冠军)"]["acc"] * 100,
     data["add_results"]["TuiLi (冠军)"]["acc"] * 100,
     data["cls_results"]["TuiLi (冠军)"]["params"] / 1000),
    ("归元推理 (融合)", data["cls_results"]["归元推理 (融合)"]["acc"] * 100,
     data["add_results"]["归元推理 (融合)"]["acc"] * 100,
     data["cls_results"]["归元推理 (融合)"]["params"] / 1000),
]

colors_scatter = ["#4A90D9", "#FF6B35"]
for i, (name, cls, add, params) in enumerate(models):
    ax.scatter(params, cls, s=250, c=colors_scatter[i], edgecolors="white",
              linewidth=2, zorder=5, label=name)
    ax.annotate(f"{name}\n({cls:.1f}% / {add:.1f}%)", (params, cls),
               textcoords="offset points", xytext=(12, 8), fontsize=11, fontweight="bold")

ax.set_xlabel("参数量 (K)", fontsize=13)
ax.set_ylabel("分类准确率 (%)", fontsize=13)
ax.set_title("参数效率对比", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(110, 145)
ax.set_ylim(94, 97.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fusion_param_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ fusion_param_efficiency.png")

# ============================================================
# 图4: 与历届冠军总排行
# ============================================================
all_champs = [
    ("归元推理", 96.4, 44.4, 132752),
    ("TuiLi", 95.6, 44.9, 116496),
    ("Chrome", 95.2, 35.4, 112175),
    ("Scrambled\nSPARK", 95.5, 36.3, 126649),
    ("SPARK", 94.8, 34.9, 126649),
    ("Agent", 93.9, 30.1, 94878),
    ("TRAE", 92.8, 29.1, 100371),
    ("Kimi", 90.8, 27.9, 83498),
    ("DeepSeek", 90.5, 28.7, 182298),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 分类排行
ax = axes[0]
names = [c[0] for c in all_champs]
cls_vals = [c[1] for c in all_champs]
colors_bar = ["#FF6B35" if c[0] == "归元推理" else ("#2ECC71" if "TuiLi" in c[0] else "#4A90D9") for c in all_champs]
bars = ax.barh(range(len(names)), cls_vals, color=colors_bar, edgecolor="white", height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("分类准确率 (%)", fontsize=12)
ax.set_title("MNIST 分类排行", fontsize=14, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, cls_vals):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(88, 98)

# 推理排行
ax = axes[1]
add_vals = [c[2] for c in all_champs]
bars = ax.barh(range(len(names)), add_vals, color=colors_bar, edgecolor="white", height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("推理准确率 (%)", fontsize=12)
ax.set_title("MNIST 加法推理排行", fontsize=14, fontweight="bold")
ax.invert_yaxis()
for bar, val in zip(bars, add_vals):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(25, 48)

fig.suptitle("历届字母架构冠军总排行 (橙色=归元推理, 绿色=TuiLi原冠军)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fusion_champion_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ fusion_champion_ranking.png")

print("\n所有图表已生成到:", OUT)
