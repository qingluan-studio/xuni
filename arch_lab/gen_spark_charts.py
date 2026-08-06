"""SPARK 实验图表生成 — 训练曲线 + 历届冠军排行 + 参数效率"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体
plt.rcParams["font.family"] = "Noto Sans CJK SC"
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/arch_lab/runs/spark"

# ---- 加载SPARK结果 ----
with open(os.path.join(OUT, "spark_results.json")) as f:
    spark = json.load(f)

# ---- 历届冠军数据 ----
champions = [
    {"name": "TuiLi",   "label": "TuiLi\n(T→U→I→L→I)",   "cls": 0.956, "add": 0.449, "params": 116496},
    {"name": "Chrome",  "label": "Chrome\n(C→H→R→O→M→E)", "cls": 0.952, "add": 0.354, "params": 112175},
    {"name": "SPARK",   "label": "SPARK\n(S→P→A→R→K)",    "cls": 0.948, "add": 0.349, "params": 126649},
    {"name": "Agent",   "label": "Agent\n(A→G→E→N→T)",    "cls": 0.939, "add": 0.301, "params":  94878},
    {"name": "TRAE",    "label": "TRAE\n(T→R→A→E)",       "cls": 0.928, "add": 0.291, "params": 100371},
    {"name": "Kimi",    "label": "Kimi\n(K→i→m→i)",       "cls": 0.908, "add": 0.279, "params":  83498},
    {"name": "DeepSeek","label": "DeepSeek\n(D→E→E→P→S→E→E→K)","cls": 0.905, "add": 0.287, "params": 182298},
]

# ============================================================
# 图1: SPARK 训练曲线 (分类 + 推理)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 分类任务
ax = axes[0]
for name, color in [("SPARK (S→P→A→R→K)", "#FF6B35"), ("Scrambled (R→P→S→A→K)", "#7B7B7B")]:
    hist = spark["cls_results"][name]["hist"]
    epochs = [h["epoch"] + 1 for h in hist]
    accs = [h["acc"] for h in hist]
    ax.plot(epochs, accs, "o-", color=color, linewidth=2.5, markersize=7, label=name)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("准确率", fontsize=12)
ax.set_title("MNIST 分类任务", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.35, 1.0)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 9))

# 推理任务
ax = axes[1]
for name, color in [("SPARK (S→P→A→R→K)", "#FF6B35"), ("Scrambled (R→P→S→A→K)", "#7B7B7B")]:
    hist = spark["add_results"][name]["hist"]
    epochs = [h["epoch"] + 1 for h in hist]
    accs = [h["acc"] for h in hist]
    ax.plot(epochs, accs, "o-", color=color, linewidth=2.5, markersize=7, label=name)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("准确率", fontsize=12)
ax.set_title("MNIST 加法推理任务", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0.08, 0.42)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, 9))

fig.suptitle("SPARK 字母架构 — 训练曲线对比", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spark_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ spark_training_curves.png")

# ============================================================
# 图2: 历届字母架构冠军排行
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 分类排行
ax = axes[0]
names = [c["label"] for c in champions]
cls_accs = [c["cls"] * 100 for c in champions]
colors = ["#FF6B35" if c["name"] == "SPARK" else "#4A90D9" for c in champions]
bars = ax.barh(range(len(names)), cls_accs, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("分类准确率 (%)", fontsize=12)
ax.set_title("MNIST 分类排行", fontsize=14, fontweight="bold")
ax.invert_yaxis()
for i, (bar, acc) in enumerate(zip(bars, cls_accs)):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(88, 97)

# 推理排行
ax = axes[1]
add_accs = [c["add"] * 100 for c in champions]
bars = ax.barh(range(len(names)), add_accs, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("推理准确率 (%)", fontsize=12)
ax.set_title("MNIST 加法推理排行", fontsize=14, fontweight="bold")
ax.invert_yaxis()
for i, (bar, acc) in enumerate(zip(bars, add_accs)):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{acc:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(25, 48)

fig.suptitle("历届字母架构冠军总排行 (橙色 = SPARK)", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spark_champion_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ spark_champion_ranking.png")

# ============================================================
# 图3: 参数效率散点图 (分类 + 推理)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

for c in champions:
    color = "#FF6B35" if c["name"] == "SPARK" else "#4A90D9"
    size = 200 if c["name"] == "SPARK" else 120
    ax.scatter(c["params"] / 1000, c["cls"] * 100, s=size, c=color,
              edgecolors="white", linewidth=1.5, zorder=5)
    ax.annotate(c["name"], (c["params"] / 1000, c["cls"] * 100),
               textcoords="offset points", xytext=(8, 5), fontsize=11, fontweight="bold")

ax.set_xlabel("参数量 (K)", fontsize=13)
ax.set_ylabel("分类准确率 (%)", fontsize=13)
ax.set_title("参数效率: 准确率 vs 参数量", fontsize=15, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.set_xlim(75, 195)
ax.set_ylim(89, 97)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "spark_param_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ spark_param_efficiency.png")

# ============================================================
# 图4: 综合热力图 (所有字母架构 x 两任务)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# 包含 SPARK + Scrambled + 历届冠军
all_models = [
    ("TuiLi (正确)",        0.956, 0.449),
    ("Scrambled SPARK",     0.955, 0.363),
    ("Chrome (正确)",       0.952, 0.354),
    ("SPARK (正确)",        0.948, 0.349),
    ("Agent (正确)",        0.939, 0.301),
    ("TRAE (正确)",         0.928, 0.291),
    ("Scrambled TuiLi",     0.956, 0.406),
    ("Scrambled Chrome",    0.940, 0.337),
    ("Scrambled DeepSeek",  0.925, 0.242),
    ("DeepSeek (正确)",     0.905, 0.287),
    ("Kimi (正确)",         0.908, 0.279),
    ("Scrambled Kimi",      0.908, 0.288),
    ("Scrambled Agent",     0.883, 0.269),
    ("Scrambled TRAE",      0.878, 0.232),
]

model_names = [m[0] for m in all_models]
cls_vals = [m[1] for m in all_models]
add_vals = [m[2] for m in all_models]
data = np.array([cls_vals, add_vals])

im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0.2, vmax=1.0)
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["分类", "推理"], fontsize=12)

for i in range(2):
    for j in range(len(model_names)):
        val = data[i, j]
        text_color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.1%}", ha="center", va="center",
               fontsize=7, color=text_color, fontweight="bold")

plt.colorbar(im, ax=ax, label="准确率")
ax.set_title("所有字母架构实验热力图", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "spark_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ spark_heatmap.png")

print("\n所有图表已生成到:", OUT)
