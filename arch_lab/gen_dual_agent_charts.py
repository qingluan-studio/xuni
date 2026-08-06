"""双Agent协作实验图表生成"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for fp in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/arch_lab/runs/dual_agent"
with open(os.path.join(OUT, "dual_agent_results.json")) as f:
    data = json.load(f)

colors = {
    "Solo-TRAE": "#4A90D9",
    "Solo-Kimi": "#E74C3C",
    "Parallel-Vote": "#2ECC71",
    "Parallel-Learned": "#9B59B6",
    "Sequential-Fuse": "#F39C12",
}

# ============================================================
# 图1: 训练曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (task_key, task_title) in zip(axes, [("cls_results", "MNIST 分类"), ("add_results", "MNIST 加法推理")]):
    for name, d in data[task_key].items():
        epochs = [h["epoch"] for h in d["hist"]]
        accs = [h["acc"] * 100 for h in d["hist"]]
        ax.plot(epochs, accs, marker="o", label=name, color=colors.get(name, "gray"),
                linewidth=2, markersize=4, alpha=0.85)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title(task_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 100)

plt.suptitle("TRAE × Kimi 双Agent协作 — 训练曲线", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dual_agent_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ dual_agent_training_curves.png")

# ============================================================
# 图2: 对比柱状图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (task_key, task_title) in zip(axes, [("cls_results", "MNIST 分类"), ("add_results", "MNIST 加法推理")]):
    names = list(data[task_key].keys())
    accs = [data[task_key][n]["acc"] * 100 for n in names]
    x = np.arange(len(names))
    w = 0.55
    bars = ax.bar(x, accs, w, color=[colors.get(n, "gray") for n in names], edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title(task_title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, max(accs) * 1.15)
    ax.grid(True, alpha=0.2, axis="y")

plt.suptitle("双Agent协作 vs 单Agent — 5种模式对比", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dual_agent_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ dual_agent_comparison.png")

# ============================================================
# 图3: 增益分析 (相对Solo-TRAE的增益)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
modes = ["Solo-TRAE", "Solo-Kimi", "Parallel-Vote", "Parallel-Learned", "Sequential-Fuse"]
cls_baseline = data["cls_results"]["Solo-TRAE"]["acc"] * 100
add_baseline = data["add_results"]["Solo-TRAE"]["acc"] * 100

cls_gains = [data["cls_results"][m]["acc"] * 100 - cls_baseline for m in modes]
add_gains = [data["add_results"][m]["acc"] * 100 - add_baseline for m in modes]

x = np.arange(len(modes))
w = 0.35
bars1 = ax.bar(x - w/2, cls_gains, w, color="#4A90D9", label="分类增益", edgecolor="white")
bars2 = ax.bar(x + w/2, add_gains, w, color="#E74C3C", label="推理增益", edgecolor="white")

for bar, val in zip(bars1, cls_gains):
    yoff = 0.05 if val >= 0 else -0.15
    ax.text(bar.get_x() + bar.get_width()/2, val + yoff,
            f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold",
            color="#2ECC71" if val > 0 else "#E74C3C")
for bar, val in zip(bars2, add_gains):
    yoff = 0.05 if val >= 0 else -0.15
    ax.text(bar.get_x() + bar.get_width()/2, val + yoff,
            f"{val:+.1f}%", ha="center", fontsize=10, fontweight="bold",
            color="#2ECC71" if val > 0 else "#E74C3C")

ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.set_ylabel("相对 Solo-TRAE 的增益 (%)", fontsize=12)
ax.set_title("协作模式增益分析 — 正数=比单独TRAE好", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dual_agent_gain_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ dual_agent_gain_analysis.png")

# ============================================================
# 图4: 对比之前的8字母融合翻车
# ============================================================
# 加载8字母融合结果
tk_path = "/workspace/arch_lab/runs/trae_kimi/trae_kimi_results.json"
tk_data = {}
if os.path.exists(tk_path):
    with open(tk_path) as f:
        tk_data = json.load(f)

fig, ax = plt.subplots(figsize=(10, 6))
# 数据准备
approaches = []
cls_vals = []
add_vals = []
colors_bar = []

# 单Agent
approaches.append("Solo-TRAE")
cls_vals.append(data["cls_results"]["Solo-TRAE"]["acc"] * 100)
add_vals.append(data["add_results"]["Solo-TRAE"]["acc"] * 100)
colors_bar.append("#4A90D9")

approaches.append("Solo-Kimi")
cls_vals.append(data["cls_results"]["Solo-Kimi"]["acc"] * 100)
add_vals.append(data["add_results"]["Solo-Kimi"]["acc"] * 100)
colors_bar.append("#E74C3C")

# 8字母融合 (翻车)
if tk_data and "TRAE×Kimi (TK→Ri→Am→Ei)" in tk_data["cls_results"]:
    approaches.append("8字母融合 (翻车)")
    cls_vals.append(tk_data["cls_results"]["TRAE×Kimi (TK→Ri→Am→Ei)"]["acc"] * 100)
    add_vals.append(tk_data["add_results"]["TRAE×Kimi (TK→Ri→Am→Ei)"]["acc"] * 100)
    colors_bar.append("#E8463A")

# 双Agent协作冠军
approaches.append("双Agent-Learned")
cls_vals.append(data["cls_results"]["Parallel-Learned"]["acc"] * 100)
add_vals.append(data["add_results"]["Parallel-Learned"]["acc"] * 100)
colors_bar.append("#9B59B6")

x = np.arange(len(approaches))
w = 0.35
bars1 = ax.bar(x - w/2, cls_vals, w, color=[c for c in colors_bar], alpha=0.7, label="分类", edgecolor="white")
bars2 = ax.bar(x + w/2, add_vals, w, color=[c for c in colors_bar], alpha=1.0, label="推理",
               edgecolor="white", hatch="///")

for bar, val in zip(bars1, cls_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
for bar, val in zip(bars2, add_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(approaches, fontsize=11)
ax.set_ylabel("准确率 (%)", fontsize=12)
ax.set_title("拆散重组 vs 保持完整协作 — 关键对比", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dual_agent_vs_fusion.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ dual_agent_vs_fusion.png")

print("\n全部图表生成完成!")
