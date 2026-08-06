"""TRAE×Kimi 融合实验图表生成"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 字体
for fp in ["/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/workspace/arch_lab/runs/trae_kimi"
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(OUT, "trae_kimi_results.json")) as f:
    data = json.load(f)

# ============================================================
# 图1: 训练曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = {"TRAE (T→R→A→E)": "#4A90D9", "Kimi (K→i→m→i)": "#E74C3C",
          "TRAE×Kimi (TK→Ri→Am→Ei)": "#9B59B6"}

for ax, (task_key, task_title) in zip(axes, [("cls_results", "MNIST 分类"), ("add_results", "MNIST 加法推理")]):
    for name, d in data[task_key].items():
        epochs = [h["epoch"] for h in d["hist"]]
        accs = [h["acc"] * 100 for h in d["hist"]]
        ax.plot(epochs, accs, marker="o", label=name, color=colors.get(name, "gray"), linewidth=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title(task_title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 100)

plt.suptitle("TRAE × Kimi 融合实验 — 训练曲线", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "trae_kimi_training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ trae_kimi_training_curves.png")

# ============================================================
# 图2: 对比柱状图 (分类 + 推理)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (task_key, task_title) in zip(axes, [("cls_results", "MNIST 分类"), ("add_results", "MNIST 加法推理")]):
    names = list(data[task_key].keys())
    accs = [data[task_key][n]["acc"] * 100 for n in names]
    params = [data[task_key][n]["params"] / 1000 for n in names]
    x = np.arange(len(names))
    w = 0.5
    bars = ax.bar(x, accs, w, color=[colors.get(n, "gray") for n in names], edgecolor="white", alpha=0.85)
    for bar, val, p in zip(bars, accs, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5,
                f"{p:.0f}K", ha="center", fontsize=9, color="white", fontweight="bold")
    ax.set_ylabel("准确率 (%)", fontsize=12)
    ax.set_title(task_title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    short = [n.split(" (")[0] for n in names]
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylim(0, max(accs) * 1.2)
    ax.grid(True, alpha=0.2, axis="y")

plt.suptitle("TRAE × Kimi — 翻车现场", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "trae_kimi_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ trae_kimi_comparison.png")

# ============================================================
# 图3: 参数效率散点图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for task_key, marker, task_label in [("cls_results", "o", "分类"), ("add_results", "s", "推理")]:
    for name, d in data[task_key].items():
        ax.scatter(d["params"]/1000, d["acc"]*100, marker=marker, s=150,
                   color=colors.get(name, "gray"), edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(f"{name.split(' (')[0]}\n{d['acc']*100:.1f}%",
                    (d["params"]/1000, d["acc"]*100),
                    textcoords="offset points", xytext=(10, 8), fontsize=9)

ax.set_xlabel("参数量 (K)", fontsize=13)
ax.set_ylabel("准确率 (%)", fontsize=13)
ax.set_title("参数效率 — 越大越往左上越好", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.2)

# 图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="分类任务"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="推理任务"),
]
ax.legend(handles=legend_elements, fontsize=11, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "trae_kimi_efficiency.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ trae_kimi_efficiency.png")

# ============================================================
# 图4: 全冠军排行榜 (含历史冠军)
# ============================================================
# 加载所有历史结果
all_champs = {}
# 归元推理 (当前冠军)
gy_path = "/workspace/arch_lab/runs/guiyuan_tuili/guiyuan_tuili_results.json"
if os.path.exists(gy_path):
    with open(gy_path) as f:
        gy = json.load(f)
    all_champs["归元推理"] = {
        "cls": gy["cls_results"]["归元推理 (融合)"]["acc"] * 100,
        "add": gy["add_results"]["归元推理 (融合)"]["acc"] * 100,
        "params": gy["cls_results"]["归元推理 (融合)"]["params"],
    }
# TuiLi
tl_path = "/workspace/arch_lab/runs/tuili/tuili_results.json"
if os.path.exists(tl_path):
    with open(tl_path) as f:
        tl = json.load(f)
    tui_name = [k for k in tl["cls_results"] if "TuiLi" in k][0]
    all_champs["TuiLi"] = {
        "cls": tl["cls_results"][tui_name]["acc"] * 100,
        "add": tl["add_results"][tui_name]["acc"] * 100,
        "params": tl["cls_results"][tui_name]["params"],
    }
# Chrome
ch_path = "/workspace/arch_lab/runs/chrome/chrome_results.json"
if os.path.exists(ch_path):
    with open(ch_path) as f:
        ch = json.load(f)
    cname = [k for k in ch["cls_results"] if "Chrome" in k][0]
    all_champs["Chrome"] = {
        "cls": ch["cls_results"][cname]["acc"] * 100,
        "add": ch["add_results"][cname]["acc"] * 100,
        "params": ch["cls_results"][cname]["params"],
    }
# TRAE, Kimi, TRAE×Kimi
for name, d in data["cls_results"].items():
    short = name.split(" (")[0]
    add_d = data["add_results"][name]
    all_champs[short] = {
        "cls": d["acc"] * 100,
        "add": add_d["acc"] * 100,
        "params": d["params"],
    }

# 排行: 按分类+推理总分
ranked = sorted(all_champs.items(), key=lambda x: x[1]["cls"] + x[1]["add"], reverse=True)

fig, ax = plt.subplots(figsize=(10, 6))
names = [n for n, _ in ranked]
cls_vals = [d["cls"] for _, d in ranked]
add_vals = [d["add"] for _, d in ranked]
totals = [c + a for c, a in zip(cls_vals, add_vals)]

y = np.arange(len(names))
h = 0.35
bars1 = ax.barh(y - h/2, cls_vals, h, color="#4A90D9", label="分类准确率", edgecolor="white")
bars2 = ax.barh(y + h/2, add_vals, h, color="#E74C3C", label="推理准确率", edgecolor="white")

for i, (c, a, t) in enumerate(zip(cls_vals, add_vals, totals)):
    ax.text(c + 0.5, i - h/2, f"{c:.1f}%", va="center", fontsize=10, fontweight="bold", color="#4A90D9")
    ax.text(a + 0.5, i + h/2, f"{a:.1f}%", va="center", fontsize=10, fontweight="bold", color="#E74C3C")
    ax.text(max(c, a) + 5, i, f"总分 {t:.1f}", va="center", fontsize=9, color="#666")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel("准确率 (%)", fontsize=12)
ax.set_title("全冠军排行榜 (含 TRAE×Kimi 翻车)", fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.set_xlim(0, 115)
ax.grid(True, alpha=0.2, axis="x")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUT, "trae_kimi_champion_ranking.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ trae_kimi_champion_ranking.png")

print("\n全部图表生成完成!")
