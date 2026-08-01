"""
train_v7_fresh.py —— V7 全新训练（轻量版，避免 OOM）

策略：
    不加载 V6（太大），直接从 V3 开始 + 海量新语料
    训练 300,000 轮，batch_size=20
    不保存中间检查点，只保存最终结果

运行：
    cd /workspace/xuni
    python examples/train_v7_fresh.py
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

# ============================================================ #
# 配置
# ============================================================ #

V3_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v3")
V7_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v7")

TRAINING_EPOCHS = 100
BATCH_SIZE = 500
REPORT_INTERVAL = 10

# ============================================================ #
# 语料收集
# ============================================================ #

def scan_files(root: str, exts: set, skip: set) -> list:
    texts = []
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                fp = os.path.join(dp, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    if len(text.strip()) > 30:
                        texts.append(text)
                except Exception:
                    pass
    return texts


def extract_code(text: str, max_lines: int = 20) -> list:
    lines = text.split("\n")
    frags = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "async def ")):
            indent = len(line) - len(stripped)
            frag_lines = [line]
            j = i + 1
            while j < len(lines) and j - i < max_lines:
                nl = lines[j]
                ns = nl.lstrip()
                if ns and not ns.startswith("#"):
                    ni = len(nl) - len(ns)
                    if ni <= indent and ns.startswith(("def ", "class ", "async def ", "@", "import ", "from ")):
                        break
                frag_lines.append(nl)
                j += 1
            while frag_lines and not frag_lines[-1].strip():
                frag_lines.pop()
            if len(frag_lines) >= 2:
                frags.append("\n".join(frag_lines))
            i = j
        else:
            i += 1
    return frags


def extract_text(text: str, max_len: int = 200) -> list:
    paragraphs = re.split(r'\n\s*\n', text)
    frags = []
    for p in paragraphs:
        p = p.strip()
        if 30 < len(p) <= max_len:
            frags.append(p)
        elif len(p) > max_len:
            sentences = re.split(r'(?<=[。！？.!?])\s+', p)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) < max_len:
                    cur += s
                else:
                    if cur:
                        frags.append(cur)
                    cur = s
            if cur:
                frags.append(cur)
    return frags


def load_gz(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 V7 全新训练 🔥🔥🔥")
    print("=" * 70)

    # 1. 加载 V3 检查点
    print("\n[1/5] 加载 V3 检查点...")
    if os.path.exists(os.path.join(V3_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V3_CKPT)
        print(f"  ✅ 已加载 V3")
    else:
        print(f"  🌱 创建新模型")
        model = Harmonia13Virtual(scale="large")

    start_frags = len(model._lite._learned_fragments)
    print(f"  起始片段: {start_frags:,}")

    # 2. 收集海量语料
    print("\n[2/5] 收集海量语料...")
    all_frags = []
    skip = {"__pycache__", ".git", "node_modules", "vendor", "dist", "build",
            ".next", ".cache", "venv", "env", ".venv"}

    # 扫描整个项目
    print("  📁 扫描项目全部文件...")
    texts = scan_files(
        os.path.join(os.path.dirname(__file__), ".."),
        {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".rst"},
        skip
    )
    for t in texts:
        if t.strip().startswith(("def ", "class ")):
            all_frags.extend(extract_code(t))
        else:
            all_frags.extend(extract_text(t))
    print(f"     → {len(texts)} 文件, {len(all_frags):,} 片段")

    # 加载所有语料文件
    sources = [
        ("/workspace/xuni/examples/real_corpus.json", "real_corpus"),
        ("/workspace/xuni/corpus_clean.json", "corpus_clean"),
    ]
    for path, name in sources:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_frags.extend(data)
                print(f"  📝 {name}: {len(data):,}")

    # 加载压缩语料
    for gz in ["/workspace/xuni/corpus_particles_large.json.gz",
               "/workspace/xuni/corpus_clean_large.json.gz"]:
        if os.path.exists(gz):
            data = load_gz(gz)
            all_frags.extend(data)
            print(f"  📦 {os.path.basename(gz)}: {len(data):,}")

    # 加载 V2 学习片段
    v2_path = "/workspace/xuni/examples/checkpoints/harmonia_v2/harmonia_lite.json.gz"
    if os.path.exists(v2_path):
        with gzip.open(v2_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        learned = data.get("learned_fragments", [])
        all_frags.extend(learned)
        print(f"  📦 V2 学习片段: {len(learned):,}")

    # 去重
    before = len(all_frags)
    seen = set()
    unique = []
    for f in all_frags:
        key = f.strip()[:150]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    all_frags = unique
    print(f"\n  📚 去重: {before:,} → {len(all_frags):,} 片段")

    # 如果语料不够，用种子补充
    if len(all_frags) < 500:
        seeds = [
            "合鸣-13 是 xuni 虚拟生态的旗舰对话模型，由 13 位专家组成",
            "虚拟电场将采样点密度转化为能量，驱动整个生态系统",
            "MoE 混合专家架构用关键词共振实现非传统路由",
            "双态系统让虚拟模型在粒子态被真正训练，数据层被真实调用",
            "物理建模合成器用数字振荡器和共鸣滤波器生成声音",
            "超混沌采样器实时生成上亿采样点，内存 O(1)",
            "水动力学把采样点当流体粒子，有蒸发凝结和涡旋",
            "玻璃逻辑把计算当光学系统，数据是光，函数是透镜",
            "Kuramoto 振子网络模拟神经同步，Hebbian 学习强化连接",
            "虚拟凭证把场能量铸造成 24 位令牌，可验证可消耗",
        ]
        all_frags = seeds * 1000
        print(f"  ⚠️ 使用种子语料: {len(all_frags):,}")

    # 3. 训练
    print(f"\n[3/5] 开始训练 {TRAINING_EPOCHS:,} 轮...")
    start_time = time.time()
    log = []

    for epoch in range(1, TRAINING_EPOCHS + 1):
        batch = random.sample(all_frags, min(BATCH_SIZE, len(all_frags)))
        model._lite.train(batch, epochs=1)

        if epoch % REPORT_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - start_time
            learned = len(model._lite._learned_fragments)
            total = sum(len(e.get("fragments", [])) for e in model._lite.experts)
            print(f"  Epoch {epoch:7,d} | 已学: {learned:12,d} | 专家总: {total:12,d} | 用时: {elapsed:.1f}s")
            log.append({"epoch": epoch, "learned": learned, "total": total, "elapsed": round(elapsed, 2)})

    total_time = time.time() - start_time
    print(f"\n  ✅ 训练完成！用时: {total_time:.2f}s")

    # 4. 保存（跳过评估，避免卡住）
    print(f"\n[5/5] 保存 V7...")
    os.makedirs(V7_CKPT, exist_ok=True)
    result = model.save(V7_CKPT)
    print(f"  ✅ 保存完成: {result['lite_path']}")
    print(f"     片段数: {result['fragments']:,}")

    # 报告
    report = {
        "version": "v7",
        "base": "v3_fresh",
        "epochs": TRAINING_EPOCHS,
        "batch": BATCH_SIZE,
        "corpus": len(all_frags),
        "fragments_learned": len(model._lite._learned_fragments),
        "time": round(total_time, 2),
        "log": log,
        "experts": {e["name"]: len(e.get("fragments", [])) for e in model._lite.experts},
    }
    rp = os.path.join(os.path.dirname(__file__), "trainer_v7_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {rp}")

    # 总结
    final = len(model._lite._learned_fragments)
    print("\n" + "=" * 70)
    print("  🎉 V7 训练完成")
    print("=" * 70)
    print(f"""
  起始: {start_frags:,} 片段
  训练: {TRAINING_EPOCHS:,} 轮
  语料: {len(all_frags):,} 条
  最终: {final:,} 片段
  增长: +{final - start_frags:,} 条
  用时: {total_time:.2f}s
""")


if __name__ == "__main__":
    main()
