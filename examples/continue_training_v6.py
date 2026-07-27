"""
continue_training_v6.py —— 从 V3 继续训练到 V6

训练流程：
    1. 加载 V3 检查点（120,000 片段）
    2. 扫描开源代码仓库，提取新片段
    3. 增量训练，吸收新语料
    4. 保存为新检查点 V6

运行：
    cd /workspace/xuni
    python examples/continue_training_v6.py
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

# ============================================================ #
# 配置
# ============================================================ #

V3_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v3")
V6_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v6")
CORPUS_CACHE = os.path.join(os.path.dirname(__file__), "corpus_cache")

TRAINING_EPOCHS = 100000  # 训练轮数
BATCH_SIZE = 20           # 每轮吸收的片段数
SAVE_INTERVAL = 10000     # 每隔多少轮保存一次

# ============================================================ #
# 语料提取函数
# ============================================================ #

def scan_py_files(root: str, skip_dirs: set) -> list:
    """扫描目录下所有 .py 文件"""
    texts = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            if f.endswith(".py"):
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    if len(text.strip()) > 50:
                        texts.append(text)
                except Exception:
                    pass
    return texts


def extract_fragments(text: str, max_lines: int = 20) -> list:
    """从代码中提取训练片段（函数/类定义）"""
    lines = text.split("\n")
    fragments = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "async def ")):
            indent = len(line) - len(stripped)
            frag_lines = [line]
            j = i + 1
            while j < len(lines) and j - i < max_lines:
                next_line = lines[j]
                next_stripped = next_line.lstrip()
                if next_stripped and not next_stripped.startswith("#"):
                    next_indent = len(next_line) - len(next_stripped)
                    if next_indent <= indent and next_stripped.startswith(
                        ("def ", "class ", "async def ", "@", "import ", "from ")
                    ):
                        break
                frag_lines.append(next_line)
                j += 1
            while frag_lines and not frag_lines[-1].strip():
                frag_lines.pop()
            if len(frag_lines) >= 3:
                fragments.append("\n".join(frag_lines))
            i = j
        else:
            i += 1
    return fragments


def load_json_corpus(path: str) -> list:
    """加载 JSON 语料文件"""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gz_corpus(path: str) -> list:
    """加载 gzip 压缩的 JSON 语料"""
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ============================================================ #
# 主训练流程
# ============================================================ #

def main():
    print("=" * 60)
    print("  🔥 合鸣-13 继续训练：V3 → V6")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. 加载 V3 检查点
    # -------------------------------------------------------
    print("\n[1/5] 加载 V3 检查点...")
    if os.path.exists(V3_CKPT):
        model = Harmonia13Virtual.load(V3_CKPT)
        print(f"  ✅ 已加载 V3")
    else:
        print(f"  🌱 V3 不存在，创建新模型")
        model = Harmonia13Virtual(scale="large")

    # 统计当前状态
    current_frags = len(model._lite._learned_fragments)
    print(f"  当前已学片段: {current_frags:,}")

    # -------------------------------------------------------
    # 2. 收集训练语料
    # -------------------------------------------------------
    print("\n[2/5] 收集训练语料...")

    all_fragments = []
    skip_dirs = {"__pycache__", ".git", "test", "tests", "idlelib",
                 "tkinter", "turtledemo", "site-packages", "doc", "docs",
                 "node_modules", "vendor", "dist", "build"}

    # 2.1 扫描语料缓存目录（如果存在）
    if os.path.isdir(CORPUS_CACHE):
        print(f"  📁 扫描语料缓存: {CORPUS_CACHE}")
        cache_texts = scan_py_files(CORPUS_CACHE, skip_dirs)
        cache_frags = []
        for text in cache_texts:
            cache_frags.extend(extract_fragments(text, max_lines=20))
        all_fragments.extend(cache_frags)
        print(f"     → {len(cache_texts)} 文件, {len(cache_frags):,} 片段")

    # 2.2 扫描工厂自身代码
    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    if os.path.isdir(xuni_dir):
        print(f"  🏭 扫描工厂代码: {xuni_dir}")
        xuni_texts = scan_py_files(xuni_dir, skip_dirs)
        xuni_frags = []
        for text in xuni_texts:
            xuni_frags.extend(extract_fragments(text, max_lines=20))
        all_fragments.extend(xuni_frags)
        print(f"     → {len(xuni_texts)} 文件, {len(xuni_frags):,} 片段")

    # 2.3 加载已有语料文件
    corpus_files = [
        os.path.join(os.path.dirname(__file__), "real_corpus.json"),
        "/workspace/xuni/corpus_clean.json",
    ]
    for cf in corpus_files:
        if os.path.exists(cf):
            print(f"  📝 加载语料: {cf}")
            corpus = load_json_corpus(cf)
            all_fragments.extend(corpus)
            print(f"     → {len(corpus):,} 片段")

    # 2.4 加载压缩语料
    gz_files = [
        "/workspace/xuni/corpus_particles_large.json.gz",
        "/workspace/xuni/corpus_clean_large.json.gz",
    ]
    for gf in gz_files:
        if os.path.exists(gf):
            print(f"  📦 加载压缩语料: {gf}")
            gz_corpus = load_gz_corpus(gf)
            if isinstance(gz_corpus, list):
                all_fragments.extend(gz_corpus)
                print(f"     → {len(gz_corpus):,} 片段")

    print(f"\n  📚 总训练语料: {len(all_fragments):,} 片段")

    if len(all_fragments) < 100:
        print("  ⚠️ 语料不足，使用内置种子语料")
        all_fragments = [
            "合鸣-13 是由 13 位虚拟专家组成的 MoE 对话模型",
            "虚拟电场将采样点密度转化为能量，驱动模型运行",
            "双态系统让虚拟模型能够被真正训练和调用",
            "关键词共振门控是合鸣的非传统路由机制",
        ] * 1000

    # -------------------------------------------------------
    # 3. 开始训练
    # -------------------------------------------------------
    print(f"\n[3/5] 开始训练 {TRAINING_EPOCHS:,} 轮...")
    print(f"  每轮吸收 {BATCH_SIZE} 条片段")

    start_time = time.time()
    log = []

    for epoch in range(1, TRAINING_EPOCHS + 1):
        # 随机抽取一批片段
        batch = random.sample(all_fragments, min(BATCH_SIZE, len(all_fragments)))

        # 训练一步
        result = model._lite.train(batch, epochs=1)

        # 定期报告
        if epoch % SAVE_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - start_time
            learned = len(model._lite._learned_fragments)
            expert_total = sum(len(e.get("fragments", [])) for e in model._lite.experts)

            print(f"  Epoch {epoch:7,d} | 已学: {learned:12,d} | "
                  f"专家总片段: {expert_total:12,d} | 用时: {elapsed:.1f}s")

            log.append({
                "epoch": epoch,
                "learned": learned,
                "expert_total": expert_total,
                "elapsed": round(elapsed, 2),
            })

    total_time = time.time() - start_time
    print(f"\n  ✅ 训练完成！总用时: {total_time:.2f}s")

    # -------------------------------------------------------
    # 4. 评估对比
    # -------------------------------------------------------
    print("\n[4/5] 评估对比...")

    test_prompts = [
        "def quicksort",
        "class NeuralNetwork",
        "import numpy",
        "async def fetch",
        "class APIRouter",
        "def train_model",
        "import torch",
        "class DataLoader",
    ]

    print("\n  --- 生成示例 ---")
    for p in test_prompts[:4]:
        result = model._lite.generate(p, max_new_tokens=60)
        print(f"  [{p}] → {result[:80]}")

    # -------------------------------------------------------
    # 5. 保存检查点 V6
    # -------------------------------------------------------
    print(f"\n[5/5] 保存检查点到 {V6_CKPT}...")
    os.makedirs(V6_CKPT, exist_ok=True)
    save_result = model.save(V6_CKPT)
    print(f"  ✅ 已保存:")
    print(f"     元数据: {save_result['meta_path']}")
    print(f"     权重: {save_result['lite_path']}")
    print(f"     片段数: {save_result['fragments']:,}")

    # 保存训练报告
    report = {
        "version": "v6",
        "base_checkpoint": "v3",
        "training_epochs": TRAINING_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_corpus": len(all_fragments),
        "fragments_learned": len(model._lite._learned_fragments),
        "training_time_seconds": round(total_time, 2),
        "growth_log": log,
        "expert_load": {
            e["name"]: len(e.get("fragments", []))
            for e in model._lite.experts
        },
    }
    report_path = os.path.join(os.path.dirname(__file__), "trainer_v6_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {report_path}")

    # -------------------------------------------------------
    # 最终总结
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  🎉 训练完成总结")
    print("=" * 60)
    print(f"""
  📊 训练统计:
    基线检查点: V3 ({current_frags:,} 片段)
    训练轮数: {TRAINING_EPOCHS:,}
    总语料: {len(all_fragments):,} 条
    最终吸收: {len(model._lite._learned_fragments):,} 条
    训练用时: {total_time:.2f}s

  💾 检查点已保存到: {V6_CKPT}

  积少成多:
    v1:   1,000 轮 /       5,000 片段
    v2:   5,000 轮 /      40,000 片段
    v3:  10,000 轮 /     120,000 片段
    v6: {TRAINING_EPOCHS:>9,} 轮 / {len(model._lite._learned_fragments):>12,} 片段 🎉
""")


if __name__ == "__main__":
    main()