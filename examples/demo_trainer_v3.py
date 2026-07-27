"""
demo_trainer_v3.py —— 工厂一键获取真实代码 + 10000 轮训练

工厂闭环 v3（zip 极速版）：
    fetch_python_zip() → 一次下载 CPython 标准库 → 扫描 .py 文件
    → get_fragments()  → 切成训练片段
    → train()          → 10000 轮增量训练

运行：
  cd /workspace/xuni
  python examples/demo_trainer_v3.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual, CorpusDownloader


def main():
    print("=" * 60)
    print("  🚀 xuni v3 —— 工厂一键获取真实代码 + 10000 轮训练")
    print("=" * 60)

    # ---------------------------------------------------------------------
    # 1. 工厂一键获取 CPython 标准库
    # ---------------------------------------------------------------------
    print("\n[1/6] 工厂一键获取 CPython 标准库 zip...")
    dl = CorpusDownloader()

    t0 = time.time()
    lib_dir = dl.fetch_python_zip()
    t1 = time.time()

    if lib_dir is None:
        print("  ❌ zip 下载失败，回退到并发 raw 下载...")
        dl2 = CorpusDownloader()
        dl2.fetch_cpython_stdlib(max_files=80, parallel=12)
        dl = dl2
        t1 = time.time()

    texts = dl.get_texts()
    stats = dl.get_stats()

    print(f"\n  📥 下载完成 ({t1-t0:.1f}s):")
    print(f"    文件数: {stats['success']} 个 .py 文件")
    print(f"    总量: {stats['total_kb']} KB")

    # 提取训练片段
    print("\n  ✂️ 提取训练片段...")
    real_fragments = dl.get_fragments(max_lines_per_fragment=20)
    print(f"    提取片段: {len(real_fragments)} 条")

    # 加载已有语料
    v2_corpus_path = os.path.join(os.path.dirname(__file__), "real_corpus.json")
    if os.path.exists(v2_corpus_path):
        with open(v2_corpus_path, "r", encoding="utf-8") as f:
            v2_corpus = json.load(f)
        print(f"    v2 语料: {len(v2_corpus)} 条")
    else:
        v2_corpus = []

    all_corpus = real_fragments + v2_corpus
    print(f"    总语料: {len(all_corpus)} 条")

    if len(all_corpus) < 100:
        print("  ⚠ 语料不足 100 条，可能 zip 下载受限。尝试备用方案...")
        # 备用：使用 v2 语料
        if len(v2_corpus) >= 100:
            all_corpus = v2_corpus
            print(f"    使用 v2 语料: {len(v2_corpus)} 条")
        else:
            print("    ❌ 语料严重不足，退出")
            return

    # ---------------------------------------------------------------------
    # 2. 创建模型 + 训练前基线
    # ---------------------------------------------------------------------
    print("\n[2/6] 创建模型 + 训练前基线...")
    model = Harmonia13Virtual(scale="mini")
    print(f"    初始专家: {len(model._lite.experts)} 位")

    baseline_prompts = [
        "def quicksort",
        "class Counter",
        "def lru_cache",
        "import functools",
        "class deque",
        "def namedtuple",
        "class OrderedDict",
        "def partial",
        "async def",
        "class Event",
    ]

    print("\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        result = model._lite.generate(p, max_new_tokens=40)
        baseline[p] = result
        print(f"  [{p}] → {result[:70]}")

    # ---------------------------------------------------------------------
    # 3. 10000 轮训练
    # ---------------------------------------------------------------------
    print(f"\n[3/6] 开始 10000 轮训练...")
    print("  (每 1000 轮报告)")

    start = time.time()
    batch_size = 12
    num_epochs = 10000
    log = []

    for epoch in range(num_epochs):
        batch = random.sample(all_corpus, min(batch_size, len(all_corpus)))
        model._lite.train(batch, epochs=1)

        if (epoch + 1) % 1000 == 0:
            elapsed = time.time() - start
            learned = len(model._lite._learned_fragments)
            frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg = sum(frags) / max(1, len(frags))
            active = sum(1 for f in frags if f > 0)

            print(f"  Epoch {epoch+1:6d} | 已学: {learned:8d} | "
                  f"活跃: {active:2d} | 均载: {avg:.0f} | "
                  f"用时: {elapsed:.1f}s")

            log.append({
                "epoch": epoch + 1,
                "learned": learned,
                "active": active,
                "avg_load": round(avg, 1),
            })

    total_time = time.time() - start
    print(f"\n  ✅ 训练完成！用时: {total_time:.2f}s")

    # ---------------------------------------------------------------------
    # 4. 训练后评估
    # ---------------------------------------------------------------------
    print("\n[4/6] 训练后评估...")

    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)

    print(f"\n  📊 模型规模:")
    print(f"    已学语料: {learned:,} 条")
    print(f"    活跃专家: {active} / {len(model._lite.experts)}")
    print(f"    各专家负载:")
    for name, frags in expert_frags:
        bar = "█" * min(50, frags // 200)
        print(f"      {name:12s} [{bar}] {frags:6d}")

    # ---------------------------------------------------------------------
    # 5. 前后对比
    # ---------------------------------------------------------------------
    print(f"\n[5/6] 训练前后对比...")

    improved = 0
    for p in baseline_prompts:
        before = baseline[p]
        after = model._lite.generate(p, max_new_tokens=40)
        if len(after) > len(before):
            improved += 1
        print(f"\n  [{p}]")
        print(f"    前: {before[:70]}")
        print(f"    后: {after[:70]}")

    # ---------------------------------------------------------------------
    # 6. 保存
    # ---------------------------------------------------------------------
    print(f"\n[6/6] 保存...")

    report = {
        "version": "v3",
        "downloader_stats": stats,
        "real_texts_count": len(texts),
        "real_fragments": len(real_fragments),
        "v2_corpus": len(v2_corpus),
        "total_corpus": len(all_corpus),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {name: frags for name, frags in expert_frags},
        "training_time": round(total_time, 2),
        "download_time": round(t1 - t0, 2),
        "growth_log": log,
        "improved": improved,
        "total_compared": len(baseline_prompts),
        "comparison": {
            p: {
                "before": baseline[p],
                "after": model._lite.generate(p, max_new_tokens=40),
            }
            for p in baseline_prompts
        },
    }

    report_path = os.path.join(os.path.dirname(__file__), "trainer_v3_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    ckpt = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v3")
    model._lite.save(ckpt)
    print(f"  检查点: {ckpt}")

    print("\n" + "=" * 60)
    print("  📈 v3 训练总结")
    print("=" * 60)
    print(f"""
  📥 下载方式: CPython zip 一键获取 ({t1-t0:.1f}s)
  📚 真实源码: {len(texts)} 个 .py 文件 ({stats['total_kb']} KB)
  ✂️ 训练片段: {len(real_fragments):,} 条
  🔄 训练轮次: {num_epochs:,} × {batch_size}条/轮 = {num_epochs*batch_size:,} 条
  🧠 吸收片段: {learned:,} 条
  👥 活跃专家: {active} / {len(model._lite.experts)}
  📈 生成提升: {improved}/{len(baseline_prompts)}

  积少成多进度:
    v1: 1,000 轮   / 5,000 片段
    v2: 5,000 轮   / 40,000 片段
    v3: 10,000 轮  / {learned:,} 片段 ✅

  工厂闭环:
    一键zip获取 → 扫描 → 粒子化 → 训练 → 生成
    完全自主、免费、极速
""")


if __name__ == "__main__":
    main()
