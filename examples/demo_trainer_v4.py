"""
demo_trainer_v4.py —— 逆天而行：多仓库 + 50000 轮

策略：直接从已下载的目录扫描 .py 文件，不走大 JSON 缓存。

运行：
  cd /workspace/xuni
  python examples/demo_trainer_v4.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual, CorpusDownloader


CACHE_DIR = os.path.join(os.path.dirname(__file__), "corpus_cache")


def _scan_py_files(root: str) -> list[str]:
    """扫描目录下所有 .py 文件内容"""
    texts = []
    skip_dirs = {"__pycache__", ".git", "test", "tests", "idlelib",
                 "tkinter", "turtledemo", "site-packages", "doc", "docs"}
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


def _extract_fragments(text: str, max_lines: int = 20):
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


def main():
    print("=" * 60)
    print("  🔥 xuni v4 —— 逆天而行：多仓库 + 50000 轮")
    print("=" * 60)

    # ---------------------------------------------------------------------
    # 1. 从已下载目录扫描代码
    # ---------------------------------------------------------------------
    print(f"\n[1/6] 从缓存目录扫描代码...")

    # 已知的仓库目录
    repo_dirs = [
        (os.path.join(CACHE_DIR, "python_cpython_main"), "CPython 标准库"),
        (os.path.join(CACHE_DIR, "psf_requests_main"), "requests HTTP"),
        (os.path.join(CACHE_DIR, "pallets_flask_main"), "flask Web"),
        (os.path.join(CACHE_DIR, "pallets_click_main"), "click CLI"),
        (os.path.join(CACHE_DIR, "pydantic_pydantic_main"), "pydantic 验证"),
    ]

    all_fragments = []
    repo_stats = []

    for repo_dir, desc in repo_dirs:
        if not os.path.isdir(repo_dir):
            print(f"  ❌ {desc}: 目录不存在")
            repo_stats.append({"desc": desc, "files": 0, "frags": 0, "ok": False})
            continue

        texts = _scan_py_files(repo_dir)
        frags = []
        for text in texts:
            frags.extend(_extract_fragments(text, max_lines=20))
        all_fragments.extend(frags)

        total_kb = sum(len(t.encode("utf-8")) for t in texts) / 1024
        print(f"  ✅ {desc}: {len(texts)} 文件 → {len(frags):,} 片段 ({total_kb:.0f} KB)")
        repo_stats.append({"desc": desc, "files": len(texts), "frags": len(frags), "ok": True})

    # 工厂自身代码
    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    xuni_texts = _scan_py_files(xuni_dir)
    xuni_frags = []
    for text in xuni_texts:
        xuni_frags.extend(_extract_fragments(text, max_lines=20))
    all_fragments.extend(xuni_frags)
    print(f"  🏭 工厂自身: {len(xuni_texts)} 文件 → {len(xuni_frags)} 片段")

    # v2 语料
    v2_path = os.path.join(os.path.dirname(__file__), "real_corpus.json")
    if os.path.exists(v2_path):
        with open(v2_path, "r", encoding="utf-8") as f:
            v2_corpus = json.load(f)
        all_fragments.extend(v2_corpus)
        print(f"  📝 v2 语料: {len(v2_corpus)} 条")

    print(f"\n  📚 总训练片段: {len(all_fragments):,} 条")

    if len(all_fragments) < 100:
        print("  ⚠ 片段不足，退出")
        return

    # ---------------------------------------------------------------------
    # 2. 创建模型 + 基线
    # ---------------------------------------------------------------------
    print("\n[2/6] 创建模型 + 基线...")
    model = Harmonia13Virtual(scale="mini")

    baseline_prompts = [
        "def quicksort", "class Counter", "def lru_cache",
        "import functools", "class deque", "def namedtuple",
        "class OrderedDict", "def partial", "async def",
        "class Event", "class Flask", "def request",
        "import numpy", "class BaseModel", "def command",
    ]

    print("\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        r = model._lite.generate(p, max_new_tokens=50)
        baseline[p] = r
        print(f"  [{p}] → {r[:60]}")

    # ---------------------------------------------------------------------
    # 3. 50000 轮训练
    # ---------------------------------------------------------------------
    print(f"\n[3/6] 50000 轮训练...")

    start = time.time()
    batch_size = 16
    num_epochs = 50000
    log = []

    for epoch in range(num_epochs):
        batch = random.sample(all_fragments, min(batch_size, len(all_fragments)))
        model._lite.train(batch, epochs=1)

        if (epoch + 1) % 5000 == 0:
            elapsed = time.time() - start
            learned = len(model._lite._learned_fragments)
            frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg = sum(frags) / max(1, len(frags))
            active = sum(1 for f in frags if f > 0)

            print(f"  Epoch {epoch+1:6d} | 已学: {learned:10,d} | "
                  f"活跃: {active:2d} | 均载: {avg:,.0f} | "
                  f"用时: {elapsed:.1f}s")

            log.append({
                "epoch": epoch + 1, "learned": learned,
                "active": active, "avg_load": round(avg, 1),
                "elapsed": round(elapsed, 2),
            })

    total_time = time.time() - start
    print(f"\n  ✅ 训练完成！用时: {total_time:.2f}s")

    # ---------------------------------------------------------------------
    # 4. 评估
    # ---------------------------------------------------------------------
    print("\n[4/6] 评估...")

    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)

    print(f"\n  📊 模型规模:")
    print(f"    已学: {learned:,} 条")
    print(f"    活跃: {active} / {len(model._lite.experts)}")
    for name, frags in expert_frags:
        bar = "█" * min(50, frags // 1000)
        print(f"    {name:12s} [{bar}] {frags:7,d}")

    # ---------------------------------------------------------------------
    # 5. 对比
    # ---------------------------------------------------------------------
    print(f"\n[5/6] 前后对比...")

    improved = 0
    for p in baseline_prompts:
        before = baseline[p]
        after = model._lite.generate(p, max_new_tokens=50)
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
        "version": "v4",
        "repo_stats": repo_stats,
        "total_fragments": len(all_fragments),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {n: f for n, f in expert_frags},
        "training_time": round(total_time, 2),
        "growth_log": log,
        "improved": improved,
        "total_compared": len(baseline_prompts),
        "comparison": {
            p: {"before": baseline[p], "after": model._lite.generate(p, max_new_tokens=50)}
            for p in baseline_prompts
        },
    }

    report_path = os.path.join(os.path.dirname(__file__), "trainer_v4_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    # 不保存大检查点（800k片段太大），只保存元信息
    meta_path = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v4_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta = {
        "version": "v4",
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {n: f for n, f in expert_frags},
        "training_time": round(total_time, 2),
        "epochs": num_epochs,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元信息: {meta_path}")

    print("\n" + "=" * 60)
    print("  🔥 v4 逆天训练总结")
    print("=" * 60)
    print(f"""
  📚 片段: {len(all_fragments):,} 条
  🔄 训练: {num_epochs:,} × {batch_size} = {num_epochs*batch_size:,}
  🧠 吸收: {learned:,} 条
  👥 活跃: {active} / {len(model._lite.experts)}
  ⏱️ 用时: {total_time:.2f}s
  📈 提升: {improved}/{len(baseline_prompts)}

  积少成多:
    v1:  1,000 轮 /     5,000
    v2:  5,000 轮 /    40,000
    v3: 10,000 轮 /   120,000
    v4: 50,000 轮 / {learned:>10,} 🔥

  只要不停，只是时间问题。
""")


if __name__ == "__main__":
    main()
