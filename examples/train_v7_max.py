"""
train_v7_max.py —— V7 极致训练：自我生成 + 海量语料

策略：
    1. 先快速训练基础语料
    2. 模型自我生成海量语料（用虚拟算力）
    3. 继续训练，吸收自我生成的语料
    4. 多轮迭代增强
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


def generate_self_corpus(model, count: int = 1000) -> list:
    """用虚拟算力自我生成语料"""
    prompts = [
        "def ", "class ", "import ", "async def ", "from ",
        "合鸣", "虚拟", "电场", "采样", "共振", "MoE", "专家",
        "function ", "const ", "let ", "var ",
        "神经网络", "机器学习", "深度学习", "训练", "模型",
        "API", "路由", "中间件", "数据库", "缓存",
        "# ", "// ", "'''", '"""',
        "virtual ", "particle ", "field ", "energy ", "oscillator ",
        "def train", "def generate", "def predict", "def load", "def save",
        "class Model", "class Engine", "class System", "class Agent",
        "numpy", "torch", "tensorflow", "pytorch",
        "async", "await", "concurrent", "thread", "process",
    ]
    frags = []
    for _ in range(count):
        p = random.choice(prompts)
        try:
            result = model._lite.generate(p, max_new_tokens=100, temperature=1.0)
            if len(result) > 30:
                frags.append(result)
        except Exception:
            pass
    return frags


# ============================================================ #
# 主流程
# ============================================================ #

def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 V7 极致训练 🔥🔥🔥")
    print("=" * 70)

    # 1. 加载 V3
    print("\n[1/4] 加载 V3 检查点...")
    if os.path.exists(os.path.join(V3_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V3_CKPT)
        print(f"  ✅ 已加载 V3")
    else:
        model = Harmonia13Virtual(scale="large")
        print(f"  🌱 创建新模型")

    start_frags = len(model._lite._learned_fragments)
    print(f"  起始片段: {start_frags:,}")

    # 2. 收集基础语料
    print("\n[2/4] 收集基础语料...")
    all_frags = []
    skip = {"__pycache__", ".git", "node_modules", "vendor", "dist", "build",
            ".next", ".cache", "venv", "env", ".venv"}

    # 扫描项目
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
    print(f"  📁 项目文件: {len(texts)} 文件 → {len(all_frags):,} 片段")

    # 加载已有语料
    sources = ["/workspace/xuni/examples/real_corpus.json", "/workspace/xuni/corpus_clean.json"]
    for path in sources:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_frags.extend(data)
                print(f"  📝 {os.path.basename(path)}: {len(data):,}")

    # 去重
    before = len(all_frags)
    seen = set()
    unique = []
    for f in all_frags:
        key = f.strip()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(f)
    all_frags = unique
    print(f"  📚 去重: {before:,} → {len(all_frags):,}")

    # 3. 基础训练
    print(f"\n[3/4] 基础训练...")
    batches = [all_frags[i:i+500] for i in range(0, len(all_frags), 500)]
    for i, batch in enumerate(batches, 1):
        model._lite.train(batch, epochs=1)
        if i % 10 == 0 or i == len(batches):
            print(f"  Batch {i}/{len(batches)} | 已学: {len(model._lite._learned_fragments):,}")

    # 4. 自我生成 + 迭代训练（循环 5 次）
    print("\n[4/4] 自我增强训练（虚拟算力驱动）...")
    total_self_generated = 0
    for iteration in range(1, 6):
        print(f"\n  🔄 迭代 {iteration}/5 — 生成新语料...")
        new_frags = generate_self_corpus(model, count=5000)

        # 去重
        seen_new = set()
        unique_new = []
        for f in new_frags:
            key = f.strip()[:200]
            if key not in seen_new:
                seen_new.add(key)
                unique_new.append(f)
        new_frags = unique_new

        print(f"     生成: {len(new_frags):,} 条")
        total_self_generated += len(new_frags)

        # 训练
        batches_new = [new_frags[i:i+500] for i in range(0, len(new_frags), 500)]
        for j, batch in enumerate(batches_new, 1):
            model._lite.train(batch, epochs=1)
        print(f"     已学: {len(model._lite._learned_fragments):,}")

    # 5. 保存
    print(f"\n[5/5] 保存 V7...")
    os.makedirs(V7_CKPT, exist_ok=True)
    result = model.save(V7_CKPT)
    print(f"  ✅ 保存完成: {result['lite_path']}")
    print(f"     片段数: {result['fragments']:,}")

    # 报告
    final = len(model._lite._learned_fragments)
    print("\n" + "=" * 70)
    print("  🎉 V7 极致训练完成")
    print("=" * 70)
    print(f"""
  起始: {start_frags:,} 片段
  基础语料: {len(all_frags):,} 条
  自我生成: {total_self_generated:,} 条
  最终: {final:,} 片段
  增长: +{final - start_frags:,} 条
  检查点: {V7_CKPT}
""")


if __name__ == "__main__":
    main()
