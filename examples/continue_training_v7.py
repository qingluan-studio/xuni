"""
continue_training_v7.py —— 全方位增强：V6 → V7

增强维度：
    1. 更多语料：扫描整个项目 + 外部语料 + 自我生成语料
    2. 更多轮次：500,000 轮
    3. 专家分化：把通用兜底内容按关键词分发到各专家
    4. 自我增强：用模型生成新语料，再喂给自己

运行：
    cd /workspace/xuni
    python examples/continue_training_v7.py
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

V6_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v6")
V7_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v7")

TRAINING_EPOCHS = 500_000   # 50万轮
BATCH_SIZE = 25             # 每轮吸收更多
SAVE_INTERVAL = 50_000      # 每隔5万轮保存
SELF_ENHANCE_EVERY = 10_000  # 每1万轮自我增强一次

# ============================================================ #
# 语料收集
# ============================================================ #

def scan_all_files(root: str, extensions: set, skip_dirs: set) -> list:
    """扫描所有指定扩展名的文件"""
    texts = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    if len(text.strip()) > 30:
                        texts.append(text)
                except Exception:
                    pass
    return texts


def extract_code_fragments(text: str, max_lines: int = 25) -> list:
    """从代码提取函数/类/注释片段"""
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
            if len(frag_lines) >= 2:
                fragments.append("\n".join(frag_lines))
            i = j
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            # 提取 docstring
            quote = stripped[:3]
            frag_lines = [line]
            j = i + 1
            while j < len(lines) and quote not in lines[j]:
                frag_lines.append(lines[j])
                j += 1
            if j < len(lines):
                frag_lines.append(lines[j])
            if len(frag_lines) >= 3:
                fragments.append("\n".join(frag_lines))
            i = j + 1
        else:
            i += 1
    return fragments


def extract_text_fragments(text: str, max_len: int = 250) -> list:
    """从纯文本提取段落"""
    paragraphs = re.split(r'\n\s*\n', text)
    frags = []
    for p in paragraphs:
        p = p.strip()
        if len(p) > 30 and len(p) <= max_len:
            frags.append(p)
        elif len(p) > max_len:
            # 按句子切分
            sentences = re.split(r'(?<=[。！？.!?])\s+', p)
            current = ""
            for s in sentences:
                if len(current) + len(s) < max_len:
                    current += s
                else:
                    if current:
                        frags.append(current)
                    current = s
            if current:
                frags.append(current)
    return frags


def load_json_corpus(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_gz_corpus(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def generate_self_corpus(model, count: int = 500) -> list:
    """让模型自己生成语料，再喂给自己"""
    prompts = [
        "def ", "class ", "import ", "async def ", "from ",
        "合鸣", "虚拟", "电场", "采样", "共振", "MoE", "专家",
        "function ", "const ", "let ", "var ",
        "if __name__", "try:", "with ", "for ", "while ",
        "神经网络", "机器学习", "深度学习", "训练", "模型",
        "API", "路由", "中间件", "数据库", "缓存",
        "# ", "// ", "/* ", "'''", '"""',
    ]
    frags = []
    for _ in range(count):
        p = random.choice(prompts)
        try:
            result = model._lite.generate(p, max_new_tokens=80, temperature=0.9)
            if len(result) > 20:
                frags.append(result)
        except Exception:
            pass
    return frags


# ============================================================ #
# 专家分化
# ============================================================ #

def distribute_fragments(model):
    """
    把 general 专家的片段按关键词分发到对应领域专家。
    让专家真正专业化。
    """
    general = None
    for e in model._lite.experts:
        if e["id"] == "general":
            general = e
            break
    if general is None:
        return 0

    moved = 0
    new_general = []
    keywords_map = {}
    for e in model._lite.experts:
        if e["id"] != "general":
            keywords_map[e["id"]] = [k.lower() for k in e.get("keywords", [])]

    for frag in general["fragments"]:
        frag_lower = frag.lower()
        best_expert = None
        best_score = 0
        for eid, kws in keywords_map.items():
            score = sum(1 for kw in kws if kw in frag_lower)
            if score > best_score:
                best_score = score
                best_expert = eid

        if best_expert and best_score >= 2:
            for e in model._lite.experts:
                if e["id"] == best_expert:
                    if frag not in e["fragments"]:
                        e["fragments"].append(frag)
                        moved += 1
                    break
        else:
            new_general.append(frag)

    general["fragments"] = new_general
    return moved


# ============================================================ #
# 主训练流程
# ============================================================ #

def main():
    print("=" * 70)
    print("  🔥🔥🔥 合鸣-13 全方位增强：V6 → V7 🔥🔥🔥")
    print("=" * 70)

    # -------------------------------------------------------
    # 1. 加载 V6 检查点
    # -------------------------------------------------------
    print("\n[1/6] 加载 V6 检查点...")
    if os.path.exists(os.path.join(V6_CKPT, "harmonia_lite.json.gz")):
        model = Harmonia13Virtual.load(V6_CKPT)
        print(f"  ✅ 已加载 V6")
    else:
        print(f"  ❌ V6 不存在")
        return

    current_frags = len(model._lite._learned_fragments)
    print(f"  当前已学片段: {current_frags:,}")

    # -------------------------------------------------------
    # 2. 全方位收集语料
    # -------------------------------------------------------
    print("\n[2/6] 全方位收集语料...")

    all_fragments = []
    skip_dirs = {"__pycache__", ".git", "test", "tests", "idlelib",
                 "tkinter", "turtledemo", "site-packages", "doc", "docs",
                 "node_modules", "vendor", "dist", "build", ".next", ".cache",
                 "venv", "env", ".venv"}

    # 2.1 扫描整个项目所有代码
    project_root = os.path.join(os.path.dirname(__file__), "..")
    print(f"  📁 扫描项目全部代码...")
    code_texts = scan_all_files(
        project_root,
        extensions={".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".rst"},
        skip_dirs=skip_dirs
    )
    for text in code_texts:
        if text.strip().startswith("def ") or text.strip().startswith("class "):
            all_fragments.extend(extract_code_fragments(text, max_lines=25))
        else:
            all_fragments.extend(extract_text_fragments(text, max_len=300))
    print(f"     → {len(code_texts)} 文件, 提取 {len(all_fragments):,} 片段")

    # 2.2 加载所有已有语料
    corpus_sources = [
        ("/workspace/xuni/examples/real_corpus.json", "real_corpus"),
        ("/workspace/xuni/corpus_clean.json", "corpus_clean"),
    ]
    for path, name in corpus_sources:
        if os.path.exists(path):
            corpus = load_json_corpus(path)
            all_fragments.extend(corpus)
            print(f"  📝 {name}: {len(corpus):,} 片段")

    gz_sources = [
        "/workspace/xuni/corpus_particles_large.json.gz",
        "/workspace/xuni/corpus_clean_large.json.gz",
    ]
    for path in gz_sources:
        if os.path.exists(path):
            corpus = load_gz_corpus(path)
            all_fragments.extend(corpus)
            print(f"  📦 {os.path.basename(path)}: {len(corpus):,} 片段")

    # 2.3 加载 V2/V3 检查点语料（如果有）
    for ver in ["v2", "v3"]:
        ckpt_path = f"/workspace/xuni/examples/checkpoints/harmonia_{ver}/harmonia_lite.json.gz"
        if os.path.exists(ckpt_path):
            try:
                with gzip.open(ckpt_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
                learned = data.get("learned_fragments", [])
                all_fragments.extend(learned)
                print(f"  📦 V{ver} 学习片段: {len(learned):,}")
            except Exception:
                pass

    # 2.4 去重
    before_dedup = len(all_fragments)
    seen = set()
    unique_frags = []
    for f in all_fragments:
        key = f.strip()[:200]
        if key not in seen:
            seen.add(key)
            unique_frags.append(f)
    all_fragments = unique_frags
    print(f"\n  📚 去重前: {before_dedup:,} | 去重后: {len(all_fragments):,} 片段")

    if len(all_fragments) < 100:
        print("  ⚠️ 语料不足，使用增强种子")
        seeds = [
            "合鸣-13 是 xuni 虚拟生态的旗舰对话模型，由 13 位专家组成",
            "虚拟电场通过采样点密度转化为能量，驱动整个生态系统",
            "MoE 混合专家架构用关键词共振实现非传统路由",
            "双态系统让虚拟模型在粒子态被真正训练，数据层被真实调用",
            "物理建模合成器用数字振荡器和共鸣滤波器生成声音",
            "超混沌采样器实时生成上亿采样点，内存 O(1)",
            "水动力学把采样点当流体粒子，有蒸发凝结和涡旋",
            "玻璃逻辑把计算当光学系统，数据是光，函数是透镜",
            "Kuramoto 振子网络模拟神经同步，Hebbian 学习强化连接",
            "虚拟凭证把场能量铸造成 24 位令牌，可验证可消耗",
        ]
        all_fragments = seeds * 500

    # -------------------------------------------------------
    # 3. 专家分化（预处理）
    # -------------------------------------------------------
    print("\n[3/6] 专家分化...")
    moved = distribute_fragments(model)
    print(f"  ✅ 已分化 {moved:,} 条片段到专业专家")

    # 统计分化后
    for e in model._lite.experts:
        print(f"    {e['name']:12s}: {len(e['fragments']):8,} 片段")

    # -------------------------------------------------------
    # 4. 大规模训练
    # -------------------------------------------------------
    print(f"\n[4/6] 开始大规模训练 {TRAINING_EPOCHS:,} 轮...")
    print(f"  每轮吸收 {BATCH_SIZE} 条 | 每 {SAVE_INTERVAL:,} 轮保存 | 每 {SELF_ENHANCE_EVERY:,} 轮自我增强")

    start_time = time.time()
    log = []
    self_enhance_frags = []

    for epoch in range(1, TRAINING_EPOCHS + 1):
        # 随机抽取语料
        batch = random.sample(all_fragments, min(BATCH_SIZE, len(all_fragments)))

        # 如果有自我增强语料，混入
        if self_enhance_frags and epoch % 100 == 0:
            batch.extend(random.sample(self_enhance_frags, min(5, len(self_enhance_frags))))

        # 训练
        model._lite.train(batch, epochs=1)

        # 自我增强
        if epoch % SELF_ENHANCE_EVERY == 0 and epoch > 0:
            print(f"\n  🤖 Epoch {epoch:,} — 自我增强中...")
            new_frags = generate_self_corpus(model, count=200)
            self_enhance_frags.extend(new_frags)
            # 自我生成的也加入训练池
            all_fragments.extend(new_frags)
            print(f"     生成 {len(new_frags)} 条新语料")

        # 定期报告
        if epoch % SAVE_INTERVAL == 0 or epoch == 1:
            elapsed = time.time() - start_time
            learned = len(model._lite._learned_fragments)
            expert_total = sum(len(e.get("fragments", [])) for e in model._lite.experts)

            print(f"  Epoch {epoch:7,d} | 已学: {learned:12,d} | "
                  f"专家总: {expert_total:12,d} | 自增: {len(self_enhance_frags):5,d} | "
                  f"用时: {elapsed:.1f}s")

            log.append({
                "epoch": epoch,
                "learned": learned,
                "expert_total": expert_total,
                "self_enhanced": len(self_enhance_frags),
                "elapsed": round(elapsed, 2),
            })

            # 中间保存
            mid_ckpt = os.path.join(V7_CKPT + f"_epoch{epoch}")
            os.makedirs(mid_ckpt, exist_ok=True)
            model.save(mid_ckpt)

    total_time = time.time() - start_time
    print(f"\n  ✅ 训练完成！总用时: {total_time:.2f}s")

    # -------------------------------------------------------
    # 5. 最终专家分化
    # -------------------------------------------------------
    print("\n[5/6] 最终专家分化...")
    final_moved = distribute_fragments(model)
    print(f"  ✅ 最终分化 {final_moved:,} 条片段")

    for e in model._lite.experts:
        print(f"    {e['name']:12s}: {len(e['fragments']):10,} 片段")

    # -------------------------------------------------------
    # 6. 评估
    # -------------------------------------------------------
    print("\n[6/6] 评估生成质量...")
    test_prompts = [
        "def quicksort",
        "class NeuralNetwork",
        "import numpy",
        "async def fetch",
        "class APIRouter",
        "def train_model",
        "import torch",
        "class DataLoader",
        "合鸣是什么",
        "虚拟电场",
    ]

    print("\n  --- 生成示例 ---")
    for p in test_prompts:
        result = model._lite.generate(p, max_new_tokens=80)
        print(f"  [{p:20s}] → {result[:90]}")

    # -------------------------------------------------------
    # 7. 保存 V7
    # -------------------------------------------------------
    print(f"\n[7/7] 保存 V7 检查点到 {V7_CKPT}...")
    os.makedirs(V7_CKPT, exist_ok=True)
    save_result = model.save(V7_CKPT)
    print(f"  ✅ 已保存:")
    print(f"     元数据: {save_result['meta_path']}")
    print(f"     权重: {save_result['lite_path']}")
    print(f"     片段数: {save_result['fragments']:,}")

    # 保存报告
    report = {
        "version": "v7",
        "base_checkpoint": "v6",
        "training_epochs": TRAINING_EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_corpus_unique": len(all_fragments),
        "self_enhanced_frags": len(self_enhance_frags),
        "fragments_learned": len(model._lite._learned_fragments),
        "training_time_seconds": round(total_time, 2),
        "growth_log": log,
        "expert_load": {
            e["name"]: len(e.get("fragments", []))
            for e in model._lite.experts
        },
    }
    report_path = os.path.join(os.path.dirname(__file__), "trainer_v7_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告: {report_path}")

    # -------------------------------------------------------
    # 最终总结
    # -------------------------------------------------------
    print("\n" + "=" * 70)
    print("  🎉🎉🎉 全方位增强完成 🎉🎉🎉")
    print("=" * 70)

    final_frags = len(model._lite._learned_fragments)
    growth = final_frags - current_frags

    print(f"""
  📊 训练统计:
    基线检查点: V6 ({current_frags:,} 片段)
    训练轮数: {TRAINING_EPOCHS:,}
    去重语料: {len(all_fragments):,} 条
    自我增强: {len(self_enhance_frags):,} 条
    最终吸收: {final_frags:,} 条
    净增长: +{growth:,} 条
    训练用时: {total_time:.2f}s

  💾 检查点已保存到: {V7_CKPT}

  积少成多:
    v1:        5,000 片段
    v2:       40,000 片段
    v3:      120,000 片段
    v6:    2,240,000 片段
    v7: {final_frags:>12,} 片段 🔥🔥🔥
""")


if __name__ == "__main__":
    main()
