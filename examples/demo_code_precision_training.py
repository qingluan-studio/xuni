"""
代码精度专项训练 — 提升代码准确性、注释规范、符号正确性
扫描所有本地高质量代码库 + 强化代码质量点
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def collect_all_repos():
    """收集所有本地代码库"""
    repos = []
    workspace = "/workspace"
    skip = {
        ".git", "node_modules", "__pycache__", "venv", ".venv",
        "dist", "build", "kosong", "coze_temp", "examples",
        "tests", ".github", "docs", "figures", "assets",
    }

    # 一级目录
    for item in os.listdir(workspace):
        fp = os.path.join(workspace, item)
        if not os.path.isdir(fp) or item.startswith(".") or item in skip:
            continue
        # 检查是否有代码
        has_code = False
        for root, dirs, files in os.walk(fp):
            dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
            for f in files:
                if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp",
                               ".h", ".hpp", ".java", ".cu", ".mjs", ".cjs")):
                    has_code = True
                    break
            if has_code:
                break
        if has_code:
            repos.append(fp)

    # 二级目录：china_ai_repos, coze_repos, kimi-agent-sdk
    for parent in ["china_ai_repos", "coze_repos"]:
        d = os.path.join(workspace, parent)
        if os.path.isdir(d):
            for item in os.listdir(d):
                fp = os.path.join(d, item)
                if os.path.isdir(fp) and not item.startswith("."):
                    repos.append(fp)

    # kimi-agent-sdk 子目录
    kas = os.path.join(workspace, "kimi-agent-sdk")
    if os.path.isdir(kas):
        for sub in ["python", "go", "node"]:
            fp = os.path.join(kas, sub)
            if os.path.isdir(fp):
                repos.append(fp)

    return repos


def main():
    print("\n" + "=" * 70)
    print("  🎯 代码精度专项训练")
    print("  目标：提升代码准确性 / 注释规范性 / 符号正确性")
    print("=" * 70 + "\n")

    repos = collect_all_repos()
    print(f"📦 待扫描代码库：{len(repos)} 个")
    for r in repos:
        print(f"  • {os.path.basename(r)}")
    print()

    trainer = BlackHoleTrainer(model_id="xenith-code-precision-v2", streaming=True)
    factory = MultiverseResourceFactory()

    # 精度专项训练：更多圈数 + 更高质量阈值
    result = trainer.absorb_and_forge(
        repo_paths=repos,
        factory=factory,
        max_files_per_repo=10000,
        spin_rounds=15,
        quality_threshold=0.3,
        knowledge_domains=[
            "computer_science", "engineering", "math",
        ],
        knowledge_count_per_domain=50000,
    )

    # 统计
    type_counts = result["phases"]["absorb"].get("type_counts", {})
    source_counts = result["phases"]["absorb"].get("source_counts", {})

    print("\n" + "=" * 70)
    print("  📊 代码精度专项训练报告")
    print("=" * 70)
    print()
    print(f"  扫描代码库：{len(repos)} 个")
    print(f"  总吸收素材：{result['phases']['absorb']['total_absorbed']:,} 份")
    print(f"  精华核心：{result['phases']['hawking_radiation']['kept_core']:,} 份")
    print(f"  吐出渣滓：{result['phases']['hawking_radiation']['total_ejected']:,} 份")
    print(f"  提纯率：{result['phases']['hawking_radiation']['purification_ratio']}")
    print()
    print(f"  初始质量：{result['phases']['spin_forge']['initial_quality']:.4f}")
    print(f"  最终质量：{result['phases']['spin_forge']['final_quality']:.4f}")
    print(f"  质量提升：+{result['phases']['spin_forge']['quality_improvement']*100:.1f}%")
    print(f"  锻造圈数：{result['phases']['spin_forge']['rounds']} 圈")
    print()
    print(f"  压缩后：{result['core']['compressed_size_bytes']:,}B")
    print(f"  压缩比：{result['core']['compression_ratio']:,.0f}x")
    print()
    print(f"  素材类型分布：")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    • {t:<25} {c:>10,}")
    print()
    print(f"  总耗时：{result['total_elapsed_seconds']:.2f}s")
    print()

    # 保存报告
    out = "/workspace/xuni/examples/code_precision_training_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"📦 报告：{out}\n")


if __name__ == "__main__":
    main()
