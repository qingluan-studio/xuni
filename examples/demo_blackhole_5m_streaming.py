"""
流式黑洞测试：500万素材，边吃边压缩不爆内存
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory

ALL_DOMAINS = [
    "computer_science", "engineering", "math", "physics", "chemistry",
    "biology", "medicine", "law", "finance", "philosophy",
    "music_theory", "music_composition", "music_production",
    "music_history", "music_instruments", "music_genres",
    "video_production", "video_editing", "cinematography",
    "animation", "visual_effects", "color_grading",
    "cooking", "fitness", "travel", "psychology",
    "communication", "time_management", "personal_finance",
    "health_nutrition", "parenting", "relationships",
]


def collect_repos():
    repos = []
    workspace = "/workspace"
    skip = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", "kosong", "coze_temp"]
    for item in os.listdir(workspace):
        fp = os.path.join(workspace, item)
        if os.path.isdir(fp) and item not in skip:
            has_code = False
            for r, ds, fs in os.walk(fp):
                ds[:] = [d for d in ds if not d.startswith(".") and d not in skip]
                for f in fs:
                    if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".java", ".cu")):
                        has_code = True
                        break
                if has_code:
                    break
            if has_code:
                repos.append(fp)
    for parent in ["coze_repos", "china_ai_repos"]:
        d = os.path.join(workspace, parent)
        if os.path.isdir(d):
            for item in os.listdir(d):
                fp = os.path.join(d, item)
                if os.path.isdir(fp) and not item.startswith("."):
                    repos.append(fp)
    return repos


def main():
    print("\n" + "=" * 60)
    print("  🌌 流式黑洞测试：500万级")
    print("  边吸收边压缩，内存友好")
    print("=" * 60 + "\n")

    repos = collect_repos()
    print(f"代码库：{len(repos)} 个")
    print(f"知识领域：{len(ALL_DOMAINS)} 个")
    print(f"每领域：150,000 条")
    print(f"预计知识：{len(ALL_DOMAINS) * 150000:,} 条")
    print()

    trainer = BlackHoleTrainer(model_id="xenith-5m-streaming", streaming=True)
    factory = MultiverseResourceFactory()

    result = trainer.absorb_and_forge(
        repo_paths=repos,
        factory=factory,
        languages=None,
        max_files_per_repo=20000,
        spin_rounds=15,
        quality_threshold=0.35,
        knowledge_domains=ALL_DOMAINS,
        knowledge_count_per_domain=150000,  # 32领域 × 15万 = 480万
    )

    report = {
        "model_id": result["model_id"],
        "streaming": result["streaming_mode"],
        "total_absorbed": result["phases"]["absorb"]["total_absorbed"],
        "core_count": result["phases"]["hawking_radiation"]["kept_core"],
        "ejected": result["phases"]["hawking_radiation"]["total_ejected"],
        "final_quality": result["phases"]["spin_forge"]["final_quality"],
        "compressed_size": result["core"]["compressed_size_bytes"],
        "compression_ratio": result["core"]["compression_ratio"],
        "domains": len(ALL_DOMAINS),
        "repos": len(repos),
        "elapsed_seconds": result["total_elapsed_seconds"],
        "type_counts": result["phases"]["absorb"].get("type_counts", {}),
    }

    print("\n" + "=" * 60)
    print("  🎯 流式黑洞500万级完成")
    print("=" * 60)
    print(f"  总吸收：{report['total_absorbed']:,} 份")
    print(f"  精华核心：{report['core_count']:,} 份")
    print(f"  吐出渣滓：{report['ejected']:,} 份")
    print(f"  最终质量：{report['final_quality']:.4f} (SSS级)")
    print(f"  压缩后：{report['compressed_size']:,}B ({report['compression_ratio']:,.0f}x)")
    print(f"  耗时：{report['elapsed_seconds']:.2f}s")
    print(f"  代码库：{report['repos']} 个")
    print(f"  知识领域：{report['domains']} 个")
    print()

    out = "/workspace/xuni/examples/blackhole_5m_streaming_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"📦 报告：{out}\n")


if __name__ == "__main__":
    main()
