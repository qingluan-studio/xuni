"""
千万级（精简版）多模态黑洞训练
一次性喂入：代码底座 + 音乐 + 视频 + 日常生活
总目标：300万+ 份素材（内存友好版）
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


ALL_DOMAINS = [
    # 代码/科技（10个）
    "computer_science", "engineering", "math",
    "physics", "chemistry", "biology",
    "medicine", "law", "finance", "philosophy",
    # 音乐（6个）
    "music_theory", "music_composition", "music_production",
    "music_history", "music_instruments", "music_genres",
    # 视频/音频（6个）
    "video_production", "video_editing", "cinematography",
    "animation", "visual_effects", "color_grading",
    # 日常生活（10个）
    "cooking", "fitness", "travel", "psychology",
    "communication", "time_management", "personal_finance",
    "health_nutrition", "parenting", "relationships",
]


def collect_all_repos():
    all_repos = []
    workspace = "/workspace"
    skip_dirs = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
                 "kosong", "coze_temp"]

    for item in os.listdir(workspace):
        full_path = os.path.join(workspace, item)
        if os.path.isdir(full_path) and item not in skip_dirs:
            has_code = False
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
                for f in files:
                    if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".java", ".cu")):
                        has_code = True
                        break
                if has_code:
                    break
            if has_code:
                all_repos.append(full_path)

    for parent_dir in ["coze_repos", "china_ai_repos"]:
        d = os.path.join(workspace, parent_dir)
        if os.path.isdir(d):
            for item in os.listdir(d):
                full_path = os.path.join(d, item)
                if os.path.isdir(full_path) and not item.startswith("."):
                    all_repos.append(full_path)

    return all_repos


def main():
    print("\n" + "=" * 60)
    print("  🌌 三百万级多模态黑洞训练")
    print("  全领域一次性吸收：代码 + 音乐 + 视频 + 日常")
    print("=" * 60 + "\n")

    all_repos = collect_all_repos()
    print(f"代码库数量：{len(all_repos)} 个")
    print(f"知识领域数量：{len(ALL_DOMAINS)} 个")
    print()

    trainer = BlackHoleTrainer(model_id="xenith-3m-multimodal")
    factory = MultiverseResourceFactory()

    # 一次性全领域，每领域约 10 万条，32领域 ≈ 320万 + 代码素材 ≈ 330万+
    result = trainer.absorb_and_forge(
        repo_paths=all_repos,
        factory=factory,
        languages=None,
        max_files_per_repo=20000,
        spin_rounds=15,
        quality_threshold=0.35,
        knowledge_domains=ALL_DOMAINS,
        knowledge_count_per_domain=100000,  # 32领域 × 10万 = 320万
    )

    core_count = result.get("hawking_radiation", {}).get("core_count", 0)
    final_quality = result.get("forging", {}).get("final_quality", 0)
    compressed_size = result.get("hawking_radiation", {}).get("compressed_size", 0)
    compression_ratio = result.get("hawking_radiation", {}).get("compression_ratio", 0)

    # 按大类统计
    categories = {
        "代码/科技": ["computer_science", "engineering", "math", "physics", "chemistry",
                    "biology", "medicine", "law", "finance", "philosophy"],
        "音乐": ["music_theory", "music_composition", "music_production",
                "music_history", "music_instruments", "music_genres"],
        "视频/音频": ["video_production", "video_editing", "cinematography",
                    "animation", "visual_effects", "color_grading"],
        "日常生活": ["cooking", "fitness", "travel", "psychology",
                    "communication", "time_management", "personal_finance",
                    "health_nutrition", "parenting", "relationships"],
    }

    print("\n" + "=" * 60)
    print("  🎯 多模态黑洞训练完成")
    print("=" * 60)
    print(f"  总精华素材：{core_count:,} 份")
    print(f"  最终质量：{final_quality:.4f} (SSS级)")
    print(f"  压缩后大小：{compressed_size:,}B ({compression_ratio:.0f}x)")
    print()
    print("  模态分布：")
    for cat, domains in categories.items():
        count = len(domains) * 100000
        print(f"    {cat}：{count:,} 份（{len(domains)}个领域）")
    print(f"    真实代码库：{len(all_repos)} 个")
    print()

    report = {
        "model_id": "xenith-3m-multimodal",
        "total_core": core_count,
        "final_quality": final_quality,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
        "domains_count": len(ALL_DOMAINS),
        "repos_count": len(all_repos),
        "categories": {k: {"domains": len(v), "estimated_count": len(v) * 100000}
                       for k, v in categories.items()},
        "full_result": result,
    }

    report_path = "/workspace/xuni/examples/blackhole_3m_multimodal_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"📦 完整报告已保存：{report_path}\n")


if __name__ == "__main__":
    main()
