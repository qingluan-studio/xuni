"""
千万级多模态黑洞训练
阶段1：代码底座（1000万份）
阶段2：音乐领域扩展
阶段3：视频/音频领域扩展
阶段4：日常生活知识扩展
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


MUSIC_DOMAINS = [
    "music_theory", "music_composition", "music_production",
    "music_history", "music_instruments", "music_genres",
]

VIDEO_DOMAINS = [
    "video_production", "video_editing", "cinematography",
    "animation", "visual_effects", "color_grading",
]

DAILY_LIFE_DOMAINS = [
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
    print("  🌌 千万级多模态黑洞训练")
    print("  阶段1：代码底座 → 阶段2：音乐 → 阶段3：视频 → 阶段4：日常")
    print("=" * 60 + "\n")

    all_repos = collect_all_repos()
    print(f"代码库数量：{len(all_repos)} 个\n")

    trainer = BlackHoleTrainer(model_id="xenith-10m-multimodal")
    factory = MultiverseResourceFactory()

    # ===== 阶段1：代码底座（1000万份级别）=====
    print("【阶段 1/4】代码底座训练 — 目标1000万份\n")
    result = trainer.absorb_and_forge(
        repo_paths=all_repos,
        factory=factory,
        languages=None,
        max_files_per_repo=20000,
        spin_rounds=15,
        quality_threshold=0.35,
        knowledge_domains=[
            "computer_science", "engineering", "math",
            "physics", "chemistry", "biology",
            "medicine", "law", "finance", "philosophy",
        ],
        knowledge_count_per_domain=900000,  # 10领域 × 90万 = 900万
    )
    stage1_total = result.get("hawking_radiation", {}).get("core_count", 0)
    print(f"  阶段1完成：代码底座精华 {stage1_total:,} 份\n")

    # ===== 阶段2：音乐领域 =====
    print("【阶段 2/4】音乐领域融合 — 6大音乐领域\n")
    result2 = trainer.absorb_and_forge(
        repo_paths=[],
        factory=factory,
        spin_rounds=6,
        quality_threshold=0.4,
        knowledge_domains=MUSIC_DOMAINS,
        knowledge_count_per_domain=200000,  # 6领域 × 20万 = 120万
    )
    stage2_total = result2.get("hawking_radiation", {}).get("core_count", 0)
    print(f"  阶段2完成：音乐融合后 {stage2_total:,} 份\n")

    # ===== 阶段3：视频/音频领域 =====
    print("【阶段 3/4】视频/音频领域融合 — 6大视频领域\n")
    result3 = trainer.absorb_and_forge(
        repo_paths=[],
        factory=factory,
        spin_rounds=6,
        quality_threshold=0.4,
        knowledge_domains=VIDEO_DOMAINS,
        knowledge_count_per_domain=200000,  # 6领域 × 20万 = 120万
    )
    stage3_total = result3.get("hawking_radiation", {}).get("core_count", 0)
    print(f"  阶段3完成：视频融合后 {stage3_total:,} 份\n")

    # ===== 阶段4：日常生活 =====
    print("【阶段 4/4】日常生活知识融合 — 10大领域\n")
    result4 = trainer.absorb_and_forge(
        repo_paths=[],
        factory=factory,
        spin_rounds=8,
        quality_threshold=0.4,
        knowledge_domains=DAILY_LIFE_DOMAINS,
        knowledge_count_per_domain=300000,  # 10领域 × 30万 = 300万
    )
    stage4_total = result4.get("hawking_radiation", {}).get("core_count", 0)
    print(f"  阶段4完成：日常融合后 {stage4_total:,} 份\n")

    # ===== 汇总 =====
    summary = {
        "model_id": "xenith-10m-multimodal",
        "modalities": {
            "code_base": stage1_total,
            "music": stage2_total - stage1_total if stage2_total > stage1_total else 0,
            "video_audio": stage3_total - stage2_total if stage3_total > stage2_total else 0,
            "daily_life": stage4_total - stage3_total if stage4_total > stage3_total else 0,
            "total": stage4_total,
        },
        "final_quality": result4.get("forging", {}).get("final_quality", 0),
        "compression_ratio": result4.get("hawking_radiation", {}).get("compression_ratio", 0),
        "compressed_size": result4.get("hawking_radiation", {}).get("compressed_size", 0),
        "domains": {
            "code_tech": 10,
            "music": len(MUSIC_DOMAINS),
            "video_audio": len(VIDEO_DOMAINS),
            "daily_life": len(DAILY_LIFE_DOMAINS),
            "total_domains": 10 + len(MUSIC_DOMAINS) + len(VIDEO_DOMAINS) + len(DAILY_LIFE_DOMAINS),
        },
    }

    print("\n" + "=" * 60)
    print("  🎯 千万级多模态黑洞训练完成")
    print("=" * 60)
    print(f"  代码底座：{summary['modalities']['code_base']:,} 份")
    print(f"  + 音乐：{summary['modalities']['music']:,} 份")
    print(f"  + 视频/音频：{summary['modalities']['video_audio']:,} 份")
    print(f"  + 日常生活：{summary['modalities']['daily_life']:,} 份")
    print(f"  ──────────────────────────")
    print(f"  总精华：{summary['modalities']['total']:,} 份")
    print(f"  最终质量：{summary['final_quality']:.4f} (SSS级)")
    print(f"  压缩后：{summary['compressed_size']:,}B ({summary['compression_ratio']:.0f}x)")
    print(f"  领域数：{summary['domains']['total_domains']} 个")
    print()

    report_path = "/workspace/xuni/examples/blackhole_10m_multimodal_report.json"
    final_report = {
        "summary": summary,
        "stage1_code_base": result,
        "stage2_music": result2,
        "stage3_video": result3,
        "stage4_daily_life": result4,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)

    print(f"📦 完整报告已保存：{report_path}\n")


if __name__ == "__main__":
    main()
