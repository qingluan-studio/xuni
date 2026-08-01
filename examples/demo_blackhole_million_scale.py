"""
百万级黑洞训练 — 100万+素材起步
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def main():
    print("\n🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌")
    print("  百万级黑洞启动 — 目标：1,000,000+ 素材")
    print("🌌🌌🌌🌌🌌🌌🌌🌌🌌🌌\n")

    # 收集所有本地代码库
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

    # coze_repos 子目录
    coze_dir = os.path.join(workspace, "coze_repos")
    if os.path.isdir(coze_dir):
        for item in os.listdir(coze_dir):
            full_path = os.path.join(coze_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                all_repos.append(full_path)

    # china_ai_repos 子目录
    china_ai_dir = os.path.join(workspace, "china_ai_repos")
    if os.path.isdir(china_ai_dir):
        for item in os.listdir(china_ai_dir):
            full_path = os.path.join(china_ai_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                all_repos.append(full_path)

    print(f"代码库数量：{len(all_repos)} 个")
    print()

    trainer = BlackHoleTrainer(model_id="xenith-million-scale")
    factory = MultiverseResourceFactory()

    # 全领域 + 大数量，目标100万+
    result = trainer.absorb_and_forge(
        repo_paths=all_repos,
        factory=factory,
        languages=None,
        max_files_per_repo=10000,
        spin_rounds=12,
        quality_threshold=0.4,
        knowledge_domains=[
            "computer_science",
            "engineering",
            "math",
            "physics",
            "chemistry",
            "biology",
            "medicine",
            "law",
            "finance",
            "philosophy",
        ],
        knowledge_count_per_domain=100000,  # 每个领域10万条，10个领域就是100万
    )

    result["total_repos"] = len(all_repos)

    report_path = "/workspace/xuni/examples/blackhole_million_scale_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📦 百万级报告已保存：{report_path}\n")

    return result


if __name__ == "__main__":
    main()
