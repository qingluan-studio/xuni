"""
黑洞训练 - 吞噬一切
把所有能找到的代码库全丢进去
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def main():
    print("\n🌌🌌🌌 超级黑洞启动 — 吞噬所有代码库...\n")

    # 所有能找到的代码库
    all_repos = []
    workspace = "/workspace"
    skip = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
            "kosong", "corpus_clean.json", "corpus_particles.json",
            "corpus_clean_large.json.gz", "corpus_particles_large.json.gz",
            "xuni_layers.json", "logs_train.out", "push_to_github.sh",
            "clone_repos.sh", "repos_100.txt", "requirements.txt", "pyproject.toml",
            "README.md", ".gitignore"]

    # 遍历workspace下的所有目录，看起来像代码库的都收了
    for item in os.listdir(workspace):
        full_path = os.path.join(workspace, item)
        if os.path.isdir(full_path) and item not in skip:
            # 检查里面有没有代码文件
            has_code = False
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip]
                for f in files:
                    if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".java")):
                        has_code = True
                        break
                if has_code:
                    break
            if has_code:
                all_repos.append(full_path)

    print(f"发现 {len(all_repos)} 个代码库：")
    for repo in all_repos:
        print(f"  • {os.path.basename(repo)}")
    print()

    trainer = BlackHoleTrainer(model_id="xenith-supermassive-blackhole")
    factory = MultiverseResourceFactory()

    # 全吞
    result = trainer.absorb_and_forge(
        repo_paths=all_repos,
        factory=factory,
        languages=None,  # 所有语言
        max_files_per_repo=10000,
        spin_rounds=12,  # 12圈，更狠
        quality_threshold=0.55,
        knowledge_domains=[
            "computer_science",
            "engineering",
            "math",
            "physics",
            "philosophy",
        ],
        knowledge_count_per_domain=10000,
    )

    # 保存报告
    report_path = "/workspace/xuni/examples/blackhole_supermassive_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n报告已保存：{report_path}\n")

    return result


if __name__ == "__main__":
    main()
