"""
终极黑洞训练 - 吞噬GitHub Top 项目精髓
freeCodeCamp / OpenClaw / developer-roadmap / coding-interview-university / Vue / n8n / VS Code
没有真实代码就用知识领域+概念注入，黑洞照样吸收精髓
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def main():
    print("\n🌌🌌🌌🌌🌌 终极黑洞启动 — 吞噬 GitHub Stars Top 项目精髓...\n")

    # 先吸收所有本地代码库
    local_repos = []
    workspace = "/workspace"
    skip_dirs = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", "kosong"]

    for item in os.listdir(workspace):
        full_path = os.path.join(workspace, item)
        if os.path.isdir(full_path) and item not in skip_dirs:
            has_code = False
            for root, dirs, files in os.walk(full_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in skip_dirs]
                for f in files:
                    if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".java")):
                        has_code = True
                        break
                if has_code:
                    break
            if has_code:
                local_repos.append(full_path)

    print(f"本地代码库：{len(local_repos)} 个")

    trainer = BlackHoleTrainer(model_id="xenith-ultimate-blackhole")
    factory = MultiverseResourceFactory()

    # 全领域知识注入（覆盖图里所有项目的领域）
    # freeCodeCamp → 全栈编程 + 计算机科学
    # OpenClaw → AI 助手 + 工程
    # developer-roadmap → 软件工程 + 职业发展
    # coding-interview-university → 算法 + 数据结构 + 面试
    # Vue → 前端 + 软件工程
    # n8n → 工作流 + 自动化 + 工程
    # VS Code → IDE + 编辑器 + 软件工程

    all_knowledge_domains = [
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
    ]

    # 一键终极吞噬
    result = trainer.absorb_and_forge(
        repo_paths=local_repos,
        factory=factory,
        languages=None,
        max_files_per_repo=10000,
        spin_rounds=15,  # 15圈，终极锻造
        quality_threshold=0.5,
        knowledge_domains=all_knowledge_domains,
        knowledge_count_per_domain=15000,  # 每个领域15,000条，量大管饱
    )

    # 保存报告
    report_path = "/workspace/xuni/examples/blackhole_ultimate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n终极报告已保存：{report_path}\n")

    return result


if __name__ == "__main__":
    main()
