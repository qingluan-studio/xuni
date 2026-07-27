"""
黑洞训练示例 - 一键吸收所有代码库，旋转锻造，吐渣滓
直接出结果，不搞假进度条
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def main():
    # 收集所有要吸收的代码库
    repos = []

    # Linux 内核四大核心模块
    linux_base = "/workspace/linux"
    linux_subdirs = ["kernel", "mm", "fs", "net", "crypto", "security", "drivers"]
    for sub in linux_subdirs:
        path = os.path.join(linux_base, sub)
        if os.path.exists(path):
            repos.append(path)

    # 之前训练过的其他代码库
    other_repos = [
        "/workspace/xuni",
        "/workspace/openclaw",
        "/workspace/kimi-cli",
        "/workspace/kimi-code",
        "/workspace/kimi-agent-sdk",
        "/workspace/MoBA",
        "/workspace/FlashKDA",
        "/workspace/Moonlight",
        "/workspace/checkpoint-engine",
    ]
    for repo in other_repos:
        if os.path.exists(repo):
            repos.append(repo)

    print(f"\n准备吸收 {len(repos)} 个代码库进入黑洞...\n")

    # 初始化黑洞训练器
    trainer = BlackHoleTrainer(model_id="xenith-blackhole-v1")
    factory = MultiverseResourceFactory()

    # 一键黑洞训练
    result = trainer.absorb_and_forge(
        repo_paths=repos,
        factory=factory,
        languages=None,  # 全部语言
        max_files_per_repo=3000,
        spin_rounds=9,  # 9圈锻造（极致）
        quality_threshold=0.65,
        knowledge_domains=[
            "computer_science",
            "engineering",
            "math",
        ],
        knowledge_count_per_domain=5000,
    )

    # 保存报告
    report_path = "/workspace/xuni/examples/blackhole_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n报告已保存：{report_path}")
    print()

    return result


if __name__ == "__main__":
    main()
