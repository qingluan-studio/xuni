"""
黑洞训练 - 吞噬整个Linux内核
直接丢进去，训练完拉结果
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


def main():
    print("\n🌌 黑洞启动 — 准备吞噬整个 Linux 内核...\n")

    trainer = BlackHoleTrainer(model_id="xenith-linux-kernel-blackhole")
    factory = MultiverseResourceFactory()

    # 直接吞整个Linux内核仓库
    result = trainer.absorb_and_forge(
        repo_paths=["/workspace/linux"],
        factory=factory,
        languages=["c", "c_header", "cpp", "cpp_header", "python", "rust"],
        max_files_per_repo=50000,  # 全吃
        spin_rounds=9,
        quality_threshold=0.6,
        knowledge_domains=["computer_science", "engineering"],
        knowledge_count_per_domain=10000,
    )

    # 保存报告
    report_path = "/workspace/xuni/examples/blackhole_linux_kernel_full_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n报告已保存：{report_path}\n")

    return result


if __name__ == "__main__":
    main()
