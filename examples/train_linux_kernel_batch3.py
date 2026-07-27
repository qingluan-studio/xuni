"""
Linux内核分批训练 - 第3批：net/ 网络协议栈
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import XenithModel
from xuni.multiverse_resources import MultiverseResourceFactory


def train_batch(batch_name: str, repo_subdirs: list, max_files: int = 2000):
    """训练一批Linux内核模块"""
    print(f"\n{'='*60}")
    print(f"  开始训练：{batch_name}")
    print(f"  子目录：{repo_subdirs}")
    print(f"{'='*60}\n")

    factory = MultiverseResourceFactory()
    model = XenithModel("xenith-linux-kernel")

    all_results = []
    total_functions = 0
    total_files = 0

    for subdir in repo_subdirs:
        repo_path = f"/workspace/linux/{subdir}"
        if not os.path.exists(repo_path):
            print(f"[跳过] 路径不存在: {repo_path}")
            continue

        print(f"\n--- 扫描 {subdir} ---")
        result = model.train_on_codebase(
            repo_path=repo_path,
            factory=factory,
            languages=["c"],
            max_files=max_files,
            augment_multiplier=5,
        )

        if "error" in result:
            print(f"[错误] {subdir}: {result['error']}")
            continue

        all_results.append({
            "subdir": subdir,
            "result": result,
        })

        if "training_log" in result:
            for log in result["training_log"]:
                print(f"  {log}")

        details = result.get("details", {})
        total_functions += details.get("functions_extracted", 0)
        total_files += details.get("files_scanned", 0)

        model.training_details[f"linux_{subdir.replace('/', '_')}"] = {
            "trained_at": time.time(),
            "result_summary": {
                "files_scanned": details.get("files_scanned", 0),
                "functions_extracted": details.get("functions_extracted", 0),
                "training_samples": details.get("total_training_items", 0),
            }
        }

    report = {
        "batch_name": batch_name,
        "subdirs": repo_subdirs,
        "total_files_scanned": total_files,
        "total_functions_extracted": total_functions,
        "model_id": model.model_id,
        "trained_at": time.time(),
        "details": all_results,
    }

    report_path = f"/workspace/xuni/examples/linux_kernel_batch3_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  {batch_name} 训练完成！")
    print(f"  扫描文件总数：{total_files}")
    print(f"  提取函数总数：{total_functions}")
    print(f"  报告已保存：{report_path}")
    print(f"{'='*60}\n")

    return model, report


if __name__ == "__main__":
    model, report = train_batch(
        batch_name="Linux内核第3批 - net/ 网络协议栈",
        repo_subdirs=["net"],
        max_files=2000,
    )
