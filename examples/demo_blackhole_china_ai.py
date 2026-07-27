"""
黑洞自动觅食器 — 给它位置，自己去训练
中国开源大模型全家桶，自动克隆 → 自动吸收 → 自动锻造 → 吐结果
"""

import sys
import os
import json
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import BlackHoleTrainer, MultiverseResourceFactory


# 精选中国开源 AI 项目（少而精）
CHINA_OPENSOURCE_AI_PROJECTS = [
    # 大模型推理框架
    ("QwenLM/qwen.cpp", "qwen-cpp"),
    ("THUDM/ChatGLM.cpp", "chatglm-cpp"),

    # Agent / 应用框架
    ("geekan/MetaGPT", "metagpt"),
    ("OpenBMB/ChatDev", "chatdev"),

    # 推理/服务框架
    ("vllm-project/vllm", "vllm"),

    # 向量数据库
    ("chroma-core/chroma", "chroma"),
    ("milvus-io/milvus", "milvus"),
]


def clone_repos(target_dir: str, repos: list) -> dict:
    """
    给黑洞目标列表，让它自己去觅食（克隆）
    能成功的就吃，不成功的跳过
    """
    os.makedirs(target_dir, exist_ok=True)
    results = {"success": [], "failed": [], "skipped": []}

    for repo_path, local_name in repos:
        local_path = os.path.join(target_dir, local_name)
        if os.path.exists(local_path):
            results["skipped"].append(local_name)
            continue

        url = f"https://github.com/{repo_path}.git"
        print(f"  🍽️  觅食中：{repo_path} ...", end=" ", flush=True)

        try:
            r = subprocess.run(
                ["git", "clone", "--depth", "1", url, local_path],
                capture_output=True, text=True, timeout=120, cwd=target_dir
            )
            if r.returncode == 0 and os.path.exists(local_path):
                # 看看有没有代码
                has_code = False
                for root, dirs, files in os.walk(local_path):
                    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
                    for f in files:
                        if f.endswith((".py", ".js", ".ts", ".go", ".rs", ".c", ".cpp", ".h", ".java", ".cu")):
                            has_code = True
                            break
                    if has_code:
                        break
                if has_code:
                    results["success"].append(local_name)
                    print("✅ 入库")
                else:
                    results["failed"].append(f"{local_name} (无代码)")
                    print("❌ 空壳")
            else:
                results["failed"].append(local_name)
                print("❌ 失败")
        except Exception as e:
            results["failed"].append(f"{local_name} ({str(e)[:30]})")
            print("❌ 超时/错误")

    return results


def main():
    target_dir = "/workspace/china_ai_repos"

    print("\n🌌🌌🌌🌌🌌🌌🌌")
    print("  黑洞自动觅食启动")
    print("  目标：中国开源大模型全家桶")
    print("🌌🌌🌌🌌🌌🌌🌌\n")

    # 阶段一：自动觅食（克隆）
    print("【阶段 0/3】自动觅食 — 克隆目标代码库...\n")
    clone_result = clone_repos(target_dir, CHINA_OPENSOURCE_AI_PROJECTS)
    print()
    print(f"  觅食结果：")
    print(f"    ✅ 成功：{len(clone_result['success'])} 个")
    print(f"    ❌ 失败：{len(clone_result['failed'])} 个")
    print(f"    ⏭️  已存在：{len(clone_result['skipped'])} 个")
    print()

    # 收集所有本地代码库 + 新觅食到的
    all_repos = []

    # 原有本地代码库
    workspace = "/workspace"
    skip_dirs = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
                 "kosong", "coze_temp", "china_ai_repos"]
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

    # 新觅食到的（china_ai_repos下的子目录）
    if os.path.isdir(target_dir):
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                all_repos.append(full_path)

    # coze_repos 下的子目录
    coze_dir = os.path.join(workspace, "coze_repos")
    if os.path.isdir(coze_dir):
        for item in os.listdir(coze_dir):
            full_path = os.path.join(coze_dir, item)
            if os.path.isdir(full_path) and not item.startswith("."):
                all_repos.append(full_path)

    print(f"  共计 {len(all_repos)} 个代码库进入黑洞菜单\n")

    # 阶段二~四：黑洞训练（吸收→锻造→吐渣滓）
    trainer = BlackHoleTrainer(model_id="xenith-china-ai-supermassive")
    factory = MultiverseResourceFactory()

    result = trainer.absorb_and_forge(
        repo_paths=all_repos,
        factory=factory,
        languages=None,
        max_files_per_repo=3000,
        spin_rounds=9,  # 9圈，够用就行
        quality_threshold=0.55,
        knowledge_domains=[
            "computer_science", "engineering", "math",
        ],
        knowledge_count_per_domain=8000,  # 适量
    )

    # 加觅食统计
    result["foraging"] = clone_result
    result["total_repos"] = len(all_repos)

    # 保存报告
    report_path = "/workspace/xuni/examples/blackhole_china_ai_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📦 完整报告已保存：{report_path}\n")

    return result


if __name__ == "__main__":
    main()
