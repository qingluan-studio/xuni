"""
向训练好的Xenith模型提问，验证代码质量
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import XenithModel, MultiverseResourceFactory, BlackHoleTrainer


def main():
    print("=" * 60)
    print("  向 Xenith 模型提问测试")
    print("=" * 60 + "\n")

    # 1. 创建并训练模型（快速版）
    print("【初始化模型】")
    model = XenithModel(model_id="xenith-qa-test")
    factory = MultiverseResourceFactory()

    # 快速训练（用本地代码库 + 少量知识）
    repos = []
    workspace = "/workspace"
    skip = [".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build",
            "kosong", "coze_temp"]
    for item in ["xuni", "kimi-cli", "MonkeyCode"]:
        fp = os.path.join(workspace, item)
        if os.path.isdir(fp):
            repos.append(fp)

    print(f"训练代码库：{[os.path.basename(r) for r in repos]}")

    # 用黑洞训练器流式模式快速训练
    trainer = BlackHoleTrainer(model_id="xenith-qa-test", streaming=True)
    trainer.absorb_and_forge(
        repo_paths=repos,
        factory=factory,
        max_files_per_repo=2000,
        spin_rounds=5,
        quality_threshold=0.5,
        knowledge_domains=["computer_science", "engineering", "math"],
        knowledge_count_per_domain=5000,
    )

    # 把黑洞训练结果同步到模型
    model.absorb_blackhole_result(trainer)

    print(f"模型训练状态：{model.training_state}")
    print(f"模型质量分：{model.quality_score:.4f}")
    print(f"知识评分：{model.xenith_capabilities.knowledge_score:.4f}")
    print(f"代码强化等级：{model.code_refinement_level}/10")
    print(f"子代理数量：{model.agent_army_size}")
    print()

    # 2. 提问测试
    questions = [
        ("用Python写一个快速排序", "code"),
        ("解释一下什么是闭包，举个JavaScript例子", "code"),
        ("如何优化数据库查询性能？", "normal"),
        ("写一个Python装饰器，统计函数执行时间", "code"),
        ("解释HTTP和HTTPS的区别", "normal"),
    ]

    results = []
    for question, mode in questions:
        print(f"{'─' * 50}")
        print(f"📝 问题：{question}")
        print(f"   模式：{mode}")
        print()

        result = model.ask(question, mode=mode)

        if "error" in result:
            print(f"❌ {result['error']}")
            print()
            results.append({"question": question, "error": result["error"]})
            continue

        answer = result.get("answer", "")
        print(f"🤖 回答：")
        print(answer)
        print()
        print(f"   领域：{result.get('domain', '?')}")
        print(f"   置信度：{result.get('confidence', 0)}")
        print(f"   耗时：{result.get('latency_ms', 0):.1f}ms")
        print()

        results.append({
            "question": question,
            "mode": mode,
            "domain": result.get("domain"),
            "answer": answer,
            "confidence": result.get("confidence"),
            "latency_ms": result.get("latency_ms"),
        })

    # 3. 代码强化测试
    print(f"{'═' * 50}")
    print("🔧 代码强化测试")
    print(f"{'═' * 50}\n")

    test_code = """def qsort(a):
    if len(a)<=1: return a
    p=a[0]
    l=[x for x in a[1:] if x<p]
    r=[x for x in a[1:] if x>=p]
    return qsort(l)+[p]+qsort(r)
"""

    print("原始代码：")
    print(test_code)
    print()

    refine_result = model.refine_code(test_code, language="python")
    if "error" in refine_result:
        print(f"❌ {refine_result['error']}")
    else:
        print("强化后代码：")
        print(refine_result.get("refined_code", "（无）"))
        print()
        print(f"强化前评分：{refine_result['before']['score']} ({refine_result['before']['grade']})")
        print(f"强化后评分：{refine_result['after']['score']} ({refine_result['after']['grade']})")
        print(f"修改数：{refine_result.get('total_modifications', 0)}")

    # 保存结果
    out = "/workspace/xuni/examples/qa_test_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "questions": results,
            "refine_test": refine_result if "error" not in refine_result else {},
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📦 结果已保存：{out}\n")


if __name__ == "__main__":
    main()
