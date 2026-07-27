"""
代码精度验证 — 生成代码并实际运行测试
"""
import sys
import os
import json
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xuni import XenithModel, BlackHoleTrainer, MultiverseResourceFactory


def train_model():
    """加载或训练模型"""
    model = XenithModel(model_id="xenith-precision-test")
    factory = MultiverseResourceFactory()

    # 快速训练（用几个高质量代码库）
    repos = []
    for item in ["xuni", "kimi-cli", "MonkeyCode", "openclaw"]:
        fp = os.path.join("/workspace", item)
        if os.path.isdir(fp):
            repos.append(fp)

    trainer = BlackHoleTrainer(model_id="xenith-precision-test", streaming=True)
    trainer.absorb_and_forge(
        repo_paths=repos,
        factory=factory,
        max_files_per_repo=1000,
        spin_rounds=9,
        quality_threshold=0.4,
        knowledge_domains=["computer_science"],
        knowledge_count_per_domain=5000,
    )

    model.absorb_blackhole_result(trainer)
    return model


def extract_code(answer: str, lang: str = "python") -> str:
    """从回答中提取代码块"""
    marker = f"```{lang}"
    start = answer.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = answer.find("```", start)
    if end == -1:
        return ""
    return answer[start:end].strip()


def test_code_execution(code: str, test_name: str) -> dict:
    """执行代码并返回结果"""
    result = {
        "test": test_name,
        "passed": False,
        "error": None,
        "output": "",
    }
    try:
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, {"__name__": "__main__"})
        result["output"] = f.getvalue().strip()
        result["passed"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
    return result


def main():
    print("\n" + "=" * 70)
    print("  🧪 代码精度验证测试")
    print("  验证生成代码的语法正确性、可执行性、注释规范性")
    print("=" * 70 + "\n")

    print("【训练模型...】")
    model = train_model()
    print(f"模型质量: {model.quality_score:.4f}")
    print(f"代码强化等级: {model.code_refinement_level}/10")
    print()

    # 测试用例
    test_cases = [
        ("快速排序", "code", "python"),
        ("二分查找", "code", "python"),
        ("单例模式", "code", "python"),
        ("装饰器 计时", "code", "python"),
        ("LRU缓存", "code", "python"),
        ("BFS 图遍历", "code", "python"),
        ("观察者模式 事件总线", "code", "python"),
    ]

    results = []
    for question, mode, lang in test_cases:
        print(f"{'─' * 50}")
        print(f"📝 测试: {question}")

        answer = model.ask(question, mode=mode)
        code = extract_code(answer["answer"], lang)

        if not code:
            print(f"  ❌ 未提取到代码")
            results.append({"question": question, "error": "no code found"})
            continue

        # 语法检查
        syntax_ok = True
        syntax_err = None
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            syntax_ok = False
            syntax_err = str(e)

        # 运行测试
        exec_result = test_code_execution(code, question)

        # 统计注释/类型注解
        lines = code.split("\n")
        comment_lines = sum(1 for l in lines if l.strip().startswith("#") or '"""' in l or "'''" in l)
        type_annotations = sum(1 for l in lines if "->" in l and "def " in l)
        docstrings = code.count('"""') // 2

        print(f"  行数: {len(lines)}")
        print(f"  语法检查: {'✅' if syntax_ok else '❌ ' + str(syntax_err)}")
        print(f"  运行测试: {'✅' if exec_result['passed'] else '❌ ' + exec_result['error']}")
        print(f"  注释/文档字符串: {comment_lines} 行 / {docstrings} 个docstring")
        print(f"  类型注解函数: {type_annotations} 个")
        if exec_result["output"]:
            preview = exec_result["output"][:200].replace("\n", " | ")
            print(f"  输出预览: {preview}")
        print()

        results.append({
            "question": question,
            "lines": len(lines),
            "syntax_ok": syntax_ok,
            "exec_ok": exec_result["passed"],
            "error": exec_result.get("error"),
            "comment_lines": comment_lines,
            "docstrings": docstrings,
            "type_annotations": type_annotations,
        })

    # 汇总
    print("=" * 70)
    print("  📊 精度测试汇总")
    print("=" * 70)
    print()

    total = len(results)
    syntax_pass = sum(1 for r in results if r.get("syntax_ok"))
    exec_pass = sum(1 for r in results if r.get("exec_ok"))

    print(f"  测试用例：{total} 个")
    print(f"  语法正确：{syntax_pass}/{total} ({syntax_pass/total*100:.0f}%)")
    print(f"  运行通过：{exec_pass}/{total} ({exec_pass/total*100:.0f}%)")
    print()

    avg_lines = sum(r["lines"] for r in results) / total
    avg_comments = sum(r["comment_lines"] for r in results) / total
    avg_docs = sum(r["docstrings"] for r in results) / total
    avg_types = sum(r["type_annotations"] for r in results) / total

    print(f"  平均代码行数：{avg_lines:.0f} 行")
    print(f"  平均注释行：{avg_comments:.0f} 行")
    print(f"  平均 docstring：{avg_docs:.1f} 个")
    print(f"  平均类型注解：{avg_types:.1f} 个")
    print()

    # 保存结果
    out = "/workspace/xuni/examples/code_precision_test_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"📦 详细报告：{out}\n")


if __name__ == "__main__":
    main()
