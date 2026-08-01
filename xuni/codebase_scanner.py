"""
CodebaseScanner — 真实代码库扫描器

从现实代码库中提取训练素材，用于训练 Xenith 模型。

功能：
1. 扫描目录，收集代码文件
2. 用 AST 提取函数、类、方法
3. 生成训练素材（代码片段 + 文档注释 + 质量评分）
4. 支持多语言：Python / JavaScript / TypeScript / Java / Go / Rust / C++
5. 自动检测真实代码 → 存入种子库
"""

import ast
import os
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class CodebaseScanner:
    """
    真实代码库扫描器。

    用法：
        scanner = CodebaseScanner()
        result = scanner.scan_repo("/path/to/repo")
        # result 里有提取出的训练素材，可直接喂给 Xenith
    """

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
        ".c": "c",
        ".h": "c_header", ".hpp": "cpp_header",
    }

    def __init__(self, max_file_size_kb: int = 500):
        self.max_file_size_kb = max_file_size_kb
        self._rng = np.random.default_rng(42)
        self.scanned_files: List[str] = []
        self.extracted_functions: List[Dict[str, Any]] = []
        self.extracted_classes: List[Dict[str, Any]] = []
        self.code_seeds: List[str] = []  # 真实代码种子（用于种子库）

    # ---- 主入口 ----

    def scan_repo(
        self,
        repo_path: str,
        languages: Optional[List[str]] = None,
        max_files: int = 1000,
    ) -> Dict[str, Any]:
        """
        扫描一个代码库，提取训练素材。

        Args:
            repo_path: 代码库根目录
            languages: 语言过滤，None=全部支持的语言
            max_files: 最多扫描的文件数

        Returns:
            扫描结果 + 训练素材
        """
        repo_path = os.path.abspath(repo_path)
        if not os.path.exists(repo_path):
            return {"error": f"路径不存在: {repo_path}"}

        start = time.time()
        self.scanned_files = []
        self.extracted_functions = []
        self.extracted_classes = []
        self.code_seeds = []

        # 1. 收集文件
        files = self._collect_files(repo_path, languages, max_files)

        # 2. 逐文件解析提取
        for fpath in files:
            try:
                size = os.path.getsize(fpath)
                if size > self.max_file_size_kb * 1024:
                    continue

                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    continue

                self.scanned_files.append(fpath)

                # 按语言提取
                ext = os.path.splitext(fpath)[1].lower()
                lang = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

                if lang == "python":
                    self._extract_python(content, fpath)
                else:
                    self._extract_generic(content, fpath, lang)

            except Exception:
                continue

        # 3. 生成训练素材
        training_data = self._generate_training_data()

        elapsed = time.time() - start

        return {
            "repo_path": repo_path,
            "files_scanned": len(self.scanned_files),
            "functions_extracted": len(self.extracted_functions),
            "classes_extracted": len(self.extracted_classes),
            "code_seeds": len(self.code_seeds),
            "languages_found": self._count_languages(),
            "training_items": len(training_data["texts"]),
            "avg_quality": training_data["avg_quality"],
            "elapsed_seconds": round(elapsed, 2),
            "training_data": training_data,
        }

    # ---- 文件收集 ----

    def _collect_files(
        self,
        root: str,
        languages: Optional[List[str]],
        max_files: int,
    ) -> List[str]:
        """收集代码文件"""
        exts = set()
        if languages:
            for ext, lang in self.SUPPORTED_EXTENSIONS.items():
                if lang in languages:
                    exts.add(ext)
        else:
            exts = set(self.SUPPORTED_EXTENSIONS.keys())

        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过隐藏目录、node_modules、__pycache__ 等
            dirnames[:] = [d for d in dirnames if not d.startswith(".")
                          and d not in ("node_modules", "__pycache__", ".git",
                                       "venv", ".venv", "dist", "build")]

            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in exts:
                    files.append(os.path.join(dirpath, fname))
                    if len(files) >= max_files:
                        return files

        return files

    # ---- Python 代码提取（用 AST）----

    def _extract_python(self, content: str, filepath: str):
        """用 AST 提取 Python 函数和类"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # 语法错误的也加入种子库（真实代码就是有好有坏）
            if len(content) > 50:
                self.code_seeds.append(content)
            return

        # 存入种子库（取前 500 字符）
        if len(content) > 100:
            seed = content[:1000] if len(content) > 1000 else content
            self.code_seeds.append(seed)

        lines = content.splitlines()

        for node in ast.walk(tree):
            # 函数提取
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function_info(node, lines, filepath)
                self.extracted_functions.append(func_info)

            # 类提取
            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, lines, filepath)
                self.extracted_classes.append(class_info)

    def _extract_function_info(
        self, node: ast.AST, lines: List[str], filepath: str
    ) -> Dict[str, Any]:
        """提取函数信息"""
        name = node.name
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno + 10)

        # 取函数代码
        func_code = "\n".join(lines[lineno - 1:end_lineno])

        # docstring
        docstring = ast.get_docstring(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

        # 参数
        args = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]

        # 复杂度估计（行数 + 分支数）
        complexity = 1
        for sub in ast.walk(node):
            if isinstance(sub, (ast.If, ast.For, ast.While, ast.Try, ast.With,
                              ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp)):
                complexity += 1

        # 质量粗略评估
        quality = self._quick_quality_score(func_code, docstring, complexity)

        return {
            "name": name,
            "type": "function",
            "language": "python",
            "filepath": filepath,
            "lineno": lineno,
            "code": func_code,
            "docstring": docstring or "",
            "args": args,
            "complexity": complexity,
            "lines": end_lineno - lineno + 1,
            "quality": quality,
        }

    def _extract_class_info(
        self, node: ast.ClassDef, lines: List[str], filepath: str
    ) -> Dict[str, Any]:
        """提取类信息"""
        name = node.name
        lineno = node.lineno
        end_lineno = getattr(node, "end_lineno", lineno + 20)

        class_code = "\n".join(lines[lineno - 1:end_lineno])
        docstring = ast.get_docstring(node)

        # 统计方法数
        methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

        # 质量评估
        quality = self._quick_quality_score(class_code, docstring, len(methods))

        return {
            "name": name,
            "type": "class",
            "language": "python",
            "filepath": filepath,
            "lineno": lineno,
            "code": class_code,
            "docstring": docstring or "",
            "methods": methods,
            "n_methods": len(methods),
            "lines": end_lineno - lineno + 1,
            "quality": quality,
        }

    # ---- 通用提取（非 Python 语言，用简单启发式）----

    def _extract_generic(self, content: str, filepath: str, lang: str):
        """非 Python 语言的简单提取"""
        if len(content) > 100:
            seed = content[:1000] if len(content) > 1000 else content
            self.code_seeds.append(seed)

        # 简单的函数识别（启发式）
        lines = content.splitlines()
        func_patterns = {
            "javascript": [r"function\s+(\w+)", r"const\s+(\w+)\s*=\s*(?:async\s+)?\("],
            "typescript": [r"function\s+(\w+)", r"(?:public|private|protected)?\s+\w+\s*\("],
            "java": [r"(?:public|private|protected)\s+\w+\s+(\w+)\s*\("],
            "go": [r"func\s+(\w+)\s*\("],
            "rust": [r"fn\s+(\w+)\s*\("],
            "cpp": [r"\w+\s+(\w+)\s*\([^)]*\)\s*\{"],
            "c": [r"\w+\s+(\w+)\s*\([^)]*\)\s*\{"],
        }

        import re
        patterns = func_patterns.get(lang, [])
        found = 0

        for i, line in enumerate(lines):
            for pat in patterns:
                m = re.match(r"^\s*" + pat, line)
                if m:
                    func_name = m.group(1)
                    # 简单截取函数体（到下一个同级 { 闭合或 50 行）
                    end = min(i + 50, len(lines))
                    func_code = "\n".join(lines[i:end])
                    quality = self._quick_quality_score(func_code, "", 3)
                    self.extracted_functions.append({
                        "name": func_name,
                        "type": "function",
                        "language": lang,
                        "filepath": filepath,
                        "lineno": i + 1,
                        "code": func_code,
                        "docstring": "",
                        "args": [],
                        "complexity": 3,
                        "lines": end - i,
                        "quality": quality,
                    })
                    found += 1
                    break
            if found >= 20:  # 每文件最多20个
                break

    # ---- 质量快速评估 ----

    def _quick_quality_score(
        self, code: str, docstring: str, complexity: int
    ) -> float:
        """快速评估代码质量（0~1）"""
        score = 0.5  # 基础分

        # 有文档注释加分
        if docstring:
            score += min(0.2, len(docstring) / 500)

        # 代码长度适中加分（太短太简单，太长太复杂）
        lines = code.count("\n") + 1
        if 5 <= lines <= 100:
            score += 0.1
        elif lines > 200:
            score -= 0.1

        # 复杂度适中
        if 2 <= complexity <= 10:
            score += 0.1
        elif complexity > 20:
            score -= 0.1

        # 有注释加分
        comment_ratio = code.count("#") / max(1, lines) if "#" in code else 0
        if comment_ratio > 0.1:
            score += 0.05

        return min(1.0, max(0.1, score))

    # ---- 生成训练素材 ----

    def _generate_training_data(self) -> Dict[str, Any]:
        """
        将提取的代码转为训练素材格式。

        训练素材格式：
        - texts: 文本数组（"函数名 + 文档 + 代码" 格式）
        - scores: 质量分数
        - grades: 等级（D~SSS）
        """
        all_items = self.extracted_functions + self.extracted_classes
        if not all_items:
            return {"texts": np.array([], dtype=object),
                    "scores": np.array([], dtype=np.float32),
                    "grades": np.array([], dtype=np.uint8),
                    "avg_quality": 0.0}

        n = len(all_items)
        texts = np.empty(n, dtype=object)
        scores = np.empty(n, dtype=np.float32)

        for i, item in enumerate(all_items):
            # 构造训练文本
            header = f"[{item['language']} {item['type']}] {item['name']}"
            doc = item.get("docstring", "")
            code = item["code"]
            fp = hashlib.md5(item["filepath"].encode()).hexdigest()[:8]

            if doc:
                text = f"{header}\n\"\"\"\n{doc}\n\"\"\"\n{code}\n[source:{fp}]"
            else:
                text = f"{header}\n{code}\n[source:{fp}]"

            texts[i] = text
            scores[i] = item["quality"]

        # 等级划分
        grade_bounds = [0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 0.97]
        grade_labels = ["D", "C", "B", "A", "S", "SS", "SSS"]
        grades = np.zeros(n, dtype=np.uint8)
        for bi, bound in enumerate(grade_bounds):
            grades[scores >= bound] = bi + 1

        avg_q = float(scores.mean()) if n > 0 else 0.0

        return {
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "avg_quality": avg_q,
            "grade_distribution": {
                grade_labels[i]: int(np.sum(grades == i))
                for i in range(len(grade_labels))
            },
        }

    def _count_languages(self) -> Dict[str, int]:
        """统计扫描到的各语言文件数"""
        lang_count: Dict[str, int] = {}
        for fpath in self.scanned_files:
            ext = os.path.splitext(fpath)[1].lower()
            lang = self.SUPPORTED_EXTENSIONS.get(ext, "other")
            lang_count[lang] = lang_count.get(lang, 0) + 1
        return lang_count

    # ---- 导出工具 ----

    def get_seed_library(self) -> List[str]:
        """获取真实代码种子库"""
        return self.code_seeds[:]

    def get_top_functions(self, n: int = 20) -> List[Dict[str, Any]]:
        """获取质量最高的 n 个函数"""
        sorted_funcs = sorted(
            self.extracted_functions, key=lambda x: x["quality"], reverse=True
        )
        return sorted_funcs[:n]

    def get_top_classes(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取质量最高的 n 个类"""
        sorted_classes = sorted(
            self.extracted_classes, key=lambda x: x["quality"], reverse=True
        )
        return sorted_classes[:n]
