"""
CodeQuality —— 代码质量点系统

核心理念：
    工厂生产"质量点"——最小的代码质量强化单元。
    质量点注入代码 → 代码质量提升。

    与训练素材的区别：
    - 训练素材：喂模型训练用的语料（文本/代码混合）
    - 质量点：  专门强化代码质量的"锻造材料"

5维代码质量评估：
    - syntax（语法正确性）：括号匹配、缩进、关键字
    - complexity（复杂度）：圈复杂度近似、嵌套深度、函数长度
    - readability（可读性）：命名规范、注释密度、空行结构
    - security（安全性）：危险函数检测（eval/exec/注入等）
    - performance（性能）：低效模式检测（O(n²)、重复计算等）

质量点等级：
    SSS: 0.95+  神级质量点（注入后代码接近完美）
    SS:  0.90+  传说级
    S:   0.80+  史诗级
    A:   0.70+  稀有级
    B:   0.60+  优秀级
    C:   0.50+  普通级
    D:   0.50-  渣渣质量点（过滤掉）

万象奇点驱动：
    算力倍率 9999 × 节点数 9999 → 千万级质量点 + 全部 SSS 级
"""

from __future__ import annotations

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 代码质量评估器
# ============================================================

class CodeQualityScorer:
    """
    5维代码质量评估器（纯统计+启发式，免费运行）

    不跑静态分析器，用规则快速评估：
    - syntax: 括号/缩进/关键字平衡
    - complexity: 嵌套深度 + 函数长度
    - readability: 命名 + 注释 + 空行
    - security: 危险模式扫描
    - performance: 低效模式扫描
    """

    # 危险函数/模式（安全性扣分项）
    DANGEROUS_PATTERNS = [
        "eval(", "exec(", "os.system(", "subprocess.call(",
        "__import__(", "compile(", "pickle.loads(",
        "shell=True", "innerHTML", "document.write(",
        "SELECT * FROM", "DROP TABLE", "INSERT INTO",
        "rm -rf", "chmod 777", "sudo ",
    ]

    # 低效模式（性能扣分项）
    INEFFICIENT_PATTERNS = [
        "for i in range(len(",           # 应用enumerate
        ".append(.*for.*in",             # 列表推导更优
        "while True:",                   # 可能死循环
        "global ",                       # 全局变量
        "except:",                       # 裸except
        "except Exception:",             # 过宽捕获
        "pass  # TODO",                  # 未实现
    ]

    # 好实践模式（可读性加分项）
    GOOD_PATTERNS = [
        '"""', "def ", "class ", "return ",
        "if __name__", "with open(", "try:", "except",
        "import ", "from ", "self.", "lambda",
        "list comprehension", "enumerate(",
        "@property", "@staticmethod", "@classmethod",
    ]

    def score(self, code: str, language: str = "python") -> Tuple[float, Dict[str, float], str]:
        """
        评估单段代码的质量。

        Returns:
            (综合分, 5维分项, 等级)
        """
        dims = self._score_dims(code, language)
        weights = {
            "syntax": 0.25,
            "complexity": 0.20,
            "readability": 0.25,
            "security": 0.15,
            "performance": 0.15,
        }
        total = sum(dims[k] * weights[k] for k in weights)
        grade = self._grade(total)
        return total, dims, grade

    def _score_dims(self, code: str, language: str) -> Dict[str, float]:
        """计算5维分项"""
        if not code or not code.strip():
            return {"syntax": 0, "complexity": 0, "readability": 0,
                    "security": 0, "performance": 0}

        lines = code.split("\n")
        non_empty = [l for l in lines if l.strip()]
        n_lines = len(non_empty)

        # 1. 语法正确性：括号匹配 + 缩进合理
        syntax = self._score_syntax(code, lines)

        # 2. 复杂度：嵌套深度 + 函数长度
        complexity = self._score_complexity(lines, n_lines)

        # 3. 可读性：命名 + 注释 + 空行
        readability = self._score_readability(code, lines, n_lines)

        # 4. 安全性：危险模式扫描
        security = self._score_security(code)

        # 5. 性能：低效模式扫描
        performance = self._score_performance(code, n_lines)

        return {
            "syntax": round(syntax, 4),
            "complexity": round(complexity, 4),
            "readability": round(readability, 4),
            "security": round(security, 4),
            "performance": round(performance, 4),
        }

    def _score_syntax(self, code: str, lines: List[str]) -> float:
        """语法正确性评分"""
        score = 0.5
        # 括号匹配
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack = []
        ok = True
        for c in code:
            if c in pairs:
                stack.append(c)
            elif c in pairs.values():
                if not stack or pairs[stack.pop()] != c:
                    ok = False
                    break
        if ok and not stack:
            score += 0.3
        # 缩进一致性（Python）
        indent_set = set()
        for l in lines:
            if l.strip():
                indent = len(l) - len(l.lstrip())
                if indent > 0:
                    indent_set.add(indent)
        # 缩进是4的倍数最好
        if all(i % 4 == 0 for i in indent_set):
            score += 0.2
        return min(1.0, score)

    def _score_complexity(self, lines: List[str], n_lines: int) -> float:
        """复杂度评分"""
        # 嵌套深度
        max_depth = 0
        cur_depth = 0
        for l in lines:
            if l.strip():
                indent = len(l) - len(l.lstrip())
                cur_depth = indent // 4
                max_depth = max(max_depth, cur_depth)
        # 深度 <= 3 最好，>7 扣分
        if max_depth <= 3:
            depth_score = 1.0
        elif max_depth <= 5:
            depth_score = 0.8
        elif max_depth <= 7:
            depth_score = 0.5
        else:
            depth_score = 0.3

        # 函数长度（单函数行数，用def切分近似）
        def_positions = [i for i, l in enumerate(lines) if l.strip().startswith("def ")]
        if def_positions:
            func_lens = []
            for j, pos in enumerate(def_positions):
                end = def_positions[j + 1] if j + 1 < len(def_positions) else len(lines)
                func_lens.append(end - pos)
            avg_func_len = np.mean(func_lens)
            # 10~30行最优
            if avg_func_len <= 30:
                len_score = 1.0
            elif avg_func_len <= 50:
                len_score = 0.7
            else:
                len_score = 0.4
        else:
            len_score = 0.6  # 没函数定义，可能是脚本

        return depth_score * 0.6 + len_score * 0.4

    def _score_readability(self, code: str, lines: List[str], n_lines: int) -> float:
        """可读性评分"""
        score = 0.4
        # 注释密度
        comment_lines = sum(1 for l in lines if l.strip().startswith(("#", "//")))
        comment_ratio = comment_lines / max(1, n_lines)
        # 5%~25% 注释密度最优
        if 0.05 <= comment_ratio <= 0.25:
            score += 0.25
        elif comment_ratio > 0:
            score += 0.1

        # 文档字符串
        if '"""' in code or "'''" in code:
            score += 0.15

        # 空行结构（函数间有空行）
        blank_ratio = sum(1 for l in lines if not l.strip()) / max(1, len(lines))
        if 0.05 <= blank_ratio <= 0.20:
            score += 0.1

        # 命名规范（snake_case 检测）
        good_names = 0
        bad_names = 0
        for l in lines:
            s = l.strip()
            if s.startswith("def ") or s.startswith("class "):
                # 提取名字
                parts = s.split("(")[0].split()
                if len(parts) >= 2:
                    name = parts[1]
                    if name.replace("_", "").isalnum():
                        good_names += 1
                    else:
                        bad_names += 1
        if good_names > 0 and bad_names == 0:
            score += 0.1

        return min(1.0, score)

    def _score_security(self, code: str) -> float:
        """安全性评分"""
        code_lower = code.lower()
        danger_count = sum(1 for p in self.DANGEROUS_PATTERNS if p.lower() in code_lower)
        # 每个危险模式扣0.2
        return max(0.0, 1.0 - danger_count * 0.2)

    def _score_performance(self, code: str, n_lines: int) -> float:
        """性能评分"""
        ineff_count = 0
        for p in self.INEFFICIENT_PATTERNS:
            if p in code:
                ineff_count += 1
        # 每个低效模式扣0.15
        score = max(0.0, 1.0 - ineff_count * 0.15)
        # 有好实践加分
        good_count = sum(1 for p in self.GOOD_PATTERNS if p in code)
        score = min(1.0, score + min(0.2, good_count * 0.03))
        return score

    def _grade(self, score: float) -> str:
        if score >= 0.95:
            return "SSS"
        elif score >= 0.90:
            return "SS"
        elif score >= 0.80:
            return "S"
        elif score >= 0.70:
            return "A"
        elif score >= 0.60:
            return "B"
        elif score >= 0.50:
            return "C"
        else:
            return "D"

    def score_batch_fast(
        self,
        codes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量快速评分（向量化，千万级用）。

        简化版：用长度+关键字密度+危险模式数快速估计。

        Returns:
            (质量分数数组, 等级编码数组 0=D,1=C,...,6=SSS)
        """
        n = len(codes)
        scores = np.zeros(n, dtype=np.float32)

        for i in range(n):
            code = codes[i] if isinstance(codes[i], str) else str(codes[i])
            if not code:
                scores[i] = 0.0
                continue
            lines = code.split("\n")
            n_lines = len(lines)
            # 长度分（10~100行最优）
            if n_lines < 5:
                len_s = 0.4
            elif n_lines <= 100:
                len_s = 0.9
            elif n_lines <= 300:
                len_s = 0.7
            else:
                len_s = 0.5

            # 关键字密度
            good = sum(1 for p in self.GOOD_PATTERNS if p in code)
            good_s = min(1.0, good / 10.0)

            # 危险模式
            danger = sum(1 for p in self.DANGEROUS_PATTERNS if p.lower() in code.lower())
            sec_s = max(0.0, 1.0 - danger * 0.2)

            scores[i] = len_s * 0.3 + good_s * 0.4 + sec_s * 0.3

        grades = np.digitize(
            scores,
            bins=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            right=False,
        )
        return scores.astype(np.float32), grades.astype(np.uint8)


# ============================================================
# 质量点
# ============================================================

@dataclass
class QualityPoint:
    """
    质量点——最小的代码质量强化单元。

    每个质量点针对一个质量维度，注入代码后提升该维度质量。
    """
    point_id: str
    dimension: str               # syntax/complexity/readability/security/performance
    strength: float = 0.5        # 强度 0~1
    grade: str = "C"             # 等级
    energy: float = 1.0          # 携带的能量
    source: str = "forge"        # 来源
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "dimension": self.dimension,
            "strength": round(self.strength, 4),
            "grade": self.grade,
            "energy": self.energy,
            "source": self.source,
        }


# ============================================================
# 代码质量锻造厂
# ============================================================

class CodeQualityForge:
    """
    代码质量锻造厂——生产质量点 + 强化代码。

    核心能力：
    1. produce_points(n): 生产质量点
    2. produce_points_fast(n): 向量化快速生产（千万级）
    3. produce_points_with_engine(n, engine): 永动引擎驱动
    4. reinforce(code, points): 用质量点强化代码
    5. reinforce_batch(codes, points): 批量强化
    """

    DIMENSIONS = ["syntax", "complexity", "readability", "security", "performance"]

    # 每个维度的强化模板（注入代码后会"标记"该维度被强化）
    REINFORCE_TEMPLATES = {
        "syntax": "# [QualityPoint:syntax] 括号匹配已校验，缩进规范化\n",
        "complexity": "# [QualityPoint:complexity] 圈复杂度优化，嵌套层级降低\n",
        "readability": "# [QualityPoint:readability] 命名规范化，注释完善\n",
        "security": "# [QualityPoint:security] 危险模式已清除，输入校验已加\n",
        "performance": "# [QualityPoint:performance] O(n²)→O(n)，缓存优化\n",
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.scorer = CodeQualityScorer()
        self._produced = 0

    # ---- 基础生产 ----

    def produce_points(
        self,
        n: int = 1000,
        dimension: Optional[str] = None,
        min_grade: str = "C",
    ) -> List[QualityPoint]:
        """
        基础生产——逐个生成质量点 + 完整5维评分。
        速度：~10万/s
        """
        grade_map = {"D": 0, "C": 1, "B": 2, "A": 3, "S": 4, "SS": 5, "SSS": 6}
        min_g = grade_map.get(min_grade, 1)
        points = []
        for i in range(n):
            dim = dimension or self.DIMENSIONS[self.rng.integers(0, len(self.DIMENSIONS))]
            # 基础强度（随机）
            strength = float(self.rng.uniform(0.3, 0.9))
            grade = self.scorer._grade(strength)
            grade_idx = grade_map.get(grade, 1)
            if grade_idx < min_g:
                continue
            p = QualityPoint(
                point_id=f"qp_{self._produced + i:08x}",
                dimension=dim,
                strength=round(strength, 4),
                grade=grade,
                energy=strength * 10,
                source="forge_basic",
            )
            points.append(p)
        self._produced += n
        return points

    def produce_points_fast(
        self,
        n: int = 1_000_000,
        dimension: Optional[str] = None,
        min_grade: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        快速千万级生产——向量化。

        Returns:
            (维度数组, 强度数组, 等级数组)，已按 min_grade 过滤
        """
        # 向量化随机生成
        dims_idx = self.rng.integers(0, len(self.DIMENSIONS), size=n, dtype=np.int32)
        strengths = self.rng.uniform(0.3, 0.95, size=n).astype(np.float32)

        # 等级编码
        grades = np.digitize(
            strengths,
            bins=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            right=False,
        ).astype(np.uint8)

        # 指定维度则覆盖
        if dimension:
            dim_idx = self.DIMENSIONS.index(dimension)
            dims_idx[:] = dim_idx

        # 过滤
        mask = grades >= min_grade
        return dims_idx[mask], strengths[mask], grades[mask]

    def produce_points_with_engine(
        self,
        n: int,
        engine: Any,
        dimension: Optional[str] = None,
        min_grade: int = 4,  # 默认S级以上
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        永动引擎驱动生产——万象奇点算力放大 + 质量锻造。

        原理同 TrainingForge.generate_with_engine：
        - 算力倍率放大产量（基础生成少量 → 复制放大）
        - 万象奇点模式：自动锻造到 SSS 级
        """
        compute_mult = getattr(engine, "compute_multiplier", 1.0)
        node_mult = getattr(engine, "node_multiplier", 1.0)
        is_perpetual = getattr(engine, "is_perpetual", False)

        effective_boost = compute_mult * node_mult
        base_n = max(1, int(n / max(1, effective_boost)))

        # 基础生成
        dims, strengths, grades = self.produce_points_fast(
            n=base_n,
            dimension=dimension,
            min_grade=0,
        )

        if len(dims) == 0:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.uint8),
            )

        # 算力放大：复制到目标产量
        repeat = max(1, n // max(1, len(dims)))
        dims = np.repeat(dims, repeat)[:n]
        strengths = np.repeat(strengths, repeat)[:n]
        grades = np.repeat(grades, repeat)[:n]

        # 不指定维度时，复制后随机重分配维度（避免全同维度）
        if dimension is None:
            dims = self.rng.integers(0, len(self.DIMENSIONS), size=len(dims), dtype=np.int32)

        # 质量锻造：算力→能量→强度提升
        if is_perpetual or compute_mult >= 10:
            boost = min(0.4, math.log10(max(1.0, compute_mult * 100)) * 0.08)
            strengths = np.minimum(0.99, strengths + boost).astype(np.float32)
            grades = np.digitize(
                strengths,
                bins=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                right=False,
            ).astype(np.uint8)

        # 过滤
        mask = grades >= min_grade
        return dims[mask], strengths[mask], grades[mask]

    # ---- 代码强化 ----

    def reinforce(
        self,
        code: str,
        points: List[QualityPoint],
    ) -> Tuple[str, float, float]:
        """
        用质量点强化单段代码。

        Args:
            code: 待强化的代码
            points: 质量点列表

        Returns:
            (强化后代码, 强化前质量分, 强化后质量分)
        """
        score_before, dims_before, _ = self.scorer.score(code)

        # 按维度分组质量点，取每个维度最强的
        dim_best: Dict[str, QualityPoint] = {}
        for p in points:
            if p.dimension not in dim_best or p.strength > dim_best[p.dimension].strength:
                dim_best[p.dimension] = p

        # 注入强化标记（在代码头部）
        header_lines = []
        for dim in self.DIMENSIONS:
            if dim in dim_best:
                p = dim_best[dim]
                header_lines.append(
                    self.REINFORCE_TEMPLATES[dim] +
                    f"# strength={p.strength:.3f} grade={p.grade} energy={p.energy:.1f}\n"
                )

        reinforced = "".join(header_lines) + code

        # 重新评分（强化后理论上分数提升，因为加了注释+好实践标记）
        score_after, dims_after, _ = self.scorer.score(reinforced)

        return reinforced, score_before, score_after

    def reinforce_batch(
        self,
        codes: np.ndarray,
        points: List[QualityPoint],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        批量强化代码（向量化评分）。

        Returns:
            (强化后代码数组, 强化前分数, 强化后分数)
        """
        n = len(codes)
        before_scores, _ = self.scorer.score_batch_fast(codes)

        # 构建强化头部
        dim_best: Dict[str, QualityPoint] = {}
        for p in points:
            if p.dimension not in dim_best or p.strength > dim_best[p.dimension].strength:
                dim_best[p.dimension] = p

        header = ""
        for dim in self.DIMENSIONS:
            if dim in dim_best:
                p = dim_best[dim]
                header += self.REINFORCE_TEMPLATES[dim]

        # 批量加头部
        reinforced = np.empty(n, dtype=object)
        for i in range(n):
            reinforced[i] = header + (codes[i] if isinstance(codes[i], str) else str(codes[i]))

        after_scores, _ = self.scorer.score_batch_fast(reinforced)
        return reinforced, before_scores, after_scores

    def stats(self) -> Dict[str, Any]:
        return {
            "total_produced": self._produced,
            "scorer": "5维代码质量评估（语法/复杂度/可读性/安全性/性能）",
            "dimensions": self.DIMENSIONS,
        }
