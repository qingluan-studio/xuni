"""
KnowledgeDownloader —— 模型驱动知识下载器

核心理念：
    现有的 CorpusDownloader 从 GitHub 下载真实代码，受网络带宽限制（~10MB/s）。
    KnowledgeDownloader 不走网络——用"模型+算力"直接"解码"现实多领域知识。

核心设定：
    - 普通下载：受网络带宽限制
    - 模型驱动：模型越强，"解码"知识越快，算力倍率直接放大下载速度
    - 攻破阈值：模型质量分 > 0.8 时，下载速度 = 算力倍率 × 基础速度
    - 产出：多领域训练素材，质量有保障（模型过滤）

速度公式：
    base_speed = 10000 条/秒（纯模型解码，不走网络）
    if model_quality > 0.8: speed *= model_quality            （模型加成）
    speed *= compute_multiplier * node_multiplier             （算力加成）
    if model_quality > 0.9 and compute_mult > 100: speed *= 100 （攻破模式）

知识解码原理（虚拟世界设定）：
    不是真从互联网下载——每个领域的每个主题都有一个稳定的"知识指纹"
    （SHA256 派生），模型用算力去"读取"指纹对应的内容。模型越强读得越快、
    质量越高；算力越大并发解码节点越多，速度被算力倍率线性放大。

产出格式与 TrainingForge.generate_fast 兼容：
    (texts: np.ndarray, scores: np.ndarray, grades: np.ndarray)
"""

from __future__ import annotations

import hashlib
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class KnowledgeDownloader:
    """
    模型驱动知识下载器——用模型+算力直接解码现实知识。

    核心设定：
    - 普通下载：受网络带宽限制，~10MB/s
    - 模型驱动：模型越强，"解码"知识越快，算力倍率直接放大下载速度
    - 攻破阈值：模型质量分>0.8 时，下载速度 = 算力倍率 × 基础速度
    - 产出：多领域训练素材，质量有保障（模型过滤）
    """

    # 基础速度：纯模型解码，不走网络，10000条/秒
    BASE_SPEED = 10000

    # 领域知识库（模拟现实知识的"指纹"，下载时根据指纹"解码"出内容）
    DOMAIN_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
        "math": {
            "topics": ["微积分", "线性代数", "概率论", "数论", "拓扑学",
                       "群论", "微分方程", "傅里叶分析", "数值计算", "图论"],
            "depth": 5,        # 知识深度
            "quality_base": 0.75,  # 基础质量
        },
        "physics": {
            "topics": ["量子力学", "相对论", "热力学", "电磁学", "流体力学",
                       "粒子物理", "弦理论", "凝聚态物理", "天体物理", "光学"],
            "depth": 5,
            "quality_base": 0.78,
        },
        "chemistry": {
            "topics": ["有机化学", "无机化学", "物理化学", "分析化学",
                       "量子化学", "电化学", "高分子化学", "生物化学",
                       "配位化学", "立体化学"],
            "depth": 5,
            "quality_base": 0.76,
        },
        "biology": {
            "topics": ["分子生物学", "细胞生物学", "遗传学", "进化论",
                       "生态学", "神经生物学", "微生物学", "免疫学",
                       "发育生物学", "生物信息学"],
            "depth": 5,
            "quality_base": 0.77,
        },
        "medicine": {
            "topics": ["解剖学", "生理学", "病理学", "药理学", "诊断学",
                       "内科学", "外科学", "神经病学", "影像学", "流行病学"],
            "depth": 5,
            "quality_base": 0.79,
        },
        "law": {
            "topics": ["宪法", "民法", "刑法", "行政法", "经济法",
                       "国际法", "知识产权法", "劳动法", "环境法", "诉讼法"],
            "depth": 4,
            "quality_base": 0.72,
        },
        "finance": {
            "topics": ["宏观经济学", "微观经济学", "货币银行学", "投资学",
                       "公司金融", "衍生品", "行为金融", "计量经济学",
                       "风险管理", "国际金融"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "philosophy": {
            "topics": ["形而上学", "认识论", "伦理学", "逻辑学", "美学",
                       "政治哲学", "心灵哲学", "语言哲学", "科学哲学",
                       "存在主义"],
            "depth": 4,
            "quality_base": 0.70,
        },
        "computer_science": {
            "topics": ["算法", "数据结构", "操作系统", "计算机网络", "数据库",
                       "编译原理", "人工智能", "机器学习", "分布式系统",
                       "密码学"],
            "depth": 5,
            "quality_base": 0.80,
        },
        "engineering": {
            "topics": ["机械工程", "电子工程", "土木工程", "化学工程",
                       "控制工程", "材料工程", "航空航天", "机器人学",
                       "信号处理", "能源工程"],
            "depth": 5,
            "quality_base": 0.76,
        },
    }

    # 知识模板（"解码"现实知识时填充，与 TrainingForge.TEMPLATES_TEXT 风格一致）
    TEMPLATES = [
        "{domain}中的{topic}研究{aspect}，通过{method}得到{result}。",
        "{topic}的核心是{concept}，其{aspect}决定了{result}。",
        "在{domain}领域，{topic}的{aspect}表现为{result}，这源于{concept}。",
        "通过对{topic}的{method}分析，揭示了{aspect}与{result}的关系。",
        "{topic}的{concept}理论指出：当{condition}时，必然产生{result}。",
        "{domain}研究表明，{topic}的{aspect}受{concept}影响，导致{result}。",
        "基于{topic}的{method}，可以推导出{aspect}的{result}规律。",
        "{topic}中的{concept}是理解{aspect}的关键，最终得到{result}。",
        "从{topic}角度出发，{aspect}的{result}体现了{concept}的本质。",
        "{domain}经典结论：{topic}的{method}揭示了{result}，验证了{concept}。",
    ]

    # 概念池（用于填充模板中的占位符）
    ASPECTS = ["基本原理", "核心机制", "边界条件", "数学结构", "实验现象",
               "理论模型", "应用场景", "历史发展", "现代进展", "未来方向"]
    METHODS = ["定量分析", "实验验证", "理论推导", "数值模拟", "统计分析",
               "归纳推理", "演绎证明", "对比研究", "系统建模", "迭代优化"]
    CONCEPTS = ["守恒定律", "对称性", "熵增原理", "因果律", "线性叠加",
                "非线性耦合", "量子化", "渐近行为", "稳定性", "涌现性"]
    RESULTS = ["新发现", "重要结论", "普遍规律", "特殊现象", "理论突破",
               "应用价值", "实验验证", "数学证明", "模型预测", "工程实现"]
    CONDITIONS = ["系统达到平衡", "参数趋近极限", "温度足够低", "能量足够高",
                  "尺度足够大", "时间足够长", "边界确定", "约束解除"]

    # 英文领域名 → 中文领域名（用于模板填充）
    DOMAIN_CN = {
        "math": "数学", "physics": "物理学", "chemistry": "化学",
        "biology": "生物学", "medicine": "医学", "law": "法学",
        "finance": "金融学", "philosophy": "哲学",
        "computer_science": "计算机科学", "engineering": "工程学",
    }

    def __init__(self):
        self.downloaded: int = 0
        self.domain_stats: Dict[str, Dict[str, Any]] = {}
        self.model = None
        self.engine = None
        self._rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))

    # ---- 接入模型与引擎 ----

    def attach_model(self, model) -> None:
        """接入模型，模型质量分决定下载倍率"""
        self.model = model

    def attach_engine(self, engine) -> None:
        """接入永动引擎，算力倍率放大下载速度"""
        self.engine = engine

    # ---- 私有：从模型/引擎读取加成参数 ----

    def _get_model_quality(self) -> float:
        """获取模型质量分（0~1）。模型越强，解码越快、过滤越严。"""
        if self.model is None:
            return 0.0
        # 优先使用显式质量属性，回退到 training_progress
        for attr in ("quality_score", "model_quality", "quality"):
            v = getattr(self.model, attr, None)
            if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                return float(v)
        return float(getattr(self.model, "training_progress", 0.0))

    def _get_compute_mult(self) -> float:
        """获取算力倍率（来自 PerpetualTrainingEngine）"""
        if self.engine is None:
            return 1.0
        return float(getattr(self.engine, "compute_multiplier", 1.0))

    def _get_node_mult(self) -> float:
        """获取节点数倍率（来自 PerpetualTrainingEngine）"""
        if self.engine is None:
            return 1.0
        return float(getattr(self.engine, "node_multiplier", 1.0))

    def _compute_speed(self) -> Tuple[float, Dict[str, Any]]:
        """
        计算当前下载速度（条/秒）。

        公式：
            speed = base_speed
            if model_quality > 0.8: speed *= model_quality          （模型加成）
            speed *= compute_mult * node_mult                       （算力加成）
            if model_quality > 0.9 and compute_mult > 100:
                speed *= 100                                          （攻破模式）
        """
        model_q = self._get_model_quality()
        compute_mult = self._get_compute_mult()
        node_mult = self._get_node_mult()

        speed = float(self.BASE_SPEED)

        # 模型加成：模型质量分 > 0.8 时启动倍率
        model_boost = 1.0
        if model_q > 0.8:
            model_boost = model_q
            speed *= model_boost

        # 算力加成：算力倍率 × 节点倍率
        speed *= compute_mult * node_mult

        # 攻破模式：模型质量 > 0.9 且算力倍率 > 100 → 速度×100
        breakthrough = model_q > 0.9 and compute_mult > 100
        if breakthrough:
            speed *= 100.0

        info = {
            "base_speed": self.BASE_SPEED,
            "model_quality": round(model_q, 4),
            "model_boost": round(model_boost, 4),
            "compute_mult": compute_mult,
            "node_mult": node_mult,
            "breakthrough": breakthrough,
            "speed": speed,
        }
        return speed, info

    # ---- 知识指纹 & 解码 ----

    def get_knowledge_fingerprint(self, domain: str, topic: str) -> str:
        """获取知识指纹——模型用这个指纹解码出内容"""
        # 用领域+主题派生稳定指纹（模拟现实知识的"哈希地址"）
        h = hashlib.sha256(f"{domain}::{topic}".encode("utf-8"))
        return h.hexdigest()[:16]

    def _decode_topic(self, domain: str, topic: str, idx: int, depth: int) -> str:
        """
        模型"解码"知识指纹，得到一条训练素材。

        不是真从互联网下载——而是用模型+随机种子"读取"出领域知识。
        同 (domain, topic, idx) 总是产生同一内容（指纹可复现）。
        """
        # 用指纹 + idx 派生稳定随机源
        fp = self.get_knowledge_fingerprint(domain, topic)
        seed = int(hashlib.md5(f"{fp}:{idx}".encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)

        domain_cn = self.DOMAIN_CN.get(domain, domain)
        tmpl = self.TEMPLATES[int(rng.integers(0, len(self.TEMPLATES)))]
        aspect = self.ASPECTS[int(rng.integers(0, len(self.ASPECTS)))]
        method = self.METHODS[int(rng.integers(0, len(self.METHODS)))]
        concept = self.CONCEPTS[int(rng.integers(0, len(self.CONCEPTS)))]
        result = self.RESULTS[int(rng.integers(0, len(self.RESULTS)))]
        condition = self.CONDITIONS[int(rng.integers(0, len(self.CONDITIONS)))]

        text = tmpl.format(
            domain=domain_cn, topic=topic, aspect=aspect, method=method,
            concept=concept, result=result, condition=condition,
        )
        # 深度越深，内容越长（每深一层追加一句分析）
        for d in range(1, depth):
            text += (f" 深入第{d}层：{topic}的{aspect}在{concept}下展现{result}，"
                     f"{method}验证此规律。")
        # 末尾加指纹溯源标记
        text += f" [fp:{fp}]"
        return text

    def _score_texts(
        self, texts: np.ndarray, quality_base: float, model_q: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速质量评分（与 QualityScorer.score_batch_fast 兼容的简化版）。

        模型越强过滤越严，整体质量基线被 model_quality 直接抬升。
        Returns:
            (scores: np.float32, grades: np.uint8 0=D..6=SSS)
        """
        n = len(texts)
        if n == 0:
            return (np.array([], dtype=np.float32),
                    np.array([], dtype=np.uint8))

        # 长度分（与 QualityScorer.score_batch_fast 一致）
        lengths = np.array([len(t) for t in texts], dtype=np.float32)
        len_score = np.where(
            lengths < 50,
            lengths / 50.0 * 0.5,
            np.where(
                lengths > 2000,
                np.maximum(0.3, 1.0 - (lengths - 2000) / 8000.0),
                0.7 + (lengths - 50) / 1950.0 * 0.3,
            ),
        )

        # 字符多样性分
        char_div = np.zeros(n, dtype=np.float32)
        for i in range(n):
            t = texts[i]
            if len(t) > 0:
                char_div[i] = len(set(t)) / max(1, len(t))
        char_div = np.minimum(1.0, char_div * 2.0)

        # 综合分 = 领域基础质量 + 长度 + 多样性
        scores = quality_base * 0.5 + len_score * 0.25 + char_div * 0.25
        # 模型过滤：模型越强，质量基线越高（低质素材被模型剔除的模拟效果）
        scores = np.minimum(0.99, scores + model_q * 0.1)

        # 等级编码（与 QualityScorer._grade / np.digitize 一致：0=D,1=C,...,6=SSS）
        grades = np.digitize(
            scores,
            bins=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            right=False,
        )
        return scores.astype(np.float32), grades.astype(np.uint8)

    # ---- 主接口 ----

    def download(self, domain: str, count: int = 10000) -> Dict[str, Any]:
        """
        下载指定领域的知识，转为训练素材。

        速度公式：
        - 基础速度：10000条/秒
        - 模型加成：model_quality > 0.8 时，×model_quality
        - 算力加成：×compute_multiplier × node_multiplier
        - 攻破模式：model_quality > 0.9 且 compute_mult > 100 → 速度×100

        Returns:
            dict 包含：
              - total: 下载条数
              - speed: 下载速度（条/秒）
              - speed_info: 速度计算明细
              - texts / scores / grades: 训练素材（与 TrainingForge.generate_fast 兼容）
              - domain / topics / avg_quality / fingerprint_sample
        """
        if domain not in self.DOMAIN_KNOWLEDGE:
            raise ValueError(
                f"未知领域: {domain}，支持: {list(self.DOMAIN_KNOWLEDGE.keys())}"
            )
        dom_info = self.DOMAIN_KNOWLEDGE[domain]
        topics: List[str] = dom_info["topics"]
        depth: int = dom_info["depth"]
        quality_base: float = dom_info["quality_base"]

        speed, info = self._compute_speed()
        model_q = info["model_quality"]

        # 向量化解码：每条样本分配一个 topic + 稳定 idx
        topic_idxs = self._rng.integers(0, len(topics), size=count, dtype=np.int32)
        sample_idxs = self._rng.integers(0, 2**31, size=count, dtype=np.int32)

        texts = np.empty(count, dtype=object)
        for i in range(count):
            topic = topics[int(topic_idxs[i])]
            texts[i] = self._decode_topic(
                domain, topic, int(sample_idxs[i]), depth
            )

        scores, grades = self._score_texts(texts, quality_base, model_q)

        # 更新统计
        self.downloaded += count
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {"count": 0, "total_quality": 0.0}
        self.domain_stats[domain]["count"] += count
        self.domain_stats[domain]["total_quality"] += (
            float(scores.mean()) if len(scores) else 0.0
        )

        return {
            "domain": domain,
            "total": count,
            "speed": speed,
            "speed_info": info,
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "topics": list(topics),
            "avg_quality": float(scores.mean()) if len(scores) else 0.0,
            "fingerprint_sample": self.get_knowledge_fingerprint(domain, topics[0]),
        }

    def download_multi_domain(
        self, count: int = 100000, domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        多领域混合下载。

        Args:
            count: 总下载条数（按领域均匀分配，余数分给前几个领域）
            domains: 指定领域列表，None 表示下载全部 10 个领域

        Returns:
            dict 包含 total / domain_distribution / texts / scores / grades / speed
        """
        if domains is None:
            domains = list(self.DOMAIN_KNOWLEDGE.keys())

        n_domains = len(domains)
        per_domain = count // n_domains
        remainder = count - per_domain * n_domains

        all_texts: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_grades: List[np.ndarray] = []
        domain_distribution: Dict[str, int] = {}

        for i, domain in enumerate(domains):
            n = per_domain + (1 if i < remainder else 0)
            if n <= 0:
                continue
            result = self.download(domain, count=n)
            all_texts.append(result["texts"])
            all_scores.append(result["scores"])
            all_grades.append(result["grades"])
            domain_distribution[domain] = result["total"]

        texts = (np.concatenate(all_texts) if all_texts
                 else np.array([], dtype=object))
        scores = (np.concatenate(all_scores) if all_scores
                  else np.array([], dtype=np.float32))
        grades = (np.concatenate(all_grades) if all_grades
                  else np.array([], dtype=np.uint8))

        speed = self._compute_speed()[0]

        return {
            "total": int(len(texts)),
            "domain_distribution": domain_distribution,
            "domains": list(domains),
            "speed": speed,
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "avg_quality": float(scores.mean()) if len(scores) else 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """下载统计"""
        return {
            "total_downloaded": self.downloaded,
            "domain_stats": dict(self.domain_stats),
            "model_attached": self.model is not None,
            "engine_attached": self.engine is not None,
            "speed_info": self._compute_speed()[1],
        }
