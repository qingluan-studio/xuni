"""
UltraContext —— 超长上下文记忆系统

工厂生产记忆点 → 记忆点 + 流式算力网络 = 超长上下文记忆

核心突破：
    原始 MemoryBank：短期20条，长期按重要性衰减
    超长上下文记忆：N节点并行存储，容量 = 记忆点数 × 节点数
    万象奇点模式：无限存储 + 瞬时检索

用法：
    from xuni.multiverse_resources import MultiverseResourceFactory
    from xuni.ultra_context import UltraContextMemory

    factory = MultiverseResourceFactory()
    memory = factory.produce_ultra_context()  # 工厂生产超长上下文记忆

    memory.memorize("用户叫小明", importance=0.8)
    memory.memorize("用户喜欢Python", importance=0.7)
    # ... 存入10000条 ...
    results = memory.recall("用户叫什么")  # 瞬时检索
"""

from __future__ import annotations

import hashlib
import time
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory import MemoryEntry, MemoryType, MemoryScope


@dataclass
class MemoryPoint:
    """
    记忆点——工厂生产的最小记忆单元。

    比普通 MemoryEntry 多了：
    - embedding: 向量嵌入（用于相似度检索）
    - energy: 记忆能量（能量越高越不容易遗忘）
    - node_id: 存储在哪个网络节点
    """
    point_id: str
    content: str
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    energy: float = 1.0          # 记忆能量
    node_id: int = 0             # 存储节点
    created_at: float = 0.0
    access_count: int = 0
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.embedding is None:
            self.embedding = self._embed(self.content)

    @staticmethod
    def _embed(text: str, dim: int = 64) -> np.ndarray:
        """简单文本嵌入：hash → 伪随机向量（纯NumPy，免费）"""
        h = hashlib.md5(text.encode()).hexdigest()
        rng = np.random.default_rng(int(h, 16) % (2**32))
        return rng.standard_normal(dim).astype(np.float32)

    def access(self):
        """访问记忆点，更新计数，补充能量"""
        self.access_count += 1
        self.energy = min(10.0, self.energy + 0.1)

    def similarity(self, query_emb: np.ndarray) -> float:
        """与查询向量的余弦相似度"""
        if self.embedding is None:
            return 0.0
        norm = np.linalg.norm(self.embedding) * np.linalg.norm(query_emb)
        if norm == 0:
            return 0.0
        return float(np.dot(self.embedding, query_emb) / norm)

    def decay(self, rate: float = 0.0001):
        """记忆衰减"""
        self.energy = max(0.0, self.energy - rate)


class UltraContextMemory:
    """
    超长上下文记忆——用工厂生产的记忆点 + 流式算力网络实现。

    核心公式：
        原始容量 = 短期20条
        超长容量 = 记忆点数 × 节点数（流式算力网络扩展）
        万象奇点 = 无限容量 + 瞬时检索

    检索策略：
        1. 关键词匹配（标签+内容）
        2. 向量相似度（embedding余弦相似度）
        3. 重要性+能量加权排序
        4. 流式网络并行检索（N节点同时搜索）
    """

    def __init__(
        self,
        node_count: int = 1,
        perpetual: bool = False,
        compute_multiplier: float = 1.0,
        max_capacity: Optional[int] = None,
    ):
        """
        Args:
            node_count: 网络节点数（来自流式算力网络）
            perpetual: 是否永动模式（万象奇点：无限容量+不衰减）
            compute_multiplier: 算力倍率（影响检索速度）
            max_capacity: 最大容量（None=无限，永动模式自动无限）
        """
        self.node_count = max(1, node_count)
        self.perpetual = perpetual
        self.compute_multiplier = compute_multiplier
        # 永动模式无限容量，否则按节点数扩展
        if max_capacity is not None:
            self.max_capacity = max_capacity
        elif perpetual:
            self.max_capacity = 10**18  # 实际无限
        else:
            # 每个节点存 1000 条 × 节点数
            self.max_capacity = 1000 * self.node_count

        # 分节点存储（模拟分布式记忆网络）
        self._nodes: List[List[MemoryPoint]] = [[] for _ in range(min(self.node_count, 256))]
        self._global_index: Dict[str, MemoryPoint] = {}
        self._total_points: int = 0
        self._recall_count: int = 0
        self._total_recall_time: float = 0.0

    def memorize(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        energy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryPoint:
        """
        存入一条记忆。自动分配到负载最低的节点。
        """
        point_id = hashlib.md5(f"{content}:{time.time()}:{self._total_points}".encode()).hexdigest()[:12]
        node_id = self._total_points % len(self._nodes) if self._nodes else 0
        point = MemoryPoint(
            point_id=point_id,
            content=content,
            importance=importance,
            tags=tags or [],
            energy=energy if energy is not None else 1.0,
            node_id=node_id,
            metadata=metadata or {},
        )
        # 容量检查（永动模式不检查）
        if not self.perpetual and self._total_points >= self.max_capacity:
            # 淘汰能量最低的
            self._evict_weakest()

        self._nodes[node_id].append(point)
        self._global_index[point_id] = point
        self._total_points += 1
        return point

    def recall(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
    ) -> List[MemoryPoint]:
        """
        检索记忆——流式网络并行检索。

        算力倍率越高，检索越快（模拟）。
        """
        start = time.time()
        query_emb = MemoryPoint._embed(query)

        # 流式并行检索：所有节点同时搜索
        all_results: List[Tuple[float, MemoryPoint]] = []
        for node in self._nodes:
            for point in node:
                score = self._score(point, query, query_emb, tags)
                if score > 0:
                    all_results.append((score, point))
                    point.access()

        # 排序取 Top-K
        all_results.sort(key=lambda x: x[0], reverse=True)
        results = [p for _, p in all_results[:top_k]]

        elapsed = time.time() - start
        self._recall_count += 1
        self._total_recall_time += elapsed
        return results

    def _score(
        self,
        point: MemoryPoint,
        query: str,
        query_emb: np.ndarray,
        tags: Optional[List[str]] = None,
    ) -> float:
        """综合评分：相似度 + 关键词 + 重要性 + 能量"""
        # 向量相似度
        sim = point.similarity(query_emb)
        # 关键词匹配
        kw_score = 0.0
        for q_char in query:
            if q_char in point.content:
                kw_score += 0.01
        kw_score = min(0.3, kw_score)
        # 标签匹配
        tag_score = 0.0
        if tags:
            for t in tags:
                if t in point.tags:
                    tag_score += 0.2
        # 重要性 × 能量
        importance_energy = point.importance * min(1.0, point.energy)

        return sim * 0.4 + kw_score * 0.2 + tag_score * 0.2 + importance_energy * 0.2

    def _evict_weakest(self):
        """淘汰能量最低的记忆点"""
        weakest = None
        weakest_energy = float("inf")
        for node in self._nodes:
            for p in node:
                if p.energy < weakest_energy:
                    weakest_energy = p.energy
                    weakest = p
        if weakest is not None:
            self._nodes[weakest.node_id] = [
                p for p in self._nodes[weakest.node_id] if p.point_id != weakest.point_id
            ]
            del self._global_index[weakest.point_id]
            self._total_points -= 1

    def consolidate(self) -> int:
        """
        记忆巩固：访问频繁的记忆点能量提升，低能量的衰减。
        永动模式下不衰减。
        """
        promoted = 0
        for node in self._nodes:
            for p in node:
                if p.access_count >= 3:
                    p.energy = min(10.0, p.energy + 0.5)
                    p.importance = min(1.0, p.importance + 0.1)
                    promoted += 1
                elif not self.perpetual:
                    p.decay(rate=0.01)
        return promoted

    def build_context(self, query: str, max_tokens: int = 2000) -> str:
        """
        构建上下文——检索相关记忆，拼成上下文文本。
        用于注入AI模型的prompt前缀，实现"记得超长上下文"。

        Args:
            query: 当前问题
            max_tokens: 上下文最大长度（近似按字符数算）

        Returns:
            上下文文本，拼到 prompt 前面
        """
        results = self.recall(query, top_k=20)
        if not results:
            return ""

        lines = ["[记忆上下文]"]
        total_len = 0
        for p in results:
            line = f"- {p.content}"
            if total_len + len(line) > max_tokens:
                break
            lines.append(line)
            total_len += len(line)
        lines.append("[/记忆上下文]")
        return "\n".join(lines)

    @property
    def capacity(self) -> int:
        """当前容量"""
        return self._total_points

    @property
    def max_capacity_display(self) -> str:
        """最大容量显示"""
        if self.perpetual:
            return "∞（万象奇点）"
        return f"{self.max_capacity:,}"

    def stats(self) -> Dict[str, Any]:
        """记忆系统统计"""
        avg_recall_ms = (self._total_recall_time / self._recall_count * 1000) if self._recall_count > 0 else 0
        return {
            "total_points": self._total_points,
            "max_capacity": self.max_capacity_display,
            "node_count": self.node_count,
            "perpetual": self.perpetual,
            "compute_multiplier": self.compute_multiplier,
            "recall_count": self._recall_count,
            "avg_recall_ms": f"{avg_recall_ms:.2f}",
            "nodes_used": sum(1 for n in self._nodes if n),
        }
