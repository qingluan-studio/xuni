"""
HarmoniaMemory —— 记忆增强合鸣模型

核心理念：
    把 xuni 工厂生产的"记忆点"接入合鸣-13，让合鸣从「无状态生成」
    升级为「有状态、可累积经验」的对话模型。

    工厂产能链：
        采样点 → 场能量 → 合鸣生成 → 记忆点（存储） → 长期记忆（晋升）
                                       ↓
                                  下次生成时召回 → 注入上下文 → 提升回答质量

增强点：
    1. 生成前 recall：按 prompt 关键词/标签从 MemoryBank 召回 Top-K 记忆
    2. 上下文注入：把召回的记忆作为前缀拼到 prompt 前，让合鸣"想起"相关事实
    3. 生成后 memorize：把 (prompt, answer) 作为新记忆点存入，重要性由命中度评估
    4. 周期性 consolidate：把短期记忆中访问频繁的晋升为长期记忆
    5. 质量评估：对比有/无记忆时的回答重叠度、事实命中率

完全免费、纯 NumPy，是 xuni 工厂产物反哺合鸣模型的最小可验证实验。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .memory import (
    MemoryBank,
    MemoryEntry,
    MemoryType,
    MemoryScope,
)
from .harmonia13 import Harmonia13Virtual, HarmoniaLiteEngine


# --------------------------------------------------------------------------- #
# 重要性评估器：根据回答长度、关键词命中、专家多样性打分
# --------------------------------------------------------------------------- #
def _estimate_importance(prompt: str, answer: str, experts_used: List[str]) -> float:
    """评估一次对话的重要性，用于决定是否晋升为长期记忆。"""
    score = 0.3  # 基础分
    # 回答越长越重要（信息量大）
    score += min(0.3, len(answer) / 400.0)
    # 命中专家越多越重要（跨领域）
    score += min(0.2, len(experts_used) * 0.05)
    # prompt 包含问号/是什么/为什么 → 事实性问题，更重要
    factual_signals = ["?", "？", "是什么", "为什么", "怎么", "如何", "什么是"]
    if any(s in prompt for s in factual_signals):
        score += 0.15
    # 回答包含具体术语 → 专业知识
    if any(kw in answer for kw in ["MoE", "共振", "虚拟", "采样", "xuni", "合鸣"]):
        score += 0.1
    return min(1.0, score)


def _extract_tags(prompt: str, experts_used: List[str]) -> List[str]:
    """从 prompt 和命中专家中提取标签。"""
    tags = list(experts_used)
    # 简单关键词提取
    for kw in ["合鸣", "harmonia", "记忆", "memory", "子代理", "agent",
               "MoE", "电场", "field", "音乐", "music", "采样", "sampler"]:
        if kw.lower() in prompt.lower():
            tags.append(kw)
    return list(dict.fromkeys(tags))  # 去重保序


def _extract_keywords(prompt: str) -> List[str]:
    """从 prompt 提取检索关键词（与 HarmoniaLiteEngine._tokenize 对齐）。"""
    import re
    kws: List[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", prompt):
        if len(raw) >= 2:
            kws.append(raw.lower())
    for seg in re.findall(r"[\u4e00-\u9fff]+", prompt):
        chars = list(seg)
        for i in range(len(chars) - 1):
            kws.append(chars[i] + chars[i + 1])
    return kws


# --------------------------------------------------------------------------- #
# 记忆增强合鸣
# --------------------------------------------------------------------------- #
class HarmoniaMemory:
    """
    记忆增强合鸣模型。

    用法：
        hm = HarmoniaMemory(Harmonia13Virtual(scale="medium"))
        hm.memorize_seed("合鸣-13 是 13 位专家的 MoE", tags=["harmonia", "moe"])
        answer = hm.chat("合鸣是什么？")
        # 下次再问相关问题，合鸣会"想起"之前的对话
        answer2 = hm.chat("它有多少位专家？")

    增强原理：
        chat(prompt)
          ├── recall(prompt) → Top-K 相关记忆 → 拼成上下文前缀
          ├── harmonia.generate(context + prompt) → 回答
          └── memorize(prompt, answer) → 存入 MemoryBank，按重要性打分
    """

    def __init__(
        self,
        harmonia: Harmonia13Virtual,
        stm_capacity: int = 50,
        recall_top_k: int = 3,
        context_max_chars: int = 200,
        auto_consolidate_every: int = 5,
    ):
        self.harmonia = harmonia
        self.bank = MemoryBank(stm_capacity=stm_capacity)
        self.recall_top_k = recall_top_k
        self.context_max_chars = context_max_chars
        self.auto_consolidate_every = auto_consolidate_every

        self._call_count = 0
        self._last_recalled: List[MemoryEntry] = []
        self._last_context_prefix: str = ""

    # ----------------------- 记忆操作 ----------------------- #

    def memorize_seed(
        self,
        content: str,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """手动注入种子记忆（用于预置事实）。"""
        return self.bank.memorize(
            content=content,
            memory_type=MemoryType.EXPERIENCE,
            importance=importance,
            tags=tags or [],
        )

    def recall(self, prompt: str) -> List[MemoryEntry]:
        """按 prompt 关键词召回相关记忆。"""
        kws = _extract_keywords(prompt)
        results: List[MemoryEntry] = []

        # 关键词检索
        for kw in kws:
            results.extend(self.bank.search(keyword=kw))

        # 标签检索（用 prompt 中可能出现的专家标签）
        for tag in ["harmonia", "moe", "field", "music", "memory", "agent",
                    "合鸣", "电场", "音乐", "记忆", "子代理"]:
            if tag.lower() in prompt.lower():
                results.extend(self.bank.search(tag=tag))

        # 去重 + 按重要性排序
        seen: set = set()
        unique: List[MemoryEntry] = []
        for e in results:
            if e.memory_id not in seen:
                seen.add(e.memory_id)
                unique.append(e)
        unique.sort(key=lambda e: e.importance, reverse=True)
        return unique[: self.recall_top_k]

    def _build_context_prefix(self, memories: List[MemoryEntry]) -> str:
        """把召回的记忆拼成上下文前缀。"""
        if not memories:
            return ""
        parts: List[str] = []
        total = 0
        for m in memories:
            snippet = m.content[:120]
            parts.append(f"[记忆:{snippet}]")
            total += len(parts[-1])
            if total >= self.context_max_chars:
                break
        prefix = " ".join(parts) + " "
        return prefix[: self.context_max_chars]

    # ----------------------- 主对话 API ----------------------- #

    def chat(self, prompt: str, **params) -> Dict[str, Any]:
        """
        记忆增强对话。

        Returns:
            dict with: answer, recalled_memories, importance, experts_used,
            memory_id (新记忆), with_memory (是否真的注入了记忆)
        """
        start = time.time()
        self._call_count += 1

        # 1. 召回
        recalled = self.recall(prompt)
        context_prefix = self._build_context_prefix(recalled)
        self._last_recalled = recalled
        self._last_context_prefix = context_prefix

        # 2. 拼上下文 + 生成
        full_prompt = context_prefix + prompt if context_prefix else prompt
        answer = self.harmonia.generate(full_prompt, **params)

        # 3. 记录命中的专家
        experts_used = list(self.harmonia._last_experts_used)

        # 4. 评估重要性 + 提取标签
        importance = _estimate_importance(prompt, answer, experts_used)
        tags = _extract_tags(prompt, experts_used)

        # 5. 存为新记忆点
        memory_content = f"Q: {prompt}\nA: {answer}"
        new_entry = self.bank.memorize(
            content=memory_content,
            memory_type=MemoryType.EXPERIENCE,
            importance=importance,
            tags=tags,
            metadata={
                "experts_used": experts_used,
                "recall_count": len(recalled),
                "timestamp": time.time(),
            },
        )

        # 6. 周期性巩固
        if self._call_count % self.auto_consolidate_every == 0:
            promoted = self.bank.consolidate()
        else:
            promoted = 0

        latency_ms = (time.time() - start) * 1000
        return {
            "answer": answer,
            "recalled_memories": [
                {"id": m.memory_id, "importance": round(m.importance, 3),
                 "content": m.content[:80], "tags": m.tags}
                for m in recalled
            ],
            "importance": round(importance, 3),
            "experts_used": experts_used,
            "memory_id": new_entry.memory_id,
            "with_memory": bool(context_prefix),
            "consolidated": promoted,
            "latency_ms": round(latency_ms, 2),
        }

    # ----------------------- 评估对比 ----------------------- #

    def chat_no_memory(self, prompt: str, **params) -> Dict[str, Any]:
        """无记忆基线：直接调用合鸣，不召回、不存储。用于 A/B 对比。"""
        start = time.time()
        answer = self.harmonia.generate(prompt, **params)
        experts_used = list(self.harmonia._last_experts_used)
        latency_ms = (time.time() - start) * 1000
        return {
            "answer": answer,
            "recalled_memories": [],
            "importance": 0.0,
            "experts_used": experts_used,
            "memory_id": None,
            "with_memory": False,
            "consolidated": 0,
            "latency_ms": round(latency_ms, 2),
        }

    # ----------------------- 状态报告 ----------------------- #

    def report(self) -> Dict[str, Any]:
        """记忆增强合鸣的状态报告。"""
        bank_report = self.bank.report()
        return {
            "harmonia_info": self.harmonia.get_info(),
            "memory_bank": bank_report,
            "call_count": self._call_count,
            "recall_top_k": self.recall_top_k,
            "last_recalled_count": len(self._last_recalled),
            "last_context_prefix": self._last_context_prefix,
        }

    def forget(self, threshold: float = 0.1) -> int:
        """遗忘重要性低于阈值的长期记忆，返回遗忘数量。"""
        return self.bank.ltm.forget_below(threshold)
