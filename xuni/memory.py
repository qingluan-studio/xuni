"""
XuniMemory —— 共振记忆系统

将 XuniBrain 的稳定共振模式存储为"记忆"，支持：
- 保存：捕获网络的共振快照
- 回忆：加载快照并自由演化
- 融合：多个记忆的加权叠加
- 梦境：记忆的随机组合与变形，产生超现实音乐

新增（从 CEE 提取）：
- 分层记忆银行：短期工作记忆 + 长期重要性记忆
- 记忆条目：带重要性评分、标签、时间戳的语义记忆

记忆不是静态数据，而是动态 attractor——
加载后，网络会收敛到该 attractor 附近，但不会完全复制，
每次回忆都是独特的。
"""

import numpy as np
import time
import hashlib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from .brain import XuniBrain


# ═══════════════════════════════════════════════════════════════════
# 分层记忆银行（从 CEE Memory Bank 提取适配）
# ═══════════════════════════════════════════════════════════════════

class MemoryScope(Enum):
    """记忆范围"""
    SESSION = "session"
    GLOBAL = "global"


class MemoryType(Enum):
    """记忆类型"""
    RESONANCE = "resonance"
    AUDIO = "audio"
    EXPERIENCE = "experience"
    PATTERN = "pattern"


@dataclass
class MemoryEntry:
    """语义记忆条目"""
    memory_id: str
    content: str
    memory_type: MemoryType = MemoryType.RESONANCE
    scope: MemoryScope = MemoryScope.GLOBAL
    importance: float = 0.5  # 0~1
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """访问记忆，更新计数"""
        self.access_count += 1
        self.last_accessed = time.time()

    def decay_importance(self, decay_rate: float = 0.001) -> float:
        """时间衰减重要性"""
        elapsed = time.time() - self.timestamp
        self.importance *= np.exp(-decay_rate * elapsed)
        return self.importance


class ShortTermMemory:
    """
    短期工作记忆 — 固定容量环形缓冲区，聚焦最近上下文。
    从 CEE 的 ShortTermMemory 提取核心思想。
    """

    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._buffer: list[MemoryEntry] = []
        self._index: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._buffer)

    def store(self, entry: MemoryEntry) -> None:
        if entry.memory_id in self._index:
            idx = self._index[entry.memory_id]
            self._buffer[idx] = entry
        else:
            if len(self._buffer) >= self._capacity:
                oldest = self._buffer.pop(0)
                self._index.pop(oldest.memory_id, None)
                # 重建索引
                self._index = {e.memory_id: i for i, e in enumerate(self._buffer)}
            self._buffer.append(entry)
            self._index[entry.memory_id] = len(self._buffer) - 1

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        if memory_id in self._index:
            entry = self._buffer[self._index[memory_id]]
            entry.touch()
            return entry
        return None

    def retrieve_by_tag(self, tag: str) -> list[MemoryEntry]:
        return [e for e in self._buffer if tag in e.tags]

    def clear(self) -> None:
        self._buffer.clear()
        self._index.clear()

    def to_list(self) -> list[MemoryEntry]:
        return list(self._buffer)


class LongTermMemory:
    """
    长期记忆 — 基于重要性和标签的持久化记忆。
    从 CEE 的 LongTermMemory 提取核心思想，简化实现。
    """

    def __init__(self, decay_rate: float = 0.0001):
        self._store: dict[str, MemoryEntry] = {}
        self._tag_index: dict[str, set[str]] = {}
        self.decay_rate = decay_rate

    def store(self, entry: MemoryEntry) -> None:
        self._store[entry.memory_id] = entry
        for tag in entry.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(entry.memory_id)

    def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        entry = self._store.get(memory_id)
        if entry:
            entry.touch()
        return entry

    def search_by_tag(self, tag: str, min_importance: float = 0.0) -> list[MemoryEntry]:
        ids = self._tag_index.get(tag, set())
        results = []
        for mid in ids:
            entry = self._store[mid]
            entry.decay_importance(self.decay_rate)
            if entry.importance >= min_importance:
                results.append(entry)
        # 按重要性排序
        results.sort(key=lambda e: e.importance, reverse=True)
        return results

    def search_by_content(self, keyword: str) -> list[MemoryEntry]:
        results = []
        for entry in self._store.values():
            if keyword.lower() in entry.content.lower():
                results.append(entry)
        return results

    def get_top_k(self, k: int = 10) -> list[MemoryEntry]:
        entries = list(self._store.values())
        for e in entries:
            e.decay_importance(self.decay_rate)
        entries.sort(key=lambda e: e.importance * (1 + e.access_count * 0.1), reverse=True)
        return entries[:k]

    def forget_below(self, threshold: float = 0.1) -> int:
        """遗忘重要性低于阈值的记忆，返回遗忘数量"""
        to_remove = [mid for mid, e in self._store.items() if e.importance < threshold]
        for mid in to_remove:
            entry = self._store.pop(mid)
            for tag in entry.tags:
                self._tag_index[tag].discard(mid)
        return len(to_remove)

    def to_list(self) -> list[MemoryEntry]:
        return list(self._store.values())


class MemoryBank:
    """
    统一记忆银行 — 短期 + 长期一体化。
    """

    def __init__(
        self,
        stm_capacity: int = 20,
        decay_rate: float = 0.0001,
    ):
        self.stm = ShortTermMemory(capacity=stm_capacity)
        self.ltm = LongTermMemory(decay_rate=decay_rate)

    def memorize(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.RESONANCE,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """记忆新内容：先存入短期，重要内容晋升长期"""
        memory_id = hashlib.md5(f"{content}:{time.time()}".encode()).hexdigest()[:12]
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
        )
        self.stm.store(entry)
        if importance >= 0.6:
            self.ltm.store(entry)
        return entry

    def recall(self, memory_id: str) -> Optional[MemoryEntry]:
        """先查短期，再查长期"""
        entry = self.stm.retrieve(memory_id)
        if entry is None:
            entry = self.ltm.retrieve(memory_id)
        return entry

    def search(self, tag: Optional[str] = None, keyword: Optional[str] = None) -> list[MemoryEntry]:
        """搜索记忆"""
        results = []
        if tag:
            results.extend(self.stm.retrieve_by_tag(tag))
            results.extend(self.ltm.search_by_tag(tag))
        if keyword:
            results.extend(self.ltm.search_by_content(keyword))
        # 去重
        seen = set()
        unique = []
        for e in results:
            if e.memory_id not in seen:
                seen.add(e.memory_id)
                unique.append(e)
        return unique

    def consolidate(self) -> int:
        """
        记忆巩固：将短期记忆中访问频繁的内容晋升到长期记忆。
        返回晋升数量。
        """
        promoted = 0
        for entry in self.stm.to_list():
            if entry.access_count >= 3 and entry.memory_id not in self.ltm._store:
                entry.importance = min(1.0, entry.importance + 0.2)
                self.ltm.store(entry)
                promoted += 1
        return promoted

    def report(self) -> dict:
        """记忆银行状态报告"""
        return {
            "stm_size": self.stm.size,
            "ltm_size": len(self.ltm._store),
            "ltm_top3": [
                {"id": e.memory_id, "importance": round(e.importance, 3), "content": e.content[:50]}
                for e in self.ltm.get_top_k(3)
            ],
        }


@dataclass
class ResonanceMemory:
    """单个共振记忆"""
    name: str
    phi: np.ndarray
    freq: np.ndarray
    amp: np.ndarray
    W: np.ndarray
    W_structural: np.ndarray
    tags: List[str]
    audio: Optional[np.ndarray] = None
    created_at: int = 0  # step count
    evocations: int = 0

    def to_dict(self) -> dict:
        """序列化为字典"""
        result = {
            "name": self.name,
            "phi": self.phi.tolist(),
            "freq": self.freq.tolist(),
            "amp": self.amp.tolist(),
            "W": self.W.tolist(),
            "W_structural": self.W_structural.tolist(),
            "tags": self.tags,
            "created_at": self.created_at,
            "evocations": self.evocations,
        }
        if self.audio is not None:
            result["audio"] = self.audio.tolist()
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "ResonanceMemory":
        return cls(
            name=d["name"],
            phi=np.array(d["phi"]),
            freq=np.array(d["freq"]),
            amp=np.array(d["amp"]),
            W=np.array(d["W"]),
            W_structural=np.array(d["W_structural"]),
            audio=np.array(d["audio"]) if "audio" in d else None,
            tags=d.get("tags", []),
            created_at=d.get("created_at", 0),
            evocations=d.get("evocations", 0),
        )


class XuniMemory:
    """
    共振记忆库。
    """

    def __init__(self, brain: XuniBrain):
        self.brain = brain
        self.memories: List[ResonanceMemory] = []
        self._step_counter = 0

    def capture(self, name: str, audio: Optional[np.ndarray] = None, tags: Optional[List[str]] = None) -> ResonanceMemory:
        """
        捕获当前网络状态为一个记忆。
        """
        state = self.brain.get_state()
        mem = ResonanceMemory(
            name=name,
            phi=state["phi"],
            freq=state["freq"],
            amp=state["amp"],
            W=state["W"],
            W_structural=state["W_structural"],
            tags=tags or [],
            audio=audio,
            created_at=self._step_counter,
        )
        self.memories.append(mem)
        self._step_counter += 1
        return mem

    def recall(self, name: str, duration: float = 5.0, perturbation: float = 0.02) -> np.ndarray:
        """
        回忆一个记忆：加载其状态并自由演化。

        Args:
            name: 记忆名称
            duration: 回忆时长
            perturbation: 扰动强度（0=完美复制，越大越变异）
        """
        mem = self._find_memory(name)
        if mem is None:
            raise ValueError(f"Memory '{name}' not found")

        mem.evocations += 1

        # 加载状态
        self.brain.set_state({
            "phi": mem.phi.copy(),
            "freq": mem.freq.copy(),
            "amp": mem.amp.copy(),
            "W": mem.W.copy(),
            "W_structural": mem.W_structural.copy(),
        })

        # 施加扰动——记忆不是完美的
        rng = np.random.default_rng()
        self.brain.phi += rng.normal(0, perturbation, self.brain.n)
        self.brain.freq *= (1 + rng.normal(0, perturbation * 0.5, self.brain.n))
        self.brain.freq = np.clip(self.brain.freq, 20, 8000)

        return self.brain.stimulate(duration=duration)

    def dream(
        self,
        duration: float = 10.0,
        n_memories: int = 3,
        morph_speed: float = 0.5,
    ) -> np.ndarray:
        """
        梦境：随机选择多个记忆，缓慢变形过渡，产生超现实音乐。

        Args:
            duration: 梦境时长
            n_memories: 参与梦境的记忆数
            morph_speed: 变形速度（0~1，越大切换越快）
        """
        if len(self.memories) < 2:
            raise ValueError("Need at least 2 memories to dream")

        samples = int(self.brain.sr * duration)
        output = np.zeros(samples)

        # 随机选择记忆序列
        rng = np.random.default_rng()
        mem_seq = rng.choice(self.memories, size=n_memories, replace=False)

        # 分段变形
        segment_samples = samples // n_memories
        for seg_idx, mem in enumerate(mem_seq):
            start = seg_idx * segment_samples
            end = start + segment_samples if seg_idx < n_memories - 1 else samples

            # 加载记忆并添加随机性
            self.brain.set_state({
                "phi": mem.phi.copy(),
                "freq": mem.freq.copy(),
                "amp": mem.amp.copy(),
                "W": mem.W.copy(),
                "W_structural": mem.W_structural.copy(),
            })
            self.brain.phi += rng.normal(0, 0.1, self.brain.n)

            # 生成这段
            for i in range(start, end):
                self.brain._dynamics_step(target_signal=None, field_energy=0.5)
                output[i] = np.sum(self.brain.amp * np.sin(self.brain.phi))

                # 在段内缓慢变形频率
                if rng.random() < morph_speed * 0.01:
                    self.brain.freq *= (1 + rng.normal(0, 0.02, self.brain.n))
                    self.brain.freq = np.clip(self.brain.freq, 20, 8000)

        max_val = np.max(np.abs(output))
        if max_val > 0:
            output /= max_val
        return output

    def fuse(self, names: List[str], weights: Optional[List[float]] = None) -> dict:
        """
        融合多个记忆为一个新状态（不是生成音频，而是产生混合状态）。
        """
        mems = [self._find_memory(n) for n in names]
        if any(m is None for m in mems):
            missing = [n for n, m in zip(names, mems) if m is None]
            raise ValueError(f"Memories not found: {missing}")

        if weights is None:
            weights = [1.0 / len(mems)] * len(mems)
        weights = np.array(weights)
        weights /= weights.sum()

        fused_phi = np.zeros(self.brain.n)
        fused_freq = np.zeros(self.brain.n)
        fused_amp = np.zeros(self.brain.n)
        fused_W = np.zeros((self.brain.n, self.brain.n))

        for mem, w in zip(mems, weights):
            fused_phi += w * mem.phi
            fused_freq += w * mem.freq
            fused_amp += w * mem.amp
            fused_W += w * mem.W

        # 相位需要特殊处理：加权平均后归一化
        fused_phi = np.mod(fused_phi, 2.0 * np.pi)

        return {
            "phi": fused_phi,
            "freq": fused_freq,
            "amp": fused_amp,
            "W": fused_W,
            "W_structural": fused_W.copy(),
        }

    def list_memories(self) -> List[Dict]:
        """列出所有记忆"""
        return [
            {
                "name": m.name,
                "tags": m.tags,
                "created_at": m.created_at,
                "evocations": m.evocations,
            }
            for m in self.memories
        ]

    def save_to_file(self, path: str):
        """保存记忆库到 JSON"""
        data = {
            "memories": [m.to_dict() for m in self.memories],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, path: str):
        """从 JSON 加载记忆库"""
        with open(path, "r") as f:
            data = json.load(f)
        self.memories = [ResonanceMemory.from_dict(m) for m in data["memories"]]

    def _find_memory(self, name: str) -> Optional[ResonanceMemory]:
        for m in self.memories:
            if m.name == name:
                return m
        return None
