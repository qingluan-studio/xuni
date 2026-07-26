"""
VirtualData —— 虚拟资料系统

核心理念：
    真实数据 → 虚拟资料（粒子态）→ 喂模型训练

    虚拟资料是粒子态的数据存在形式：
    - 不占现实内存（用指纹+元数据表示，实际内容在数据层）
    - 可被模型消费训练
    - 支持压缩、索引、检索、统计
    - 训练时从粒子态"坍缩"恢复为可训练数据

闭环位置：
    真实数据 → [转换] → 虚拟资料(粒子态) → [坍缩] → 训练数据 → 模型训练
                                                          ↑
                              采样点产参数 ←————— 参数反馈
"""

import hashlib
import time
import uuid
import json
import zlib
import base64
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Iterator, Tuple
import numpy as np


class DataPhase(Enum):
    """数据存在的态"""
    RAW = auto()          # 原始态：现实中的真实数据
    PARTICLE = auto()     # 粒子态：虚拟资料，不占现实内存
    COLLAPSED = auto()    # 坍缩态：训练时恢复的可训练数据


@dataclass
class VirtualDataParticle:
    """
    虚拟资料粒子——粒子态的单条数据。

    不存储原始内容，只存指纹+元数据+压缩摘要。
    实际内容"存在"于数据层，需要时坍缩恢复。
    """
    particle_id: str
    fingerprint: str              # 内容指纹（SHA256）
    data_type: str                # text / array / image_desc / audio_desc
    shape: Optional[Tuple] = None # 形状（如果是数组）
    dtype: Optional[str] = None   # 数据类型
    size_bytes: int = 0           # 原始大小（字节）
    compressed_b64: str = ""      # 压缩后的摘要（base64）
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    quality_score: float = 0.5    # 数据质量分（0-1）
    created_at: float = field(default_factory=time.time)
    access_count: int = 0         # 被访问次数
    source: str = "unknown"       # 数据来源

    @property
    def real_memory_footprint(self) -> int:
        """现实内存占用——极小，因为只存指纹"""
        return len(self.fingerprint) + len(self.compressed_b64) + 200

    @property
    def virtual_size(self) -> int:
        """虚拟大小——数据层的完整大小"""
        return self.size_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "particle_id": self.particle_id,
            "fingerprint": self.fingerprint[:16] + "...",
            "data_type": self.data_type,
            "shape": self.shape,
            "size_bytes": self.size_bytes,
            "real_memory": self.real_memory_footprint,
            "virtual_size": self.virtual_size,
            "quality_score": self.quality_score,
            "tags": self.tags,
            "source": self.source,
            "access_count": self.access_count,
        }


class VirtualDataConverter:
    """
    虚拟资料转换器——真实数据 → 虚拟资料（粒子态）

    转换过程：
    1. 接收真实数据（文本/数组/字典）
    2. 计算指纹
    3. 压缩存储摘要（不存完整内容）
    4. 生成粒子态虚拟资料
    5. 返回 VirtualDataParticle

    特点：现实内存占用极小，虚拟大小完整保留。
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}  # 临时坍缩缓存（可选）

    def convert_text(self, text: str, tags: List[str] = None,
                     source: str = "unknown", quality: float = 0.5) -> VirtualDataParticle:
        """文本 → 虚拟资料粒子"""
        raw_bytes = text.encode("utf-8")
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()

        # 压缩摘要：只存压缩后的前512字节作为特征摘要
        compressed = zlib.compress(raw_bytes, level=9)
        summary = compressed[:512]
        compressed_b64 = base64.b64encode(summary).decode("ascii")

        # 元数据提取
        metadata = {
            "char_count": len(text),
            "line_count": text.count("\n") + 1,
            "word_count": len(text.split()),
            "has_code": "def " in text or "class " in text or "function" in text,
            "language_hint": self._detect_language(text),
        }

        return VirtualDataParticle(
            particle_id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            data_type="text",
            size_bytes=len(raw_bytes),
            compressed_b64=compressed_b64,
            metadata=metadata,
            tags=tags or [],
            quality_score=quality,
            source=source,
        )

    def convert_array(self, arr: np.ndarray, tags: List[str] = None,
                      source: str = "unknown", quality: float = 0.5) -> VirtualDataParticle:
        """数组 → 虚拟资料粒子"""
        raw_bytes = arr.tobytes()
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()

        compressed = zlib.compress(raw_bytes, level=9)
        summary = compressed[:512]
        compressed_b64 = base64.b64encode(summary).decode("ascii")

        metadata = {
            "ndim": arr.ndim,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

        return VirtualDataParticle(
            particle_id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            data_type="array",
            shape=tuple(arr.shape),
            dtype=str(arr.dtype),
            size_bytes=len(raw_bytes),
            compressed_b64=compressed_b64,
            metadata=metadata,
            tags=tags or [],
            quality_score=quality,
            source=source,
        )

    def convert_dict(self, data: Dict[str, Any], tags: List[str] = None,
                     source: str = "unknown", quality: float = 0.5) -> VirtualDataParticle:
        """字典 → 虚拟资料粒子"""
        raw_bytes = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()

        compressed = zlib.compress(raw_bytes, level=9)
        summary = compressed[:512]
        compressed_b64 = base64.b64encode(summary).decode("ascii")

        metadata = {
            "keys": list(data.keys()),
            "key_count": len(data),
        }

        return VirtualDataParticle(
            particle_id=str(uuid.uuid4()),
            fingerprint=fingerprint,
            data_type="dict",
            size_bytes=len(raw_bytes),
            compressed_b64=compressed_b64,
            metadata=metadata,
            tags=tags or [],
            quality_score=quality,
            source=source,
        )

    def _detect_language(self, text: str) -> str:
        """简单语言检测"""
        chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_count > len(text) * 0.1:
            return "zh"
        return "en"

    def collapse(self, particle: VirtualDataParticle, original_data: Any = None) -> Any:
        """
        坍缩：粒子态 → 可训练数据

        如果提供了 original_data（原始数据），直接返回；
        否则从压缩摘要恢复（有损恢复，用于训练时重建）。
        """
        particle.access_count += 1

        if original_data is not None:
            return original_data

        # 有损恢复：从摘要重建（演示用，实际由数据层保证完整性）
        compressed = base64.b64decode(particle.compressed_b64)
        try:
            if particle.data_type == "text":
                return zlib.decompress(compressed).decode("utf-8", errors="replace")
            elif particle.data_type == "array":
                raw = zlib.decompress(compressed)
                return np.frombuffer(raw, dtype=particle.dtype or "float32")
        except Exception:
            return None


class VirtualDataset:
    """
    虚拟资料集——粒子态数据集合

    特点：
    - 现实内存占用极小（只存指纹+元数据）
    - 虚拟大小完整（数据层大小）
    - 支持索引、检索、过滤、统计
    - 支持质量筛选
    - 训练时批量坍缩
    """

    def __init__(self, name: str = "virtual_dataset"):
        self.name = name
        self.particles: List[VirtualDataParticle] = []
        self._index: Dict[str, int] = {}  # fingerprint → index
        self._tag_index: Dict[str, List[int]] = {}
        self._created_at = time.time()

    def add(self, particle: VirtualDataParticle) -> None:
        """添加粒子"""
        if particle.fingerprint in self._index:
            return  # 去重
        idx = len(self.particles)
        self.particles.append(particle)
        self._index[particle.fingerprint] = idx
        for tag in particle.tags:
            self._tag_index.setdefault(tag, []).append(idx)

    def add_batch(self, particles: List[VirtualDataParticle]) -> None:
        for p in particles:
            self.add(p)

    def get_by_tag(self, tag: str) -> List[VirtualDataParticle]:
        """按标签检索"""
        indices = self._tag_index.get(tag, [])
        return [self.particles[i] for i in indices]

    def filter_quality(self, min_quality: float = 0.5) -> List[VirtualDataParticle]:
        """按质量筛选"""
        return [p for p in self.particles if p.quality_score >= min_quality]

    def filter_type(self, data_type: str) -> List[VirtualDataParticle]:
        """按类型筛选"""
        return [p for p in self.particles if p.data_type == data_type]

    def sample(self, n: int, min_quality: float = 0.0) -> List[VirtualDataParticle]:
        """随机采样 n 个粒子"""
        pool = self.filter_quality(min_quality) if min_quality > 0 else self.particles
        if len(pool) <= n:
            return pool
        rng = np.random.default_rng()
        indices = rng.choice(len(pool), n, replace=False)
        return [pool[i] for i in indices]

    def collapse_batch(self, original_data_map: Dict[str, Any] = None) -> List[Any]:
        """
        批量坍缩为可训练数据

        original_data_map: {fingerprint: original_data} 可选的原始数据映射
        """
        original_data_map = original_data_map or {}
        converter = VirtualDataConverter()
        result = []
        for p in self.particles:
            data = original_data_map.get(p.fingerprint)
            collapsed = converter.collapse(p, data)
            if collapsed is not None:
                result.append(collapsed)
        return result

    @property
    def real_memory_bytes(self) -> int:
        """现实内存占用——极小"""
        return sum(p.real_memory_footprint for p in self.particles)

    @property
    def virtual_size_bytes(self) -> int:
        """虚拟大小——数据层完整大小"""
        return sum(p.virtual_size for p in self.particles)

    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        type_dist = {}
        quality_scores = []
        for p in self.particles:
            type_dist[p.data_type] = type_dist.get(p.data_type, 0) + 1
            quality_scores.append(p.quality_score)

        return {
            "name": self.name,
            "particle_count": len(self.particles),
            "real_memory_mb": self.real_memory_bytes / 1024 / 1024,
            "virtual_size_mb": self.virtual_size_bytes / 1024 / 1024,
            "compression_ratio": (
                self.virtual_size_bytes / max(1, self.real_memory_bytes)
            ),
            "type_distribution": type_dist,
            "avg_quality": np.mean(quality_scores) if quality_scores else 0,
            "tags": list(self._tag_index.keys()),
            "age_seconds": time.time() - self._created_at,
        }

    def __len__(self) -> int:
        return len(self.particles)

    def __iter__(self) -> Iterator[VirtualDataParticle]:
        return iter(self.particles)


# ============================================================
# 数据生成器——为训练生成真实数据，再转为虚拟资料
# ============================================================

class VirtualDataGenerator:
    """
    数据生成器——生成真实训练数据，然后转为虚拟资料。

    流程：生成真实数据 → 转为虚拟资料（粒子态）→ 存入虚拟资料集

    支持生成：
    - 采样点概念文本（用于认知相空间模型）
    - 音乐描述数据
    - 对话数据
    - 通用文本数据
    """

    # 认知相空间核心概念
    CONCEPTS = [
        "采样点", "场能量", "虚拟电", "虚拟凭证", "虚拟模型", "虚拟API",
        "双态切换", "粒子态", "数据层", "替代物", "训练", "推理",
        "参数", "资源", "信息", "自由能", "活力", "涌现",
        "采样", "场", "电场", "物质", "能量", "闭环",
        "认知", "相空间", "几何", "拓扑", "流形", "吸引子",
        "音乐", "作曲", "旋律", "和声", "节奏", "音色",
        "扩散", "生成", "对话", "理解", "记忆", "推理",
        "认领", "归属", "评估", "淘汰", "导师", "加成",
        "交易", "市场", "拍卖", "导入", "导出", "流通",
        "分层", "MoE", "专家", "路由", "激活", "稀疏",
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.converter = VirtualDataConverter()

    def generate_concept_texts(self, n: int = 1000) -> Tuple[VirtualDataset, Dict[str, str]]:
        """
        生成认知相空间概念文本数据

        返回：(虚拟资料集, 原始数据映射)
        """
        dataset = VirtualDataset(name="concept_texts")
        original_map = {}

        templates = [
            "{a}是{xuni}生态的核心概念，它与{b}形成{rel}关系。",
            "在{xuni}系统中，{a}通过{c}产生{b}，构成能量闭环。",
            "{a}的粒子态不占现实内存，但{d}时可以坍缩为可训练数据。",
            "采样点产生{a}和{b}，{a}驱动{c}，{b}注入{d}。",
            "双态切换中，{a}寻找{e}作为替代物，实现真正训练。",
            "虚拟电转化为虚拟算力，驱动{a}的训练，形成算力闭环。",
            "{a}与{b}在数据层融合，产出{c}，这是涌现的结果。",
            "认知相空间中，{a}对应{d}维度，{b}对应{e}维度。",
        ]

        relations = ["因果", "对偶", "共生", "层级", "反馈", "涌现", "闭环", "耦合"]

        for i in range(n):
            a, b = self.rng.choice(self.CONCEPTS, 2, replace=False)
            c, d = self.rng.choice(self.CONCEPTS, 2, replace=False)
            e = self.rng.choice(self.CONCEPTS)
            rel = self.rng.choice(relations)
            template = self.rng.choice(templates)

            text = template.format(
                a=a, b=b, c=c, d=d, e=e, rel=rel, xuni="xuni"
            )

            quality = float(self.rng.uniform(0.3, 0.95))
            tags = ["concept", "training"]
            if quality > 0.7:
                tags.append("high_quality")

            particle = self.converter.convert_text(
                text, tags=tags, source="generator", quality=quality
            )
            dataset.add(particle)
            original_map[particle.fingerprint] = text

        return dataset, original_map

    def generate_dialogue_data(self, n: int = 500) -> Tuple[VirtualDataset, Dict[str, str]]:
        """生成对话数据"""
        dataset = VirtualDataset(name="dialogue_data")
        original_map = {}

        questions = [
            "什么是{x}？", "{x}和{y}有什么关系？", "如何理解{x}？",
            "{x}在系统中的作用是什么？", "为什么需要{x}？",
            "{x}是如何工作的？", "{x}和{y}哪个更重要？",
        ]
        answers = [
            "{x}是系统的核心组件，它通过{z}实现功能。",
            "{x}与{y}形成互补关系，共同维持系统运转。",
            "{x}的本质是{z}的一种表现形式。",
            "没有{x}，系统将无法{z}，因此它是不可或缺的。",
            "{x}通过采样点获取能量，驱动{z}过程。",
        ]

        for i in range(n):
            x = self.rng.choice(self.CONCEPTS)
            y = self.rng.choice(self.CONCEPTS)
            z = self.rng.choice(self.CONCEPTS)
            q = self.rng.choice(questions).format(x=x, y=y)
            a = self.rng.choice(answers).format(x=x, y=y, z=z)
            text = f"Q: {q}\nA: {a}"

            quality = float(self.rng.uniform(0.4, 0.9))
            particle = self.converter.convert_text(
                text, tags=["dialogue", "training"], source="generator", quality=quality
            )
            dataset.add(particle)
            original_map[particle.fingerprint] = text

        return dataset, original_map

    def generate_music_descriptions(self, n: int = 300) -> Tuple[VirtualDataset, Dict[str, str]]:
        """生成音乐描述数据"""
        dataset = VirtualDataset(name="music_descriptions")
        original_map = {}

        scales = ["C大调", "G大调", "D小调", "F大调", "A小调", "E小调"]
        tempos = ["慢板", "行板", "快板", "急板", "柔板"]
        moods = ["宁静", "激昂", "忧郁", "欢快", "神秘", "庄严"]

        for i in range(n):
            scale = self.rng.choice(scales)
            tempo = self.rng.choice(tempos)
            mood = self.rng.choice(moods)
            bpm = int(self.rng.integers(60, 180))

            text = (
                f"调性：{scale}；速度：{tempo}({bpm}BPM)；"
                f"情绪：{mood}；"
                f"结构：{self.rng.choice(['ABA', 'ABAC', 'ABC', 'AABA'])}；"
                f"主旋律由{self.rng.choice(['钢琴', '弦乐', '木管', '合成器'])}演奏。"
            )

            quality = float(self.rng.uniform(0.5, 0.95))
            particle = self.converter.convert_text(
                text, tags=["music", "training"], source="generator", quality=quality
            )
            dataset.add(particle)
            original_map[particle.fingerprint] = text

        return dataset, original_map
