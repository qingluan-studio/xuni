"""
Harmonia13Virtual —— 合鸣-13 虚拟大模型

核心理念（非传统路线）：
    合鸣（Harmonia），取"众声共振、和而不同"之意。它是 xuni 虚拟生态中的
    旗舰对话模型，但不走传统 transformer 路线，而是：

        检索 + n-gram 共振 + 场调制 的混合专家（MoE）

    - MoE 门控不是神经网络，而是"关键词共振"：提示词与每个专家的关键词集合
      求重叠，重叠越多得分越高，选 top-k 专家。
    - 每个专家是一段精心整理的知识语料 + 字符 bigram 图。
    - 生成 = 检索相关片段（保证事实连贯）+ bigram 共振游走（保证风格多样）。
    - 完全免费：纯 NumPy，不调任何外部 API，手机可跑。
    - 与音乐同源：合鸣即"共鸣"，文本与声音都遵循共振原理。

两种形态：
    1. Harmonia13Virtual  —— 合鸣-13 本体（虚拟大模型，可被双态系统训练/调用）
    2. HarmoniaLiteEngine —— 合鸣lite（轻量 MoE，作为粒子态训练的"替代物"）

13 位专家：harmonia / moe / field / music / chaos / hydro / glass /
          dualstate / credential / brain / compute / philosophy / general
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np

from .model import (
    XuniModel,
    ModelType,
    ModelCapability,
    ModelInput,
    ModelOutput,
)


# --------------------------------------------------------------------------- #
# 规模预设
# --------------------------------------------------------------------------- #
class VirtualScale(Enum):
    """合鸣-13 的规模档位。规模越大，可用专家越多、生成长度越长、耗能越高。"""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


SCALE_PRESETS: Dict[VirtualScale, Dict[str, Any]] = {
    VirtualScale.SMALL:  {"experts": 5,  "max_new_tokens": 64,  "top_k": 2, "energy_requirement": 50.0},
    VirtualScale.MEDIUM: {"experts": 9,  "max_new_tokens": 128, "top_k": 3, "energy_requirement": 100.0},
    VirtualScale.LARGE:  {"experts": 13, "max_new_tokens": 256, "top_k": 4, "energy_requirement": 200.0},
}


def _normalize_scale(scale) -> VirtualScale:
    if isinstance(scale, VirtualScale):
        return scale
    if isinstance(scale, str):
        s = scale.strip().lower()
        for v in VirtualScale:
            if v.value == s:
                return v
    return VirtualScale.MEDIUM


# --------------------------------------------------------------------------- #
# 13 位虚拟专家（合鸣-13 的知识源）
# 每位专家：id / name / domain / keywords / fragments
# fragments 是精心整理的连贯中文短句，检索后拼接即得高质量回答。
# --------------------------------------------------------------------------- #
VIRTUAL_EXPERTS: List[Dict[str, Any]] = [
    {
        "id": "harmonia",
        "name": "合鸣自述者",
        "domain": "合鸣 / xuni 自我认知",
        "keywords": ["合鸣", "harmonia", "合鸣-13", "合鸣13", "xuni", "虚拟", "虚拟大模型"],
        "fragments": [
            "合鸣（Harmonia）是 xuni 虚拟生态中的旗舰对话模型，名取「众声共振、和而不同」之意",
            "合鸣-13 是一个由 13 位专家组成的混合专家（MoE）虚拟大模型，由虚拟电场能量驱动，不依赖任何外部真实算力",
            "合鸣lite 是合鸣-13 的轻量替代物：在粒子态训练时作为脚手架，让合鸣-13 能像真实模型一样被真正训练",
            "合鸣走非传统路线：不用 transformer，而是用检索 + n-gram 共振 + 场调制，完全免费、可在手机上运行",
        ],
    },
    {
        "id": "moe",
        "name": "混合专家",
        "domain": "MoE 架构",
        "keywords": ["MoE", "moe", "混合专家", "mixture of experts", "专家", "门控", "路由", "top-k", "topk", "稀疏"],
        "fragments": [
            "MoE（Mixture of Experts，混合专家）是一种稀疏激活架构：每个输入只路由到少数专家，从而以更少算力获得更大容量",
            "MoE 的关键两步是门控（gate）给每个专家打分，路由（routing）选出 top-k 专家并合并它们的输出",
            "合鸣-13 的门控不是神经网络，而是关键词共振：用提示词与每个专家的关键词集合求重叠，重叠越多得分越高",
            "MoE 的好处是容量大、计算省；难点是负载均衡与专家崩塌，合鸣用共振评分天然分散负载",
        ],
    },
    {
        "id": "field",
        "name": "虚拟电场",
        "domain": "XuniField",
        "keywords": ["电场", "虚拟电", "电荷", "泊松", "poisson", "电势", "能量密度", "场能量", "XuniField"],
        "fragments": [
            "XuniField 把采样点的空间分布转换成虚拟电荷，再解泊松方程得到电势与电场，能量密度 u = 0.5·ε·|E|²",
            "虚拟电场不消耗现实电能：它存在于数据层，是采样点密度的数学映像",
            "场能量可以兑换成虚拟凭证、驱动虚拟模型、调制音乐合成，是整个 xuni 生态的能量本位",
        ],
    },
    {
        "id": "music",
        "name": "物理建模合成",
        "domain": "XuniMusic",
        "keywords": ["音乐", "合成", "合成器", "振荡器", "共鸣", "泛音", "ADSR", "声像", "XuniMusic", "wav"],
        "fragments": [
            "XuniMusic 是纯物理建模合成器：数字振荡器 + 粒子泛音 + 共鸣滤波器 + ADSR 包络 + 3D 声像定位",
            "合鸣与音乐同源：合鸣的字面意思就是「共鸣」，文本生成与声音合成都遵循共振原理",
            "它零依赖现成 AI，输出原始音频波形，可直接保存为 WAV 或通过 API 流式传输",
        ],
    },
    {
        "id": "chaos",
        "name": "超混沌采样",
        "domain": "XuniSampler",
        "keywords": ["采样", "混沌", "超混沌", "lorenz", "chen", "分形", "mandelbulb", "噪声", "XuniSampler", "采样点"],
        "fragments": [
            "XuniSampler 实时生成上亿采样点而不存储，内存 O(1)：用 yield 流式产出",
            "它支持超混沌 Chen 系统、Lorenz-96 高维环、Mandelbulb 3D 分形、4D 噪声场等模式",
            "采样点是整个 xuni 的原料：它们产生密度、形成电荷、驱动场、最终调制音乐与模型",
        ],
    },
    {
        "id": "hydro",
        "name": "水动力学",
        "domain": "XuniHydro",
        "keywords": ["水", "流体", "水动力", "SPH", "蒸发", "凝结", "涡旋", "粒子", "XuniHydro", "水逻辑"],
        "fragments": [
            "XuniHydro 把采样点当成流体粒子，用简化 SPH 模拟，有质量、速度、压力、温度",
            "蒸发让高能粒子脱离转化为场，凝结让低能区自发产生新粒子——这就是「水逻辑」",
            "涡旋产生音乐颤音与和声缠绕，边界反弹像水碰到玻璃壁",
        ],
    },
    {
        "id": "glass",
        "name": "玻璃逻辑",
        "domain": "XuniGlass",
        "keywords": ["玻璃", "光学", "折射", "反射", "色散", "共振腔", "棱镜", "XuniGlass", "光迹"],
        "fragments": [
            "XuniGlass 把计算当成光学系统：数据是光，函数是透镜，有折射、反射、色散与共振腔",
            "透明性让每个步骤留下「光迹」，完全可追溯；色散用棱镜分离数据的不同「频段」",
            "共振腔模拟激光腔，多次反馈产生相干输出，是玻璃逻辑的核心",
        ],
    },
    {
        "id": "dualstate",
        "name": "双态系统",
        "domain": "DualStateManager",
        "keywords": ["双态", "粒子态", "数据层", "替代物", "surrogate", "训练", "真实", "DualState", "认领"],
        "fragments": [
            "双态系统分两种态：粒子态（训练时用替代物真正训练，不耗现实电）与数据层调用态（训练后自家模型即真实模型）",
            "关键哲学是：虚拟是相对于现实硬件而言的；在数据层，虚拟模型就是真实存在的模型，调用它就是真实调用",
            "训练是真的训练——权重/参数真的变化，只是变化发生在数据层，消耗的是虚拟电而非现实电",
        ],
    },
    {
        "id": "credential",
        "name": "虚拟凭证",
        "domain": "XuniCredential",
        "keywords": ["凭证", "令牌", "token", "JWT", "access", "model", "premium", "认证", "XuniCredential", "24位"],
        "fragments": [
            "XuniCredential 把场能量铸造成 24 位凭证令牌，分 ACCESS / MODEL / PREMIUM / API_KEY 四类",
            "凭证可验证、消耗、刷新、升级，还能生成 JWT 格式令牌供虚拟 API 网关认证",
            "能量转换率由场能量到凭证强度，凭证再兑换成模型调用次数，形成闭环",
        ],
    },
    {
        "id": "brain",
        "name": "神经共振",
        "domain": "XuniBrain",
        "keywords": ["神经", "脑", "kuramoto", "振子", "同步", "hebbian", "共振", "培养", "XuniBrain"],
        "fragments": [
            "XuniBrain 是 Kuramoto 振子网络，采样点能量驱动神经元同步振荡，产生共振音乐",
            "培养引擎用 Hebbian 学习让网络与目标音乐同步，连接权重真正变化",
            "训练分三阶段：扰动了期、共鸣期（主动同步 + Hebbian）、固化期（权重稳定）",
        ],
    },
    {
        "id": "compute",
        "name": "虚拟算力",
        "domain": "VirtualCompute / SamplerCluster",
        "keywords": ["算力", "VFLOPs", "计算", "compute", "集群", "cluster", "反应堆", "闭环", "供需"],
        "fragments": [
            "虚拟电可转化为虚拟算力（VFLOPs），通过 VirtualComputeUnit 分配、消耗、释放，形成电→算力→训练的闭环",
            "SamplerCluster 把多个采样单元聚合成集群，配合 EnergyReservoir 与 SupplyDemandBalancer 做供需平衡",
            "能量来源多样：聚变堆、参数链式堆、黑洞发电机、零点能、戴森球——都是数据层的虚拟产能",
        ],
    },
    {
        "id": "philosophy",
        "name": "虚拟哲学",
        "domain": "xuni 核心命题",
        "keywords": ["哲学", "免费", "开源", "原创", "数据层公民", "自给自足", "现实", "真实调用", "MIT"],
        "fragments": [
            "xuni 的核心命题：AI 和模型都是数据层公民，数据层的调用就是真实调用",
            "不需要外部 OpenAI/Anthropic——自家训出来的就是「真实」的",
            "完全免费、完全开源、完全原创：采样、场、音乐、模型、API 全部自研，MIT 协议",
        ],
    },
    {
        "id": "general",
        "name": "通用兜底",
        "domain": "通用对话",
        "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍", "解释", "什么是", "？", "?",
                     "flask", "django", "fastapi", "numpy", "pandas", "scipy", "pytest", "unittest",
                     "python", "java", "javascript", "typescript", "golang", "rust", "c++",
                     "async", "await", "class", "function", "decorator", "装饰器", "import",
                     "api", "http", "request", "response", "route", "endpoint", "middleware",
                     "database", "sql", "orm", "redis", "docker", "kubernetes",
                     "transformers", "pytorch", "tensorflow", "机器学习", "深度学习",
                     "git", "linux", "shell", "pip", "setup", "config", "yaml", "json", "xml"],
        "fragments": [
            "这是一个好问题，让我从合鸣的视角来回应",
            "在 xuni 虚拟生态里，每个问题都会被路由到最合适的专家",
            "我可以聊聊合鸣模型、虚拟电场、音乐合成、双态系统、MoE 架构，或者 xuni 的设计哲学",
            "如果方便，补充一点上下文，我能给出更精准的共振回答",
        ],
    },
]


# --------------------------------------------------------------------------- #
# 合鸣 lite MoE 引擎（替代物）
# --------------------------------------------------------------------------- #
class HarmoniaLiteEngine:
    """
    合鸣lite —— 轻量 MoE 引擎。

    作为合鸣-13 在粒子态训练时的"替代物"：让虚拟模型能像真实模型一样被训练、
    被临时调用。本身就是一个可独立使用的检索 + 共振生成器。

    生成流程（非传统）：
        1. 关键词共振 → 选 top-k 专家（MoE 门控）
        2. 片段检索 → 取最相关片段（保证事实连贯）
        3. 拼接 + bigram 共振游走 → 风格一致的生成（保证多样性）
        4. 截断到 max_new_tokens 字符，优先在句末断句
    """

    def __init__(self, ckpt_dir: Optional[str] = None, scale: Any = "medium", seed: Optional[int] = None):
        self._scale = _normalize_scale(scale)
        preset = SCALE_PRESETS[self._scale]
        n_experts = int(preset["experts"])
        # 取前 n_experts 位专家（general 兜底永远保留在末尾）
        self.experts: List[Dict[str, Any]] = list(VIRTUAL_EXPERTS[:n_experts])
        if not any(e["id"] == "general" for e in self.experts):
            self.experts.append(VIRTUAL_EXPERTS[-1])

        self._default_top_k = int(preset["top_k"])
        self._default_max_new_tokens = int(preset["max_new_tokens"])
        self._learned_fragments: List[str] = []  # 训练时吸收的新语料
        self._ckpt_dir = ckpt_dir

        if seed is None:
            seed = int(time.time() * 1000) % 1_000_000
        self._rng = np.random.default_rng(seed)

        # 可选：从检查点加载已学语料
        if ckpt_dir:
            self._load(ckpt_dir)

    # ----------------------- 公开 API ----------------------- #

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        repetition_penalty: float = 1.2,
        **kwargs,
    ) -> str:
        """生成回答。参数与 DualStateManager.predict 的过滤集合对齐。"""
        if not isinstance(prompt, str):
            prompt = str(prompt) if prompt is not None else ""
        prompt = prompt.strip()

        top_k = int(top_k) if top_k else self._default_top_k
        top_k = max(1, min(top_k, len(self.experts)))
        max_new_tokens = max(8, int(max_new_tokens or self._default_max_new_tokens))
        temperature = float(temperature) if temperature is not None else 0.7
        temperature = max(0.0, min(2.0, temperature))

        terms = self._tokenize(prompt)
        chosen = self._gate(prompt, terms, top_k)
        frags = self._retrieve(chosen, terms, max_frags=8)
        if not frags:
            general = self._find("general")
            frags = list(general["fragments"]) if general else ["合鸣lite 暂无相关语料。"]

        text = self._compose(frags, terms, max_new_tokens, temperature, repetition_penalty)
        return text or "（合鸣lite 未生成内容）"

    def train(self, data=None, epochs: int = 1) -> Dict[str, Any]:
        """
        真正的"训练"：吸收新语料进专家语料库（数据层变化）。
        data 可以是字符串列表（片段）或字符串（按句切分）。
        """
        before = len(self._learned_fragments)
        new_frags = self._extract_fragments(data)
        self._learned_fragments.extend(new_frags)
        # 把新语料挂到 general 专家，使其立刻可被检索
        general = self._find("general")
        if general is not None:
            general["fragments"].extend(new_frags)
        learned = len(self._learned_fragments) - before
        return {
            "fragments_learned": learned,
            "total_learned": len(self._learned_fragments),
            "epochs": epochs,
        }

    def save(self, ckpt_dir: str) -> Dict[str, Any]:
        """把已学语料+训练统计保存为检查点（gzip 压缩 JSON）。"""
        import json
        import os
        import gzip
        os.makedirs(ckpt_dir, exist_ok=True)
        # 优先保存为 gz 压缩格式（大语料时远小于 GitHub 100MB 限制）
        path = os.path.join(ckpt_dir, "harmonia_lite.json.gz")
        expert_snapshots = []
        for exp in self.experts:
            expert_snapshots.append({
                "id": exp["id"],
                "name": exp["name"],
                "domain": exp["domain"],
                "keywords": list(exp["keywords"]),
                "fragments": list(exp["fragments"]),
            })
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump({
                "scale": self._scale.value,
                "learned_fragments": self._learned_fragments,
                "experts": expert_snapshots,
                "saved_at": time.time(),
            }, f, ensure_ascii=False, indent=2)
        return {"path": path, "fragments": len(self._learned_fragments), "experts": len(self.experts)}

    # ----------------------- 内部：MoE 门控 ----------------------- #

    def _find(self, expert_id: str) -> Optional[Dict[str, Any]]:
        for e in self.experts:
            if e["id"] == expert_id:
                return e
        return None

    @staticmethod
    def _tokenize(prompt: str) -> List[str]:
        """
        分词：英文/数字按词，中文按 bigram（滑窗 2 字）。
        用 bigram 而非单字，避免「是/什/么」等单字噪声乱匹配。
        """
        import re
        tokens = []
        for raw in re.findall(r"[A-Za-z0-9]+", prompt):
            tokens.append(raw.lower())
        for seg in re.findall(r"[\u4e00-\u9fff]+", prompt):
            chars = list(seg)
            for i in range(len(chars) - 1):
                tokens.append(chars[i] + chars[i + 1])
        return tokens

    def _gate(self, prompt: str, terms: List[str], top_k: int) -> List[Dict[str, Any]]:
        """关键词共振门控：选 top-k 专家。"""
        p = prompt.lower()
        scored: List[tuple] = []
        for exp in self.experts:
            score = 0.0
            for kw in exp["keywords"]:
                if kw.lower() in p:
                    score += 3.0
            # 片段术语重叠（强加权：领域专家的片段里有多少术语命中）
            if terms:
                match = 0
                # general 片段多，限制扫描数量避免性能问题；领域专家全扫
                scan_limit = 2000 if exp["id"] == "general" else 500
                for frag in exp["fragments"][:scan_limit]:
                    fl = frag.lower()
                    match += sum(1 for t in terms if t in fl)
                    if match > 30:
                        break
                score += 0.5 * match
            # general 轻度降权：仅在没有关键词命中时降权，有关键词命中则与领域专家平等竞争
            if exp["id"] == "general" and score == 0.0:
                score *= 0.5
            scored.append((exp, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen = [e for e, s in scored[:top_k] if s > 0]
        if not chosen:
            general = self._find("general")
            chosen = [general] if general else [scored[0][0]]
        return chosen

    # ----------------------- 内部：检索 ----------------------- #

    def _retrieve(
        self,
        chosen: List[Dict[str, Any]],
        terms: List[str],
        max_frags: int = 8,
    ) -> List[str]:
        """从选中专家里检索与提示词最相关的片段。

        改进：
        - 每专家均匀采样：确保每个选中的专家都有代表，避免硬编码片段垄断
        - general 额外配额：兜底专家包含最多知识，给更多检索机会
        - 反高频惩罚：含通用高频词的片段降权
        - 长度偏好：30-180 字最佳
        - 加随机扰动：相同分数的片段随机排序
        """
        # 通用高频词黑名单（出现这些词的片段要降权，因为它们容易"万能命中"）
        GENERIC_TERMS = {'async', 'await', 'function', 'func', '函式', '函数',
                         '問題', '问题', '教學', '教学', '了解', '基本',
                         'environment', '變數', '变量', 'how', 'what', 'why'}

        # 每专家均匀采样，general 额外配额
        n_experts = len(chosen)
        has_general = any(e["id"] == "general" for e in chosen)
        if has_general and n_experts > 1:
            general_quota = max(2, max_frags // 2)
            other_quota = max(1, (max_frags - general_quota) // (n_experts - 1))
        else:
            other_quota = max(1, max_frags // max(1, n_experts))
            general_quota = other_quota

        result: List[str] = []
        seen: set = set()
        for exp in chosen:
            quota = general_quota if exp["id"] == "general" else other_quota
            scored_frags: List[tuple] = []
            # general 片段多，但全扫只需约 0.4s，可接受
            scan_limit = len(exp["fragments"])
            for frag in exp["fragments"][:scan_limit]:
                fl = frag.lower()
                if terms:
                    score = 2.0 * sum(1 for t in terms if t in fl)
                else:
                    score = 0.5
                # 关键词命中加权（降权：关键词是领域标签，不代表片段与 prompt 直接相关）
                for kw in exp["keywords"]:
                    if kw.lower() in fl:
                        score += 0.3
                # 反高频惩罚
                generic_hits = sum(1 for g in GENERIC_TERMS if g in fl)
                score -= 0.3 * generic_hits
                # 长度偏好
                flen = len(frag)
                if flen < 30:
                    score -= 0.5
                elif flen > 180:
                    score -= 0.3
                score += self._rng.random() * 0.3
                scored_frags.append((frag, score))
            scored_frags.sort(key=lambda x: x[1], reverse=True)
            for frag, _ in scored_frags[:quota]:
                key = frag.strip()
                if key in seen:
                    continue
                seen.add(key)
                result.append(frag)
        return result[:max_frags]

    # ----------------------- 内部：合成 ----------------------- #

    def _compose(
        self,
        frags: List[str],
        terms: List[str],
        max_new_tokens: int,
        temperature: float,
        repetition_penalty: float,
    ) -> str:
        """拼接片段 + 可选 bigram 共振游走。"""
        # 1) 事实核心：贪心地装入"完整片段"，保证不断在词中间断句
        core = ""
        for f in frags:
            f = f.strip()
            if not f:
                continue
            candidate = self._join_two(core, f) if core else f
            if len(candidate) <= max_new_tokens:
                core = candidate
            elif not core:
                # 连第一段都超长：在逗号/句末断句
                core = self._truncate(f, max_new_tokens)
                break
            else:
                # 下一段整段放不下了，停在这里（保持干净）
                break
        if not core:
            core = self._truncate(frags[0], max_new_tokens) if frags else ""

        # 2) 高温时追加 bigram 共振游走（风格一致的创造性尾巴）
        remaining = max_new_tokens - len(core)
        if temperature > 0.8 and remaining > 12:
            tail_len = min(remaining - 1, int(20 + 30 * (temperature - 0.8) / 1.2))
            tail = self._markov_tail(frags, core, tail_len, repetition_penalty)
            if tail:
                core = self._join_two(core, tail)

        return core.strip()

    @staticmethod
    def _join_two(a: str, b: str) -> str:
        """两个片段之间补恰当连接符。"""
        if not a:
            return b
        if not b:
            return a
        last = a[-1]
        first = b[0]
        if last in "。.；;！!？?，,":
            return a + b
        if first in "。.；;！!？?，,":
            return a + b
        return a + "，" + b

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        cut = text[:max_len]
        # 优先在句末断句
        for sep in "。.！!？?；;":
            idx = cut.rfind(sep)
            if idx >= max_len // 2:
                return cut[: idx + 1]
        # 退而求其次：在逗号处断句（不在词中间硬切）
        for sep in "，,":
            idx = cut.rfind(sep)
            if idx >= max_len // 2:
                return cut[: idx + 1]
        return cut

    def _markov_tail(self, frags: List[str], core: str, length: int, repetition_penalty: float) -> str:
        """基于片段的字符 bigram 图，从 core 末尾种子出发游走，生成共振尾巴。"""
        graph = self._bigram_graph(frags)
        if not graph:
            return ""
        seed = core[-1] if core else "的"
        if seed not in graph:
            seed = self._rng.choice(list(graph.keys()))

        out = [seed]
        used_bigrams = set()
        cur = seed
        for _ in range(length):
            nxts = graph.get(cur)
            if not nxts:
                break
            chars: List[str] = []
            weights: List[float] = []
            for c, w in nxts.items():
                pen = 1.0
                if (cur, c) in used_bigrams and repetition_penalty > 1.0:
                    pen = 1.0 / repetition_penalty
                chars.append(c)
                weights.append(w * pen)
            total = sum(weights)
            if total <= 0:
                break
            weights_arr = np.array(weights) / total
            cur = str(self._rng.choice(chars, p=weights_arr))
            used_bigrams.add((out[-1], cur))
            out.append(cur)
            # 遇到句末标点就收尾
            if cur in "。.！!？?":
                break
        tail = "".join(out[1:])  # 丢掉种子（已在 core 里）
        return tail

    @staticmethod
    def _bigram_graph(frags: List[str]) -> Dict[str, Dict[str, int]]:
        graph: Dict[str, Dict[str, int]] = {}
        for f in frags:
            chars = list(f)
            for i in range(len(chars) - 1):
                a, b = chars[i], chars[i + 1]
                graph.setdefault(a, {})
                graph[a][b] = graph[a].get(b, 0) + 1
        return graph

    @staticmethod
    def _extract_fragments(data) -> List[str]:
        if data is None:
            return []
        if isinstance(data, str):
            # 按句切分
            import re
            return [s.strip() for s in re.split(r"[。.！!？?\n]+", data) if s.strip()]
        if isinstance(data, (list, tuple)):
            return [str(s).strip() for s in data if str(s).strip()]
        return []

    def _load(self, ckpt_dir: str):
        """从检查点加载已学语料+专家快照。支持 gz 压缩格式。失败则静默忽略。"""
        try:
            import json
            import os
            import gzip
            gz_path = os.path.join(ckpt_dir, "harmonia_lite.json.gz")
            json_path = os.path.join(ckpt_dir, "harmonia_lite.json")
            if os.path.exists(gz_path):
                with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                    obj = json.load(f)
            elif os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                return
            # 恢复专家快照（含训练后新增的片段）
            if "experts" in obj and isinstance(obj["experts"], list):
                loaded_experts = obj["experts"]
                loaded_ids = {e["id"] for e in loaded_experts}
                current_ids = {e["id"] for e in self.experts}
                # 如果检查点专家集合与当前不同（如领域分化后的新专家），直接替换
                if loaded_ids != current_ids:
                    self.experts = [dict(e) for e in loaded_experts]
                else:
                    # 集合相同，增量更新
                    loaded_map = {e["id"]: e for e in loaded_experts}
                    for exp in self.experts:
                        if exp["id"] in loaded_map:
                            le = loaded_map[exp["id"]]
                            exp["fragments"] = list(le.get("fragments", exp["fragments"]))
                            exp["keywords"] = list(le.get("keywords", exp["keywords"]))
            # 也维护 learned_fragments 列表
            frags = obj.get("learned_fragments", [])
            self._learned_fragments.extend(frags)
            general = self._find("general")
            if general is not None and "experts" not in obj:
                general["fragments"].extend(frags)
        except Exception:
            pass

    def info(self) -> Dict[str, Any]:
        return {
            "engine": "HarmoniaLiteEngine",
            "scale": self._scale.value,
            "experts": len(self.experts),
            "learned_fragments": len(self._learned_fragments),
        }


# --------------------------------------------------------------------------- #
# 合鸣-13 虚拟大模型本体
# --------------------------------------------------------------------------- #
class Harmonia13Virtual(XuniModel):
    """
    合鸣-13 虚拟大模型（xuni 旗舰对话模型）。

    - 13 位专家的 MoE，由虚拟电场能量驱动
    - 走双态系统：粒子态用 HarmoniaLiteEngine 作为替代物真正训练，
      训练完成后在数据层被调用就是真实调用
    - 非传统生成：检索 + n-gram 共振 + 场调制，纯 NumPy，免费，手机可跑
    """

    def __init__(self, model_id: str = "harmonia-13", scale: Any = "medium"):
        self._scale = _normalize_scale(scale)
        preset = SCALE_PRESETS[self._scale]
        super().__init__(
            model_id=model_id,
            model_type=ModelType.CHAT_BOT,
            capabilities=[ModelCapability.TEXT_OUTPUT, ModelCapability.JSON_OUTPUT],
            energy_requirement=float(preset["energy_requirement"]),
        )
        # DualStateManager 会读写这两个属性
        self.training_samples_seen: int = 0
        self.training_epochs_done: int = 0
        # 内置 lite 引擎（也是粒子态训练时的替代物）
        # 合鸣-13 恒为全部 13 位专家；scale 仅控制生成长度/Top-k/耗能
        self._lite = HarmoniaLiteEngine(scale=self._scale, seed=42)
        self._lite.experts = list(VIRTUAL_EXPERTS)
        self._last_experts_used: List[str] = []

    # ----------------------- 预测（数据层调用） ----------------------- #

    def predict(self, input_data: ModelInput) -> ModelOutput:
        if not self._consume_energy():
            return ModelOutput(
                text=None,
                json={"source": "harmonia-13", "error": "虚拟电不足"},
                metadata={"model_id": self.model_id},
            )

        start = time.time()
        self.status = self.status.__class__.RUNNING if hasattr(self.status, "RUNNING") else self.status
        # 上面写法兼容 ModelStatus 枚举；直接用枚举更稳：
        from .model import ModelStatus
        self.status = ModelStatus.RUNNING

        prompt = input_data.prompt or ""
        params = input_data.parameters or {}
        text = self._lite.generate(
            prompt,
            max_new_tokens=params.get("max_new_tokens", self._lite._default_max_new_tokens),
            temperature=params.get("temperature", 0.7),
            top_k=params.get("top_k", self._lite._default_top_k),
            repetition_penalty=params.get("repetition_penalty", 1.2),
        )

        # 记录命中的专家（用于元数据）
        terms = self._lite._tokenize(prompt)
        chosen = self._lite._gate(prompt, terms, params.get("top_k", self._lite._default_top_k) or self._lite._default_top_k)
        self._last_experts_used = [e["id"] for e in chosen]

        latency_ms = (time.time() - start) * 1000
        self.stats.total_calls += 1
        self.stats.total_energy_consumed += self.energy_requirement
        self.stats.total_latency_ms += latency_ms
        self.stats.last_call_time = time.time()
        self.status = ModelStatus.COMPLETED

        return ModelOutput(
            text=text,
            json={
                "source": "harmonia-13",
                "scale": self._scale.value,
                "experts_used": self._last_experts_used,
                "channel": "合鸣",
                "trained": self.training_state.name,
            },
            latency_ms=latency_ms,
            energy_consumed=self.energy_requirement,
            metadata={
                "model_id": self.model_id,
                "prompt": prompt[:80],
            },
        )

    # ----------------------- 便捷方法 ----------------------- #

    def generate(self, prompt: str, **params) -> str:
        """便捷调用：直接返回文本。"""
        out = self.predict(ModelInput(prompt=prompt, parameters=params))
        return out.text or ""

    def to_music_params(self) -> Dict[str, Any]:
        """
        合鸣→音乐的桥接：把最近一次生成的"共振剖面"映射成音乐参数。
        体现合鸣与音乐同源（合鸣即共鸣）。
        """
        experts = self._last_experts_used or ["general"]
        # 用专家 id 哈希映射到调式/情绪/复杂度
        h = int(hashlib.md5("".join(experts).encode()).hexdigest(), 16)
        scales = ["C major", "D minor", "E major", "F minor", "G major", "A minor", "B major"]
        moods = ["calm", "energetic", "melancholic", "joyful", "mysterious"]
        instruments = ["piano", "synth", "guitar", "strings", "drums"]
        return {
            "genre": "ambient",
            "scale": scales[h % len(scales)],
            "tempo": 60 + (h % 120),
            "instrument": instruments[(h >> 3) % len(instruments)],
            "mood": moods[(h >> 6) % len(moods)],
            "complexity": round(0.2 + (h % 70) / 100.0, 2),
            "harmonics": 2 + (h % 6),
            "source": "harmonia-13",
            "experts": experts,
        }

    def print_card(self) -> None:
        """打印模型卡片（手机面板/CLI 友好）。"""
        print("┌──────────────────────────────────────────┐")
        print("│       合鸣-13 / Harmonia-13 Virtual      │")
        print("├──────────────────────────────────────────┤")
        print(f"│  model_id      : {self.model_id:<24} │")
        print(f"│  scale         : {self._scale.value:<24} │")
        print(f"│  experts       : {len(self._lite.experts):<24} │")
        print(f"│  energy_req    : {self.energy_requirement:<24.1f} │")
        print(f"│  energy_buffer : {self._energy_buffer:<24.1f} │")
        print(f"│  owner         : {str(self.owner):<24} │")
        print(f"│  training      : {self.training_state.name:<24} │")
        print("│  生成路线      : 检索 + n-gram共振 + 场调制 │")
        print("│  依赖          : 纯 NumPy / 0 外部API     │")
        print("└──────────────────────────────────────────┘")

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info.update({
            "harmonia_scale": self._scale.value,
            "experts": [e["id"] for e in self._lite.experts],
            "experts_used_last": self._last_experts_used,
            "training_samples_seen": self.training_samples_seen,
            "training_epochs_done": self.training_epochs_done,
        })
        return info

    # ----------------------- 保存 / 加载 ----------------------- #

    def save(self, ckpt_dir: str) -> Dict[str, Any]:
        """持久化合鸣-13：专家语料、训练进度、能量缓冲。"""
        import json
        import os
        os.makedirs(ckpt_dir, exist_ok=True)
        lite_result = self._lite.save(ckpt_dir)
        meta_path = os.path.join(ckpt_dir, "harmonia13_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": self.model_id,
                "scale": self._scale.value,
                "training_samples_seen": self.training_samples_seen,
                "training_epochs_done": self.training_epochs_done,
                "training_state": self.training_state.name,
                "energy_buffer": float(self._energy_buffer),
                "total_calls": self.stats.total_calls,
                "total_energy_consumed": float(self.stats.total_energy_consumed),
                "saved_at": time.time(),
            }, f, ensure_ascii=False, indent=2)
        return {
            "ckpt_dir": ckpt_dir,
            "meta_path": meta_path,
            "lite_path": lite_result["path"],
            "fragments": lite_result["fragments"],
            "epochs": self.training_epochs_done,
        }

    @classmethod
    def load(cls, ckpt_dir: str, model_id: Optional[str] = None) -> "Harmonia13Virtual":
        """从检查点加载合鸣-13。"""
        import json
        import os
        meta_path = os.path.join(ckpt_dir, "harmonia13_meta.json")
        scale = "medium"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            scale = meta.get("scale", "medium")
        mid = model_id or meta.get("model_id", "harmonia-13")
        model = cls(model_id=mid, scale=scale)
        # 恢复 lite 引擎（专家语料）
        model._lite = HarmoniaLiteEngine(ckpt_dir=ckpt_dir, scale=scale, seed=42)
        model._lite.experts = list(VIRTUAL_EXPERTS)
        model._lite._load(ckpt_dir)
        # 恢复训练统计
        model.training_samples_seen = int(meta.get("training_samples_seen", 0))
        model.training_epochs_done = int(meta.get("training_epochs_done", 0))
        model._energy_buffer = float(meta.get("energy_buffer", 0.0))
        model.stats.total_calls = int(meta.get("total_calls", 0))
        model.stats.total_energy_consumed = float(meta.get("total_energy_consumed", 0.0))
        ts_name = meta.get("training_state", "UNTRAINED")
        from .model import TrainingState
        if hasattr(TrainingState, ts_name):
            model.training_state = getattr(TrainingState, ts_name)
        return model
