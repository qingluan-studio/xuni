"""
PhaseSpaceModel —— 认知相空间虚拟模型

一个轻量的主题化虚拟模型，演示"通道已开启：cognitive-phase-space ↔ xuni"。
本身是 XuniModel 子类，可被双态系统训练与调用。

设计为最小可用：内置一组关于"认知相空间"的问答片段，按关键词检索回答。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .model import (
    XuniModel,
    ModelType,
    ModelCapability,
    ModelInput,
    ModelOutput,
    ModelStatus,
)


# 主题问答库（关键词, 回答）
_QA: List[Dict[str, Any]] = [
    {
        "keys": ["认知相空间", "相空间", "phase space", "是什么"],
        "answer": (
            "认知相空间是把一个认知系统的所有可能状态张成的多维空间：每个轴是一个状态变量，"
            "思维/感知/记忆的演化就是其中一条轨迹。xuni 把它作为外部认知层，通过通道接入虚拟生态。"
        ),
    },
    {
        "keys": ["通道", "channel", "开启", "连接"],
        "answer": (
            "通道已开启：cognitive-phase-space 与 xuni 虚拟生态双向相连。"
            "相空间里的轨迹可以坍缩成 xuni 的采样点，xuni 的场能量也能反哺相空间的演化。"
        ),
    },
    {
        "keys": ["轨迹", "吸引子", "演化", "动力学"],
        "answer": (
            "在相空间里，认知轨迹受吸引子牵引：稳定的吸引子对应习惯，混沌吸引子对应创造，"
            "极限环对应节律——这与 xuni 的超混沌采样天然共鸣。"
        ),
    },
    {
        "keys": ["坍缩", "collapse", "观测", "采样"],
        "answer": (
            "相空间轨迹坍缩为采样点，就是从连续认知流中取出离散的可计算样本。"
            "xuni 的采样层正是这个坍缩过程的执行者。"
        ),
    },
]


class PhaseSpaceModel(XuniModel):
    """认知相空间虚拟模型（XuniModel 子类，可走双态系统）。"""

    def __init__(self, model_id: str = "phase-space-001"):
        super().__init__(
            model_id=model_id,
            model_type=ModelType.CHAT_BOT,
            capabilities=[ModelCapability.TEXT_OUTPUT, ModelCapability.JSON_OUTPUT],
            energy_requirement=6.0,
        )
        self.training_samples_seen: int = 0
        self.training_epochs_done: int = 0
        self._qa = [dict(item) for item in _QA]

    def predict(self, input_data: ModelInput) -> ModelOutput:
        if not self._consume_energy():
            return ModelOutput(json={"source": "phase_space", "error": "虚拟电不足"})

        start = __import__("time").time()
        self.status = ModelStatus.RUNNING

        prompt = (input_data.prompt or "").lower()
        best: Optional[Dict[str, Any]] = None
        best_score = -1
        for item in self._qa:
            score = sum(1 for k in item["keys"] if k.lower() in prompt)
            if score > best_score:
                best_score = score
                best = item

        if best is None or best_score <= 0:
            text = (
                "认知相空间模型：我专注于相空间、通道、轨迹与坍缩这类话题。"
                "可以问「什么是认知相空间」或「通道是如何开启的」。"
            )
        else:
            text = best["answer"]

        latency_ms = (__import__("time").time() - start) * 1000
        self.stats.total_calls += 1
        self.stats.total_energy_consumed += self.energy_requirement
        self.stats.total_latency_ms += latency_ms
        self.stats.last_call_time = __import__("time").time()
        self.status = ModelStatus.COMPLETED

        return ModelOutput(
            text=text,
            json={
                "source": "phase_space",
                "channel": "cognitive-phase-space↔xuni",
                "matched": best is not None and best_score > 0,
            },
            latency_ms=latency_ms,
            energy_consumed=self.energy_requirement,
            metadata={"model_id": self.model_id},
        )

    def print_card(self) -> None:
        print("┌──────────────────────────────────────────┐")
        print("│   PhaseSpaceModel / 认知相空间虚拟模型    │")
        print("├──────────────────────────────────────────┤")
        print(f"│  model_id      : {self.model_id:<24} │")
        print(f"│  energy_req    : {self.energy_requirement:<24.1f} │")
        print(f"│  energy_buffer : {self._energy_buffer:<24.1f} │")
        print(f"│  owner         : {str(self.owner):<24} │")
        print(f"│  training      : {self.training_state.name:<24} │")
        print("│  channel       : cognitive-phase-space↔xuni│")
        print("└──────────────────────────────────────────┘")


def create_phase_space_model(model_id: str = "phase-space-001") -> PhaseSpaceModel:
    """工厂函数：创建一个认知相空间虚拟模型。"""
    return PhaseSpaceModel(model_id=model_id)
