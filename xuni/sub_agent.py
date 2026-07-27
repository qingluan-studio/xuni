"""
SubAgent —— 子代理系统

核心理念：
    把 xuni 工厂生产的"子代理"接入项目，让一个合鸣实例能"分裂"成多个
    具名的子代理，每个子代理有自己的记忆、可认领自己的模型、执行自己的任务。

    工厂产能链：
        AI_NAME_POOL → SubAgent.spawn() → 子代理（具名、可认领模型）
                                          ↓
        任务派发 → 子代理执行（recall 经验 + 合鸣生成 + memorize 结果）
                                          ↓
        经验沉淀 → 代理经验（按代理名分组，可复用）

子代理 vs 普通调用：
    普通合鸣调用：无状态、无归属、无经验累积
    子代理调用  ：有名字、有归属模型、有经验记忆、可跨任务复用

设计要点：
    1. 每个子代理从 AI_NAME_POOL 取一个独特名字（Aria/Bolt/Coda...）
    2. 子代理可跨层认领模型（layer.claim_model），形成"自己的工具箱"
    3. 子代理有独立的 MemoryBank，存储自己的经验
    4. 执行任务时先 recall 自己的经验，再生成，再 memorize
    5. 编排器（Orchestrator）负责派发任务、汇总结果、负载均衡

完全免费、纯 NumPy，是 xuni 工厂子代理产物反哺 AI 的最小可验证实验。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .layer import AI_NAME_POOL, ModelLayer, LayeredModelSystem
from .memory import MemoryBank, MemoryEntry, MemoryType
from .harmonia13 import Harmonia13Virtual
from .harmonia_memory import HarmoniaMemory, _extract_keywords, _extract_tags, _estimate_importance


# --------------------------------------------------------------------------- #
# 子代理
# --------------------------------------------------------------------------- #
@dataclass
class AgentTask:
    """一个待执行的任务。"""
    task_id: str
    prompt: str
    tags: List[str] = field(default_factory=list)
    priority: float = 0.5  # 0~1
    deadline: float = 0.0  # 0 表示无 deadline
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """子代理执行结果。"""
    task_id: str
    agent_name: str
    answer: str
    success: bool
    latency_ms: float
    experts_used: List[str]
    recalled_count: int
    memory_id: Optional[str]
    timestamp: float = field(default_factory=time.time)


class SubAgent:
    """
    单个子代理。

    每个子代理：
    - 有名字（来自 AI_NAME_POOL）
    - 有自己的 MemoryBank（经验库）
    - 可认领多个模型（跨层）
    - 执行任务时：recall → generate → memorize
    """

    def __init__(
        self,
        name: str,
        harmonia: Harmonia13Virtual,
        stm_capacity: int = 30,
        recall_top_k: int = 2,
    ):
        self.name = name
        self.harmonia = harmonia
        self.memory = HarmoniaMemory(
            harmonia=harmonia,
            stm_capacity=stm_capacity,
            recall_top_k=recall_top_k,
        )
        self.claimed_models: List[str] = []  # 跨层认领的 model_id
        self.tasks_done: int = 0
        self.tasks_failed: int = 0
        self.total_latency_ms: float = 0.0
        self.created_at: float = time.time()
        self.specialty_tags: List[str] = []  # 擅长的领域标签（动态学习）

    # ----------------------- 模型认领 ----------------------- #

    def claim_model_in_layer(self, layer: ModelLayer, model_id: Optional[str] = None) -> Optional[str]:
        """在指定层认领一个模型。"""
        if model_id is None:
            # 自动认领一个未认领的
            unclaimed = layer.get_unclaimed()
            if not unclaimed:
                return None
            model_id = unclaimed[0].model_id
        if layer.claim_model(model_id, self.name):
            self.claimed_models.append(model_id)
            return model_id
        return None

    def release_all_models(self, layer_system: LayeredModelSystem) -> int:
        """释放所有认领的模型，返回释放数量。"""
        count = 0
        for model_id in list(self.claimed_models):
            for layer in layer_system.layers.values():
                if layer.release_model(model_id):
                    count += 1
                    self.claimed_models.remove(model_id)
                    break
        return count

    # ----------------------- 任务执行 ----------------------- #

    def execute(self, task: AgentTask, **params) -> AgentResult:
        """执行一个任务：recall → generate → memorize。"""
        start = time.time()
        try:
            result = self.memory.chat(task.prompt, **params)
            answer = result["answer"]
            success = bool(answer) and "未生成内容" not in answer

            # 学习任务标签作为专长
            for tag in task.tags:
                if tag not in self.specialty_tags:
                    self.specialty_tags.append(tag)
                    if len(self.specialty_tags) > 10:
                        self.specialty_tags = self.specialty_tags[-10:]

            if success:
                self.tasks_done += 1
            else:
                self.tasks_failed += 1

            self.total_latency_ms += result["latency_ms"]

            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                answer=answer,
                success=success,
                latency_ms=result["latency_ms"],
                experts_used=result["experts_used"],
                recalled_count=len(result["recalled_memories"]),
                memory_id=result["memory_id"],
            )
        except Exception as e:
            self.tasks_failed += 1
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                answer=f"[子代理 {self.name} 执行失败] {e}",
                success=False,
                latency_ms=(time.time() - start) * 1000,
                experts_used=[],
                recalled_count=0,
                memory_id=None,
            )

    def execute_no_memory(self, task: AgentTask, **params) -> AgentResult:
        """无记忆基线执行，用于 A/B 对比。"""
        start = time.time()
        result = self.memory.chat_no_memory(task.prompt, **params)
        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            answer=result["answer"],
            success=bool(result["answer"]),
            latency_ms=result["latency_ms"],
            experts_used=result["experts_used"],
            recalled_count=0,
            memory_id=None,
        )

    # ----------------------- 经验复用 ----------------------- #

    def recall_experience(self, prompt: str, top_k: int = 3) -> List[MemoryEntry]:
        """召回本代理的相关经验。"""
        return self.memory.recall(prompt)[:top_k]

    def memorize_experience(
        self,
        content: str,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> MemoryEntry:
        """手动注入经验。"""
        return self.memory.memorize_seed(content, importance, tags)

    # ----------------------- 报告 ----------------------- #

    def report(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "claimed_models": list(self.claimed_models),
            "tasks_done": self.tasks_done,
            "tasks_failed": self.tasks_failed,
            "success_rate": round(
                self.tasks_done / max(1, self.tasks_done + self.tasks_failed), 3
            ),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "specialty_tags": self.specialty_tags,
            "memory": self.memory.report()["memory_bank"],
        }


# --------------------------------------------------------------------------- #
# 子代理编排器
# --------------------------------------------------------------------------- #
class SubAgentOrchestrator:
    """
    子代理编排器。

    职责：
    1. 从 AI_NAME_POOL 派生子代理（每个名字只能用一个）
    2. 按任务标签路由到最合适的子代理（专长匹配）
    3. 支持并行（同步模拟）/ 串行执行
    4. 汇总所有子代理的结果与统计

    用法：
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        orch.spawn("Bolt")
        result = orch.dispatch(AgentTask(task_id="t1", prompt="合鸣是什么？", tags=["harmonia"]))
        report = orch.report()
    """

    def __init__(self, harmonia: Harmonia13Virtual, layer_system: Optional[LayeredModelSystem] = None):
        self.harmonia = harmonia
        self.layer_system = layer_system
        self.agents: Dict[str, SubAgent] = {}
        self._available_names = list(AI_NAME_POOL)
        self._results: List[AgentResult] = []
        self._spawn_count = 0

    # ----------------------- 派生 / 注销 ----------------------- #

    def spawn(
        self,
        name: Optional[str] = None,
        stm_capacity: int = 30,
        recall_top_k: int = 2,
    ) -> SubAgent:
        """派生一个新子代理。"""
        if name is None:
            if not self._available_names:
                raise RuntimeError("AI_NAME_POOL 已耗尽，无法再派生子代理")
            name = self._available_names.pop(0)
        elif name in self.agents:
            raise ValueError(f"子代理 {name} 已存在")
        elif name in AI_NAME_POOL:
            self._available_names.remove(name)
        else:
            # 不在池里也允许，但记录为自定义名
            pass

        agent = SubAgent(
            name=name,
            harmonia=self.harmonia,
            stm_capacity=stm_capacity,
            recall_top_k=recall_top_k,
        )
        self.agents[name] = agent
        self._spawn_count += 1

        # 如果有分层系统，自动让新代理认领一个模型
        if self.layer_system is not None:
            for layer in self.layer_system.layers.values():
                if layer.get_unclaimed():
                    agent.claim_model_in_layer(layer)
                    break

        return agent

    def spawn_batch(self, count: int) -> List[SubAgent]:
        """批量派生子代理。"""
        return [self.spawn() for _ in range(min(count, len(self._available_names)))]

    def retire(self, name: str) -> bool:
        """注销一个子代理，释放其模型与记忆。"""
        agent = self.agents.get(name)
        if agent is None:
            return False
        if self.layer_system is not None:
            agent.release_all_models(self.layer_system)
        # 释放名字回池
        if name in AI_NAME_POOL and name not in self._available_names:
            self._available_names.append(name)
        del self.agents[name]
        return True

    # ----------------------- 任务路由 ----------------------- #

    def _route(self, task: AgentTask) -> SubAgent:
        """按专长匹配路由任务到最合适的子代理。"""
        if not self.agents:
            raise RuntimeError("没有可用子代理，请先 spawn")
        # 单代理直接返回
        if len(self.agents) == 1:
            return next(iter(self.agents.values()))

        # 按专长标签匹配打分
        scored: List[Tuple[SubAgent, float]] = []
        for agent in self.agents.values():
            score = 0.0
            for tag in task.tags:
                if tag in agent.specialty_tags:
                    score += 1.0
            # 负载均衡：任务少的代理加分
            score -= 0.3 * (agent.tasks_done + agent.tasks_failed) / 10.0
            # 优先级高的任务给成功率高的代理
            score += 0.2 * agent.report()["success_rate"]
            scored.append((agent, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def dispatch(self, task: AgentTask, **params) -> AgentResult:
        """派发任务到最合适的子代理。"""
        agent = self._route(task)
        result = agent.execute(task, **params)
        self._results.append(result)
        return result

    def dispatch_batch(self, tasks: List[AgentTask], **params) -> List[AgentResult]:
        """批量派发任务（同步串行）。"""
        return [self.dispatch(t, **params) for t in tasks]

    def broadcast(self, task: AgentTask, **params) -> Dict[str, AgentResult]:
        """广播任务到所有子代理（用于投票/对比）。"""
        results: Dict[str, AgentResult] = {}
        for name, agent in self.agents.items():
            r = agent.execute(task, **params)
            results[name] = r
            self._results.append(r)
        return results

    def vote(self, task: AgentTask, **params) -> Dict[str, Any]:
        """
        投票式执行：所有子代理都回答，取多数一致/最长的作为最终答案。

        Returns:
            dict with: final_answer, votes, all_answers, consensus
        """
        results = self.broadcast(task, **params)
        answers = {name: r.answer for name, r in results.items()}

        # 简单多数投票：按答案前 30 字分组
        prefix_groups: Dict[str, List[str]] = {}
        for name, ans in answers.items():
            key = ans[:30] if ans else ""
            prefix_groups.setdefault(key, []).append(name)

        consensus_key = max(prefix_groups.keys(), key=lambda k: len(prefix_groups[k]))
        consensus_voters = prefix_groups[consensus_key]
        consensus = len(consensus_voters) / max(1, len(answers))

        # 最终答案：取共识组里最长的
        final = max(
            (answers[n] for n in consensus_voters),
            key=len, default="",
        )

        return {
            "final_answer": final,
            "votes": {k: len(v) for k, v in prefix_groups.items()},
            "all_answers": answers,
            "consensus": round(consensus, 3),
            "voter_count": len(consensus_voters),
        }

    # ----------------------- 查询 / 报告 ----------------------- #

    def get_agent(self, name: str) -> Optional[SubAgent]:
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())

    def results(self) -> List[AgentResult]:
        return list(self._results)

    def report(self) -> Dict[str, Any]:
        return {
            "spawn_count": self._spawn_count,
            "active_agents": len(self.agents),
            "available_names": len(self._available_names),
            "total_tasks": len(self._results),
            "total_success": sum(1 for r in self._results if r.success),
            "total_failed": sum(1 for r in self._results if not r.success),
            "agents": [a.report() for a in self.agents.values()],
        }

    # ----------------------- 经验共享 ----------------------- #

    def share_experience(self, from_agent: str, to_agent: str, top_k: int = 3) -> int:
        """把一个子代理的 Top-K 经验分享给另一个子代理，返回分享条数。"""
        src = self.agents.get(from_agent)
        dst = self.agents.get(to_agent)
        if src is None or dst is None:
            return 0
        # 取 src 长期记忆 Top-K
        top_entries = src.memory.bank.ltm.get_top_k(top_k)
        for entry in top_entries:
            dst.memorize_experience(
                content=f"[来自 {from_agent}] {entry.content}",
                importance=entry.importance * 0.9,  # 转手轻微衰减
                tags=entry.tags + [f"shared:{from_agent}"],
            )
        return len(top_entries)

    def broadcast_experience(self, top_k: int = 2) -> Dict[str, int]:
        """所有子代理互相分享 Top-K 经验，返回每个代理收到的新经验数。"""
        received: Dict[str, int] = {name: 0 for name in self.agents}
        names = list(self.agents.keys())
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j:
                    continue
                n = self.share_experience(src, dst, top_k=top_k)
                received[dst] += n
        return received
