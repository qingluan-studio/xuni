"""
Tests for HarmoniaMemory and SubAgent modules.
"""

import pytest

from xuni import (
    Harmonia13Virtual,
    HarmoniaMemory,
    SubAgent,
    SubAgentOrchestrator,
    AgentTask,
    SubstanceSystem,
)


@pytest.fixture
def harmonia():
    h = Harmonia13Virtual(scale="small")
    h.charge(5000.0)
    return h


# ═══════════════════════════════════════════════════════════════════
# HarmoniaMemory
# ═══════════════════════════════════════════════════════════════════
class TestHarmoniaMemory:
    def test_memorize_seed(self, harmonia):
        hm = HarmoniaMemory(harmonia)
        entry = hm.memorize_seed("test fact", importance=0.8, tags=["test"])
        assert entry.memory_id is not None
        assert entry.importance == 0.8
        assert "test" in entry.tags
        # importance >= 0.6 → should also be in LTM
        assert entry.memory_id in hm.bank.ltm._store

    def test_recall_returns_relevant(self, harmonia):
        hm = HarmoniaMemory(harmonia)
        hm.memorize_seed("合鸣-13 是 MoE 模型", importance=0.8, tags=["harmonia"])
        hm.memorize_seed("电场由采样点产生", importance=0.7, tags=["field"])
        recalled = hm.recall("合鸣是什么？")
        assert len(recalled) >= 1
        # 应该召回合鸣相关的记忆
        contents = [m.content for m in recalled]
        assert any("合鸣" in c for c in contents)

    def test_chat_with_memory(self, harmonia):
        hm = HarmoniaMemory(harmonia, recall_top_k=2)
        hm.memorize_seed("合鸣-13 由 13 位专家组成", 0.85, ["harmonia"])
        result = hm.chat("合鸣是什么？", max_new_tokens=50)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert result["memory_id"] is not None
        assert "harmonia" in result["experts_used"] or len(result["experts_used"]) > 0

    def test_chat_no_memory_baseline(self, harmonia):
        hm = HarmoniaMemory(harmonia)
        result = hm.chat_no_memory("合鸣是什么？", max_new_tokens=50)
        assert result["with_memory"] is False
        assert result["recalled_memories"] == []
        assert result["memory_id"] is None

    def test_auto_consolidate(self, harmonia):
        hm = HarmoniaMemory(harmonia, auto_consolidate_every=2)
        # 触发 2 次 chat 后应执行 consolidate
        hm.chat("合鸣是什么？", max_new_tokens=30)
        hm.chat("MoE 是什么？", max_new_tokens=30)
        # call_count=2, 应已触发巩固
        assert hm._call_count == 2

    def test_report(self, harmonia):
        hm = HarmoniaMemory(harmonia)
        hm.memorize_seed("test", 0.7, ["t"])
        report = hm.report()
        assert "harmonia_info" in report
        assert "memory_bank" in report
        assert report["call_count"] == 0
        assert report["recall_top_k"] == 3

    def test_forget(self, harmonia):
        hm = HarmoniaMemory(harmonia)
        # 只有 importance>=0.6 才会进入 LTM，所以手动注入低重要性到 LTM
        hm.memorize_seed("important", 0.9, ["t"])
        from xuni.memory import MemoryEntry, MemoryType, MemoryScope
        low_entry = MemoryEntry(
            memory_id="low-1",
            content="trivial",
            memory_type=MemoryType.EXPERIENCE,
            scope=MemoryScope.GLOBAL,
            importance=0.05,
            tags=["t"],
        )
        hm.bank.ltm.store(low_entry)
        forgotten = hm.forget(threshold=0.5)
        assert forgotten == 1
        # 重要的应该还在
        assert hm.bank.ltm.retrieve("low-1") is None


# ═══════════════════════════════════════════════════════════════════
# SubAgent
# ═══════════════════════════════════════════════════════════════════
class TestSubAgent:
    def test_execute_task(self, harmonia):
        agent = SubAgent(name="TestAgent", harmonia=harmonia)
        task = AgentTask(task_id="t1", prompt="合鸣是什么？", tags=["harmonia"])
        result = agent.execute(task, max_new_tokens=50)
        assert result.task_id == "t1"
        assert result.agent_name == "TestAgent"
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert result.success is True
        assert agent.tasks_done == 1

    def test_execute_no_memory(self, harmonia):
        agent = SubAgent(name="TestAgent2", harmonia=harmonia)
        task = AgentTask(task_id="t2", prompt="合鸣是什么？")
        result = agent.execute_no_memory(task, max_new_tokens=50)
        assert result.recalled_count == 0
        assert result.memory_id is None

    def test_recall_experience(self, harmonia):
        agent = SubAgent(name="TestAgent3", harmonia=harmonia)
        agent.memorize_experience("合鸣-13 是 MoE", 0.85, ["harmonia"])
        exp = agent.recall_experience("合鸣是什么？")
        assert len(exp) >= 1

    def test_specialty_learning(self, harmonia):
        agent = SubAgent(name="TestAgent4", harmonia=harmonia)
        task = AgentTask(task_id="t1", prompt="合鸣是什么？", tags=["harmonia", "moe"])
        agent.execute(task, max_new_tokens=30)
        assert "harmonia" in agent.specialty_tags
        assert "moe" in agent.specialty_tags

    def test_report(self, harmonia):
        agent = SubAgent(name="TestAgent5", harmonia=harmonia)
        agent.execute(AgentTask(task_id="t", prompt="合鸣", tags=["harmonia"]), max_new_tokens=30)
        report = agent.report()
        assert report["name"] == "TestAgent5"
        assert report["tasks_done"] == 1
        assert report["success_rate"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# SubAgentOrchestrator
# ═══════════════════════════════════════════════════════════════════
class TestSubAgentOrchestrator:
    def test_spawn_auto_name(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        agent = orch.spawn()
        assert agent.name == "Aria"  # 第一个名字
        assert "Aria" in orch.agents

    def test_spawn_named(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        agent = orch.spawn("Bolt")
        assert agent.name == "Bolt"
        # Bolt 应该从可用池中移除
        assert "Bolt" not in orch._available_names

    def test_spawn_duplicate_raises(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        with pytest.raises(ValueError):
            orch.spawn("Aria")

    def test_spawn_batch(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        agents = orch.spawn_batch(3)
        assert len(agents) == 3
        assert len(orch.agents) == 3

    def test_retire(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        assert orch.retire("Aria") is True
        assert "Aria" not in orch.agents
        # 名字应该回到池
        assert "Aria" in orch._available_names

    def test_dispatch_routes_to_agent(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        task = AgentTask(task_id="t1", prompt="合鸣", tags=["harmonia"])
        result = orch.dispatch(task, max_new_tokens=30)
        assert result.agent_name == "Aria"
        assert result.task_id == "t1"

    def test_dispatch_batch(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        orch.spawn("Bolt")
        tasks = [
            AgentTask(task_id=f"t{i}", prompt=f"问题{i}", tags=["harmonia"])
            for i in range(3)
        ]
        results = orch.dispatch_batch(tasks, max_new_tokens=30)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_broadcast(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn("Aria")
        orch.spawn("Bolt")
        task = AgentTask(task_id="b1", prompt="合鸣", tags=["harmonia"])
        results = orch.broadcast(task, max_new_tokens=30)
        assert len(results) == 2
        assert "Aria" in results
        assert "Bolt" in results

    def test_vote(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn_batch(3)
        task = AgentTask(task_id="v1", prompt="合鸣", tags=["harmonia"])
        vote = orch.vote(task, max_new_tokens=30)
        assert "final_answer" in vote
        assert "consensus" in vote
        assert "votes" in vote
        assert 0 < vote["consensus"] <= 1.0
        assert len(vote["all_answers"]) == 3

    def test_share_experience(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        a1 = orch.spawn("Aria")
        a2 = orch.spawn("Bolt")
        a1.memorize_experience("经验1", 0.85, ["harmonia"])
        # Aria 的记忆需要进入 LTM 才能 share
        a1.memory.bank.ltm.store(a1.memory.bank.stm.to_list()[0])
        shared = orch.share_experience("Aria", "Bolt", top_k=1)
        assert shared == 1

    def test_report(self, harmonia):
        orch = SubAgentOrchestrator(harmonia)
        orch.spawn_batch(2)
        orch.dispatch(AgentTask(task_id="t1", prompt="合鸣", tags=["harmonia"]), max_new_tokens=30)
        report = orch.report()
        assert report["active_agents"] == 2
        assert report["total_tasks"] == 1
        assert report["total_success"] == 1


# ═══════════════════════════════════════════════════════════════════
# Substance registry
# ═══════════════════════════════════════════════════════════════════
class TestSubstanceRegistry:
    def test_memory_point_registered(self):
        ss = SubstanceSystem()
        sub = ss.get("记忆点")
        assert sub is not None
        assert sub.name_en == "Memory Point"
        assert "MemoryBank.memorize()" in sub.production_methods

    def test_sub_agent_registered(self):
        ss = SubstanceSystem()
        sub = ss.get("子代理")
        assert sub is not None
        assert sub.name_en == "Sub Agent"
        assert "SubAgentOrchestrator.spawn()" in sub.production_methods

    def test_long_term_memory_registered(self):
        ss = SubstanceSystem()
        sub = ss.get("长期记忆")
        assert sub is not None
        assert "MemoryBank.consolidate()" in sub.production_methods

    def test_agent_experience_registered(self):
        ss = SubstanceSystem()
        sub = ss.get("代理经验")
        assert sub is not None
        assert "SubAgent.execute()" in sub.production_methods

    def test_resonance_memory_registered(self):
        ss = SubstanceSystem()
        sub = ss.get("共振记忆")
        assert sub is not None
        assert "XuniMemory.capture()" in sub.production_methods

    def test_production_chain_for_sub_agent(self):
        ss = SubstanceSystem()
        chain = ss.get_production_chain("子代理")
        assert "子代理" in chain
        assert "虚拟模型" in chain
        assert "虚拟电" in chain

    def test_production_chain_for_agent_experience(self):
        ss = SubstanceSystem()
        chain = ss.get_production_chain("代理经验")
        assert "代理经验" in chain
        assert "子代理" in chain
        assert "记忆点" in chain

    def test_total_substances_increased(self):
        ss = SubstanceSystem()
        stats = ss.statistics()
        # 原本 18，新增 5 个 → 23
        assert stats["total_substances"] >= 23
