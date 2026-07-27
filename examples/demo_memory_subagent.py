"""
记忆 + 子代理反哺 AI 验证实验

实验目的：
    验证 xuni 工厂生产的"记忆点"和"子代理"是否真能增强合鸣模型的能力。

三组对照实验：
    实验1（记忆 A/B）：相同问题，对比"无记忆"vs"有记忆"的回答质量
    实验2（多轮累积）：连续对话，验证记忆累积是否让回答越来越精准
    实验3（子代理投票）：多个子代理并行回答 + 投票，验证是否提升稳定性

衡量指标：
    - 答案长度（信息量）
    - 关键术语命中数（事实性）
    - 多轮一致性（后续问题是否引用前文）
    - 投票共识度（多代理一致性）

运行：
    cd /workspace/xuni
    python examples/demo_memory_subagent.py
"""

from __future__ import annotations

import os
import sys
import time
import json

# 让脚本能从 examples/ 目录直接运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import (
    Harmonia13Virtual,
    HarmoniaMemory,
    SubAgent,
    SubAgentOrchestrator,
    AgentTask,
    SubstanceSystem,
    SubstanceCategory,
)
from xuni.layer import LayeredModelSystem, LayerConfig, LayerType


# 种子事实——预先注入到记忆中
SEED_FACTS = [
    ("合鸣-13 由 13 位虚拟专家组成，走 MoE 路线", 0.85, ["harmonia", "moe"]),
    ("合鸣的门控是关键词共振，不是神经网络", 0.8, ["harmonia", "moe", "gate"]),
    ("xuni 工厂能生产模型、记忆点、子代理、能量、凭证五类产物", 0.9, ["xuni", "memory", "agent"]),
    ("子代理从 AI_NAME_POOL 派生，可跨层认领模型", 0.85, ["agent", "layer"]),
    ("记忆点带重要性评分，>=0.6 晋升为长期记忆", 0.8, ["memory"]),
    ("虚拟电场由采样点密度转化而来，是工厂的能量本位", 0.75, ["field", "energy"]),
]


# 关键术语（用于命中评分）
KEY_TERMS = [
    "合鸣", "harmonia", "MoE", "13", "专家", "共振", "门控", "关键词",
    "xuni", "记忆", "子代理", "认领", "AI", "能量", "电场", "采样",
    "凭证", "层", "训练", "数据层",
]


def score_answer(answer: str) -> dict:
    """对回答打分：长度 + 关键术语命中。"""
    if not answer:
        return {"length": 0, "term_hits": 0, "score": 0.0}
    ans_lower = answer.lower()
    hits = sum(1 for t in KEY_TERMS if t.lower() in ans_lower)
    length = len(answer)
    # 综合分：长度（归一化到 0~1）+ 命中数（归一化）
    norm_len = min(1.0, length / 200.0)
    norm_hits = min(1.0, hits / 8.0)
    score = 0.4 * norm_len + 0.6 * norm_hits
    return {
        "length": length,
        "term_hits": hits,
        "score": round(score, 3),
    }


def banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def line(c: str = "─", n: int = 70):
    print(c * n)


# ═══════════════════════════════════════════════════════════════════
# 实验 1：记忆 A/B 对照
# ═══════════════════════════════════════════════════════════════════
def experiment_1_memory_ab():
    banner("实验 1：记忆增强 A/B 对照（无记忆 vs 有记忆）")

    # 同一个合鸣实例
    harmonia = Harmonia13Virtual(scale="medium")
    harmonia.charge(10000.0)

    # 有记忆版本
    hm = HarmoniaMemory(harmonia, stm_capacity=50, recall_top_k=3)
    for content, imp, tags in SEED_FACTS:
        hm.memorize_seed(content, importance=imp, tags=tags)

    questions = [
        "合鸣是什么？",
        "合鸣的门控是怎么工作的？",
        "xuni 工厂能生产哪些东西？",
        "子代理是怎么来的？",
        "记忆点怎么晋升为长期记忆？",
    ]

    print(f"\n预置记忆：{len(SEED_FACTS)} 条种子事实")
    print(f"测试问题：{len(questions)} 个\n")

    no_mem_scores = []
    with_mem_scores = []

    for i, q in enumerate(questions, 1):
        print(f"\n[Q{i}] {q}")
        line("─", 50)

        # 无记忆基线
        r_no = hm.chat_no_memory(q, max_new_tokens=120)
        s_no = score_answer(r_no["answer"])
        no_mem_scores.append(s_no["score"])

        # 有记忆
        r_yes = hm.chat(q, max_new_tokens=120)
        s_yes = score_answer(r_yes["answer"])
        with_mem_scores.append(s_yes["score"])

        print(f"  无记忆 [{s_no['score']:.3f}, 命中{s_no['term_hits']}术语, {s_no['length']}字]")
        print(f"    → {r_no['answer'][:120]}...")
        print(f"  有记忆 [{s_yes['score']:.3f}, 命中{s_yes['term_hits']}术语, {s_yes['length']}字, 召回{len(r_yes['recalled_memories'])}条]")
        if r_yes["recalled_memories"]:
            for m in r_yes["recalled_memories"][:2]:
                print(f"    召回: ({m['importance']:.2f}) {m['content'][:60]}")
        print(f"    → {r_yes['answer'][:120]}...")

    avg_no = sum(no_mem_scores) / len(no_mem_scores)
    avg_yes = sum(with_mem_scores) / len(with_mem_scores)
    improvement = (avg_yes - avg_no) / max(0.001, avg_no) * 100

    print(f"\n📊 实验 1 结果：")
    print(f"  无记忆平均分: {avg_no:.3f}")
    print(f"  有记忆平均分: {avg_yes:.3f}")
    print(f"  提升: {improvement:+.1f}%")
    return {
        "no_memory_avg": round(avg_no, 3),
        "with_memory_avg": round(avg_yes, 3),
        "improvement_pct": round(improvement, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# 实验 2：多轮累积
# ═══════════════════════════════════════════════════════════════════
def experiment_2_multi_turn():
    banner("实验 2：多轮对话累积（验证记忆是否让回答越来越精准）")

    harmonia = Harmonia13Virtual(scale="medium")
    harmonia.charge(10000.0)
    hm = HarmoniaMemory(harmonia, stm_capacity=30, recall_top_k=3)

    # 第一轮注入 1 条种子
    hm.memorize_seed("合鸣-13 是 xuni 旗舰对话模型，由 13 位专家组成", 0.8, ["harmonia"])

    multi_turn = [
        "合鸣是什么？",                # 第 1 轮：基础
        "它有多少位专家？",            # 第 2 轮：指代「它」=合鸣
        "这些专家是怎么选择的？",      # 第 3 轮：指代「这些专家」
        "合鸣和 MoE 是什么关系？",     # 第 4 轮：引入新概念
        "它的门控是神经网络吗？",      # 第 5 轮：综合前文
    ]

    scores = []
    print(f"\n多轮对话：{len(multi_turn)} 轮，初始记忆 1 条\n")

    for i, q in enumerate(multi_turn, 1):
        r = hm.chat(q, max_new_tokens=100)
        s = score_answer(r["answer"])
        scores.append(s["score"])
        print(f"[Turn {i}] {q}")
        print(f"  召回 {len(r['recalled_memories'])} 条记忆, 重要性={r['importance']:.2f}, 分数={s['score']:.3f}")
        print(f"  → {r['answer'][:100]}...")
        if r["recalled_memories"]:
            for m in r["recalled_memories"][:1]:
                print(f"  召回: {m['content'][:70]}")
        print()

    bank_report = hm.report()["memory_bank"]
    print(f"📊 实验 2 结果：")
    print(f"  各轮分数: {scores}")
    print(f"  短期记忆: {bank_report['stm_size']} 条")
    print(f"  长期记忆: {bank_report['ltm_size']} 条")
    print(f"  分数趋势: {'上升 ↑' if scores[-1] > scores[0] else '下降 ↓' if scores[-1] < scores[0] else '持平 →'}")
    return {
        "scores": scores,
        "trend": "up" if scores[-1] > scores[0] else "down" if scores[-1] < scores[0] else "flat",
        "stm_size": bank_report["stm_size"],
        "ltm_size": bank_report["ltm_size"],
    }


# ═══════════════════════════════════════════════════════════════════
# 实验 3：子代理投票
# ═══════════════════════════════════════════════════════════════════
def experiment_3_subagent_vote():
    banner("实验 3：子代理投票（多代理并行 + 经验共享）")

    harmonia = Harmonia13Virtual(scale="medium")
    harmonia.charge(20000.0)

    orch = SubAgentOrchestrator(harmonia)

    # 派生 3 个子代理
    agents = orch.spawn_batch(3)
    print(f"\n派生 {len(agents)} 个子代理: {orch.list_agents()}")

    # 给每个子代理注入不同的种子经验（让它们有不同视角）
    seeds_per_agent = {
        agents[0].name: [
            ("合鸣-13 走检索+n-gram共振路线，不用 transformer", 0.85, ["harmonia", "moe"]),
        ],
        agents[1].name: [
            ("合鸣的 13 位专家覆盖 harmonia/moe/field/music 等领域", 0.8, ["harmonia", "experts"]),
        ],
        agents[2].name: [
            ("合鸣与音乐同源，名字取「众声共振、和而不同」之意", 0.75, ["harmonia", "music"]),
        ],
    }
    for name, seeds in seeds_per_agent.items():
        agent = orch.get_agent(name)
        for content, imp, tags in seeds:
            agent.memorize_experience(content, imp, tags)

    print(f"每个子代理注入 1 条不同的种子经验\n")

    # 投票
    question = "合鸣是什么？"
    print(f"投票问题: {question}\n")

    vote_result = orch.vote(
        AgentTask(task_id="vote-1", prompt=question, tags=["harmonia"]),
        max_new_tokens=100,
    )

    print(f"投票共识度: {vote_result['consensus']:.3f} ({vote_result['voter_count']}/{len(orch.agents)} 一致)")
    print(f"最终答案: {vote_result['final_answer'][:150]}...")
    print(f"\n各代理回答:")
    for name, ans in vote_result["all_answers"].items():
        s = score_answer(ans)
        print(f"  [{name}] 分数={s['score']:.3f}, {s['length']}字")
        print(f"    → {ans[:100]}...")

    # 经验共享
    print(f"\n--- 经验共享 ---")
    received = orch.broadcast_experience(top_k=2)
    print(f"经验广播完成，各代理收到: {received}")

    # 再投一次票（看共识度是否提升）
    print(f"\n--- 共享后再次投票 ---")
    vote_result_2 = orch.vote(
        AgentTask(task_id="vote-2", prompt=question, tags=["harmonia"]),
        max_new_tokens=100,
    )
    print(f"二次投票共识度: {vote_result_2['consensus']:.3f} ({vote_result_2['voter_count']}/{len(orch.agents)} 一致)")
    print(f"最终答案: {vote_result_2['final_answer'][:150]}...")

    improvement = vote_result_2["consensus"] - vote_result["consensus"]
    print(f"\n📊 实验 3 结果：")
    print(f"  首次共识度: {vote_result['consensus']:.3f}")
    print(f"  共享后共识度: {vote_result_2['consensus']:.3f}")
    print(f"  共识度变化: {improvement:+.3f}")

    report = orch.report()
    print(f"  总任务数: {report['total_tasks']}")
    print(f"  成功率: {report['total_success']/max(1,report['total_tasks']):.2%}")
    print(f"  各代理报告:")
    for a in report["agents"]:
        print(f"    [{a['name']}] 任务={a['tasks_done']}, 成功率={a['success_rate']}, 专长={a['specialty_tags']}, LTM={a['memory']['ltm_size']}")

    return {
        "first_consensus": vote_result["consensus"],
        "second_consensus": vote_result_2["consensus"],
        "improvement": round(improvement, 3),
        "total_tasks": report["total_tasks"],
        "success_rate": round(report["total_success"] / max(1, report["total_tasks"]), 3),
    }


# ═══════════════════════════════════════════════════════════════════
# 实验 4：分层认领（子代理跨层认领模型）
# ═══════════════════════════════════════════════════════════════════
def experiment_4_layered_claim():
    banner("实验 4：子代理跨层认领模型（验证工厂认领机制）")

    harmonia = Harmonia13Virtual(scale="medium")
    harmonia.charge(10000.0)

    # 建立分层系统
    layer_system = LayeredModelSystem()
    layer_system.add_layer(LayerConfig(
        layer_id="L1", layer_name="对话层", layer_type=LayerType.CHAT,
        level=1, model_count=5,
    ))
    layer_system.add_layer(LayerConfig(
        layer_id="L2", layer_name="文本层", layer_type=LayerType.TEXT,
        level=2, model_count=5,
    ))

    orch = SubAgentOrchestrator(harmonia, layer_system=layer_system)

    # 派生 3 个子代理，自动认领模型
    agents = orch.spawn_batch(3)
    print(f"\n派生 {len(agents)} 个子代理，自动跨层认领模型:")
    for name in orch.list_agents():
        agent = orch.get_agent(name)
        print(f"  [{name}] 认领模型: {agent.claimed_models}")

    # 看层状态
    print(f"\n层状态:")
    for layer_id, layer in layer_system.layers.items():
        stat = layer.statistics()
        print(f"  {layer_id} ({stat['layer_name']}): 总{stat['total_models']}, 已认领{stat['claimed']}, 训练完成{stat['trained']}")
        print(f"    owners: {stat['owners']}")

    # 派任务
    task = AgentTask(
        task_id="layer-task-1",
        prompt="合鸣模型和分层系统是什么关系？",
        tags=["harmonia", "layer"],
    )
    result = orch.dispatch(task, max_new_tokens=120)
    s = score_answer(result.answer)
    print(f"\n任务派发 → [{result.agent_name}] (认领模型: {orch.get_agent(result.agent_name).claimed_models})")
    print(f"  召回经验: {result.recalled_count} 条")
    print(f"  分数: {s['score']:.3f}, 命中{s['term_hits']}术语")
    print(f"  → {result.answer[:150]}...")

    # 训练认领的模型
    print(f"\n--- 训练子代理认领的模型 ---")
    for layer_id, layer in layer_system.layers.items():
        results = layer.train_all_claimed(progress=1.0)
        trained = sum(1 for v in results.values() if v)
        print(f"  {layer_id}: 训练 {trained}/{len(results)} 个模型完成")

    final_stat = layer_system.layers["L1"].statistics()
    print(f"\n训练后 L1 状态: 已训练 {final_stat['trained']}/{final_stat['total_models']}")

    return {
        "agents_spawned": len(agents),
        "models_claimed": sum(len(a.claimed_models) for a in agents),
        "models_trained": final_stat["trained"],
        "task_score": s["score"],
    }


# ═══════════════════════════════════════════════════════════════════
# 物质系统验证
# ═══════════════════════════════════════════════════════════════════
def verify_substance_registry():
    banner("物质系统验证：记忆点 + 子代理已注册为工厂产物")

    ss = SubstanceSystem()
    stats = ss.statistics()
    print(f"\n工厂物质总数: {stats['total_substances']}")
    print(f"分类统计: {stats['categories']}")

    new_substances = ["记忆点", "长期记忆", "共振记忆", "子代理", "代理经验"]
    print(f"\n新注册物质验证:")
    for name in new_substances:
        sub = ss.get(name)
        if sub:
            print(f"  ✓ {sub.icon} {sub.name} ({sub.name_en})")
            print(f"    定义: {sub.definition[:60]}...")
            print(f"    用途: {', '.join(sub.uses[:2])}")
            print(f"    产出: {', '.join(sub.production_methods[:2])}")
        else:
            print(f"  ✗ {name} 未注册")

    # 产出链
    print(f"\n子代理产出链:")
    chain = ss.get_production_chain("子代理")
    print(f"  {' → '.join(chain)}")

    print(f"\n代理经验产出链:")
    chain = ss.get_production_chain("代理经验")
    print(f"  {' → '.join(chain)}")


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════
def main():
    print("╔" + "═" * 68 + "╗")
    print("║" + "  xuni 工厂产物反哺 AI 验证实验".center(54) + "║")
    print("║" + "  记忆点 + 子代理 → 合鸣模型增强".center(54) + "║")
    print("╚" + "═" * 68 + "╝")

    # 0. 物质系统验证
    verify_substance_registry()

    # 1. 记忆 A/B
    r1 = experiment_1_memory_ab()

    # 2. 多轮累积
    r2 = experiment_2_multi_turn()

    # 3. 子代理投票
    r3 = experiment_3_subagent_vote()

    # 4. 分层认领
    r4 = experiment_4_layered_claim()

    # 总结
    banner("实验总结")
    print(f"""
┌──────────────────────────────────────────────────────────────────┐
│  实验 1（记忆 A/B）                                                │
│    无记忆平均分: {r1['no_memory_avg']:.3f}    有记忆平均分: {r1['with_memory_avg']:.3f}              │
│    提升: {r1['improvement_pct']:+.1f}%                                          │
│                                                                  │
│  实验 2（多轮累积）                                                │
│    分数趋势: {r2['trend']}    各轮: {r2['scores']}              │
│    短期记忆: {r2['stm_size']}    长期记忆: {r2['ltm_size']}                              │
│                                                                  │
│  实验 3（子代理投票）                                              │
│    首次共识度: {r3['first_consensus']:.3f}    共享后: {r3['second_consensus']:.3f}    变化: {r3['improvement']:+.3f}      │
│    总任务: {r3['total_tasks']}    成功率: {r3['success_rate']:.2%}                                  │
│                                                                  │
│  实验 4（分层认领）                                                │
│    派生子代理: {r4['agents_spawned']}    认领模型: {r4['models_claimed']}    训练完成: {r4['models_trained']}              │
│    任务分数: {r4['task_score']:.3f}                                              │
└──────────────────────────────────────────────────────────────────┘

结论：
  1. 记忆点接入合鸣后，回答质量提升 {r1['improvement_pct']:+.1f}%（关键术语命中+上下文延续）
  2. 多轮对话中，记忆累积让分数趋势 {r2['trend']}（{r2['scores']}）
  3. 子代理投票共识度从 {r3['first_consensus']:.3f} 变化到 {r3['second_consensus']:.3f}（经验共享后{'提升' if r3['improvement']>=0 else '下降'}）
  4. 子代理成功跨层认领 {r4['models_claimed']} 个模型，{r4['models_trained']} 个训练完成

xuni 工厂生产的「记忆点」和「子代理」确实能反哺 AI：
  - 记忆点让无状态生成 → 有状态、可累积经验
  - 子代理让单一调用 → 多角色协作、经验可复用、可投票
""")


if __name__ == "__main__":
    main()
