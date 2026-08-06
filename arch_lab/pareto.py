"""
多目标 Pareto 前沿：在 (准确率, 参数量) 二维空间里寻找非支配解。
一个个体被另一个支配 = 对方在所有目标上都不差且至少一个好。
Pareto 前沿 = 所有不被任何个体支配的个体集合。
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ParetoPoint:
    """Pareto 前沿上的一个点。"""
    genome_repr: str          # 基因组的可读表示
    acc: float                # 验证准确率(越大越好)
    params: int               # 参数量(越小越好)
    fitness: float
    dominated: bool = False


def is_dominated(a_acc, a_params, b_acc, b_params) -> bool:
    """a 是否被 b 支配(最大化 acc, 最小化 params)。"""
    return b_acc >= a_acc and b_params <= a_params and (b_acc > a_acc or b_params < a_params)


def pareto_front(individuals: List[Dict]) -> List[ParetoPoint]:
    """从个体列表计算 Pareto 前沿。
    individuals: [{"acc":float,"params":int,"fitness":float,"genome_repr":str}, ...]
    """
    points = []
    for ind in individuals:
        if ind.get("acc") is None:
            continue
        points.append(ParetoPoint(
            genome_repr=ind.get("genome_repr", "?"),
            acc=ind["acc"], params=ind["params"],
            fitness=ind.get("fitness", 0.0),
        ))
    # 标记被支配的点
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if i != j and is_dominated(p.acc, p.params, q.acc, q.params):
                p.dominated = True
                break
    front = [p for p in points if not p.dominated]
    front.sort(key=lambda p: p.params)  # 按参数量排序
    return front


def pareto_summary(front: List[ParetoPoint]) -> Dict:
    """Pareto 前沿摘要。"""
    if not front:
        return {"size": 0, "points": []}
    return {
        "size": len(front),
        "best_acc": max(p.acc for p in front),
        "min_params": min(p.params for p in front),
        "points": [{"acc": round(p.acc, 4), "params": p.params,
                     "fitness": round(p.fitness, 4)} for p in front],
    }
