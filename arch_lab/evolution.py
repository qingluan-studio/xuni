"""
进化引擎：用“变异(创造)+交叉(融合)”在基因组空间里搜索，
以适应度为环境压力，让优秀架构“被选择”并繁衍。
proxy 模式下先用免训练代理快速筛选整个种群，仅对少量精英做真实训练以校准。
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
from .config import Config
from .genome import Genome, random_genome
from .operations import mutate, crossover
from . import evaluator
from .emergence import EmergenceTracker


@dataclass
class Individual:
    genome: Genome
    fitness: float = 0.0
    proxy: Optional[float] = None
    acc: Optional[float] = None
    params: Optional[int] = None
    mode: str = "proxy"


@dataclass
class GenerationLog:
    gen: int
    best_fitness: float
    mean_fitness: float
    best_acc: Optional[float]
    best_proxy: Optional[float]
    best_params: Optional[int]
    elite_motifs: Dict[str, float]
    elapsed: float


class Evolution:
    def __init__(self, cfg: Config, device: torch.device, seeds=None):
        self.cfg = cfg
        self.device = device
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.tracker = EmergenceTracker(cfg.emergence_threshold)
        self.history: List[GenerationLog] = []
        self.population: List[Individual] = []
        self.best: Optional[Individual] = None          # 按适应度(含代理)择优
        self.best_by_acc: Optional[Individual] = None   # 按真实验证准确率择优(更贴近"好架构")
        self._loaders = None
        self._sample = None
        self.seeds = seeds or []                        # 种子基因组(来自手绘图等)
        self.all_evaluated: List[Individual] = []       # 所有做过真实训练的个体(供Pareto分析)

    def _eval(self, ind: Individual, use_full: bool):
        res = evaluator.evaluate(ind.genome, self.cfg, self.device,
                                 loaders=self._loaders, use_full=use_full, sample=self._sample)
        ind.fitness = res["fitness"]
        ind.proxy = res["proxy"]
        ind.acc = res["acc"]
        ind.params = res["params"]
        ind.mode = res["mode"]

    def _maybe_best_acc(self, ind: Individual):
        """凡是做过真实训练、拿到 acc 的个体，都参与"准确率最优"的竞选，并收集供Pareto分析。"""
        if ind.acc is not None:
            if self.best_by_acc is None or ind.acc > self.best_by_acc.acc:
                self.best_by_acc = Individual(genome=ind.genome.copy(), fitness=ind.fitness,
                                              proxy=ind.proxy, acc=ind.acc, params=ind.params, mode=ind.mode)
            # 收集到 all_evaluated 供 Pareto 分析(去重：同基因组只保留最优)
            self.all_evaluated.append(Individual(genome=ind.genome.copy(), fitness=ind.fitness,
                                                  proxy=ind.proxy, acc=ind.acc, params=ind.params, mode=ind.mode))

    def _init_population(self):
        self.population = []
        # 先注入种子基因组(来自手绘图等)
        for seed_g in self.seeds:
            if len(self.population) < self.cfg.pop_size:
                self.population.append(Individual(genome=seed_g.copy()))
        # 剩余位置用随机基因组填充
        while len(self.population) < self.cfg.pop_size:
            ind = Individual(genome=random_genome(self.cfg, self.rng))
            self.population.append(ind)

    def _tournament(self) -> Individual:
        contenders = self.rng.sample(self.population, min(self.cfg.tournament_k, len(self.population)))
        return max(contenders, key=lambda x: x.fitness)

    def _make_offspring(self) -> Individual:
        parent = self._tournament()
        if self.rng.random() < self.cfg.cross_rate and len(self.population) > 1:
            mate = self._tournament()
            child_genome = crossover(parent.genome, mate.genome, self.cfg, self.rng)
        else:
            child_genome = parent.genome.copy()
        if self.rng.random() < self.cfg.mut_rate:
            child_genome = mutate(child_genome, self.cfg, self.rng)
        return Individual(genome=child_genome)

    def run(self, log_fn=None) -> Dict:
        cfg = self.cfg
        self._loaders = evaluator.get_loaders(cfg)
        # 代理用的固定样本(降低方差)
        self._sample = torch.randn(cfg.batch_size, cfg.in_channels, 28, 28, device=self.device)
        self._init_population()
        use_full = (cfg.mode == "full")

        for gen in range(cfg.generations):
            t0 = time.time()
            # 评估整个种群
            for ind in self.population:
                self._eval(ind, use_full=use_full)
            # 排序
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            # 精英(用于选择/保留)
            elites = self.population[: max(1, cfg.elite_size)]
            # 涌现追踪用“核心群体(前 1/3)”，比仅看 2 个精英信号更稳、更易观察到 motif 涌现
            core_n = max(cfg.elite_size, len(self.population) // 3)
            core_genomes = [ind.genome for ind in self.population[:core_n]]
            elite_motifs = self.tracker.record(core_genomes)

            # proxy 模式：对前 N 个精英补做真实训练，得到可读准确率
            if not use_full and cfg.full_train_top > 0:
                for ind in elites[: cfg.full_train_top]:
                    if ind.acc is None:
                        res = evaluator.evaluate(ind.genome, cfg, self.device,
                                                 loaders=self._loaders, use_full=True)
                        ind.acc = res["acc"]
                        ind.params = res["params"]

            # 更新“按真实准确率择优”(覆盖 full 与 proxy 两种模式)
            for ind in self.population:
                self._maybe_best_acc(ind)

            best = self.population[0]
            if self.best is None or best.fitness > self.best.fitness:
                self.best = Individual(genome=best.genome.copy(), fitness=best.fitness,
                                       proxy=best.proxy, acc=best.acc, params=best.params, mode=best.mode)
            mean_fit = sum(x.fitness for x in self.population) / len(self.population)
            log = GenerationLog(
                gen=gen, best_fitness=best.fitness, mean_fitness=mean_fit,
                best_acc=best.acc, best_proxy=best.proxy, best_params=best.params,
                elite_motifs=elite_motifs, elapsed=time.time() - t0,
            )
            self.history.append(log)
            if log_fn:
                log_fn(log)
            if cfg.verbose:
                acc_s = f"{best.acc:.3f}" if best.acc is not None else "  -  "
                prx_s = f"{best.proxy:.3f}" if best.proxy is not None else "  -  "
                print(f"[gen {gen}] fit={best.fitness:.4f} acc={acc_s} "
                      f"proxy={prx_s} params={best.params} nodes={len(best.genome)} "
                      f"mean_fit={mean_fit:.4f} t={log.elapsed:.1f}s")
            # 繁衍下一代
            next_pop: List[Individual] = [Individual(genome=g.copy(), fitness=f.fitness,
                                                    proxy=f.proxy, acc=f.acc, params=f.params, mode=f.mode)
                                          for g, f in ((e.genome, e) for e in elites)]
            # 5% 随机移民维持多样性
            while len(next_pop) < cfg.pop_size:
                if self.rng.random() < 0.05:
                    next_pop.append(Individual(genome=random_genome(cfg, self.rng)))
                else:
                    next_pop.append(self._make_offspring())
            self.population = next_pop[: cfg.pop_size]

        # 末代再评估一次保证 best 准确率已校准
        if self.best and self.best.acc is None and not use_full:
            res = evaluator.evaluate(self.best.genome, cfg, self.device,
                                     loaders=self._loaders, use_full=True)
            self.best.acc = res["acc"]; self.best.params = res["params"]
            self._maybe_best_acc(self.best)
        # 若 best_by_acc 仍空(极少见：全程未做真实训练)，退化为 best
        recommended = self.best_by_acc or self.best
        return {
            "best": self.best,
            "best_by_acc": self.best_by_acc,
            "recommended": recommended,
            "history": self.history,
            "emergence": self.tracker.emerged(),
            "tracker": self.tracker,
            "all_evaluated": self.all_evaluated,
        }
