"""
实验室入口：
    python -m arch_lab.main --pop 16 --gens 6 --mode proxy
    python -m arch_lab.main --pop 8 --gens 3 --mode full --epochs 1   # 极速冒烟
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
import torch
from .config import Config
from .evolution import Evolution
from .seeds import seed_architectures
from . import viz, pareto, export, fuser


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="架构涌现实验室")
    p.add_argument("--pop", type=int, default=16)
    p.add_argument("--gens", type=int, default=6)
    p.add_argument("--mode", choices=["proxy", "full"], default="proxy")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--full-train-top", type=int, default=4)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--max-nodes", type=int, default=8)
    p.add_argument("--min-nodes", type=int, default=3)
    p.add_argument("--train-subset", type=int, default=6000)
    p.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="/workspace/arch_lab/runs/default")
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--seeds", type=int, default=1, help="注入N张手绘图作为种子(0=禁用)")
    p.add_argument("--fuse", action="store_true", help="把5张手绘图融合成新架构作为种子")
    a = p.parse_args()
    cfg = Config()
    cfg.pop_size = a.pop
    cfg.generations = a.gens
    cfg.mode = a.mode
    cfg.epochs = a.epochs
    cfg.full_train_top = a.full_train_top
    cfg.channels = a.channels
    cfg.max_nodes = a.max_nodes
    cfg.min_nodes = a.min_nodes
    cfg.train_subset = a.train_subset
    cfg.dataset = a.dataset
    cfg.in_channels = 3 if a.dataset == "cifar10" else 1
    cfg.seed = a.seed
    cfg.out_dir = a.out
    cfg.tag = a.tag
    cfg._seeds_n = a.seeds
    cfg._fuse = a.fuse
    return cfg


def genome_to_dict(g):
    return [{"op": n.op, "act": n.act, "expand": n.expand, "heads": n.heads, "inputs": n.inputs}
            for n in g.nodes]


def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 注入手绘架构图作为种子
    all_seeds = seed_architectures()
    n_seeds = getattr(cfg, "_seeds_n", 1)
    do_fuse = getattr(cfg, "_fuse", False)
    rng = random.Random(cfg.seed)

    if do_fuse:
        # ---- 融合模式：把 5 张图融合成新架构 ----
        seeds_for_fuse = all_seeds[:5]
        print(f"== 架构涌现实验室 [融合模式] ==  device={device}  mode={cfg.mode}  "
              f"pop={cfg.pop_size} gens={cfg.generations} dataset={cfg.dataset}")
        print(f"  融合 {len(seeds_for_fuse)} 张手绘图:")
        for i, s in enumerate(seeds_for_fuse):
            print(f"    图{i+1}: {len(s)}节点 " + "→".join(n.op for n in s.nodes))
        # 三种融合策略各产出一个
        fused = fuser.fuse_all(seeds_for_fuse, cfg, rng)
        print(f"\n  融合产出 {len(fused)} 个种子架构:")
        for i, f in enumerate(fused):
            print(f"    融合{i+1} ({['串联','并联','嫁接'][i]}): {len(f)}节点 " + fuser.genome_signature(f))
        seeds = fused
        # 同时保留原始 5 张图作为额外种子(丰富种群)
        seeds.extend([s.copy() for s in seeds_for_fuse[:2]])
    else:
        seeds = all_seeds[:n_seeds] if n_seeds > 0 else []
        if seeds:
            print(f"== 架构涌现实验室 ==  device={device}  mode={cfg.mode}  "
                  f"pop={cfg.pop_size} gens={cfg.generations} dataset={cfg.dataset} "
                  f"seeds={len(seeds)}张手绘图")
            for i, s in enumerate(seeds):
                print(f"  种子{i+1}: {len(s)}节点 " + "→".join(n.op for n in s.nodes))
        else:
            print(f"== 架构涌现实验室 ==  device={device}  mode={cfg.mode}  "
                  f"pop={cfg.pop_size} gens={cfg.generations} dataset={cfg.dataset} (无种子)")

    eng = Evolution(cfg, device, seeds=seeds)
    t0 = time.time()
    result = eng.run()
    elapsed = time.time() - t0

    best = result["best"]                 # 按适应度(含代理)最优
    rec = result["recommended"]           # 按真实验证准确率最优(推荐架构)
    print(f"\n== 完成 (用时 {elapsed:.1f}s) ==")
    if rec is not None and rec.acc is not None:
        print(f"推荐架构(按真实准确率) 验证准确率: {rec.acc:.4f}  参数量: {rec.params}  节点数: {len(rec.genome)}")
    if best.acc is not None:
        print(f"适应度最优个体 acc={best.acc:.4f} fitness={best.fitness:.4f} (代理可能略偏离真实精度)")
    show = rec or best
    print("推荐架构:")
    for i, n in enumerate(show.genome.nodes):
        print(f"  node{i}: {n.op:8s} act={n.act:4s} expand={n.expand} heads={n.heads} inputs={n.inputs}")

    # 涌现总结
    emerged = result["emergence"]
    print(f"\n== 涌现 motif (初始罕见→末期≥{cfg.emergence_threshold}) ==")
    if emerged:
        for e in emerged:
            print(f"  {e['motif']:36s} {e['initial_freq']:.2f} → {e['final_freq']:.2f}  (+{e['growth']:.2f})")
    else:
        print("  (本轮未观察到越过阈值的 motif，可增加代数/种群规模再看)")
    # 末期 top motif
    top_m = result["tracker"].top_motifs(8)
    print(f"\n== 末期核心群体 top motif ==")
    for m, f in top_m:
        print(f"  {m:36s} freq={f:.2f}")

    # ---- Pareto 前沿 ----
    all_eval = result.get("all_evaluated", [])
    pareto_pts = [{"acc": i.acc, "params": i.params, "fitness": i.fitness,
                   "genome_repr": "→".join(n.op for n in i.genome.nodes)} for i in all_eval]
    pfront = pareto.pareto_front(pareto_pts)
    psum = pareto.pareto_summary(pfront)
    print(f"\n== Pareto 前沿 (准确率 vs 参数量) ==")
    print(f"  前沿大小: {psum['size']}  最高准确率: {psum.get('best_acc','-')}  最小参数: {psum.get('min_params','-')}")
    for p in psum.get("points", []):
        print(f"    acc={p['acc']:.4f}  params={p['params']:,}  fitness={p['fitness']:.4f}")

    # ---- 导出推荐架构为独立 PyTorch 代码 ----
    exported_path = os.path.join(cfg.out_dir, "evolved_model.py")
    try:
        export.export_genome(show.genome, exported_path,
                             in_channels=cfg.in_channels, num_classes=cfg.num_classes,
                             channels=cfg.channels, model_name="EvolvedModel")
        print(f"\n推荐架构已导出为独立代码: {exported_path}")
    except Exception as e:
        print(f"导出跳过: {e}")

    # 保存结果
    summary = {
        "tag": getattr(cfg, "tag", "run"),
        "elapsed_sec": round(elapsed, 1),
        "best_fitness": best.fitness,
        "best_acc": best.acc,
        "best_params": best.params,
        "best_genome": genome_to_dict(best.genome),
        "recommended_acc": getattr(rec, "acc", None),
        "recommended_params": getattr(rec, "params", None),
        "recommended_genome": genome_to_dict(show.genome),
        "emerged_motifs": emerged,
        "pareto_front": psum,
        "seeds_used": len(seeds),
        "history": [{"gen": h.gen, "best_fitness": h.best_fitness, "mean_fitness": h.mean_fitness,
                     "best_acc": h.best_acc, "best_proxy": h.best_proxy, "best_params": h.best_params}
                    for h in result["history"]],
    }
    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 图表
    try:
        viz.plot_evolution(result["history"], os.path.join(cfg.out_dir, "evolution.png"))
        viz.plot_motif_frequency(result["tracker"], os.path.join(cfg.out_dir, "motifs.png"))
        viz.draw_architecture(show.genome, os.path.join(cfg.out_dir, "best_arch.png"),
                              title=f"推荐架构 acc={getattr(show,'acc',None)}")
        if pfront:
            viz.plot_pareto(pfront, os.path.join(cfg.out_dir, "pareto.png"))
        print(f"\n图表与 summary.json 已保存到: {cfg.out_dir}")
    except Exception as e:
        print(f"绘图跳过: {e}")


if __name__ == "__main__":
    main()
