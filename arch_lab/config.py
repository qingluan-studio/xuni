"""全局配置：实验室的所有超参数集中在这里，方便调参。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ---- 数据 ----
    dataset: str = "mnist"            # mnist / cifar10
    train_subset: int = 6000          # 快速评估用的小训练集
    val_subset: int = 1000            # 验证集大小
    num_classes: int = 10
    in_channels: int = 1              # mnist=1, cifar=3

    # ---- 模型骨架 ----
    channels: int = 32                # 常量宽度 C（必须能被 heads 整除）
    stem_size: int = 7                # stem 后的空间尺寸 (H=W)
    max_nodes: int = 8                # 基因组最大节点数
    min_nodes: int = 3
    skip_prob: float = 0.3            # 随机基因组里跳连概率

    # ---- 进化 ----
    pop_size: int = 16
    generations: int = 6
    elite_size: int = 2               # 精英保留数
    tournament_k: int = 3
    mut_rate: float = 0.9             # 每个后代被创造(变异)的概率
    cross_rate: float = 0.5           # 每个后代被融合(交叉)的概率

    # ---- 评估 ----
    mode: str = "proxy"               # proxy: 免训练代理为主; full: 全量训练每个
    full_train_top: int = 4           # 每代对前 N 个精英做真实训练
    epochs: int = 2                   # 真实训练轮数
    batch_size: int = 128
    lr: float = 1e-3

    # ---- 多目标适应度权重 ----
    w_acc: float = 1.0
    w_params: float = 0.02            # 参数量惩罚(归一化后)
    w_latency: float = 0.0            # 可选延迟惩罚

    # ---- 涌现追踪 ----
    emergence_threshold: float = 0.5  # motif 在精英中占比超过此值视为"涌现"

    # ---- 运行 ----
    seed: int = 0
    num_workers: int = 0
    out_dir: str = "/workspace/arch_lab/runs/default"
    verbose: bool = True
