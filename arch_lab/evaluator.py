"""
评估器：把基因组变成可训练模型，并给出“适应度”。
- proxy 模式：用免训练代理指标(梯度信号)快速给大量候选排序，逼近最新 NAS 的高效范式。
- full 模式：真实训练若干轮得到验证准确率。
- 适应度 = 准确率(或代理) - 参数量惩罚，便于在“好且小”之间权衡。
"""
from __future__ import annotations
import math
import time
import random
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from .genome import Genome, GenomeModel

_DATA_CACHE: Dict[str, object] = {}


def get_loaders(cfg: Config):
    """加载(子集)数据，带缓存。"""
    key = f"{cfg.dataset}_{cfg.train_subset}_{cfg.val_subset}_{cfg.batch_size}"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    import torchvision
    from torchvision import transforms
    if cfg.dataset == "mnist":
        ds = torchvision.datasets.MNIST
        mean, std = (0.1307,), (0.3081,)
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        ds = torchvision.datasets.CIFAR10
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5,)*3, (0.5,)*3)])
    root = "/data/user/work/torchdata"
    train_full = ds(root, train=True, download=True, transform=tf)
    test_full = ds(root, train=False, download=True, transform=tf)
    # 子集
    g = torch.Generator().manual_seed(cfg.seed)
    ti = torch.randperm(len(train_full), generator=g)[:cfg.train_subset].tolist()
    vi = torch.randperm(len(test_full), generator=g)[:cfg.val_subset].tolist()
    train_set = torch.utils.data.Subset(train_full, ti)
    val_set = torch.utils.data.Subset(test_full, vi)
    tl = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                                     num_workers=cfg.num_workers)
    vl = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False,
                                     num_workers=cfg.num_workers)
    _DATA_CACHE[key] = (tl, vl)
    return tl, vl


def build_model(genome: Genome, cfg: Config) -> GenomeModel:
    return GenomeModel(genome, cfg.in_channels, cfg.num_classes, cfg.channels, cfg.stem_size)


@torch.no_grad()
def _spatial_size(cfg: Config) -> int:
    return cfg.stem_size


def zero_cost_proxy(genome: Genome, cfg: Config, device: torch.device,
                    sample: Optional[torch.Tensor] = None) -> float:
    """SynFlow 式免训练代理：前向求和反向，统计参数梯度绝对值之和。
    数值越大通常意味着该结构在初始化附近“信号越通、越可训练”。"""
    model = build_model(genome, cfg).to(device)
    model.train()
    if sample is None:
        sample = torch.randn(cfg.batch_size, cfg.in_channels, 28, 28, device=device)
    # 让 BN 用当前 batch 统计
    out = model(sample)
    loss = out.sum()
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad],
                                create_graph=False, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.abs().sum().item()
    # 归一化(按参数量)并对极端值做对数压缩
    npar = max(1, model.num_parameters())
    score = math.log1p(total / npar)
    if not math.isfinite(score):
        score = 0.0
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return score


def train_model(genome: Genome, cfg: Config, device: torch.device,
                loaders: Optional[Tuple] = None) -> Tuple[float, int]:
    """真实训练 cfg.epochs 轮，返回(验证准确率, 参数量)。"""
    if loaders is None:
        loaders = get_loaders(cfg)
    train_loader, val_loader = loaders
    model = build_model(genome, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    model.train()
    for ep in range(cfg.epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
    # 验证
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    acc = correct / max(1, total)
    npar = model.num_parameters()
    del model
    return acc, npar


def param_penalty(npar: int) -> float:
    """参数量惩罚，归一化到 0~1 量级。"""
    return npar / 1e6


def fitness_from_acc(acc: float, npar: int, cfg: Config) -> float:
    return cfg.w_acc * acc - cfg.w_params * param_penalty(npar)


def evaluate(genome: Genome, cfg: Config, device: torch.device,
             loaders: Optional[Tuple] = None,
             use_full: bool = False, sample: Optional[torch.Tensor] = None) -> Dict:
    """统一评估入口，返回 dict(proxy, acc, params, fitness, mode)。"""
    result = {"proxy": None, "acc": None, "params": None, "fitness": 0.0, "mode": "proxy"}
    if use_full:
        acc, npar = train_model(genome, cfg, device, loaders)
        result.update(acc=acc, params=npar, fitness=fitness_from_acc(acc, npar, cfg), mode="full")
    else:
        proxy = zero_cost_proxy(genome, cfg, device, sample)
        npar = build_model(genome, cfg).num_parameters()
        # proxy 模式：用代理分主导，参数量轻度惩罚
        fit = proxy - cfg.w_params * param_penalty(npar) * 0.5
        result.update(proxy=proxy, params=npar, fitness=fit, mode="proxy")
    return result
