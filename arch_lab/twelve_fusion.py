"""
12领域架构融合实验：
  1. 单独测试12个架构在MNIST上的基线表现
  2. 四种融合策略:
     - chain:   顺序串联所有12个
     - parallel: 并联后汇聚
     - hierarchical: 分层分组融合 (4组×3架构 → 4组融合 → 最终融合)
     - evolved: 进化选择最优子集组合 (从12个中选6个最优排列)
  3. 对比所有融合架构 vs 单个架构
"""
from __future__ import annotations
import json
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .twelve_ops import get_all_ops, ARCH_INFO


# ============================================================
# 数据加载
# ============================================================
def get_loaders(batch_size: int = 128, train_subset: int = 6000, val_subset: int = 1000):
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    root = "/data/user/work/torchdata"
    train_full = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test_full = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(42)
    ti = torch.randperm(len(train_full), generator=g)[:train_subset].tolist()
    vi = torch.randperm(len(test_full), generator=g)[:val_subset].tolist()
    train_set = torch.utils.data.Subset(train_full, ti)
    val_set = torch.utils.data.Subset(test_full, vi)
    tl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)
    return tl, vl


# ============================================================
# 单架构模型
# ============================================================
class SingleArchModel(nn.Module):
    """单个架构组件包装成完整模型。"""
    def __init__(self, op_name: str, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        ops = get_all_ops(c)
        self.op = ops[op_name]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.op(x)
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 融合策略1: 串联 (Chain)
# ============================================================
class ChainFusionModel(nn.Module):
    """将12个架构顺序串联: stem → op1 → op2 → ... → op12 → head
    每层间有残差连接和梯度裁剪。"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32,
                 selected: list = None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        if selected:
            self.op_names = selected
        else:
            self.op_names = list(all_ops.keys())
        self.ops = nn.ModuleList([all_ops[name] for name in self.op_names])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x) + x  # 残差
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 融合策略2: 并联 (Parallel)
# ============================================================
class ParallelFusionModel(nn.Module):
    """将12个架构并联: stem → [op1, op2, ..., op12] → 汇聚 → head
    所有分支共享stem, 输出用加权求和汇聚。"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32,
                 selected: list = None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        if selected:
            self.op_names = selected
        else:
            self.op_names = list(all_ops.keys())
        self.ops = nn.ModuleList([all_ops[name] for name in self.op_names])
        n = len(self.op_names)
        # 可学习的汇聚权重
        self.gate = nn.Linear(c * n, n)
        self.fuse_proj = nn.Conv2d(c * n, c, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        outputs = [op(x) for op in self.ops]  # n × (B, C, H, W)
        # 门控加权
        pooled = torch.cat([self.pool(o).flatten(1) for o in outputs], dim=1)  # (B, nC)
        weights = F.softmax(self.gate(pooled), dim=-1)  # (B, n)
        # 加权求和
        stacked = torch.stack(outputs, dim=1)  # (B, n, C, H, W)
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, n, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)  # (B, C, H, W)
        # 投影
        fused = self.norm(self.fuse_proj(torch.cat(outputs, dim=1)))
        x = x + fused
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 融合策略3: 分层分组 (Hierarchical)
# ============================================================
class HierarchicalFusionModel(nn.Module):
    """分层融合: 将12个架构分成4组, 每组3个串联融合,
    然后4个组的结果并联汇聚, 最后用一个融合层整合。
    分组按架构互补性:
      组1 感知层: ViT + Conformer + Graphormer (特征提取)
      组2 注意力层: Transformer + DiT + PointTrans (注意力变体)
      组3 序列层: Mamba + RWKV + VideoMAE (时序建模)
      组4 融合层: CLIP + MoE + Perceiver (信息融合)
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        # 4个组
        self.groups = nn.ModuleList()
        self.group_names = [
            ["ViT", "Conformer", "Graphormer"],      # 感知层
            ["Transformer", "DiT", "PointTrans"],      # 注意力层
            ["Mamba", "RWKV", "VideoMAE"],             # 序列层
            ["CLIP", "MoE", "Perceiver"],              # 融合层
        ]
        for group in self.group_names:
            ops = nn.ModuleList([all_ops[name] for name in group])
            self.groups.append(ops)

        # 组间并联汇聚
        self.group_fuse = nn.Sequential(
            nn.Conv2d(c * 4, c * 2, 1, bias=False),
            nn.BatchNorm2d(c * 2), nn.GELU(),
            nn.Conv2d(c * 2, c, 1, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
        )
        self.norm = nn.BatchNorm2d(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        group_outputs = []
        for ops in self.groups:
            h = x
            for op in ops:
                h = op(h) + h  # 组内串联+残差
            group_outputs.append(h)
        # 组间并联汇聚
        fused = self.group_fuse(torch.cat(group_outputs, dim=1))
        x = x + self.norm(fused)
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 融合策略4: 进化选择 (Evolved)
# ============================================================
class EvolvedFusionModel(nn.Module):
    """从12个架构中进化选择6个最优组合, 串联+跳连。
    选择策略: 先单独评估所有12个, 选top-6, 按准确率排序串联。
    加入跨层跳连提升梯度流。"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32,
                 selected: list = None, skips: dict = None):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        if selected is None:
            # 默认选择6个互补架构
            selected = ["Perceiver", "Transformer", "Conformer", "Mamba", "CLIP", "DiT"]
        self.op_names = selected
        self.ops = nn.ModuleList([all_ops[name] for name in selected])
        # 跳连: {层索引: [源层索引列表]}
        self.skips = skips or {
            2: [0],  # 第3层接第1层
            3: [1],  # 第4层接第2层
            4: [0, 2],  # 第5层接第1、3层
            5: [1, 3],  # 第6层接第2、4层
        }
        self.norm = nn.BatchNorm2d(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        outputs = [x]
        for i, op in enumerate(self.ops):
            inp = outputs[-1]
            # 跳连
            if i in self.skips:
                for src in self.skips[i]:
                    if src < len(outputs):
                        inp = inp + outputs[src]
            out = op(inp) + inp  # 残差
            outputs.append(out)
        x = self.norm(outputs[-1])
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 训练评估
# ============================================================
def train_and_eval(model, train_loader, val_loader, device, epochs=3, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        best_acc = max(best_acc, acc)
        avg_loss = total_loss / max(1, n)
        history.append({"epoch": ep, "loss": round(avg_loss, 4), "acc": round(acc, 4)})
        print(f"    epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    npar = sum(p.numel() for p in model.parameters())
    return best_acc, npar, history


# ============================================================
# 主实验
# ============================================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/workspace/arch_lab/runs/twelve_fusion"
    os.makedirs(out_dir, exist_ok=True)

    epochs = 3
    c = 32

    print(f"\n{'='*70}")
    print(f"  12领域AI架构融合实验")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    tl, vl = get_loaders()
    results = {}

    # ---- 阶段1: 单架构基线 ----
    print(f"\n{'='*70}")
    print(f"  阶段1: 12个架构单独基线测试")
    print(f"{'='*70}")

    all_ops = get_all_ops(c)
    single_results = {}
    for name in all_ops.keys():
        print(f"\n  >> {name} ({ARCH_INFO[name]['domain']})")
        model = SingleArchModel(name, c=c)
        # 前向验证
        try:
            dummy = torch.randn(2, 1, 28, 28)
            _ = model(dummy)
        except Exception as e:
            print(f"     前向失败: {e}")
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
        elapsed = time.time() - t0
        single_results[name] = {
            "acc": round(acc, 4), "params": npar,
            "elapsed": round(elapsed, 1), "hist": hist,
            "domain": ARCH_INFO[name]["domain"],
            "score": ARCH_INFO[name]["score"],
        }
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    results["single"] = single_results

    # 排序找top-6
    ranked = sorted(single_results.items(), key=lambda x: x[1]["acc"], reverse=True)
    print(f"\n  单架构排名:")
    for i, (name, data) in enumerate(ranked):
        print(f"    {i+1}. {name:15s}  acc={data['acc']:.4f}  params={data['params']:>10,}  ({data['domain']})")
    top6 = [name for name, _ in ranked[:6]]
    print(f"\n  Top-6入选融合: {top6}")

    # ---- 阶段2: 融合实验 ----
    print(f"\n{'='*70}")
    print(f"  阶段2: 四种融合策略")
    print(f"{'='*70}")

    fusion_results = {}

    # 策略1: 全串联 (12个)
    print(f"\n  >> 融合策略1: 全串联 (Chain, 12个架构)")
    model = ChainFusionModel(c=c)
    t0 = time.time()
    acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
    elapsed = time.time() - t0
    fusion_results["Chain(12)"] = {
        "acc": round(acc, 4), "params": npar,
        "elapsed": round(elapsed, 1), "hist": hist,
        "desc": "12个架构顺序串联+残差",
    }
    print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
    del model

    # 策略2: 全并联 (12个)
    print(f"\n  >> 融合策略2: 全并联 (Parallel, 12个架构)")
    model = ParallelFusionModel(c=c)
    t0 = time.time()
    acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
    elapsed = time.time() - t0
    fusion_results["Parallel(12)"] = {
        "acc": round(acc, 4), "params": npar,
        "elapsed": round(elapsed, 1), "hist": hist,
        "desc": "12个架构并联+门控汇聚",
    }
    print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
    del model

    # 策略3: 分层分组 (4组×3)
    print(f"\n  >> 融合策略3: 分层分组 (Hierarchical, 4组×3)")
    model = HierarchicalFusionModel(c=c)
    t0 = time.time()
    acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
    elapsed = time.time() - t0
    fusion_results["Hierarchical(4×3)"] = {
        "acc": round(acc, 4), "params": npar,
        "elapsed": round(elapsed, 1), "hist": hist,
        "desc": "4组(感知/注意力/序列/融合)各3架构串联→并联汇聚",
    }
    print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
    del model

    # 策略4: 进化选择 (Top-6串联+跳连)
    print(f"\n  >> 融合策略4: 进化选择 (Evolved, Top-6+跳连)")
    print(f"     选择: {top6}")
    model = EvolvedFusionModel(c=c, selected=top6)
    t0 = time.time()
    acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
    elapsed = time.time() - t0
    fusion_results["Evolved(Top6)"] = {
        "acc": round(acc, 4), "params": npar,
        "elapsed": round(elapsed, 1), "hist": hist,
        "desc": f"Top-6({'+'.join(top6)})串联+跨层跳连",
        "selected": top6,
    }
    print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
    del model

    results["fusion"] = fusion_results

    # ---- 汇总报告 ----
    print(f"\n{'='*70}")
    print(f"  实验汇总")
    print(f"{'='*70}")

    print(f"\n  ▶ 单架构基线 (按准确率排序)")
    print(f"  {'架构':<16s} {'领域':<10s} {'准确率':>8s} {'参数量':>10s} {'评分':>6s}")
    print(f"  {'-'*56}")
    for name, data in ranked:
        print(f"  {name:<16s} {data['domain']:<10s} {data['acc']:>8.4f} {data['params']:>10,} {data['score']:>5.1f}★")

    print(f"\n  ▶ 融合策略对比")
    print(f"  {'策略':<22s} {'准确率':>8s} {'参数量':>10s} {'用时':>8s}")
    print(f"  {'-'*52}")
    for name, data in fusion_results.items():
        print(f"  {name:<22s} {data['acc']:>8.4f} {data['params']:>10,} {data['elapsed']:>7.1f}s")

    # 找最佳
    best_single = ranked[0]
    best_fusion = max(fusion_results.items(), key=lambda x: x[1]["acc"])
    print(f"\n  ▶ 最佳单架构: {best_single[0]} ({best_single[1]['acc']:.4f})")
    print(f"  ▶ 最佳融合策略: {best_fusion[0]} ({best_fusion[1]['acc']:.4f})")
    print(f"  ▶ 融合提升: {best_fusion[1]['acc'] - best_single[1]['acc']:+.4f}")

    # 保存结果
    summary = {
        "experiment": "twelve_domain_fusion",
        "epochs": epochs,
        "channels": c,
        "device": str(device),
        "single_results": single_results,
        "fusion_results": fusion_results,
        "top6": top6,
        "best_single": best_single[0],
        "best_fusion": best_fusion[0],
        "fusion_gain": round(best_fusion[1]["acc"] - best_single[1]["acc"], 4),
    }
    with open(os.path.join(out_dir, "twelve_fusion_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/twelve_fusion_results.json")

    return summary


if __name__ == "__main__":
    run_experiment()
