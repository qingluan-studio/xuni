"""
三联融合实验 (Triple Fusion)：
  串联(Chain) + 并联(Parallel) + 异联(Cross) 三种策略融为一体
  
设计理念:
  Stage 1 — 感知并联层 (Parallel): ViT + CLIP + Perceiver 并联 → 门控汇聚
            → 多模态感知，不同视角同时处理输入
  Stage 2 — 推理串联层 (Chain): Transformer → Mamba → Transformer → Mamba
            → 深度推理，4层异构串联+残差，构建深层理解
  Stage 3 — 异联融合层 (Cross): MoE路由 + 跨阶段跳连 + 迭代精炼回路
            → 异联：Stage1直通Stage3, Stage2中间结果参与, Stage3→Stage2回路

评测任务:
  1. MNIST 分类 (感知能力)
  2. MNIST 数字加法推理 (推理能力): 给两张数字图片，预测它们的和 (0-18, 19类)
     — 这要求模型: 识别两个数字 → 心算加法 → 输出结果，是真正的推理任务
"""
from __future__ import annotations
import json
import os
import time
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


def get_addition_loaders(batch_size: int = 128, train_subset: int = 6000, val_subset: int = 1000):
    """MNIST 数字加法推理任务: 两张图片 → 预测数字之和 (0-18, 19类)"""
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    root = "/data/user/work/torchdata"
    train_full = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test_full = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(42)
    ti = torch.randperm(len(train_full), generator=g)[:train_subset].tolist()
    vi = torch.randperm(len(test_full), generator=g)[:val_subset].tolist()

    class AdditionDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, seed=42):
            self.data = [(base_dataset[i][0], base_dataset[i][1]) for i in indices]
            self.rng = torch.Generator().manual_seed(seed)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            i = torch.randint(0, len(self.data), (1,), generator=self.rng).item()
            img1, label1 = self.data[idx]
            img2, label2 = self.data[i]
            # 拼接两张图片: (2, 28, 28)
            combined = torch.stack([img1.squeeze(0), img2.squeeze(0)], dim=0)
            target = label1 + label2  # 0-18
            return combined, target

    train_set = AdditionDataset(train_full, ti)
    val_set = AdditionDataset(test_full, vi, seed=99)
    tl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)
    return tl, vl


# ============================================================
# 三联融合架构 (Triple Fusion)
# ============================================================
class TripleFusionModel(nn.Module):
    """三联融合 V2: 串联 + 并联 + 异联 三种策略融为一体
    
    Stage 1 (并联): ViT + CLIP + Perceiver → 门控汇聚 [多模态感知]
    Stage 2 (串联): Transformer → Mamba → Transformer [3层深度推理]
    Stage 3 (异联): 跨阶段注意力融合 [轻量异构融合]
    
    异联机制 (无回路, 纯前向跳连):
      - Stage1 → Stage3 直通跳连 (感知信息直达融合层)
      - Stage2 中间层 → Stage3 (推理中间结果参与融合)
      - Stage2 最终 → Stage3 (推理最终结果)
    三路信息通过可学习的注意力权重融合。
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.c = c

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

        all_ops = get_all_ops(c)

        # ---- Stage 1: 感知并联层 (Parallel) ----
        self.stage1_names = ["ViT", "CLIP", "Perceiver"]
        self.stage1_ops = nn.ModuleList([all_ops[n] for n in self.stage1_names])
        self.stage1_gate = nn.Linear(c * len(self.stage1_names), len(self.stage1_names))
        self.stage1_fuse = nn.Sequential(
            nn.Conv2d(c * len(self.stage1_names), c, 1, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
        )

        # ---- Stage 2: 推理串联层 (Chain) — 3层 ----
        self.stage2_names = ["Transformer", "Mamba", "Transformer"]
        self.stage2_ops = nn.ModuleList([all_ops[n] for n in self.stage2_names])

        # ---- Stage 3: 异联融合层 (Cross) — 轻量注意力融合 ----
        # 三路输入: Stage1_out + Stage2_mid + Stage2_final
        # 用可学习的注意力权重融合三路信息
        self.cross_attn = nn.MultiheadAttention(c, num_heads=4, batch_first=True)
        self.cross_norm = nn.LayerNorm(c)
        # 路由权重: 从三路全局特征计算注意力分配
        self.route = nn.Sequential(
            nn.Linear(c * 3, c), nn.GELU(), nn.Linear(c, 3),
        )
        self.cross_proj = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c),
        )
        self.cross_out_norm = nn.LayerNorm(c)

        # ---- 分类头 ----
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def _parallel_stage(self, x):
        """Stage 1: 并联感知"""
        outputs = [op(x) for op in self.stage1_ops]
        pooled = torch.cat([self.pool(o).flatten(1) for o in outputs], dim=1)
        weights = F.softmax(self.stage1_gate(pooled), dim=-1)
        stacked = torch.stack(outputs, dim=1)
        w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gated = (stacked * w).sum(dim=1)
        fused = self.stage1_fuse(torch.cat(outputs, dim=1))
        return x + gated + fused

    def _chain_stage(self, x):
        """Stage 2: 串联推理, 返回最终输出和中间输出"""
        h = x
        mid = None
        for i, op in enumerate(self.stage2_ops):
            h = op(h) + h  # 残差
            if i == 1:
                mid = h
        return h, mid

    def _cross_stage(self, s1_out, s2_mid, s2_final):
        """Stage 3: 异联融合 — 三路信息通过注意力融合"""
        B, C, H, W = s2_final.shape
        # 展平为序列
        s1 = s1_out.flatten(2).transpose(1, 2)  # (B, HW, C)
        sm = s2_mid.flatten(2).transpose(1, 2)
        sf = s2_final.flatten(2).transpose(1, 2)

        # 路由权重: 从三路全局特征计算
        g1 = s1.mean(1)  # (B, C)
        gm = sm.mean(1)
        gf = sf.mean(1)
        route_w = F.softmax(self.route(torch.cat([g1, gm, gf], dim=-1)), dim=-1)  # (B, 3)

        # 加权融合三路信息
        r1 = route_w[:, 0:1].unsqueeze(-1)  # (B, 1, 1, 1)
        rm = route_w[:, 1:2].unsqueeze(-1)
        rf = route_w[:, 2:3].unsqueeze(-1)
        merged = (s1 * r1 + sm * rm + sf * rf)  # (B, HW, C)

        # 自注意力精炼
        h = self.cross_norm(merged)
        attn_out, _ = self.cross_attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.cross_proj(self.cross_out_norm(merged))

        return merged.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        x = self.stem(x)
        # Stage 1: 并联感知
        s1_out = self._parallel_stage(x)
        # Stage 2: 串联推理
        s2_out, s2_mid = self._chain_stage(s1_out)
        # Stage 3: 异联融合
        s3_out = self._cross_stage(s1_out, s2_mid, s2_out)
        # 分类
        feat = self.pool(s3_out).flatten(1)
        return self.head(feat)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比架构: 纯并联 (复用之前的设计)
# ============================================================
class ParallelOnlyModel(nn.Module):
    """纯并联基线: 6个架构并联，无串联无异联"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        names = ["ViT", "CLIP", "Perceiver", "Transformer", "Mamba", "MoE"]
        self.ops = nn.ModuleList([all_ops[n] for n in names])
        n = len(names)
        self.gate = nn.Linear(c * n, n)
        self.fuse_proj = nn.Conv2d(c * n, c, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        outputs = [op(x) for op in self.ops]
        pooled = torch.cat([self.pool(o).flatten(1) for o in outputs], dim=1)
        weights = F.softmax(self.gate(pooled), dim=-1)
        stacked = torch.stack(outputs, dim=1)
        w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        fused = (stacked * w).sum(dim=1)
        fused = self.norm(self.fuse_proj(torch.cat(outputs, dim=1)))
        x = x + fused
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比架构: 纯串联
# ============================================================
class ChainOnlyModel(nn.Module):
    """纯串联基线: 6个架构串联，无并联无异联"""
    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        all_ops = get_all_ops(c)
        names = ["ViT", "Transformer", "Mamba", "CLIP", "Perceiver", "MoE"]
        self.ops = nn.ModuleList([all_ops[n] for n in names])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for op in self.ops:
            x = op(x) + x
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 训练评估
# ============================================================
def train_and_eval(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
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
    out_dir = "/workspace/arch_lab/runs/triple_fusion"
    os.makedirs(out_dir, exist_ok=True)

    epochs = 8
    c = 32

    print(f"\n{'='*70}")
    print(f"  三联融合实验 (Triple Fusion)")
    print(f"  串联 + 并联 + 异联 → 推理强化")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # ---- 任务1: MNIST 分类 ----
    print(f"\n{'='*70}")
    print(f"  任务1: MNIST 分类 (感知能力)")
    print(f"{'='*70}")

    tl_cls, vl_cls = get_loaders()

    models_cls = {
        "Chain(6)": ChainOnlyModel(in_channels=1, num_classes=10, c=c),
        "Parallel(6)": ParallelOnlyModel(in_channels=1, num_classes=10, c=c),
        "TripleFusion": TripleFusionModel(in_channels=1, num_classes=10, c=c),
    }

    cls_results = {}
    for name, model in models_cls.items():
        print(f"\n  >> {name}")
        try:
            dummy = torch.randn(2, 1, 28, 28)
            _ = model(dummy)
        except Exception as e:
            print(f"     前向失败: {e}")
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_cls, vl_cls, device, epochs=epochs)
        elapsed = time.time() - t0
        cls_results[name] = {
            "acc": round(acc, 4), "params": npar,
            "elapsed": round(elapsed, 1), "hist": hist,
        }
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # ---- 任务2: MNIST 数字加法推理 ----
    print(f"\n{'='*70}")
    print(f"  任务2: MNIST 数字加法推理 (推理能力)")
    print(f"  输入: 两张数字图片 → 输出: 数字之和 (0-18, 19类)")
    print(f"{'='*70}")

    tl_add, vl_add = get_addition_loaders()

    models_add = {
        "Chain(6)": ChainOnlyModel(in_channels=2, num_classes=19, c=c),
        "Parallel(6)": ParallelOnlyModel(in_channels=2, num_classes=19, c=c),
        "TripleFusion": TripleFusionModel(in_channels=2, num_classes=19, c=c),
    }

    add_results = {}
    for name, model in models_add.items():
        print(f"\n  >> {name}")
        try:
            dummy = torch.randn(2, 2, 28, 28)
            _ = model(dummy)
        except Exception as e:
            print(f"     前向失败: {e}")
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_add, vl_add, device, epochs=epochs)
        elapsed = time.time() - t0
        add_results[name] = {
            "acc": round(acc, 4), "params": npar,
            "elapsed": round(elapsed, 1), "hist": hist,
        }
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # ---- 推理基线: 随机猜测 ----
    random_acc_cls = 1.0 / 10  # 10%
    random_acc_add = 1.0 / 19  # ~5.3%

    # ---- 汇总报告 ----
    print(f"\n{'='*70}")
    print(f"  实验汇总")
    print(f"{'='*70}")

    print(f"\n  ▶ 任务1: MNIST 分类 (随机基线={random_acc_cls:.1%})")
    print(f"  {'架构':<20s} {'准确率':>8s} {'参数量':>10s} {'用时':>8s}")
    print(f"  {'-'*50}")
    for name, data in sorted(cls_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {name:<20s} {data['acc']:>8.4f} {data['params']:>10,} {data['elapsed']:>7.1f}s")

    print(f"\n  ▶ 任务2: MNIST 数字加法推理 (随机基线={random_acc_add:.1%})")
    print(f"  {'架构':<20s} {'准确率':>8s} {'参数量':>10s} {'用时':>8s}  {'vs随机':>8s}")
    print(f"  {'-'*58}")
    for name, data in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        lift = data["acc"] / random_acc_add
        print(f"  {name:<20s} {data['acc']:>8.4f} {data['params']:>10,} {data['elapsed']:>7.1f}s  {lift:>7.1f}x")

    # 推理增益
    if "TripleFusion" in add_results and "Parallel(6)" in add_results:
        gain = add_results["TripleFusion"]["acc"] - add_results["Parallel(6)"]["acc"]
        print(f"\n  ▶ 三联融合 vs 纯并联 推理增益: {gain:+.4f}")
    if "TripleFusion" in add_results and "Chain(6)" in add_results:
        gain = add_results["TripleFusion"]["acc"] - add_results["Chain(6)"]["acc"]
        print(f"  ▶ 三联融合 vs 纯串联 推理增益: {gain:+.4f}")

    # 保存结果
    summary = {
        "experiment": "triple_fusion",
        "epochs": epochs,
        "channels": c,
        "device": str(device),
        "cls_task": "MNIST分类(10类)",
        "add_task": "MNIST数字加法推理(19类)",
        "random_baseline_cls": random_acc_cls,
        "random_baseline_add": random_acc_add,
        "cls_results": cls_results,
        "add_results": add_results,
    }
    with open(os.path.join(out_dir, "triple_fusion_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/triple_fusion_results.json")

    return summary


if __name__ == "__main__":
    run_experiment()
