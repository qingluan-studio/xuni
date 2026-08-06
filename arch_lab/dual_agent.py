"""
TRAE × Kimi 双Agent协作实验 — 借鉴扣子(Coze)多Agent协作思路

核心区别:
  之前的字母融合 = 把两个模型的组件拆散重组到一个网络里 (基因混合)
  本实验 = 保持两个模型各自完整,通过编排层协作 (团队合作)

5种协作模式:
  1. Solo-TRAE:   TRAE单独工作 (基线)
  2. Solo-Kimi:   Kimi单独工作 (基线)
  3. Parallel-Vote:  两者并行预测,logits平均投票 (民主决策)
  4. Parallel-Learned: 两者并行,学习动态路由器加权 (智能调度)
  5. Sequential-Fuse:  两者各自提取特征,融合MLP做最终决策 (特征级协作)

编排器(Orchestrator) = 扣子平台的Agent编排层
  - 决定Agent之间的信息流向
  - 决定Agent的权重分配
  - 决定何时并行、何时串联
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from trae_model import TRAEModel
from kimi_model import KimiModel


# ============================================================
# 模式3: Parallel-Vote (并行投票)
# ============================================================
class ParallelVoteModel(nn.Module):
    """两个Agent并行预测,logits平均。无需额外参数。"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.trae = TRAEModel(in_channels, num_classes, c)
        self.kimi = KimiModel(in_channels, num_classes, c)

    def forward(self, x):
        logits_trae = self.trae(x)
        logits_kimi = self.kimi(x)
        return (logits_trae + logits_kimi) / 2.0

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 模式4: Parallel-Learned (学习动态路由)
# ============================================================
class ParallelLearnedModel(nn.Module):
    """两个Agent并行,路由器根据输入动态分配权重。
    编排器学习: 什么样的输入该信TRAE,什么样的该信Kimi。"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.trae = TRAEModel(in_channels, num_classes, c)
        self.kimi = KimiModel(in_channels, num_classes, c)
        # 路由器: 从全局特征决定两个Agent的权重
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 0 + c, c),  # 输入是stem后的通道数
            nn.GELU(),
            nn.Linear(c, 2),
            nn.Softmax(dim=-1),
        )
        # stem (共享,用于路由器提取全局特征)
        self.shared_stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 路由权重
        stem_feat = self.shared_stem(x)
        w = self.router(stem_feat)  # (B, 2)
        w_trae = w[:, 0:1]  # (B, 1)
        w_kimi = w[:, 1:2]
        # 两个Agent并行预测
        logits_trae = self.trae(x)
        logits_kimi = self.kimi(x)
        # 动态加权
        return logits_trae * w_trae + logits_kimi * w_kimi

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 模式5: Sequential-Fuse (特征级融合)
# ============================================================
class SequentialFuseModel(nn.Module):
    """两个Agent各自提取特征,融合MLP做最终决策。
    TRAE特征(条件控制+路由) + Kimi特征(记忆+序列) → 融合 → 分类。"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.trae = TRAEModel(in_channels, num_classes, c)
        self.kimi = KimiModel(in_channels, num_classes, c)
        # 融合层: 两个Agent的特征向量拼接 → MLP
        self.fuse = nn.Sequential(
            nn.Linear(c * 2, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, c), nn.GELU(),
            nn.Linear(c, num_classes),
        )

    def forward(self, x):
        # 提取TRAE特征 (分类头之前的pool特征)
        trae_feat = self.trae.pool(self._trae_forward_features(x)).flatten(1)  # (B, C)
        # 提取Kimi特征
        kimi_feat = self.kimi.pool(self._kimi_forward_features(x)).flatten(1)  # (B, C)
        # 融合
        fused = torch.cat([trae_feat, kimi_feat], dim=-1)  # (B, 2C)
        return self.fuse(fused)

    def _trae_forward_features(self, x):
        """TRAE前向到分类头之前,返回特征图。"""
        x = self.trae.stem(x)
        t_out = self.trae.T(x) + x
        r_out = self.trae.R(t_out) + t_out
        a_out = self.trae.A(r_out) + r_out
        e_out = self.trae.E(t_out, r_out, a_out) + a_out
        return e_out

    def _kimi_forward_features(self, x):
        """Kimi前向到分类头之前,返回特征图。"""
        x = self.kimi.stem(x)
        x = self.kimi.K(x) + x
        x = self.kimi.i1(x) + x
        x = self.kimi.m(x) + x
        x = self.kimi.i2(x) + x
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 训练评估 (复用)
# ============================================================
def get_loaders(batch_size=128, train_subset=6000, val_subset=1000):
    import torchvision
    from torchvision import transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    root = "/data/user/work/torchdata"
    train_full = torchvision.datasets.MNIST(root, train=True, download=True, transform=tf)
    test_full = torchvision.datasets.MNIST(root, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(42)
    ti = torch.randperm(len(train_full), generator=g)[:train_subset].tolist()
    vi = torch.randperm(len(test_full), generator=g)[:val_subset].tolist()
    tl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_full, ti), batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_full, vi), batch_size=256, shuffle=False)
    return tl, vl


def get_addition_loaders(batch_size=128, train_subset=6000, val_subset=1000):
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
        def __init__(self, base, indices, seed=42):
            self.data = [(base[i][0], base[i][1]) for i in indices]
            self.rng = torch.Generator().manual_seed(seed)
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            i = torch.randint(0, len(self.data), (1,), generator=self.rng).item()
            img1, l1 = self.data[idx]
            img2, l2 = self.data[i]
            combined = torch.stack([img1.squeeze(0), img2.squeeze(0)], dim=0)
            return combined, l1 + l2

    tl = torch.utils.data.DataLoader(
        AdditionDataset(train_full, ti), batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(
        AdditionDataset(test_full, vi, seed=99), batch_size=256, shuffle=False)
    return tl, vl


def train_and_eval(model, tl, vl, device, epochs=8, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0
    hist = []
    for ep in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n += 1
        sched.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.numel()
        acc = correct / max(1, total)
        best_acc = max(best_acc, acc)
        avg = total_loss / max(1, n)
        hist.append({"epoch": ep, "loss": round(avg, 4), "acc": round(acc, 4)})
        print(f"    epoch {ep+1}/{epochs}  loss={avg:.4f}  acc={acc:.4f}")
    return best_acc, sum(p.numel() for p in model.parameters()), hist


# ============================================================
# 主实验
# ============================================================
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/workspace/arch_lab/runs/dual_agent"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  TRAE × Kimi 双Agent协作实验 (借鉴扣子多Agent思路)")
    print(f"  5种协作模式对比:")
    print(f"  1. Solo-TRAE  2. Solo-Kimi  3. Parallel-Vote")
    print(f"  4. Parallel-Learned  5. Sequential-Fuse")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    models_cls = [
        ("Solo-TRAE", TRAEModel),
        ("Solo-Kimi", KimiModel),
        ("Parallel-Vote", ParallelVoteModel),
        ("Parallel-Learned", ParallelLearnedModel),
        ("Sequential-Fuse", SequentialFuseModel),
    ]

    # ===== 任务1: MNIST 分类 =====
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}
    for name, MC in models_cls:
        print(f"\n  >> {name}")
        model = MC(in_channels=1, num_classes=10, c=c)
        try:
            _ = model(torch.randn(2, 1, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}")
            import traceback; traceback.print_exc()
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_cls, vl_cls, device, epochs)
        elapsed = time.time() - t0
        cls_results[name] = {
            "acc": round(acc, 4), "params": npar,
            "elapsed": round(elapsed, 1), "hist": hist,
        }
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # ===== 任务2: MNIST 加法推理 =====
    print(f"\n  === 任务2: MNIST 数字加法推理 ===")
    tl_add, vl_add = get_addition_loaders()
    add_results = {}
    for name, MC in models_cls:
        print(f"\n  >> {name}")
        model = MC(in_channels=2, num_classes=19, c=c)
        try:
            _ = model(torch.randn(2, 2, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}")
            import traceback; traceback.print_exc()
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_add, vl_add, device, epochs)
        elapsed = time.time() - t0
        add_results[name] = {
            "acc": round(acc, 4), "params": npar,
            "elapsed": round(elapsed, 1), "hist": hist,
        }
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # ===== 汇总 =====
    print(f"\n{'='*70}")
    print(f"  实验汇总")
    print(f"{'='*70}")
    print(f"\n  ▶ MNIST 分类 (随机基线=10%)")
    for n, d in sorted(cls_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<24s} {d['acc']:>8.4f}  {d['params']:>10,}  {d['elapsed']:>6.1f}s")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<24s} {d['acc']:>8.4f}  {d['params']:>10,}  {d['elapsed']:>6.1f}s")

    summary = {
        "experiment": "dual_agent_collaboration",
        "epochs": epochs, "channels": c, "device": str(device),
        "cls_results": cls_results, "add_results": add_results,
    }
    with open(os.path.join(out_dir, "dual_agent_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/dual_agent_results.json")
    return summary


if __name__ == "__main__":
    run()
