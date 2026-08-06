"""
终极五架构大乱斗 — 冠军大乱斗验证
1. 融合冠军 (TEXT基因, c=32)
2. RL冠军 (DualStream基因, c=32)
3. 混合公式V1 (c=32)
4. Marvis V2 (c=32)
5. Marvis 冠军精确配方 (变通道, 3.8M params)
6. Marvis V2 (c=64, 1.1M params) — 加大通道宽度

运行: python -m arch_lab.ultimate_gauntlet
"""
from __future__ import annotations
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hybrid_ops import HybridModel, FusionChampionModel, RLChampionModel
from .multiscale_ops import MarvisV2Model
from .marvis_champion import MarvisChampionModel


def get_loaders(train_subset=6000, val_subset=1000, batch_size=128):
    import torchvision
    from torchvision import transforms
    ds = torchvision.datasets.MNIST
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    root = "/data/user/work/torchdata"
    train_full = ds(root, train=True, download=True, transform=tf)
    test_full = ds(root, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(42)
    ti = torch.randperm(len(train_full), generator=g)[:train_subset].tolist()
    vi = torch.randperm(len(test_full), generator=g)[:val_subset].tolist()
    tl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_full, ti),
                                     batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.Subset(test_full, vi),
                                     batch_size=256, shuffle=False)
    return tl, vl


def train_and_eval(model, train_loader, val_loader, device, epochs=5, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc = 0.0
    history = []
    for ep in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
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
        avg_loss = total_loss / max(1, n_batches)
        history.append({"epoch": ep, "loss": round(avg_loss, 4), "acc": round(acc, 4)})
        print(f"    epoch {ep+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")
    npar = sum(p.numel() for p in model.parameters())
    return best_acc, npar, history


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5  # 比之前多训2轮

    models = {
        "融合冠军(c32)": {"factory": lambda: FusionChampionModel(1, 10, 32), "desc": "TEXT基因 c=32"},
        "RL冠军(c32)": {"factory": lambda: RLChampionModel(1, 10, 32), "desc": "DualStream基因 c=32"},
        "混合公式V1(c32)": {"factory": lambda: HybridModel(1, 10, 32), "desc": "30/25/20/25 c=32"},
        "MarvisV2(c32)": {"factory": lambda: MarvisV2Model(1, 10, 32), "desc": "MultiScale+WinAttn c=32"},
        "MarvisV2(c64)": {"factory": lambda: MarvisV2Model(1, 10, 64), "desc": "MultiScale+WinAttn c=64 加宽"},
        "Marvis冠军配方": {"factory": lambda: MarvisChampionModel(1, 10, 64), "desc": "精确复刻 变通道 3.8M"},
    }

    print(f"\n{'='*70}")
    print(f"  终极五架构大乱斗 (+Marvis冠军配方)")
    print(f"  设备: {device}  epochs={epochs}  数据集: MNIST")
    print(f"{'='*70}")

    tl, vl = get_loaders()
    results = {}

    for name, cfg in models.items():
        print(f"\n  >> {name}")
        print(f"     {cfg['desc']}")
        model = cfg["factory"]()
        try:
            dummy = torch.randn(2, 1, 28, 28)
            _ = model(dummy)
            print(f"     前向验证 ✓  params={sum(p.numel() for p in model.parameters()):,}")
        except Exception as e:
            print(f"     前向失败: {e}")
            continue

        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl, vl, device, epochs=epochs)
        elapsed = time.time() - t0
        results[name] = {"acc": round(acc, 4), "params": npar, "time": round(elapsed, 1), "hist": hist}
        print(f"     → 最佳准确率: {acc:.4f}  参数: {npar:,}  用时: {elapsed:.1f}s")
        del model

    # 汇总
    print(f"\n{'='*70}")
    print(f"  终极排名")
    print(f"{'='*70}")
    print(f"  {'架构':<20s} {'准确率':>8s} {'参数量':>12s} {'用时':>8s} {'效率':>10s}")
    print(f"  {'-'*62}")
    ranked = sorted(results.items(), key=lambda x: x[1]["acc"], reverse=True)
    for i, (name, data) in enumerate(ranked):
        eff = data["acc"] / (data["params"] / 1000) * 100  # acc per K params * 100
        medal = ["🥇", "🥈", "🥉", "  ", "  ", "  "][i]
        print(f"  {medal} {name:<18s} {data['acc']:>8.4f} {data['params']:>12,} {data['time']:>7.1f}s {eff:>9.2f}")

    # 逐 epoch 对比
    print(f"\n  ▶ 训练曲线 (各 epoch 准确率)")
    print(f"  {'架构':<20s}", end="")
    for ep in range(epochs):
        print(f"  ep{ep+1:>1d}", end="")
    print()
    for name, data in results.items():
        hist = data["hist"]
        print(f"  {name:<20s}", end="")
        for h in hist:
            print(f" {h['acc']:.3f}", end="")
        print()

    # 关键对比
    print(f"\n  ▶ 关键对比")
    if "Marvis冠军配方" in results and "混合公式V1(c32)" in results:
        champ = results["Marvis冠军配方"]
        v1 = results["混合公式V1(c32)"]
        print(f"  Marvis冠军配方 vs 混合公式V1:")
        print(f"    准确率: {champ['acc']:.4f} vs {v1['acc']:.4f} (Δ={champ['acc']-v1['acc']:+.4f})")
        print(f"    参数量: {champ['params']:,} vs {v1['params']:,} ({champ['params']/v1['params']:.1f}x)")

    if "MarvisV2(c64)" in results and "MarvisV2(c32)" in results:
        c64 = results["MarvisV2(c64)"]
        c32 = results["MarvisV2(c32)"]
        print(f"  MarvisV2 c=64 vs c=32:")
        print(f"    准确率: {c64['acc']:.4f} vs {c32['acc']:.4f} (Δ={c64['acc']-c32['acc']:+.4f})")
        print(f"    参数量: {c64['params']:,} vs {c32['params']:,} ({c64['params']/c32['params']:.1f}x)")


if __name__ == "__main__":
    run()
