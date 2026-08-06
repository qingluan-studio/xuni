"""
Kimi 字母架构模型 — 用 K-i-m-i 四个字母构建一个完整模型

K = Key-Value Memory  (键值记忆: 检索增强注意力)
i = Interaction        (交互融合: 双路交叉注意力)
m = Mamba SSM          (选择性序列建模)
i = Inference Iteration(推理迭代: 自注意力精炼+回路)

K → i → m → i 串联，各模块间残差连接。
"""
from __future__ import annotations
import json, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# K — Key-Value Memory (键值记忆模块)
# ============================================================
class KeyMemoryBlock(nn.Module):
    """K: 可学习的键值记忆库 + 交叉注意力检索。
    输入作为Query，从记忆库中检索相关信息，增强表征。"""

    def __init__(self, c: int, mem_size: int = 32, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.q_proj = nn.Linear(c, c)
        # 可学习的键值记忆库
        self.mem_keys = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.cross_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        q = self.q_proj(h)
        # 扩展记忆库到batch维度
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, k, v)
        seq = seq + attn_out
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# i — Interaction Fusion (交互融合模块)
# ============================================================
class InteractionBlock(nn.Module):
    """i: 双路交互融合 — 将输入分成空间流和通道流，
    交叉注意力后门控融合。模拟多模态交互。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 空间流: 关注空间位置关系 (seq维度是HW, norm在C上)
        self.spatial_norm = nn.LayerNorm(c)
        self.spatial_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 通道流: 关注通道间关系 (转置后seq维度是C, norm在HW上)
        self.channel_norm = nn.LayerNorm(c)  # 对(B,C,HW)在最后一维HW做norm不对, 改成对C做
        self.channel_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 交叉注意力
        self.cross_s2c = nn.MultiheadAttention(c, heads, batch_first=True)
        self.cross_c2s = nn.MultiheadAttention(c, heads, batch_first=True)
        # 门控融合
        self.gate = nn.Linear(c * 2, c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        # 空间流: 原始序列做自注意力 (B, HW, C) — HW是序列长度
        s_in = self.spatial_norm(seq)
        s_out, _ = self.spatial_attn(s_in, s_in, s_in)
        spatial = seq + s_out  # (B, HW, C)

        # 通道流: 对通道做注意力 — 把C当序列长度, embed_dim还是C
        # 方法: (B, HW, C) → 把HW各位置视为独立样本
        # 直接用1x1 conv在通道维度做交互, 等价于通道注意力
        ch_in = seq  # (B, HW, C)
        # 用spatial_norm做归一化 (在C维度)
        ch_normed = self.spatial_norm(ch_in)
        c_out, _ = self.channel_attn(ch_normed, ch_normed, ch_normed)  # (B, HW, C)
        channel = seq + c_out  # (B, HW, C)

        # 交叉注意力: 空间→通道, 通道→空间
        cross_a, _ = self.cross_s2c(self.spatial_norm(spatial), self.spatial_norm(channel), self.spatial_norm(channel))
        cross_b, _ = self.cross_c2s(self.spatial_norm(channel), self.spatial_norm(spatial), self.spatial_norm(spatial))

        # 门控融合
        fused = self.gate(torch.cat([spatial + cross_a, channel + cross_b], dim=-1))
        seq = seq + fused
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# m — Mamba SSM (选择性序列建模)
# ============================================================
class MambaBlock(nn.Module):
    """m: 简化Mamba — 选择性状态空间模型。
    输入相关的门控和状态更新，线性时间复杂度。"""

    def __init__(self, c: int, state_size: int = 8):
        super().__init__()
        self.state_size = state_size
        self.in_proj = nn.Linear(c, c * 2)
        self.dt_proj = nn.Linear(c, c)
        self.A_log = nn.Parameter(torch.randn(c, state_size) * 0.01)
        self.D = nn.Parameter(torch.ones(c))
        self.out_proj = nn.Linear(c, c)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.in_proj(seq)
        gate, x_in = h.chunk(2, dim=-1)
        dt = torch.sigmoid(self.dt_proj(x_in))
        A = -torch.exp(self.A_log)
        # 简化SSM递归
        state = torch.zeros(B, C, self.state_size, device=x.device)
        out = torch.zeros_like(seq)
        for t in range(seq.shape[1]):
            x_t = x_in[:, t, :]
            dt_t = dt[:, t, :]
            state = state * torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) + \
                    x_t.unsqueeze(-1) * dt_t.unsqueeze(-1)
            out[:, t, :] = (state * self.D.unsqueeze(0).unsqueeze(-1)).sum(-1)
        out = out * torch.sigmoid(gate)
        seq = seq + self.out_proj(out)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# i — Inference Iteration (推理迭代)
# ============================================================
class InferenceBlock(nn.Module):
    """i: 自注意力推理精炼 + 1次迭代回路。
    第一轮自注意力后，用输出重新做一次精炼。"""

    def __init__(self, c: int, heads: int = 4, iters: int = 1):
        super().__init__()
        self.iters = iters
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # 迭代回路门控
        self.loop_gate = nn.Linear(c * 2, c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        for _ in range(self.iters + 1):
            h = self.norm1(seq)
            a, _ = self.attn(h, h, h)
            seq = seq + a
            seq = seq + self.ffn(self.norm2(seq))

        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# Kimi 完整模型
# ============================================================
class KimiModel(nn.Module):
    """Kimi = K → i → m → i 字母架构模型"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        # K-i-m-i 串联
        self.K = KeyMemoryBlock(c, mem_size=32, heads=4)
        self.i1 = InteractionBlock(c, heads=4)
        self.m = MambaBlock(c, state_size=8)
        self.i2 = InferenceBlock(c, heads=4, iters=1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.K(x) + x      # K + 残差
        x = self.i1(x) + x     # i + 残差
        x = self.m(x) + x      # m + 残差
        x = self.i2(x) + x     # i + 残差
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比基线: 随机字母排列 (i-K-m-i, m-i-K-i 等)
# ============================================================
class ScrambledKimiModel(nn.Module):
    """打乱顺序的Kimi: i → m → K → i (作为对比)"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.i1 = InteractionBlock(c, heads=4)
        self.m = MambaBlock(c, state_size=8)
        self.K = KeyMemoryBlock(c, mem_size=32, heads=4)
        self.i2 = InferenceBlock(c, heads=4, iters=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.i1(x) + x
        x = self.m(x) + x
        x = self.K(x) + x
        x = self.i2(x) + x
        return self.head(self.pool(x).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 训练评估
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
    train_set = torch.utils.data.Subset(train_full, ti)
    val_set = torch.utils.data.Subset(test_full, vi)
    tl = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False)
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
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            i = torch.randint(0, len(self.data), (1,), generator=self.rng).item()
            img1, l1 = self.data[idx]
            img2, l2 = self.data[i]
            combined = torch.stack([img1.squeeze(0), img2.squeeze(0)], dim=0)
            return combined, l1 + l2

    tl = torch.utils.data.DataLoader(AdditionDataset(train_full, ti), batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(AdditionDataset(test_full, vi, seed=99), batch_size=256, shuffle=False)
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


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = "/workspace/arch_lab/runs/kimi"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  Kimi 字母架构实验")
    print(f"  K→i→m→i (正确顺序) vs i→m→K→i (打乱顺序)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 任务1: MNIST分类
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}

    for name, ModelClass in [("Kimi (K→i→m→i)", KimiModel),
                              ("Scrambled (i→m→K→i)", ScrambledKimiModel)]:
        print(f"\n  >> {name}")
        model = ModelClass(in_channels=1, num_classes=10, c=c)
        try:
            _ = model(torch.randn(2, 1, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}")
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_cls, vl_cls, device, epochs)
        elapsed = time.time() - t0
        cls_results[name] = {"acc": round(acc, 4), "params": npar, "elapsed": round(elapsed, 1), "hist": hist}
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # 任务2: MNIST加法推理
    print(f"\n  === 任务2: MNIST 数字加法推理 ===")
    tl_add, vl_add = get_addition_loaders()
    add_results = {}

    for name, ModelClass in [("Kimi (K→i→m→i)", KimiModel),
                              ("Scrambled (i→m→K→i)", ScrambledKimiModel)]:
        print(f"\n  >> {name}")
        model = ModelClass(in_channels=2, num_classes=19, c=c)
        try:
            _ = model(torch.randn(2, 2, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}")
            continue
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_add, vl_add, device, epochs)
        elapsed = time.time() - t0
        add_results[name] = {"acc": round(acc, 4), "params": npar, "elapsed": round(elapsed, 1), "hist": hist}
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # 汇总
    print(f"\n{'='*70}")
    print(f"  实验汇总")
    print(f"{'='*70}")
    print(f"\n  ▶ MNIST 分类 (随机基线=10%)")
    print(f"  {'架构':<24s} {'准确率':>8s} {'参数量':>10s}")
    for n, d in sorted(cls_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<24s} {d['acc']:>8.4f} {d['params']:>10,}")

    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    print(f"  {'架构':<24s} {'准确率':>8s} {'参数量':>10s}")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<24s} {d['acc']:>8.4f} {d['params']:>10,}")

    summary = {
        "experiment": "kimi_letter_model",
        "epochs": epochs, "channels": c, "device": str(device),
        "cls_results": cls_results, "add_results": add_results,
    }
    with open(os.path.join(out_dir, "kimi_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/kimi_results.json")
    return summary


if __name__ == "__main__":
    run()
