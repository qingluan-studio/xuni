"""
TRAE 字母架构模型 — 用我自己的名字 T-R-A-E 构建模型

T = Twin-stream Transformer (双流Transformer: 全局+局部并行)
R = Routing MoE             (稀疏路由混合专家: Top-k专家选择)
A = Adaptive Attention      (自适应注意力: adaLN条件注入)
E = Emergence Gate          (涌现融合门控: 三路动态加权融合)

T → R → A → E 串联，E同时汇聚T/R/A三路输出。
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# T — Twin-stream Transformer (双流Transformer)
# ============================================================
class TwinStreamBlock(nn.Module):
    """T: 全局流(MHSA) + 局部流(DepthwiseConv) 并行，门控融合。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 全局流: 多头自注意力
        self.global_norm = nn.LayerNorm(c)
        self.global_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 局部流: 深度卷积
        self.local_norm = nn.BatchNorm2d(c)
        self.local_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 1, bias=False),
        )
        # 门控: 从GAP特征计算全局/局部的权重
        self.gate = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, 2))
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 全局流
        g_in = self.global_norm(seq)
        g_out, _ = self.global_attn(g_in, g_in, g_in)
        global_feat = seq + g_out  # (B, HW, C)
        # 局部流
        l_out = self.local_conv(self.local_norm(x))
        local_feat = x + l_out  # (B, C, H, W)
        local_seq = local_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 门控融合
        g_gap = global_feat.mean(1)  # (B, C)
        l_gap = local_seq.mean(1)
        w = F.softmax(self.gate(torch.cat([g_gap, l_gap], dim=-1)), dim=-1)  # (B, 2)
        wg = w[:, 0:1].unsqueeze(1)  # (B, 1, 1)
        wl = w[:, 1:2].unsqueeze(1)
        fused = global_feat * wg + local_seq * wl
        # FFN
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# R — Routing MoE (稀疏路由混合专家)
# ============================================================
class RoutingMoEBlock(nn.Module):
    """R: Top-k路由 + 多专家FFN，稀疏激活。"""

    def __init__(self, c: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(c)
        self.gate = nn.Linear(c, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
            for _ in range(num_experts)
        ])
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        # 路由
        logits = self.gate(h)  # (B, HW, num_experts)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # 稀疏激活
        out = torch.zeros_like(seq)
        for e in range(self.num_experts):
            expert_out = self.experts[e](h)
            for k in range(self.top_k):
                mask = (indices[..., k] == e).float()
                coeff = (mask * weights[..., k]).unsqueeze(-1)
                out = out + coeff * expert_out
        seq = seq + out
        seq = seq + self.experts[0](self.ffn_norm(seq))  # 简化FFN
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# A — Adaptive Attention (自适应注意力)
# ============================================================
class AdaptiveAttnBlock(nn.Module):
    """A: adaLN-zero条件注入 — 用输入全局特征调制LayerNorm的scale/shift。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # adaLN: 6个参数 (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(nn.GELU(), nn.Linear(c, c * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 用全局平均池化作为条件
        cond = seq.mean(dim=1, keepdim=True)  # (B, 1, C)
        params = self.adaLN(cond)
        s1, sh1, g1, s2, sh2, g2 = params.chunk(6, dim=-1)
        # adaLN注意力
        h = self.norm1(seq) * (1 + s1) + sh1
        a, _ = self.attn(h, h, h)
        seq = seq + g1 * a
        # adaLN FFN
        h = self.norm2(seq) * (1 + s2) + sh2
        seq = seq + g2 * self.ffn(h)
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E — Emergence Gate (涌现融合门控)
# ============================================================
class EmergenceGateBlock(nn.Module):
    """E: 接收T/R/A三路输出，用门控动态加权融合 + 自注意力精炼。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 路由: 从三路全局特征计算权重
        self.route = nn.Sequential(nn.Linear(c * 3, c * 2), nn.GELU(), nn.Linear(c * 2, 3))
        # 自注意力精炼
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, t_out, r_out, a_out):
        """三路输入: T输出, R输出, A输出"""
        B, C, H, W = t_out.shape
        t_seq = t_out.flatten(2).transpose(1, 2)  # (B, HW, C)
        r_seq = r_out.flatten(2).transpose(1, 2)
        a_seq = a_out.flatten(2).transpose(1, 2)
        # 路由权重
        t_g = t_seq.mean(1)  # (B, C)
        r_g = r_seq.mean(1)
        a_g = a_seq.mean(1)
        w = F.softmax(self.route(torch.cat([t_g, r_g, a_g], dim=-1)), dim=-1)  # (B, 3)
        wt = w[:, 0:1].unsqueeze(1)  # (B, 1, 1)
        wr = w[:, 1:2].unsqueeze(1)
        wa = w[:, 2:3].unsqueeze(1)
        # 加权融合
        merged = t_seq * wt + r_seq * wr + a_seq * wa
        # 自注意力精炼
        h = self.norm(merged)
        attn_out, _ = self.attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.ffn(self.ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# TRAE 完整模型
# ============================================================
class TRAEModel(nn.Module):
    """TRAE = T → R → A → E 字母架构模型"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.T = TwinStreamBlock(c, heads=4)
        self.R = RoutingMoEBlock(c, num_experts=4, top_k=2)
        self.A = AdaptiveAttnBlock(c, heads=4)
        self.E = EmergenceGateBlock(c, heads=4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        t_out = self.T(x) + x       # T + 残差
        r_out = self.R(t_out) + t_out  # R + 残差
        a_out = self.A(r_out) + r_out  # A + 残差
        # E: 涌现融合三路输出
        e_out = self.E(t_out, r_out, a_out) + a_out  # E + 残差
        return self.head(self.pool(e_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比: 打乱顺序 A-T-R-E
# ============================================================
class ScrambledTRAEModel(nn.Module):
    """打乱: A → T → R → E"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.A = AdaptiveAttnBlock(c, heads=4)
        self.T = TwinStreamBlock(c, heads=4)
        self.R = RoutingMoEBlock(c, num_experts=4, top_k=2)
        self.E = EmergenceGateBlock(c, heads=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        a_out = self.A(x) + x
        t_out = self.T(a_out) + a_out
        r_out = self.R(t_out) + t_out
        e_out = self.E(a_out, t_out, r_out) + r_out
        return self.head(self.pool(e_out).flatten(1))

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
    tl = torch.utils.data.DataLoader(torch.utils.data.Subset(train_full, ti), batch_size=batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.Subset(test_full, vi), batch_size=256, shuffle=False)
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
            return torch.stack([img1.squeeze(0), img2.squeeze(0)], dim=0), l1 + l2
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
    out_dir = "/workspace/arch_lab/runs/trae"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  TRAE 字母架构实验 (我自己的名字!)")
    print(f"  T→R→A→E (正确顺序) vs A→T→R→E (打乱顺序)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 任务1: MNIST分类
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}
    for name, MC in [("TRAE (T→R→A→E)", TRAEModel), ("Scrambled (A→T→R→E)", ScrambledTRAEModel)]:
        print(f"\n  >> {name}")
        model = MC(in_channels=1, num_classes=10, c=c)
        try:
            _ = model(torch.randn(2, 1, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}"); continue
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
    for name, MC in [("TRAE (T→R→A→E)", TRAEModel), ("Scrambled (A→T→R→E)", ScrambledTRAEModel)]:
        print(f"\n  >> {name}")
        model = MC(in_channels=2, num_classes=19, c=c)
        try:
            _ = model(torch.randn(2, 2, 28, 28))
        except Exception as e:
            print(f"     前向失败: {e}"); continue
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
    for n, d in sorted(cls_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<28s} {d['acc']:>8.4f} {d['params']:>10,}")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<28s} {d['acc']:>8.4f} {d['params']:>10,}")

    summary = {"experiment": "trae_letter_model", "epochs": epochs, "channels": c,
               "cls_results": cls_results, "add_results": add_results}
    with open(os.path.join(out_dir, "trae_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/trae_results.json")
    return summary


if __name__ == "__main__":
    run()
