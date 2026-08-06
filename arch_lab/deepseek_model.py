"""
DeepSeek 八字母架构模型 — 终极离谱挑战!

8个字母，4个E各有不同含义 — 真正的"深度求索"

D = Dense Connection       (稠密连接: 密集跳连)
E₁= Efficient Attention    (高效注意力: 线性注意力 O(n))
E₂= Expert Routing         (专家路由: MoE稀疏路由)
P = Position-aware Attn    (位置感知: 可学习位置嵌入)
S = Sparse Window Attn     (稀疏窗口: 局部窗口注意力)
E₃= Emergence Gate         (涌现门控: 多路动态融合)
E₄= Ensemble Refinement    (集成精炼: 残差集成+自注意力)
K = Knowledge Memory       (知识记忆: 记忆库检索+最终集成)

D → E₁ → E₂ → P → S → E₃ → E₄ → K 串联
K 同时汇聚前7路输出 (七路集成!)
"""
from __future__ import annotations
import json, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# D — Dense Connection (稠密连接)
# ============================================================
class DenseConnectionBlock(nn.Module):
    """D: 稠密连接 — 每层输出拼接后投影, 信息流最大化。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 稠密: 输入→变换→拼接→投影
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # 稠密投影: 将变换前后的特征拼接后投影
        self.dense_proj = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 注意力
        h = self.norm1(seq)
        a, _ = self.attn(h, h, h)
        attn_out = seq + a
        # FFN
        ffn_out = attn_out + self.ffn(self.norm2(attn_out))
        # 稠密连接: 拼接变换前和变换后
        dense = self.dense_proj(torch.cat([seq, ffn_out], dim=-1))
        return dense.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E₁ — Efficient Attention (高效线性注意力)
# ============================================================
class EfficientAttentionBlock(nn.Module):
    """E₁: 线性注意力 — 用核函数将注意力复杂度从O(n²)降到O(n)。
    创新点: ELU+1核函数, 高效全局建模。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = c // heads
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.out_proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm1(seq)
        qkv = self.qkv(h).reshape(B, H * W, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 线性注意力: ELU+1 核函数
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        # O(n) 计算: (Q·K^T)·V → Q·(K^T·V)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (B, heads, d, d)
        z = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))  # (B, heads, HW)
        z = z.clamp(min=1e-6).unsqueeze(-1)
        attn = torch.einsum("bhnd,bhde->bhne", q, kv) / z  # (B, heads, HW, d)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        seq = seq + self.out_proj(attn)
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E₂ — Expert Routing (专家路由 MoE)
# ============================================================
class ExpertRoutingBlock(nn.Module):
    """E₂: Top-k路由 + 多专家FFN, 稀疏激活。"""

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
        self.shared_ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        # 路由
        logits = self.gate(h)
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
        seq = seq + self.shared_ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# P — Position-aware Attention (位置感知注意力)
# ============================================================
class PositionAwareBlock(nn.Module):
    """P: 可学习位置嵌入 + 注意力。
    创新点: 显式位置编码注入注意力。"""

    def __init__(self, c: int, heads: int = 4, max_len: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 可学习位置嵌入
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, c) * 0.02)
        self.pos_proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        N = seq.shape[1]
        # 注入位置信息
        pos = self.pos_proj(self.pos_embed[:, :N, :])
        seq_pos = seq + pos
        # 注意力
        h = self.norm1(seq_pos)
        a, _ = self.attn(h, h, h)
        seq = seq + a
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# S — Sparse Window Attention (稀疏窗口注意力)
# ============================================================
class SparseWindowBlock(nn.Module):
    """S: 局部窗口注意力 — 只在窗口内做注意力, 大幅降低计算量。
    创新点: 窗口内密集, 窗口间稀疏。"""

    def __init__(self, c: int, heads: int = 4, window_size: int = 4):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        # 确保H, W能被窗口整除
        Hp = (H + ws - 1) // ws * ws
        Wp = (W + ws - 1) // ws * ws
        if Hp != H or Wp != W:
            x = F.pad(x, (0, Wp - W, 0, Hp - H))
        _, _, Hp, Wp = x.shape
        # 分窗口
        windows = x.unfold(2, ws, ws).unfold(3, ws, ws)  # (B, C, nH, nW, ws, ws)
        nH, nW = windows.shape[2], windows.shape[3]
        windows = windows.contiguous().view(B, C, nH * nW, ws * ws)  # (B, C, nWindows, ws²)
        windows = windows.permute(0, 2, 3, 1)  # (B, nWindows, ws², C)
        Bw, Nw, Lw, Cw = windows.shape
        seq = windows.reshape(Bw * Nw, Lw, Cw)
        # 窗口内注意力
        h = self.norm1(seq)
        a, _ = self.attn(h, h, h)
        seq = seq + a
        seq = seq + self.ffn(self.norm2(seq))
        # 还原
        seq = seq.reshape(B, nH, nW, ws, ws, C)
        seq = seq.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = seq.reshape(B, C, Hp, Wp)
        if Hp != H or Wp != W:
            out = out[:, :, :H, :W]
        return out


# ============================================================
# E₃ — Emergence Gate (涌现门控)
# ============================================================
class EmergenceGateBlock(nn.Module):
    """E₃: 接收多路输出, 门控动态加权融合 + 自注意力精炼。"""

    def __init__(self, c: int, heads: int = 4, n_inputs: int = 5):
        super().__init__()
        self.n_inputs = n_inputs
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, *inputs):
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]
        gaps = [s.mean(1) for s in seqs]
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        h = self.norm(merged)
        attn_out, _ = self.attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.ffn(self.ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E₄ — Ensemble Refinement (集成精炼)
# ============================================================
class EnsembleRefinementBlock(nn.Module):
    """E₄: 残差集成 + 自注意力精炼。
    在E₃融合基础上做进一步精炼。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 多尺度精炼
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 门控残差
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        # 自注意力
        h = self.norm1(seq)
        a, _ = self.attn(h, h, h)
        transformed = seq + a
        transformed = transformed + self.ffn(self.norm2(transformed))
        # 门控残差: 自适应决定精炼程度
        T = self.gate(seq)
        out = T * transformed + (1 - T) * seq
        return out.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# K — Knowledge Memory (知识记忆 + 最终集成)
# ============================================================
class KnowledgeMemoryBlock(nn.Module):
    """K: 可学习记忆库检索 + 七路集成融合。
    作为最终层, 同时做记忆检索和集成所有前层输出。"""

    def __init__(self, c: int, mem_size: int = 32, heads: int = 4, n_inputs: int = 7):
        super().__init__()
        self.n_inputs = n_inputs
        # 记忆库
        self.mem_keys = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 集成路由
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        # 精炼
        self.refine_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, *inputs):
        """接收7路输出"""
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]
        # 集成路由
        gaps = [s.mean(1) for s in seqs]
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        # 记忆库检索
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        mem_out, _ = self.mem_attn(merged, k, v)
        merged = merged + mem_out
        # 精炼
        h = self.norm(merged)
        attn_out, _ = self.refine_attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.ffn(self.ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# DeepSeek 完整模型 (D→E₁→E₂→P→S→E₃→E₄→K)
# ============================================================
class DeepSeekModel(nn.Module):
    """DeepSeek = D → E₁ → E₂ → P → S → E₃ → E₄ → K 八字母架构"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.D = DenseConnectionBlock(c, heads=4)
        self.E1 = EfficientAttentionBlock(c, heads=4)
        self.E2 = ExpertRoutingBlock(c, num_experts=4, top_k=2)
        self.P = PositionAwareBlock(c, heads=4, max_len=64)
        self.S = SparseWindowBlock(c, heads=4, window_size=4)
        self.E3 = EmergenceGateBlock(c, heads=4, n_inputs=5)  # 汇聚 D,E1,E2,P,S
        self.E4 = EnsembleRefinementBlock(c, heads=4)
        self.K = KnowledgeMemoryBlock(c, mem_size=32, heads=4, n_inputs=7)  # 汇聚全部7路

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        d = self.D(x) + x
        e1 = self.E1(d) + d
        e2 = self.E2(e1) + e1
        p = self.P(e2) + e2
        s = self.S(p) + p
        # E₃: 涌现融合前5路
        e3 = self.E3(d, e1, e2, p, s) + s
        # E₄: 精炼
        e4 = self.E4(e3) + e3
        # K: 知识记忆 + 七路集成
        k = self.K(d, e1, e2, p, s, e3, e4) + e4
        return self.head(self.pool(k).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 打乱顺序: P → S → E₁ → D → E₂ → E₃ → E₄ → K
# ============================================================
class ScrambledDeepSeekModel(nn.Module):
    """打乱: P → S → E₁ → D → E₂ → E₃ → E₄ → K"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.P = PositionAwareBlock(c, heads=4, max_len=64)
        self.S = SparseWindowBlock(c, heads=4, window_size=4)
        self.E1 = EfficientAttentionBlock(c, heads=4)
        self.D = DenseConnectionBlock(c, heads=4)
        self.E2 = ExpertRoutingBlock(c, num_experts=4, top_k=2)
        self.E3 = EmergenceGateBlock(c, heads=4, n_inputs=5)
        self.E4 = EnsembleRefinementBlock(c, heads=4)
        self.K = KnowledgeMemoryBlock(c, mem_size=32, heads=4, n_inputs=7)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        p = self.P(x) + x
        s = self.S(p) + p
        e1 = self.E1(s) + s
        d = self.D(e1) + e1
        e2 = self.E2(d) + d
        e3 = self.E3(p, s, e1, d, e2) + e2
        e4 = self.E4(e3) + e3
        k = self.K(p, s, e1, d, e2, e3, e4) + e4
        return self.head(self.pool(k).flatten(1))

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
    out_dir = "/workspace/arch_lab/runs/deepseek"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  DeepSeek 八字母架构实验 (终极离谱挑战!)")
    print(f"  D→E₁→E₂→P→S→E₃→E₄→K (正确) vs P→S→E₁→D→E₂→E₃→E₄→K (打乱)")
    print(f"  4个E各有不同含义! 七路集成!")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 冒烟测试
    print(f"\n  === 冒烟测试 ===")
    for name, MC in [("DeepSeek", DeepSeekModel), ("Scrambled", ScrambledDeepSeekModel)]:
        try:
            m = MC(in_channels=1, num_classes=10, c=c)
            _ = m(torch.randn(2, 1, 28, 28))
            print(f"  {name}: OK  params={sum(p.numel() for p in m.parameters()):,}")
            del m
        except Exception as e:
            print(f"  {name}: FAIL - {e}")
            import traceback; traceback.print_exc()
            return

    # 任务1: MNIST分类
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}
    for name, MC in [("DeepSeek (D→E₁→E₂→P→S→E₃→E₄→K)", DeepSeekModel),
                      ("Scrambled (P→S→E₁→D→E₂→E₃→E₄→K)", ScrambledDeepSeekModel)]:
        print(f"\n  >> {name}")
        model = MC(in_channels=1, num_classes=10, c=c)
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
    for name, MC in [("DeepSeek (D→E₁→E₂→P→S→E₃→E₄→K)", DeepSeekModel),
                      ("Scrambled (P→S→E₁→D→E₂→E₃→E₄→K)", ScrambledDeepSeekModel)]:
        print(f"\n  >> {name}")
        model = MC(in_channels=2, num_classes=19, c=c)
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
        print(f"  {n:<48s} {d['acc']:>8.4f} {d['params']:>10,}")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<48s} {d['acc']:>8.4f} {d['params']:>10,}")

    summary = {"experiment": "deepseek_letter_model", "epochs": epochs, "channels": c,
               "cls_results": cls_results, "add_results": add_results}
    with open(os.path.join(out_dir, "deepseek_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/deepseek_results.json")
    return summary


if __name__ == "__main__":
    run()
