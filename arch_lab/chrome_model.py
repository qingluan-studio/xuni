"""
CHROME 字母架构模型 — 六字母长名挑战!

C = Cross-Attention       (交叉注意力: Q路/KV路分路交叉)
H = Highway Network       (高速网络: 可学习门控跳连)
R = Rotary Attention      (旋转注意力: RoPE位置编码)
O = Octave Attention      (八度注意力: 高频+低频分离)
M = Memory Bank           (记忆库: 可学习键值记忆检索)
E = Ensemble Gate         (集成门控: 汇聚C/H/R/O/M五路输出)

C → H → R → O → M → E 串联，E同时汇聚前五路输出。
"""
from __future__ import annotations
import json, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# C — Cross-Attention (交叉注意力)
# ============================================================
class CrossAttentionBlock(nn.Module):
    """C: 将输入分成Q路和KV路, 做交叉注意力。
    创新点: 同一输入自分成两路, 互相查询, 增强表征多样性。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        # Q路: 深度卷积提取局部特征作为Query
        self.q_proj = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.q_norm = nn.BatchNorm2d(c)
        # KV路: 1x1卷积投影
        self.k_proj = nn.Conv2d(c, c, 1, bias=False)
        self.v_proj = nn.Conv2d(c, c, 1, bias=False)
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(c)
        self.norm_kv = nn.LayerNorm(c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # Q路: 局部卷积
        q_feat = self.q_norm(self.q_proj(x))  # (B, C, H, W)
        q_seq = q_feat.flatten(2).transpose(1, 2)  # (B, HW, C)
        # KV路: 原始特征
        k_seq = self.k_proj(x).flatten(2).transpose(1, 2)
        v_seq = self.v_proj(x).flatten(2).transpose(1, 2)
        # 交叉注意力
        q = self.norm_q(q_seq)
        k = self.norm_kv(k_seq)
        v = self.norm_kv(v_seq)
        attn_out, _ = self.cross_attn(q, k, v)
        seq = q_seq + attn_out
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# H — Highway Network (高速网络)
# ============================================================
class HighwayBlock(nn.Module):
    """H: Highway Network — 学习门控决定多少信息直接通过、多少经过变换。
    创新点: 可学习的跳连比例, 让网络自适应信息流。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 变换路径
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # 门控: 决定transform vs carry的比例
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 变换路径
        h = self.norm1(seq)
        a, _ = self.attn(h, h, h)
        transformed = seq + a
        transformed = transformed + self.ffn(self.norm2(transformed))
        # 门控: T * transform + (1-T) * carry
        T = self.gate(seq)  # (B, HW, C), 值在0-1
        out = T * transformed + (1 - T) * seq
        return out.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# R — Rotary Attention (旋转位置注意力)
# ============================================================
class RotaryAttentionBlock(nn.Module):
    """R: 使用RoPE (Rotary Position Embedding) 的注意力。
    创新点: 旋转位置编码将相对位置信息编码到Q/K中。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = c // heads
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.out_proj = nn.Linear(c, c)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # 预计算旋转频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, H, W):
        """对 (B, heads, HW, head_dim) 应用RoPE"""
        B, h, N, d = x.shape
        # 生成2D位置 (行, 列)
        pos_y = torch.arange(H, device=x.device).float()
        pos_x = torch.arange(W, device=x.device).float()
        # 用行坐标的旋转编码前半, 列坐标旋转后半
        d_half = d // 2
        freqs_y = torch.einsum("i,j->ij", pos_y, self.inv_freq[:d_half//2] if d_half > 2 else self.inv_freq[:1])  # 简化
        # 简化: 用1D序列位置
        positions = torch.arange(N, device=x.device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)  # (N, d/2)
        angles = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d/2)
        # cos/sin
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (1, 1, N, d)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        # 旋转: x_even * cos - x_odd * sin, x_odd * cos + x_even * sin
        x1 = x[..., 0::2]  # 偶数位
        x2 = x[..., 1::2]  # 奇数位
        rotated = torch.stack([x1 * cos[..., 0::2] - x2 * sin[..., 0::2],
                               x1 * sin[..., 1::2] + x2 * cos[..., 1::2]], dim=-1)
        return rotated.flatten(-2)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm1(seq)
        qkv = self.qkv(h).reshape(B, H * W, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 应用RoPE
        q = self._apply_rope(q, H, W)
        k = self._apply_rope(k, H, W)
        # 注意力
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        seq = seq + self.out_proj(attn)
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# O — Octave Attention (八度注意力)
# ============================================================
class OctaveAttentionBlock(nn.Module):
    """O: 将特征分成高频(原始分辨率)和低频(下采样)两路,
    分别做注意力后融合。创新点: 频率分离注意力。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 低频路: 下采样后做注意力
        self.downsample = nn.AvgPool2d(2)
        self.low_norm = nn.LayerNorm(c)
        self.low_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 高频路: 原始分辨率做注意力
        self.high_norm = nn.LayerNorm(c)
        self.high_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 融合
        self.fuse = nn.Conv2d(c * 2, c, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 高频路: 原始分辨率
        h_seq = x.flatten(2).transpose(1, 2)
        h_in = self.high_norm(h_seq)
        h_out, _ = self.high_attn(h_in, h_in, h_in)
        high = h_seq + h_out  # (B, HW, C)
        # 低频路: 下采样
        low_feat = self.downsample(x)  # (B, C, H/2, W/2)
        _, _, Hd, Wd = low_feat.shape
        l_seq = low_feat.flatten(2).transpose(1, 2)
        l_in = self.low_norm(l_seq)
        l_out, _ = self.low_attn(l_in, l_in, l_in)
        low = l_seq + l_out  # (B, HW/4, C)
        # 上采样回原始大小
        low_up = low.transpose(1, 2).reshape(B, C, Hd, Wd)
        low_up = F.interpolate(low_up, size=(H, W), mode="bilinear", align_corners=False)
        # 融合高频+低频
        high_feat = high.transpose(1, 2).reshape(B, C, H, W)
        fused = self.fuse_norm(self.fuse(torch.cat([high_feat, low_up], dim=1)))
        fused_seq = fused.flatten(2).transpose(1, 2)
        fused_seq = fused_seq + self.ffn(self.ffn_norm(fused_seq))
        return fused_seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# M — Memory Bank (记忆库)
# ============================================================
class MemoryBankBlock(nn.Module):
    """M: 可学习的键值记忆库 + 交叉注意力检索。
    输入作为Query, 从记忆库检索相关信息增强表征。"""

    def __init__(self, c: int, mem_size: int = 32, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.q_proj = nn.Linear(c, c)
        # 可学习记忆库
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
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, k, v)
        seq = seq + attn_out
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E — Ensemble Gate (集成门控)
# ============================================================
class EnsembleGateBlock(nn.Module):
    """E: 接收C/H/R/O/M五路输出, 门控动态加权融合 + 自注意力精炼。"""

    def __init__(self, c: int, heads: int = 4, n_inputs: int = 5):
        super().__init__()
        self.n_inputs = n_inputs
        # 路由: 从五路全局特征计算权重
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        # 自注意力精炼
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, *inputs):
        """接收多路输出"""
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]  # 各 (B, HW, C)
        # 路由权重
        gaps = [s.mean(1) for s in seqs]  # 各 (B, C)
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)  # (B, n_inputs)
        # 加权融合
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        # 自注意力精炼
        h = self.norm(merged)
        attn_out, _ = self.attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.ffn(self.ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# CHROME 完整模型 (正确顺序 C→H→R→O→M→E)
# ============================================================
class ChromeModel(nn.Module):
    """CHROME = C → H → R → O → M → E 六字母架构"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.C = CrossAttentionBlock(c, heads=4)
        self.H = HighwayBlock(c, heads=4)
        self.R = RotaryAttentionBlock(c, heads=4)
        self.O = OctaveAttentionBlock(c, heads=4)
        self.M = MemoryBankBlock(c, mem_size=32, heads=4)
        self.E = EnsembleGateBlock(c, heads=4, n_inputs=5)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        c_out = self.C(x) + x
        h_out = self.H(c_out) + c_out
        r_out = self.R(h_out) + h_out
        o_out = self.O(r_out) + r_out
        m_out = self.M(o_out) + o_out
        # E: 集成融合五路输出
        e_out = self.E(c_out, h_out, r_out, o_out, m_out) + m_out
        return self.head(self.pool(e_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 打乱顺序: O→M→C→H→R→E
# ============================================================
class ScrambledChromeModel(nn.Module):
    """打乱: O → M → C → H → R → E"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.O = OctaveAttentionBlock(c, heads=4)
        self.M = MemoryBankBlock(c, mem_size=32, heads=4)
        self.C = CrossAttentionBlock(c, heads=4)
        self.H = HighwayBlock(c, heads=4)
        self.R = RotaryAttentionBlock(c, heads=4)
        self.E = EnsembleGateBlock(c, heads=4, n_inputs=5)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        o_out = self.O(x) + x
        m_out = self.M(o_out) + o_out
        c_out = self.C(m_out) + m_out
        h_out = self.H(c_out) + c_out
        r_out = self.R(h_out) + h_out
        e_out = self.E(o_out, m_out, c_out, h_out, r_out) + r_out
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
    out_dir = "/workspace/arch_lab/runs/chrome"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  CHROME 六字母架构实验 (离谱长名挑战!)")
    print(f"  C→H→R→O→M→E (正确顺序) vs O→M→C→H→R→E (打乱顺序)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 冒烟测试
    print(f"\n  === 冒烟测试 ===")
    for name, MC in [("Chrome", ChromeModel), ("Scrambled", ScrambledChromeModel)]:
        try:
            m = MC(in_channels=1, num_classes=10, c=c)
            _ = m(torch.randn(2, 1, 28, 28))
            print(f"  {name}: OK  params={sum(p.numel() for p in m.parameters()):,}")
            del m
        except Exception as e:
            print(f"  {name}: FAIL - {e}")
            return

    # 任务1: MNIST分类
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}
    for name, MC in [("Chrome (C→H→R→O→M→E)", ChromeModel), ("Scrambled (O→M→C→H→R→E)", ScrambledChromeModel)]:
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
    for name, MC in [("Chrome (C→H→R→O→M→E)", ChromeModel), ("Scrambled (O→M→C→H→R→E)", ScrambledChromeModel)]:
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
        print(f"  {n:<36s} {d['acc']:>8.4f} {d['params']:>10,}")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<36s} {d['acc']:>8.4f} {d['params']:>10,}")

    summary = {"experiment": "chrome_letter_model", "epochs": epochs, "channels": c,
               "cls_results": cls_results, "add_results": add_results}
    with open(os.path.join(out_dir, "chrome_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/chrome_results.json")
    return summary


if __name__ == "__main__":
    run()
