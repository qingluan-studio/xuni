"""
TuiLi (推理) 五字母架构模型 — T-U-I-L-I

名字就是"推理"的拼音, 用来做推理任务!

T = Twin-path Attention    (双路注意力: 全局注意力+局部卷积并行)
U = U-Shape Skip           (U型跳连: 下采样编码→上采样解码, 对称跳连)
I₁= Inception Multi-scale  (多尺度并行: 1x1/3x3/5x5卷积并行)
L = Linear Attention       (线性注意力: ELU+1核, O(n)复杂度)
I₂= Iterative Refinement   (迭代精炼: 2轮自注意力+门控残差)

T → U → I₁ → L → I₂ 串联, I₂同时汇聚前四路输出。
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# T — Twin-path Attention (双路注意力)
# ============================================================
class TwinPathAttentionBlock(nn.Module):
    """T: 全局路(自注意力) + 局部路(深度卷积) 并行, 门控融合。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 全局路: 多头自注意力
        self.global_norm = nn.LayerNorm(c)
        self.global_attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        # 局部路: 深度可分离卷积
        self.local_norm = nn.BatchNorm2d(c)
        self.local_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 1, bias=False),
        )
        # 门控融合
        self.gate = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, 2))
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 全局路
        g_in = self.global_norm(seq)
        g_out, _ = self.global_attn(g_in, g_in, g_in)
        global_feat = seq + g_out
        # 局部路
        l_out = self.local_conv(self.local_norm(x))
        local_feat = (x + l_out).flatten(2).transpose(1, 2)
        # 门控融合
        g_gap = global_feat.mean(1)
        l_gap = local_feat.mean(1)
        w = F.softmax(self.gate(torch.cat([g_gap, l_gap], dim=-1)), dim=-1)
        fused = global_feat * w[:, 0:1].unsqueeze(1) + local_feat * w[:, 1:2].unsqueeze(1)
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# U — U-Shape Skip (U型跳连)
# ============================================================
class UShapeSkipBlock(nn.Module):
    """U: 下采样编码 → 处理 → 上采样解码, 跳连保留细节。
    创新点: U-Net风格的对称结构, 多尺度特征融合。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 编码: 下采样
        self.enc_norm = nn.LayerNorm(c)
        self.enc_attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        self.downsample = nn.Conv2d(c, c, 2, stride=2, bias=False)
        self.enc_bn = nn.BatchNorm2d(c)
        # 解码: 上采样
        self.upsample = nn.ConvTranspose2d(c, c, 2, stride=2, bias=False)
        self.dec_bn = nn.BatchNorm2d(c)
        # 跳连融合: 编码特征 + 解码特征
        self.fuse = nn.Conv2d(c * 2, c, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 编码: 下采样
        skip = x  # 跳连特征
        down = F.gelu(self.enc_bn(self.downsample(x)))  # (B, C, H/2, W/2)
        # 在下采样分辨率做注意力
        down_seq = down.flatten(2).transpose(1, 2)
        h = self.enc_norm(down_seq)
        a, _ = self.enc_attn(h, h, h)
        down_seq = down_seq + a
        down_refined = down_seq.transpose(1, 2).reshape(B, C, H // 2, W // 2)
        # 解码: 上采样回原分辨率
        up = F.gelu(self.dec_bn(self.upsample(down_refined)))
        # 确保尺寸一致
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=(H, W), mode="bilinear", align_corners=False)
        # 跳连融合
        fused = self.fuse_norm(self.fuse(torch.cat([skip, up], dim=1)))
        # FFN
        seq = fused.flatten(2).transpose(1, 2)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# I₁ — Inception Multi-scale (多尺度并行)
# ============================================================
class InceptionBlock(nn.Module):
    """I₁: 1x1 + 3x3 + 5x5 卷积并行, 多尺度特征提取。
    创新点: 同一层捕获不同感受野的特征。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        c4 = c // 4
        # 分支1: 1x1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(c, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.GELU()
        )
        # 分支2: 1x1 → 3x3
        self.branch2 = nn.Sequential(
            nn.Conv2d(c, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.GELU(),
            nn.Conv2d(c4, c4, 3, padding=1, bias=False), nn.BatchNorm2d(c4), nn.GELU()
        )
        # 分支3: 1x1 → 5x5
        self.branch3 = nn.Sequential(
            nn.Conv2d(c, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.GELU(),
            nn.Conv2d(c4, c4, 5, padding=2, bias=False), nn.BatchNorm2d(c4), nn.GELU()
        )
        # 分支4: 全局池化 → 1x1
        self.branch4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c4, 1, bias=False), nn.BatchNorm2d(c4), nn.GELU()
        )
        # 融合投影
        self.fuse = nn.Conv2d(c4 * 4, c, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(c)
        # 注意力
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b4 = F.interpolate(b4, size=(H, W), mode="bilinear", align_corners=False)
        # 融合
        fused = self.fuse_norm(self.fuse(torch.cat([b1, b2, b3, b4], dim=1)))
        # 注意力
        seq = fused.flatten(2).transpose(1, 2)
        h = self.norm(seq)
        a, _ = self.attn(h, h, h)
        seq = seq + a
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# L — Linear Attention (线性注意力)
# ============================================================
class LinearAttentionBlock(nn.Module):
    """L: 线性注意力 — ELU+1核函数, O(n)复杂度。
    创新点: 高效全局建模, 适合长序列。"""

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
        seq = x.flatten(2).transpose(1, 2)
        h = self.norm1(seq)
        qkv = self.qkv(h).reshape(B, H * W, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # 线性注意力: ELU+1核
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2))
        z = z.clamp(min=1e-6).unsqueeze(-1)
        attn = torch.einsum("bhnd,bhde->bhne", q, kv) / z
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        seq = seq + self.out_proj(attn)
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# I₂ — Iterative Refinement (迭代精炼 + 四路集成)
# ============================================================
class IterativeRefinementBlock(nn.Module):
    """I₂: 2轮自注意力迭代精炼 + 四路集成融合。
    作为最终层, 同时集成前四路输出并迭代精炼。"""

    def __init__(self, c: int, heads: int = 4, n_inputs: int = 4, iters: int = 2):
        super().__init__()
        self.n_inputs = n_inputs
        self.iters = iters
        # 集成路由
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        # 迭代精炼: 每轮一个注意力+FFN
        self.refine_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(iters)])
        self.refine_attns = nn.ModuleList([
            nn.MultiheadAttention(c, num_heads=heads, batch_first=True) for _ in range(iters)
        ])
        self.refine_ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c)) for _ in range(iters)
        ])
        self.refine_ffn_norms = nn.ModuleList([nn.LayerNorm(c) for _ in range(iters)])
        # 门控残差
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())

    def forward(self, *inputs):
        """接收4路输出, 集成后迭代精炼"""
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]
        # 集成路由
        gaps = [s.mean(1) for s in seqs]
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        # 迭代精炼
        for i in range(self.iters):
            h = self.refine_norms[i](merged)
            a, _ = self.refine_attns[i](h, h, h)
            transformed = merged + a
            transformed = transformed + self.refine_ffns[i](self.refine_ffn_norms[i](transformed))
            # 门控残差
            T = self.gate(merged)
            merged = T * transformed + (1 - T) * merged
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# TuiLi 完整模型 (T→U→I₁→L→I₂)
# ============================================================
class TuiLiModel(nn.Module):
    """TuiLi (推理) = T → U → I₁ → L → I₂ 五字母架构"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.T = TwinPathAttentionBlock(c, heads=4)
        self.U = UShapeSkipBlock(c, heads=4)
        self.I1 = InceptionBlock(c, heads=4)
        self.L = LinearAttentionBlock(c, heads=4)
        self.I2 = IterativeRefinementBlock(c, heads=4, n_inputs=4, iters=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        t_out = self.T(x) + x
        u_out = self.U(t_out) + t_out
        i1_out = self.I1(u_out) + u_out
        l_out = self.L(i1_out) + i1_out
        # I₂: 四路集成 + 迭代精炼
        i2_out = self.I2(t_out, u_out, i1_out, l_out) + l_out
        return self.head(self.pool(i2_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 打乱顺序: L→I₁→T→U→I₂
# ============================================================
class ScrambledTuiLiModel(nn.Module):
    """打乱: L → I₁ → T → U → I₂"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.L = LinearAttentionBlock(c, heads=4)
        self.I1 = InceptionBlock(c, heads=4)
        self.T = TwinPathAttentionBlock(c, heads=4)
        self.U = UShapeSkipBlock(c, heads=4)
        self.I2 = IterativeRefinementBlock(c, heads=4, n_inputs=4, iters=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        l_out = self.L(x) + x
        i1_out = self.I1(l_out) + l_out
        t_out = self.T(i1_out) + i1_out
        u_out = self.U(t_out) + t_out
        i2_out = self.I2(l_out, i1_out, t_out, u_out) + u_out
        return self.head(self.pool(i2_out).flatten(1))

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
    out_dir = "/workspace/arch_lab/runs/tuili"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  TuiLi (推理) 五字母架构实验 — 名副其实的推理模型!")
    print(f"  T→U→I₁→L→I₂ (正确) vs L→I₁→T→U→I₂ (打乱)")
    print(f"  两个I各有不同含义! I₂迭代精炼2轮!")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 冒烟测试
    print(f"\n  === 冒烟测试 ===")
    for name, MC in [("TuiLi", TuiLiModel), ("Scrambled", ScrambledTuiLiModel)]:
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
    for name, MC in [("TuiLi (T→U→I₁→L→I₂)", TuiLiModel), ("Scrambled (L→I₁→T→U→I₂)", ScrambledTuiLiModel)]:
        print(f"\n  >> {name}")
        model = MC(in_channels=1, num_classes=10, c=c)
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_cls, vl_cls, device, epochs)
        elapsed = time.time() - t0
        cls_results[name] = {"acc": round(acc, 4), "params": npar, "elapsed": round(elapsed, 1), "hist": hist}
        print(f"     → acc={acc:.4f}  params={npar:,}  time={elapsed:.1f}s")
        del model

    # 任务2: MNIST加法推理
    print(f"\n  === 任务2: MNIST 数字加法推理 (推理模型的主场!) ===")
    tl_add, vl_add = get_addition_loaders()
    add_results = {}
    for name, MC in [("TuiLi (T→U→I₁→L→I₂)", TuiLiModel), ("Scrambled (L→I₁→T→U→I₂)", ScrambledTuiLiModel)]:
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

    summary = {"experiment": "tuili_letter_model", "epochs": epochs, "channels": c,
               "cls_results": cls_results, "add_results": add_results}
    with open(os.path.join(out_dir, "tuili_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/tuili_results.json")
    return summary


if __name__ == "__main__":
    run()
