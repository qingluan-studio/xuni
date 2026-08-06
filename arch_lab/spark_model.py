"""
SPARK 字母架构模型 — 用 S-P-A-R-K 五个字母构建模型

S = Split Attention     (分流注意力: 空间+通道+局部 三路门控融合)
P = Pyramid Multi-scale (金字塔多尺度: 1x1/2x2/4x4 多感受野池化)
A = Adaptive Norm Attn  (自适应归一化: SPADE风格空间条件注入)
R = Recurrent Gating    (循环门控: GRU风格动态遗忘/更新)
K = Knowledge Memory    (知识记忆: 键值记忆库+4路集成+迭代精炼)

S → P → A → R → K 串联，K同时汇聚S/P/A/R四路输出。
寓意: SPARK = 火花/灵感，象征架构融合中的涌现创新。

对比: 正确顺序 S→P→A→R→K vs 打乱顺序 R→P→S→A→K
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# S — Split Attention (分流注意力)
# ============================================================
class SplitAttentionBlock(nn.Module):
    """S: 空间路(MHSA) + 通道路(SE) + 局部路(深度卷积) 三路并行，门控融合。
    创新点: 同一输入分三路提取不同维度的特征，动态门控决定融合比例。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 空间路: 多头自注意力
        self.spatial_norm = nn.LayerNorm(c)
        self.spatial_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 通道路: SE通道注意力
        self.channel_norm = nn.LayerNorm(c)
        self.channel_fc = nn.Sequential(
            nn.Linear(c, c // 4), nn.GELU(), nn.Linear(c // 4, c), nn.Sigmoid()
        )
        # 局部路: 深度可分离卷积
        self.local_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 1, bias=False),
        )
        # 门控融合: 从三路全局特征计算权重
        self.gate = nn.Sequential(nn.Linear(c * 3, c), nn.GELU(), nn.Linear(c, 3))
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 空间路
        h = self.spatial_norm(seq)
        a, _ = self.spatial_attn(h, h, h)
        spatial = seq + a  # (B, HW, C)
        # 通道路 (SE)
        gap = seq.mean(1)  # (B, C)
        ch_w = self.channel_fc(self.channel_norm(seq).mean(1))  # (B, C)
        channel = seq * ch_w.unsqueeze(1)  # (B, HW, C)
        # 局部路
        local = self.local_conv(x).flatten(2).transpose(1, 2)  # (B, HW, C)
        # 门控融合
        gates = F.softmax(self.gate(torch.cat([
            spatial.mean(1), channel.mean(1), local.mean(1)
        ], dim=-1)), dim=-1)  # (B, 3)
        merged = (spatial * gates[:, 0:1].unsqueeze(1) +
                  channel * gates[:, 1:2].unsqueeze(1) +
                  local * gates[:, 2:3].unsqueeze(1))
        # FFN
        merged = merged + self.ffn(self.ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# P — Pyramid Multi-scale (金字塔多尺度)
# ============================================================
class PyramidBlock(nn.Module):
    """P: 1x1/2x2/4x4 自适应池化 → 上采样回原尺寸 → 残差融合 + 注意力。
    创新点: 不同感受野的全局信息并行提取，捕获多尺度上下文。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 金字塔池化: 3个尺度
        self.p1 = nn.AdaptiveAvgPool2d(1)
        self.p2 = nn.AdaptiveAvgPool2d(2)
        self.p4 = nn.AdaptiveAvgPool2d(4)
        # 各尺度投影
        self.proj1 = nn.Conv2d(c, c, 1, bias=False)
        self.proj2 = nn.Conv2d(c, c, 1, bias=False)
        self.proj4 = nn.Conv2d(c, c, 1, bias=False)
        # 融合卷积
        self.fuse = nn.Conv2d(c * 4, c, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(c)
        # 注意力
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 多尺度池化 + 上采样回原尺寸
        p1 = self.proj1(self.p1(x)).expand(B, C, H, W)
        p2 = F.interpolate(self.proj2(self.p2(x)), size=(H, W), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.proj4(self.p4(x)), size=(H, W), mode="bilinear", align_corners=False)
        # 拼接融合: 原始 + 3个尺度
        fused = self.fuse_norm(self.fuse(torch.cat([x, p1, p2, p4], dim=1)))
        fused = fused + x  # 残差
        # 注意力
        seq = fused.flatten(2).transpose(1, 2)
        h = self.norm(seq)
        a, _ = self.attn(h, h, h)
        seq = seq + a
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# A — Adaptive Norm Attention (自适应归一化注意力)
# ============================================================
class AdaptiveNormBlock(nn.Module):
    """A: SPADE风格空间自适应归一化 — 从输入生成空间gamma/beta调制特征。
    创新点: 条件归一化让每个空间位置有独立的归一化参数。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # SPADE: InstanceNorm + 空间条件调制
        self.norm_inst = nn.InstanceNorm2d(c, affine=False)
        self.spade = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.GELU(),
            nn.Conv2d(c, c * 2, 1),  # gamma + beta (空间维度)
        )
        # 注意力
        self.attn_norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # SPADE: 条件归一化
        normed = self.norm_inst(x)
        params = self.spade(x)  # (B, 2C, H, W)
        gamma, beta = params[:, :C], params[:, C:]
        x_spade = normed * (1 + gamma) + beta
        # 注意力
        seq = x_spade.flatten(2).transpose(1, 2)
        h = self.attn_norm(seq)
        a, _ = self.attn(h, h, h)
        seq = seq + a
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# R — Recurrent Gating (循环门控)
# ============================================================
class RecurrentGatingBlock(nn.Module):
    """R: GRU风格门控循环 — 更新门/重置门动态控制信息流。
    创新点: 注意力获取新信息 → GRU门控决定保留多少旧信息、融入多少新信息。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 注意力获取新信息
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # GRU门控: update gate + reset gate + candidate
        self.update_gate = nn.Linear(c * 2, c)
        self.reset_gate = nn.Linear(c * 2, c)
        self.candidate = nn.Linear(c * 2, c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C) — 旧状态
        # 注意力获取新信息
        h = self.norm(seq)
        attn_out, _ = self.attn(h, h, h)
        new_info = attn_out  # (B, HW, C)
        # GRU门控
        combined = torch.cat([seq, new_info], dim=-1)  # (B, HW, 2C)
        z = torch.sigmoid(self.update_gate(combined))   # 更新门: 多少新信息
        r = torch.sigmoid(self.reset_gate(combined))    # 重置门: 旧信息保留比例
        candidate = torch.tanh(self.candidate(torch.cat([r * seq, new_info], dim=-1)))
        seq = (1 - z) * seq + z * candidate             # GRU更新
        # FFN
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# K — Knowledge Memory + 4-way Ensemble (知识记忆集成)
# ============================================================
class KnowledgeMemoryBlock(nn.Module):
    """K: 键值记忆库检索 + 4路集成路由 + 2轮迭代精炼。
    作为最终层，同时集成前4路输出并检索外部知识。"""

    def __init__(self, c: int, heads: int = 4, mem_size: int = 32,
                 n_inputs: int = 4, iters: int = 2):
        super().__init__()
        self.n_inputs = n_inputs
        self.iters = iters
        # 集成路由: 从4路全局特征计算权重
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        # 键值记忆库
        self.mem_keys = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.q_proj = nn.Linear(c, c)
        self.cross_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(c)
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
        """接收n_inputs路输出，集成 + 记忆检索 + 迭代精炼"""
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]
        # 集成路由
        gaps = [s.mean(1) for s in seqs]
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)  # (B, n_inputs)
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        # 记忆检索
        q = self.q_proj(self.cross_norm(merged))
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        mem_out, _ = self.cross_attn(q, k, v)
        merged = merged + mem_out
        # 迭代精炼
        for i in range(self.iters):
            h = self.refine_norms[i](merged)
            a, _ = self.refine_attns[i](h, h, h)
            transformed = merged + a
            transformed = transformed + self.refine_ffns[i](
                self.refine_ffn_norms[i](transformed)
            )
            T = self.gate(merged)
            merged = T * transformed + (1 - T) * merged
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# SPARK 完整模型 (S→P→A→R→K)
# ============================================================
class SparkModel(nn.Module):
    """SPARK = S → P → A → R → K 字母架构模型"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.S = SplitAttentionBlock(c, heads=4)
        self.P = PyramidBlock(c, heads=4)
        self.A = AdaptiveNormBlock(c, heads=4)
        self.R = RecurrentGatingBlock(c, heads=4)
        self.K = KnowledgeMemoryBlock(c, heads=4, mem_size=32, n_inputs=4, iters=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        s_out = self.S(x) + x           # S + 残差
        p_out = self.P(s_out) + s_out   # P + 残差
        a_out = self.A(p_out) + p_out   # A + 残差
        r_out = self.R(a_out) + a_out   # R + 残差
        # K: 四路集成 + 记忆检索 + 迭代精炼
        k_out = self.K(s_out, p_out, a_out, r_out) + r_out  # K + 残差
        return self.head(self.pool(k_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比: 打乱顺序 R→P→S→A→K (K始终在最后做集成)
# ============================================================
class ScrambledSparkModel(nn.Module):
    """打乱: R → P → S → A → K (K始终在最后)"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.R = RecurrentGatingBlock(c, heads=4)
        self.P = PyramidBlock(c, heads=4)
        self.S = SplitAttentionBlock(c, heads=4)
        self.A = AdaptiveNormBlock(c, heads=4)
        self.K = KnowledgeMemoryBlock(c, heads=4, mem_size=32, n_inputs=4, iters=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        r_out = self.R(x) + x
        p_out = self.P(r_out) + r_out
        s_out = self.S(p_out) + p_out
        a_out = self.A(s_out) + s_out
        k_out = self.K(r_out, p_out, s_out, a_out) + a_out
        return self.head(self.pool(k_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 数据加载
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


# ============================================================
# 训练评估
# ============================================================
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
    out_dir = "/workspace/arch_lab/runs/spark"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  SPARK 字母架构实验 (火花/灵感!)")
    print(f"  S→P→A→R→K (正确顺序) vs R→P→S→A→K (打乱顺序)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # ---- 任务1: MNIST 分类 ----
    print(f"\n{'='*70}")
    print(f"  任务1: MNIST 分类 (感知能力)")
    print(f"{'='*70}")

    tl_cls, vl_cls = get_loaders()
    models_cls = {
        "SPARK (S→P→A→R→K)": SparkModel(in_channels=1, num_classes=10, c=c),
        "Scrambled (R→P→S→A→K)": ScrambledSparkModel(in_channels=1, num_classes=10, c=c),
    }

    cls_results = {}
    for name, model in models_cls.items():
        print(f"\n  >> {name}  (params={model.num_parameters()})")
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_cls, vl_cls, device, epochs=epochs)
        elapsed = time.time() - t0
        cls_results[name] = {"acc": acc, "params": npar, "elapsed": round(elapsed, 1), "hist": hist}
        print(f"  => acc={acc:.4f}  params={npar}  time={elapsed:.1f}s")

    # ---- 任务2: MNIST 加法推理 ----
    print(f"\n{'='*70}")
    print(f"  任务2: MNIST 加法推理 (推理能力, 19类)")
    print(f"{'='*70}")

    tl_add, vl_add = get_addition_loaders()
    models_add = {
        "SPARK (S→P→A→R→K)": SparkModel(in_channels=2, num_classes=19, c=c),
        "Scrambled (R→P→S→A→K)": ScrambledSparkModel(in_channels=2, num_classes=19, c=c),
    }

    add_results = {}
    for name, model in models_add.items():
        print(f"\n  >> {name}  (params={model.num_parameters()})")
        t0 = time.time()
        acc, npar, hist = train_and_eval(model, tl_add, vl_add, device, epochs=epochs)
        elapsed = time.time() - t0
        add_results[name] = {"acc": acc, "params": npar, "elapsed": round(elapsed, 1), "hist": hist}
        print(f"  => acc={acc:.4f}  params={npar}  time={elapsed:.1f}s")

    # ---- 保存结果 ----
    results = {
        "experiment": "spark_letter_model",
        "epochs": epochs,
        "channels": c,
        "device": str(device),
        "cls_results": cls_results,
        "add_results": add_results,
    }
    with open(os.path.join(out_dir, "spark_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {out_dir}/spark_results.json")

    # ---- 打印汇总 ----
    print(f"\n{'='*70}")
    print(f"  SPARK 实验汇总")
    print(f"{'='*70}")
    print(f"\n  {'模型':<30} {'分类准确率':>10} {'推理准确率':>10} {'参数量':>10}")
    print(f"  {'-'*65}")
    for name in cls_results:
        cls_acc = cls_results[name]["acc"]
        add_acc = add_results[name]["acc"]
        npar = cls_results[name]["params"]
        print(f"  {name:<30} {cls_acc:>10.4f} {add_acc:>10.4f} {npar:>10}")

    return results


if __name__ == "__main__":
    run()
