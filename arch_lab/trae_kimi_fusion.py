"""
TRAE × Kimi 融合架构 — 8字母终极融合

TRAE 组件:
  T = Twin-stream Transformer (双流: 全局注意力 + 局部卷积)
  R = Routing MoE (稀疏路由混合专家)
  A = Adaptive Attention (adaLN条件注入)
  E = Emergence Gate (三路动态融合)

Kimi 组件:
  K = Key-Value Memory (键值记忆检索)
  i = Interaction Fusion (空间+通道交叉注意力)
  m = Mamba SSM (选择性状态空间)
  i = Inference Iteration (迭代精炼)

融合设计 — "TRAE-Kimi" (TK-Ri-Am-Ei):
  TK: Twin-stream Key Memory = TRAE的T双流 + Kimi的K记忆检索
     全局路用记忆增强注意力(Query→记忆库检索), 局部路深度卷积, 门控融合

  Ri: Routing Interaction = TRAE的R路由 + Kimi的i交互
     每个专家做空间-通道交叉注意力, Top-k路由稀疏激活

  Am: Adaptive Mamba = TRAE的A自适应 + Kimi的m状态空间
     adaLN条件控制Mamba的门控和状态更新

  Ei: Emergence Inference = TRAE的E涌现门控 + Kimi的i迭代精炼
     三路门控融合(TK/Ri/Am) → 2轮迭代自注意力精炼

  TK → Ri → Am → Ei 串联
  Ei 同时汇聚 TK/Ri/Am 三路输出 (跨层跳连)
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TK — Twin-stream Key Memory (TRAE的T + Kimi的K)
# ============================================================
class TwinKeyBlock(nn.Module):
    """TK: 全局路(MHSA + 记忆库检索) + 局部路(深度卷积), 门控融合。
    融合点: TRAE双流架构的global路注入Kimi的键值记忆检索。"""

    def __init__(self, c: int, heads: int = 4, mem_size: int = 32):
        super().__init__()
        # 全局路: 自注意力 + 记忆检索
        self.global_norm = nn.LayerNorm(c)
        self.global_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # Kimi 记忆库
        self.q_proj = nn.Linear(c, c)
        self.mem_keys = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 记忆融合门控
        self.mem_gate = nn.Linear(c * 2, c)
        # 局部路: 深度卷积
        self.local_norm = nn.BatchNorm2d(c)
        self.local_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 1, bias=False),
        )
        # 双流门控融合
        self.gate = nn.Sequential(nn.Linear(c * 2, c), nn.GELU(), nn.Linear(c, 2))
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 全局路: 自注意力
        g_in = self.global_norm(seq)
        g_out, _ = self.global_attn(g_in, g_in, g_in)
        global_feat = seq + g_out  # (B, HW, C)
        # 记忆检索: Query从全局特征投影, Key/Value从记忆库
        q = self.q_proj(self.global_norm(global_feat))
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        mem_out, _ = self.mem_attn(q, k, v)
        # 记忆门控融合
        global_feat = global_feat + self.mem_gate(
            torch.cat([global_feat, mem_out], dim=-1)
        )
        # 局部路
        l_out = self.local_conv(self.local_norm(x))
        local_feat = (x + l_out).flatten(2).transpose(1, 2)  # (B, HW, C)
        # 双流门控融合
        g_gap = global_feat.mean(1)
        l_gap = local_feat.mean(1)
        w = F.softmax(self.gate(torch.cat([g_gap, l_gap], dim=-1)), dim=-1)
        wg = w[:, 0:1].unsqueeze(1)
        wl = w[:, 1:2].unsqueeze(1)
        fused = global_feat * wg + local_feat * wl
        # FFN
        fused = fused + self.ffn(self.ffn_norm(fused))
        return fused.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# Ri — Routing Interaction (TRAE的R + Kimi的i)
# ============================================================
class RoutingInteractionBlock(nn.Module):
    """Ri: Top-k路由 + 每个专家做空间-通道交叉注意力。
    融合点: TRAE的MoE路由机制 + Kimi的空间/通道交互融合。"""

    def __init__(self, c: int, heads: int = 4, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm = nn.LayerNorm(c)
        self.gate = nn.Linear(c, num_experts)
        # 每个专家: 空间注意力 + 通道注意力 + 交叉
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                "spatial_attn": nn.MultiheadAttention(c, heads, batch_first=True),
                "channel_attn": nn.MultiheadAttention(c, heads, batch_first=True),
                "cross_s2c": nn.MultiheadAttention(c, heads, batch_first=True),
                "fuse": nn.Linear(c * 2, c),
            }) for _ in range(num_experts)
        ])
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        # 路由
        logits = self.gate(h)  # (B, HW, num_experts)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        # 稀疏激活: 每个专家做空间-通道交互
        out = torch.zeros_like(seq)
        for e in range(self.num_experts):
            expert = self.experts[e]
            # 空间注意力
            s_out, _ = expert["spatial_attn"](h, h, h)
            spatial = h + s_out
            # 通道注意力 (在C维度做)
            ch_in = h  # (B, HW, C), 对C做norm
            c_out, _ = expert["channel_attn"](ch_in, ch_in, ch_in)
            channel = h + c_out
            # 交叉: 空间→通道
            cross_a, _ = expert["cross_s2c"](spatial, channel, channel)
            expert_out = expert["fuse"](torch.cat([spatial + cross_a, channel], dim=-1))
            for k in range(self.top_k):
                mask = (indices[..., k] == e).float()
                coeff = (mask * weights[..., k]).unsqueeze(-1)
                out = out + coeff * expert_out
        seq = seq + out
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# Am — Adaptive Mamba (TRAE的A + Kimi的m)
# ============================================================
class AdaptiveMambaBlock(nn.Module):
    """Am: adaLN条件注入控制Mamba SSM的门控和状态更新。
    融合点: TRAE的adaLN自适应归一化 + Kimi的Mamba选择性状态空间。"""

    def __init__(self, c: int, state_size: int = 8, heads: int = 4):
        super().__init__()
        self.state_size = state_size
        # adaLN: 用全局条件生成 scale/shift/gate
        self.adaLN = nn.Sequential(nn.GELU(), nn.Linear(c, c * 6))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        # Mamba SSM
        self.in_proj = nn.Linear(c, c * 2)
        self.dt_proj = nn.Linear(c, c)
        self.A_log = nn.Parameter(torch.randn(c, state_size) * 0.01)
        self.D = nn.Parameter(torch.ones(c))
        self.out_proj = nn.Linear(c, c)
        # 归一化 (adaLN控制)
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # adaLN条件: 全局平均池化
        cond = seq.mean(dim=1, keepdim=True)  # (B, 1, C)
        params = self.adaLN(cond)
        s1, sh1, g1, s2, sh2, g2 = params.chunk(6, dim=-1)
        # adaLN Mamba
        h = self.norm1(seq) * (1 + s1) + sh1
        # Mamba SSM
        proj = self.in_proj(h)
        gate, x_in = proj.chunk(2, dim=-1)
        dt = torch.sigmoid(self.dt_proj(x_in))
        A = -torch.exp(self.A_log)
        state = torch.zeros(B, C, self.state_size, device=x.device)
        out = torch.zeros_like(seq)
        for t in range(seq.shape[1]):
            x_t = x_in[:, t, :]
            dt_t = dt[:, t, :]
            state = state * torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) + \
                    x_t.unsqueeze(-1) * dt_t.unsqueeze(-1)
            out[:, t, :] = (state * self.D.unsqueeze(0).unsqueeze(-1)).sum(-1)
        out = out * torch.sigmoid(gate)
        seq = seq + g1 * self.out_proj(out)
        # adaLN FFN
        h = self.norm2(seq) * (1 + s2) + sh2
        seq = seq + g2 * self.ffn(h)
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# Ei — Emergence Inference (TRAE的E + Kimi的i)
# ============================================================
class EmergenceInferenceBlock(nn.Module):
    """Ei: 三路门控融合(TK/Ri/Am) → 2轮迭代自注意力精炼。
    融合点: TRAE的涌现门控动态加权 + Kimi的推理迭代回路。"""

    def __init__(self, c: int, heads: int = 4, iters: int = 2):
        super().__init__()
        self.iters = iters
        # 三路门控路由
        self.route = nn.Sequential(
            nn.Linear(c * 3, c * 2), nn.GELU(), nn.Linear(c * 2, 3)
        )
        # 迭代自注意力
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.norm2 = nn.LayerNorm(c)
        # 迭代回路门控
        self.loop_gate = nn.Linear(c * 2, c)

    def forward(self, tk_out, ri_out, am_out):
        """三路输入: TK输出, Ri输出, Am输出"""
        B, C, H, W = tk_out.shape
        tk_seq = tk_out.flatten(2).transpose(1, 2)
        ri_seq = ri_out.flatten(2).transpose(1, 2)
        am_seq = am_out.flatten(2).transpose(1, 2)
        # 三路门控
        tk_g = tk_seq.mean(1)
        ri_g = ri_seq.mean(1)
        am_g = am_seq.mean(1)
        w = F.softmax(
            self.route(torch.cat([tk_g, ri_g, am_g], dim=-1)), dim=-1
        )  # (B, 3)
        wtk = w[:, 0:1].unsqueeze(1)
        wri = w[:, 1:2].unsqueeze(1)
        wam = w[:, 2:3].unsqueeze(1)
        merged = tk_seq * wtk + ri_seq * wri + am_seq * wam
        # 迭代精炼
        for _ in range(self.iters):
            h = self.norm1(merged)
            a, _ = self.attn(h, h, h)
            merged = merged + a
            h = self.norm2(merged)
            ff = self.ffn(h)
            # 回路门控
            merged = self.loop_gate(
                torch.cat([merged, ff], dim=-1)
            )
        return merged.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# TRAE-Kimi 融合完整模型
# ============================================================
class TRAEKimiModel(nn.Module):
    """TRAE × Kimi = TK → Ri → Am → Ei 八字母融合架构"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.TK = TwinKeyBlock(c, heads=4, mem_size=32)
        self.Ri = RoutingInteractionBlock(c, heads=4, num_experts=4, top_k=2)
        self.Am = AdaptiveMambaBlock(c, state_size=8, heads=4)
        self.Ei = EmergenceInferenceBlock(c, heads=4, iters=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        tk_out = self.TK(x) + x          # TK + 残差
        ri_out = self.Ri(tk_out) + tk_out  # Ri + 残差
        am_out = self.Am(ri_out) + ri_out  # Am + 残差
        # Ei: 三路涌现融合 + 迭代精炼
        ei_out = self.Ei(tk_out, ri_out, am_out) + am_out
        return self.head(self.pool(ei_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比: 单独的 TRAE 和 Kimi (用于横向比较)
# ============================================================
from trae_model import TRAEModel
from kimi_model import KimiModel


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
        torch.utils.data.Subset(train_full, ti), batch_size=batch_size, shuffle=True
    )
    vl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_full, vi), batch_size=256, shuffle=False
    )
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
        AdditionDataset(train_full, ti), batch_size=batch_size, shuffle=True
    )
    vl = torch.utils.data.DataLoader(
        AdditionDataset(test_full, vi, seed=99), batch_size=256, shuffle=False
    )
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
    out_dir = "/workspace/arch_lab/runs/trae_kimi"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  TRAE × Kimi 8字母终极融合实验")
    print(f"  TRAE(T→R→A→E) + Kimi(K→i→m→i) = TK→Ri→Am→Ei")
    print(f"  对比: 单独TRAE / 单独Kimi / 融合模型")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    models_cls = [
        ("TRAE (T→R→A→E)", TRAEModel),
        ("Kimi (K→i→m→i)", KimiModel),
        ("TRAE×Kimi (TK→Ri→Am→Ei)", TRAEKimiModel),
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
        print(f"  {n:<30s} {d['acc']:>8.4f}  {d['params']:>10,}")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<30s} {d['acc']:>8.4f}  {d['params']:>10,}")

    summary = {
        "experiment": "trae_kimi_fusion",
        "epochs": epochs, "channels": c, "device": str(device),
        "cls_results": cls_results, "add_results": add_results,
    }
    with open(os.path.join(out_dir, "trae_kimi_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/trae_kimi_results.json")
    return summary


if __name__ == "__main__":
    run()
