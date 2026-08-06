"""
Agent 五字母架构模型 — A-G-E-N-T

A = Axial Attention     (轴向注意力: 行+列分离注意力)
G = Gated Residual      (门控残差: 可学习跳连比例)
E = External Memory     (外部记忆: 可学习记忆库检索)
N = Non-local Block     (非局部块: 空间长距离依赖)
T = Transformer XL      (递归记忆: 段级递归+相对位置)

A → G → E → N → T 串联，T同时汇聚前四路输出。
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# A — Axial Attention (轴向注意力)
# ============================================================
class AxialAttentionBlock(nn.Module):
    """A: 将2D注意力分解为行注意力+列注意力, 降低复杂度。
    创新点: 轴向分解, O(H·W·C) 而非 O(H²·W²·C)。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 行注意力 (沿宽度方向)
        self.row_norm = nn.LayerNorm(c)
        self.row_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 列注意力 (沿高度方向)
        self.col_norm = nn.LayerNorm(c)
        self.col_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 行注意力: 每行做自注意力 (B*H, W, C)
        row_seq = x.permute(0, 2, 3, 1).reshape(B * H, W, C)
        h = self.row_norm(row_seq)
        a, _ = self.row_attn(h, h, h)
        row_seq = row_seq + a
        row_feat = row_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # 列注意力: 每列做自注意力 (B*W, H, C)
        col_seq = row_feat.permute(0, 3, 2, 1).reshape(B * W, H, C)
        h = self.col_norm(col_seq)
        a, _ = self.col_attn(h, h, h)
        col_seq = col_seq + a
        col_feat = col_seq.reshape(B, W, H, C).permute(0, 3, 2, 1)
        # FFN
        seq = col_feat.flatten(2).transpose(1, 2)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# G — Gated Residual (门控残差)
# ============================================================
class GatedResidualBlock(nn.Module):
    """G: 可学习门控决定变换路径和跳连路径的比例。
    创新点: 自适应残差, 每个位置独立决定信息流。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 变换路径: 注意力 + FFN
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        # 门控: 从输入特征计算变换/跳连的比例
        self.gate = nn.Sequential(nn.Linear(c, c // 2), nn.GELU(), nn.Linear(c // 2, c), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 变换路径
        h = self.norm1(seq)
        a, _ = self.attn(h, h, h)
        transformed = seq + a
        transformed = transformed + self.ffn(self.norm2(transformed))
        # 门控: T * transformed + (1-T) * identity
        T = self.gate(seq)
        out = T * transformed + (1 - T) * seq
        return out.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# E — External Memory (外部记忆)
# ============================================================
class ExternalMemoryBlock(nn.Module):
    """E: 可学习的键值记忆库, 输入作为Query检索外部知识。
    创新点: 将"外部知识"编码为参数化记忆, 通过交叉注意力检索。"""

    def __init__(self, c: int, mem_size: int = 32, heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.q_proj = nn.Linear(c, c)
        # 可学习记忆库
        self.mem_keys = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(1, mem_size, c) * 0.02)
        # 交叉注意力检索
        self.cross_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 记忆更新: 用检索结果调制记忆 (类似Memory Network)
        self.mem_update = nn.Linear(c * 2, c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        q = self.q_proj(h)
        k = self.mem_keys.expand(B, -1, -1)
        v = self.mem_vals.expand(B, -1, -1)
        # 检索
        mem_out, _ = self.cross_attn(q, k, v)
        seq = seq + mem_out
        # FFN
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# N — Non-local Block (非局部块)
# ============================================================
class NonLocalBlock(nn.Module):
    """N: 非局部均值滤波 — 计算所有位置间的相似度, 捕获长距离依赖。
    创新点: 不受局部窗口限制, 全局空间依赖建模。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = c // heads
        # Q/K/V 投影
        self.norm = nn.LayerNorm(c)
        self.theta = nn.Linear(c, c, bias=False)  # Q
        self.phi = nn.Linear(c, c, bias=False)    # K
        self.g = nn.Linear(c, c, bias=False)      # V
        self.out_proj = nn.Linear(c, c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        # 多头非局部注意力
        q = self.theta(h).reshape(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.phi(h).reshape(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.g(h).reshape(B, H * W, self.heads, self.head_dim).permute(0, 2, 1, 3)
        # 非局部: 全位置相似度
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, H * W, C)
        seq = seq + self.out_proj(attn)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# T — Transformer XL (递归记忆)
# ============================================================
class TransformerXLBlock(nn.Module):
    """T: 段级递归 + 相对位置编码。
    创新点: 缓存上一段的隐状态, 与当前段拼接做注意力, 实现长程记忆。"""

    def __init__(self, c: int, heads: int = 4, mem_len: int = 16):
        super().__init__()
        self.heads = heads
        self.head_dim = c // heads
        self.mem_len = mem_len
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 相对位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, mem_len * 2, c) * 0.02)
        self.pos_proj = nn.Linear(c, c)
        # 缓存
        self.register_buffer("memory", torch.zeros(1, mem_len, c), persistent=False)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 获取记忆 (上一段的缓存)
        mem = self.memory.expand(B, -1, -1)  # (B, mem_len, C)
        # 拼接记忆和当前序列
        kv = torch.cat([mem, seq], dim=1)  # (B, mem_len + HW, C)
        # 相对位置注入
        N = kv.shape[1]
        pos = self.pos_proj(self.pos_embed[:, :N, :])
        kv = kv + pos
        # 注意力: query=当前序列, key/value=记忆+当前
        h = self.norm1(seq)
        kv_h = self.norm1(kv)
        a, _ = self.attn(h, kv_h, kv_h)
        seq = seq + a
        seq = seq + self.ffn(self.norm2(seq))
        # 更新记忆: 保存当前序列的最后mem_len个位置
        with torch.no_grad():
            cur = x.flatten(2).transpose(1, 2)  # (B, HW, C)
            if cur.shape[1] >= self.mem_len:
                self.memory = cur[:, -self.mem_len:, :].detach().mean(0, keepdim=True)
            else:
                self.memory = cur[:, :self.mem_len, :].detach().mean(0, keepdim=True)
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# Agent 完整模型 (A→G→E→N→T)
# ============================================================
class AgentModel(nn.Module):
    """Agent = A → G → E → N → T 五字母架构"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.A = AxialAttentionBlock(c, heads=4)
        self.G = GatedResidualBlock(c, heads=4)
        self.E = ExternalMemoryBlock(c, mem_size=32, heads=4)
        self.N = NonLocalBlock(c, heads=4)
        # T: 最终集成层 — 汇聚前四路输出
        self.T_route = nn.Sequential(nn.Linear(c * 4, c * 2), nn.GELU(), nn.Linear(c * 2, 4))
        self.T_attn = nn.MultiheadAttention(c, num_heads=4, batch_first=True)
        self.T_norm = nn.LayerNorm(c)
        self.T_ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.T_ffn_norm = nn.LayerNorm(c)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def _t_fuse(self, a_out, g_out, e_out, n_out):
        """T: 四路集成融合"""
        B, C, H, W = n_out.shape
        seqs = [a_out, g_out, e_out, n_out]
        flat = [s.flatten(2).transpose(1, 2) for s in seqs]
        gaps = [s.mean(1) for s in flat]
        w = F.softmax(self.T_route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(flat[i] * w[:, i:i+1].unsqueeze(1) for i in range(4))
        h = self.T_norm(merged)
        attn_out, _ = self.T_attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.T_ffn(self.T_ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        x = self.stem(x)
        a_out = self.A(x) + x
        g_out = self.G(a_out) + a_out
        e_out = self.E(g_out) + g_out
        n_out = self.N(e_out) + e_out
        # T: 四路集成
        t_out = self._t_fuse(a_out, g_out, e_out, n_out) + n_out
        return self.head(self.pool(t_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 打乱顺序: N→E→A→G→T
# ============================================================
class ScrambledAgentModel(nn.Module):
    """打乱: N → E → A → G → T"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.N = NonLocalBlock(c, heads=4)
        self.E = ExternalMemoryBlock(c, mem_size=32, heads=4)
        self.A = AxialAttentionBlock(c, heads=4)
        self.G = GatedResidualBlock(c, heads=4)
        # T: 同样的集成层
        self.T_route = nn.Sequential(nn.Linear(c * 4, c * 2), nn.GELU(), nn.Linear(c * 2, 4))
        self.T_attn = nn.MultiheadAttention(c, num_heads=4, batch_first=True)
        self.T_norm = nn.LayerNorm(c)
        self.T_ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.T_ffn_norm = nn.LayerNorm(c)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def _t_fuse(self, n_out, e_out, a_out, g_out):
        B, C, H, W = g_out.shape
        seqs = [n_out, e_out, a_out, g_out]
        flat = [s.flatten(2).transpose(1, 2) for s in seqs]
        gaps = [s.mean(1) for s in flat]
        w = F.softmax(self.T_route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(flat[i] * w[:, i:i+1].unsqueeze(1) for i in range(4))
        h = self.T_norm(merged)
        attn_out, _ = self.T_attn(h, h, h)
        merged = merged + attn_out
        merged = merged + self.T_ffn(self.T_ffn_norm(merged))
        return merged.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        x = self.stem(x)
        n_out = self.N(x) + x
        e_out = self.E(n_out) + n_out
        a_out = self.A(e_out) + e_out
        g_out = self.G(a_out) + a_out
        t_out = self._t_fuse(n_out, e_out, a_out, g_out) + g_out
        return self.head(self.pool(t_out).flatten(1))

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
    out_dir = "/workspace/arch_lab/runs/agent"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  Agent 五字母架构实验")
    print(f"  A→G→E→N→T (正确顺序) vs N→E→A→G→T (打乱顺序)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # 冒烟测试
    print(f"\n  === 冒烟测试 ===")
    for name, MC in [("Agent", AgentModel), ("Scrambled", ScrambledAgentModel)]:
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
    for name, MC in [("Agent (A→G→E→N→T)", AgentModel), ("Scrambled (N→E→A→G→T)", ScrambledAgentModel)]:
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
    for name, MC in [("Agent (A→G→E→N→T)", AgentModel), ("Scrambled (N→E→A→G→T)", ScrambledAgentModel)]:
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

    summary = {"experiment": "agent_letter_model", "epochs": epochs, "channels": c,
               "cls_results": cls_results, "add_results": add_results}
    with open(os.path.join(out_dir, "agent_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/agent_results.json")
    return summary


if __name__ == "__main__":
    run()
