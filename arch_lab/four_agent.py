"""
4-Agent代码AI协作实验 — 借鉴扣子多Agent团队协作

四个有Agent能力的代码AI组队:
  Agent    (A→G→E→N→T)   — 轴向注意力+外部记忆+TransformerXL (95K)
  TRAE     (T→R→A→E)     — 双流+MoE路由+adaLN+涌现门控 (100K)
  DeepSeek (D→E₁→E₂→P→S→E₃→E₄→K) — 密集+高效注意力+专家路由+知识记忆 (182K)
  Chrome   (C→H→R→O→M→E) — 交叉注意力+旋转+八度+记忆库+集成 (112K)

协作策略 (编排器设计):
  1. 4×Solo:     每个Agent单独工作 (基线)
  2. 4-Way-Vote: 4个并行,logits平均投票
  3. 4-Way-Router: 4个并行,学习动态路由权重
  4. Hierarchical: 两两配对(Agent+TRAE / DeepSeek+Chrome) → 再融合
  5. Feature-Stack: 4个特征拼接 → MLP融合
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent_model import AgentModel
from trae_model import TRAEModel
from deepseek_model import DeepSeekModel
from chrome_model import ChromeModel

AGENTS = [
    ("Agent", AgentModel),
    ("TRAE", TRAEModel),
    ("DeepSeek", DeepSeekModel),
    ("Chrome", ChromeModel),
]


# ============================================================
# 模式2: 4-Way-Vote (并行投票)
# ============================================================
class FourWayVoteModel(nn.Module):
    """4个Agent并行预测,logits平均。零额外参数。"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.agents = nn.ModuleList([MC(in_channels, num_classes, c) for _, MC in AGENTS])

    def forward(self, x):
        logits = sum(agent(x) for agent in self.agents)
        return logits / len(self.agents)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 模式3: 4-Way-Router (学习动态路由)
# ============================================================
class FourWayRouterModel(nn.Module):
    """4个Agent并行,路由器根据输入动态分配权重。
    编排器学习: 什么输入该信哪个Agent。"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.agents = nn.ModuleList([MC(in_channels, num_classes, c) for _, MC in AGENTS])
        # 共享stem用于路由器
        self.router_stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, c), nn.GELU(),
            nn.Linear(c, 4), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        w = self.router(self.router_stem(x))  # (B, 4)
        logits = torch.stack([agent(x) for agent in self.agents], dim=1)  # (B, 4, num_classes)
        return (logits * w.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 模式4: Hierarchical (分层配对融合)
# ============================================================
class HierarchicalModel(nn.Module):
    """分层协作: 
    第一层: Agent+TRAE 配对投票, DeepSeek+Chrome 配对投票
    第二层: 两个配对结果再学习路由融合"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.agent = AgentModel(in_channels, num_classes, c)
        self.trae = TRAEModel(in_channels, num_classes, c)
        self.deepseek = DeepSeekModel(in_channels, num_classes, c)
        self.chrome = ChromeModel(in_channels, num_classes, c)
        # 第二层路由器
        self.fuse_router = nn.Sequential(
            nn.Linear(num_classes * 2, c), nn.GELU(),
            nn.Linear(c, 2), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # 第一层: 两两配对投票
        pair1 = (self.agent(x) + self.trae(x)) / 2.0      # Agent+TRAE
        pair2 = (self.deepseek(x) + self.chrome(x)) / 2.0  # DeepSeek+Chrome
        # 第二层: 路由融合
        w = self.fuse_router(torch.cat([pair1, pair2], dim=-1))  # (B, 2)
        return pair1 * w[:, 0:1] + pair2 * w[:, 1:2]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 模式5: Feature-Stack (特征级拼接融合)
# ============================================================
class FeatureStackModel(nn.Module):
    """4个Agent各自提取特征,拼接后MLP融合决策。"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.agent = AgentModel(in_channels, num_classes, c)
        self.trae = TRAEModel(in_channels, num_classes, c)
        self.deepseek = DeepSeekModel(in_channels, num_classes, c)
        self.chrome = ChromeModel(in_channels, num_classes, c)
        # 融合MLP: 4个特征向量拼接
        self.fuse = nn.Sequential(
            nn.Linear(c * 4, c * 2), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(c * 2, c), nn.GELU(),
            nn.Linear(c, num_classes),
        )

    def _agent_features(self, m, x):
        """Agent: A→G→E→N→T, T通过_t_fuse实现四路集成"""
        x = m.stem(x)
        a_out = m.A(x) + x
        g_out = m.G(a_out) + a_out
        e_out = m.E(g_out) + g_out
        n_out = m.N(e_out) + e_out
        # T: 四路集成融合 (不是单独的m.T属性)
        t_out = m._t_fuse(a_out, g_out, e_out, n_out) + n_out
        return m.pool(t_out).flatten(1)

    def _trae_features(self, m, x):
        x = m.stem(x)
        t = m.T(x) + x
        r = m.R(t) + t
        a = m.A(r) + r
        e = m.E(t, r, a) + a
        return m.pool(e).flatten(1)

    def _deepseek_features(self, m, x):
        """DeepSeek: D→E₁→E₂→P→S→E₃→E₄→K, E3汇聚5路, K汇聚7路"""
        x = m.stem(x)
        d = m.D(x) + x
        e1 = m.E1(d) + d
        e2 = m.E2(e1) + e1
        p = m.P(e2) + e2
        s = m.S(p) + p
        # E₃: 汇聚 D,E1,E2,P,S 五路
        e3 = m.E3(d, e1, e2, p, s) + s
        e4 = m.E4(e3) + e3
        # K: 汇聚全部七路 (d,e1,e2,p,s,e3,e4)
        k = m.K(d, e1, e2, p, s, e3, e4) + e4
        return m.pool(k).flatten(1)

    def _chrome_features(self, m, x):
        x = m.stem(x)
        c = m.C(x) + x
        h = m.H(c) + c
        r = m.R(h) + h
        o = m.O(r) + r
        me = m.M(o) + o
        e = m.E(c, h, r, o, me) + me
        return m.pool(e).flatten(1)

    def forward(self, x):
        feats = [
            self._agent_features(self.agent, x),
            self._trae_features(self.trae, x),
            self._deepseek_features(self.deepseek, x),
            self._chrome_features(self.chrome, x),
        ]
        stacked = torch.cat(feats, dim=-1)  # (B, 4C)
        return self.fuse(stacked)

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
    out_dir = "/workspace/arch_lab/runs/four_agent"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  4-Agent代码AI协作实验 (扣子多Agent团队思路)")
    print(f"  Agent + TRAE + DeepSeek + Chrome")
    print(f"  9种实验: 4×Solo + Vote + Router + Hierarchical + FeatureStack")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    all_models = [
        ("Solo-Agent", AgentModel),
        ("Solo-TRAE", TRAEModel),
        ("Solo-DeepSeek", DeepSeekModel),
        ("Solo-Chrome", ChromeModel),
        ("4-Way-Vote", FourWayVoteModel),
        ("4-Way-Router", FourWayRouterModel),
        ("Hierarchical", HierarchicalModel),
        ("Feature-Stack", FeatureStackModel),
    ]

    # ===== 任务1: MNIST 分类 =====
    print(f"\n  === 任务1: MNIST 分类 ===")
    tl_cls, vl_cls = get_loaders()
    cls_results = {}
    for name, MC in all_models:
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
    for name, MC in all_models:
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
        print(f"  {n:<22s} {d['acc']:>8.4f}  {d['params']:>10,}  {d['elapsed']:>6.1f}s")
    print(f"\n  ▶ MNIST 加法推理 (随机基线=5.3%)")
    for n, d in sorted(add_results.items(), key=lambda x: x[1]["acc"], reverse=True):
        print(f"  {n:<22s} {d['acc']:>8.4f}  {d['params']:>10,}  {d['elapsed']:>6.1f}s")

    summary = {
        "experiment": "four_agent_collaboration",
        "epochs": epochs, "channels": c, "device": str(device),
        "cls_results": cls_results, "add_results": add_results,
    }
    with open(os.path.join(out_dir, "four_agent_results.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_dir}/four_agent_results.json")
    return summary


if __name__ == "__main__":
    run()
