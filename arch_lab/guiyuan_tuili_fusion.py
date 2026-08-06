"""
归元 × TuiLi 融合架构 — 将手绘归元模型与冠军TuiLi融合

归元核心设计 (来自手绘图):
  1. 三层嵌套立方体 = 多尺度金字塔 (1×1 / 2×2 / 4×4 池化)
  2. 中心X交叉注意力 = 瓶颈处多尺度特征融合
  3. 跨层对角线 = 跳跃连接 (Skip Connection)

TuiLi 冠军架构 (当前冠军: 分类95.6%, 推理44.9%):
  T: Twin-path Attention  (全局+局部双路)
  U: U-Shape Skip         (编码-解码跳连)
  I₁: Inception Multi-scale (多尺度并行卷积)
  L: Linear Attention     (O(n)线性注意力)
  I₂: Iterative Refinement (4路集成+2轮迭代)

融合设计 — "归元推理" (GuiYuan-TuiLi):
  G-T: 归元双路注意力 = TuiLi的Twin-path + 归元的金字塔多尺度池化
  G-U: 归元U型跳连 = TuiLi的U-Shape + 归元的三层嵌套(3级下采样/上采样)
  G-X: 归元X交叉注意力 = 归元的中心X + 多尺度特征交叉融合 (替换I₁)
  L:   线性注意力 (保留TuiLi原版)
  G-I₂: 归元迭代精炼 = TuiLi的4路集成 + 归元X注意力融合 + 2轮迭代

  G-T → G-U → G-X → L → G-I₂ 串联
  G-I₂ 同时汇聚 G-T/G-U/G-X/L 四路输出 (归元跨层跳连)
"""
from __future__ import annotations
import json, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# G-T — 归元双路注意力 (Twin-path + 金字塔多尺度)
# ============================================================
class GuiYuanTwinPathBlock(nn.Module):
    """G-T: 全局路(MHSA + 归元金字塔池化) + 局部路(深度卷积), 门控融合。
    融合点: 归元的三层嵌套 → 金字塔多尺度池化注入全局路。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # 全局路: 多头自注意力
        self.global_norm = nn.LayerNorm(c)
        self.global_attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        # 归元金字塔: 1×1 / 2×2 / 4×4 多尺度池化
        self.pyramid_p1 = nn.AdaptiveAvgPool2d(1)
        self.pyramid_p2 = nn.AdaptiveAvgPool2d(2)
        self.pyramid_p4 = nn.AdaptiveAvgPool2d(4)
        self.pyramid_proj = nn.Conv2d(c * 3, c, 1, bias=False)
        self.pyramid_norm = nn.BatchNorm2d(c)
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
        # 全局路: 自注意力
        g_in = self.global_norm(seq)
        g_out, _ = self.global_attn(g_in, g_in, g_in)
        global_feat = seq + g_out  # (B, HW, C)
        # 归元金字塔: 多尺度池化 → 注入全局特征
        p1 = F.interpolate(self.pyramid_p1(x), size=(H, W), mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.pyramid_p2(x), size=(H, W), mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.pyramid_p4(x), size=(H, W), mode="bilinear", align_corners=False)
        pyramid = self.pyramid_norm(self.pyramid_proj(torch.cat([p1, p2, p4], dim=1)))
        global_feat = global_feat + pyramid.flatten(2).transpose(1, 2)
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
# G-U — 归元U型跳连 (三层嵌套编码-解码)
# ============================================================
class GuiYuanUSkipBlock(nn.Module):
    """G-U: 三级嵌套下采样→处理→上采样, 每级跳连。
    融合点: 归元的三层立方体嵌套 → 3级U-Net结构。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        # Level 1: 下采样到 1/2
        self.down1 = nn.Conv2d(c, c, 2, stride=2, bias=False)
        self.down1_norm = nn.BatchNorm2d(c)
        self.skip1_proj = nn.Conv2d(c, c, 1, bias=False)
        # Level 2: 下采样到 1/4
        self.down2 = nn.Conv2d(c, c, 2, stride=2, bias=False)
        self.down2_norm = nn.BatchNorm2d(c)
        self.skip2_proj = nn.Conv2d(c, c, 1, bias=False)
        # Level 3 (瓶颈): 归元中心 — 注意力
        self.bottleneck_norm = nn.LayerNorm(c)
        self.bottleneck_attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        # 上采样: Level 2
        self.up2 = nn.ConvTranspose2d(c, c, 2, stride=2, bias=False)
        self.up2_norm = nn.BatchNorm2d(c)
        self.fuse2 = nn.Conv2d(c * 2, c, 1, bias=False)
        self.fuse2_norm = nn.BatchNorm2d(c)
        # 上采样: Level 1
        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2, bias=False)
        self.up1_norm = nn.BatchNorm2d(c)
        self.fuse1 = nn.Conv2d(c * 2, c, 1, bias=False)
        self.fuse1_norm = nn.BatchNorm2d(c)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # Level 0: 原始分辨率 (skip0 = x)
        skip0 = x
        # Level 1: 下采样 1/2
        d1 = F.gelu(self.down1_norm(self.down1(x)))
        skip1 = self.skip1_proj(d1)
        H1, W1 = d1.shape[2], d1.shape[3]
        # Level 2: 下采样 1/4
        d2 = F.gelu(self.down2_norm(self.down2(d1)))
        skip2 = self.skip2_proj(d2)
        # Level 3: 瓶颈 — 注意力 (归元中心)
        b_seq = d2.flatten(2).transpose(1, 2)
        h = self.bottleneck_norm(b_seq)
        a, _ = self.bottleneck_attn(h, h, h)
        b_seq = b_seq + a
        d2_refined = b_seq.transpose(1, 2).reshape(d2.shape)
        # 上采样 Level 2: 1/4 → 1/2, 跳连 skip2
        u2 = F.gelu(self.up2_norm(self.up2(d2_refined)))
        if u2.shape[2:] != skip1.shape[2:]:
            u2 = F.interpolate(u2, size=(H1, W1), mode="bilinear", align_corners=False)
        u2_fused = self.fuse2_norm(self.fuse2(torch.cat([skip1, u2], dim=1)))
        # 上采样 Level 1: 1/2 → 原始, 跳连 skip0
        u1 = F.gelu(self.up1_norm(self.up1(u2_fused)))
        if u1.shape[2:] != skip0.shape[2:]:
            u1 = F.interpolate(u1, size=(H, W), mode="bilinear", align_corners=False)
        u1_fused = self.fuse1_norm(self.fuse1(torch.cat([skip0, u1], dim=1)))
        # FFN
        seq = u1_fused.flatten(2).transpose(1, 2)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# G-X — 归元X交叉注意力 (中心X多尺度特征融合)
# ============================================================
class GuiYuanXAttentionBlock(nn.Module):
    """G-X: 归元中心X — 将特征分成4路(对应X的4个端点),
    做全交叉注意力后融合。
    融合点: 归元的X结构 → 4路特征交叉注意力。"""

    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        c4 = c // 4
        self.c4 = c4
        self.c = c
        # 4路特征提取 (X的4个端点)
        self.branch_a = nn.Conv2d(c, c4, 1, bias=False)  # 左上
        self.branch_b = nn.Conv2d(c, c4, 1, bias=False)  # 右上
        self.branch_c = nn.Conv2d(c, c4, 1, bias=False)  # 左下
        self.branch_d = nn.Conv2d(c, c4, 1, bias=False)  # 右下
        # 4路交叉注意力 (X的交叉)
        self.cross_ab = nn.MultiheadAttention(c4, num_heads=max(1, heads // 4), batch_first=True)
        self.cross_cd = nn.MultiheadAttention(c4, num_heads=max(1, heads // 4), batch_first=True)
        self.cross_ac = nn.MultiheadAttention(c4, num_heads=max(1, heads // 4), batch_first=True)
        self.cross_bd = nn.MultiheadAttention(c4, num_heads=max(1, heads // 4), batch_first=True)
        # 归一化
        self.norms = nn.ModuleList([nn.LayerNorm(c4) for _ in range(4)])
        # 融合: 4路 → 原始通道
        self.fuse = nn.Conv2d(c4 * 4, c, 1, bias=False)
        self.fuse_norm = nn.BatchNorm2d(c)
        # 自注意力精炼
        self.refine_norm = nn.LayerNorm(c)
        self.refine_attn = nn.MultiheadAttention(c, num_heads=heads, batch_first=True)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), nn.GELU(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 4路特征
        a = self.branch_a(x).flatten(2).transpose(1, 2)  # (B, HW, c4)
        b = self.branch_b(x).flatten(2).transpose(1, 2)
        c_ = self.branch_c(x).flatten(2).transpose(1, 2)
        d = self.branch_d(x).flatten(2).transpose(1, 2)
        # X交叉注意力: a↔b, c↔d, a↔c, b↔d (X形交叉)
        ab, _ = self.cross_ab(self.norms[0](a), self.norms[1](b), self.norms[1](b))
        a = a + ab
        cd, _ = self.cross_cd(self.norms[2](c_), self.norms[3](d), self.norms[3](d))
        c_ = c_ + cd
        ac, _ = self.cross_ac(self.norms[0](a), self.norms[2](c_), self.norms[2](c_))
        a = a + ac
        bd, _ = self.cross_bd(self.norms[1](b), self.norms[3](d), self.norms[3](d))
        b = b + bd
        # 融合4路
        merged = torch.cat([a, b, c_, d], dim=-1)  # (B, HW, C)
        merged = merged.transpose(1, 2).reshape(B, C, H, W)
        merged = self.fuse_norm(self.fuse(merged))
        # 自注意力精炼
        seq = merged.flatten(2).transpose(1, 2)
        h = self.refine_norm(seq)
        ra, _ = self.refine_attn(h, h, h)
        seq = seq + ra
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# L — 线性注意力 (保留TuiLi原版)
# ============================================================
class LinearAttentionBlock(nn.Module):
    """L: 线性注意力 — ELU+1核函数, O(n)复杂度。"""

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
# G-I₂ — 归元迭代精炼 (4路集成 + X注意力融合 + 2轮迭代)
# ============================================================
class GuiYuanIterativeRefinementBlock(nn.Module):
    """G-I₂: 4路集成路由 + 归元X交叉注意力融合 + 2轮迭代精炼。
    融合点: TuiLi的4路集成 + 归元的X注意力做路间融合。"""

    def __init__(self, c: int, heads: int = 4, n_inputs: int = 4, iters: int = 2):
        super().__init__()
        self.n_inputs = n_inputs
        self.iters = iters
        # 集成路由
        self.route = nn.Sequential(
            nn.Linear(c * n_inputs, c * 2), nn.GELU(), nn.Linear(c * 2, n_inputs)
        )
        # 归元X: 路间交叉注意力 (4路两两交叉)
        c4 = c // 4
        self.x_proj = nn.ModuleList([nn.Linear(c, c4) for _ in range(n_inputs)])
        self.x_cross = nn.MultiheadAttention(c4, num_heads=max(1, heads // 4), batch_first=True)
        self.x_back = nn.Linear(c4, c)
        self.x_norm = nn.LayerNorm(c)
        # 迭代精炼
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
        B, C, H, W = inputs[0].shape
        seqs = [inp.flatten(2).transpose(1, 2) for inp in inputs]
        # 集成路由
        gaps = [s.mean(1) for s in seqs]
        w = F.softmax(self.route(torch.cat(gaps, dim=-1)), dim=-1)
        merged = sum(seqs[i] * w[:, i:i+1].unsqueeze(1) for i in range(self.n_inputs))
        # 归元X: 路间交叉注意力
        x_feats = [proj(s) for proj, s in zip(self.x_proj, seqs)]  # 各 (B, HW, c4)
        # 交叉: 0→1, 2→3, 0→2, 1→3 (X形)
        x01, _ = self.x_cross(x_feats[0], x_feats[1], x_feats[1])
        x23, _ = self.x_cross(x_feats[2], x_feats[3], x_feats[3])
        x_merged = torch.cat([x01, x23], dim=-1)  # (B, HW, c4*2)
        x_out = self.x_back(x_merged.chunk(2, dim=-1)[0] + x_merged.chunk(2, dim=-1)[1])
        merged = self.x_norm(merged + x_out)
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
# 归元推理 (GuiYuan-TuiLi) 完整模型
# ============================================================
class GuiYuanTuiLiModel(nn.Module):
    """归元推理 = G-T → G-U → G-X → L → G-I₂
    归元手绘架构 × TuiLi冠军架构 融合"""

    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.GT = GuiYuanTwinPathBlock(c, heads=4)
        self.GU = GuiYuanUSkipBlock(c, heads=4)
        self.GX = GuiYuanXAttentionBlock(c, heads=4)
        self.L = LinearAttentionBlock(c, heads=4)
        self.GI2 = GuiYuanIterativeRefinementBlock(c, heads=4, n_inputs=4, iters=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(c, c * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(c * 2, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        gt_out = self.GT(x) + x           # G-T + 残差
        gu_out = self.GU(gt_out) + gt_out # G-U + 残差
        gx_out = self.GX(gu_out) + gu_out # G-X + 残差
        l_out = self.L(gx_out) + gx_out   # L + 残差
        # G-I₂: 四路集成 + 归元X融合 + 迭代精炼
        gi2_out = self.GI2(gt_out, gu_out, gx_out, l_out) + l_out
        return self.head(self.pool(gi2_out).flatten(1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 对比: 纯TuiLi (从tuili_model.py导入)
# ============================================================
class TuiLiModel(nn.Module):
    """TuiLi 原版冠军 = T → U → I₁ → L → I₂"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        from tuili_model import TwinPathAttentionBlock, UShapeSkipBlock, InceptionBlock, LinearAttentionBlock, IterativeRefinementBlock
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
        i2_out = self.I2(t_out, u_out, i1_out, l_out) + l_out
        return self.head(self.pool(i2_out).flatten(1))

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
    out_dir = "/workspace/arch_lab/runs/guiyuan_tuili"
    os.makedirs(out_dir, exist_ok=True)
    epochs, c = 8, 32

    print(f"\n{'='*70}")
    print(f"  归元 × TuiLi 融合实验")
    print(f"  归元推理 (G-T→G-U→G-X→L→G-I₂) vs TuiLi冠军 (T→U→I₁→L→I₂)")
    print(f"  设备: {device}  epochs={epochs}  c={c}")
    print(f"{'='*70}")

    # ---- 任务1: MNIST 分类 ----
    print(f"\n{'='*70}")
    print(f"  任务1: MNIST 分类 (感知能力)")
    print(f"{'='*70}")

    tl_cls, vl_cls = get_loaders()
    models_cls = {
        "TuiLi (冠军)": TuiLiModel(in_channels=1, num_classes=10, c=c),
        "归元推理 (融合)": GuiYuanTuiLiModel(in_channels=1, num_classes=10, c=c),
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
        "TuiLi (冠军)": TuiLiModel(in_channels=2, num_classes=19, c=c),
        "归元推理 (融合)": GuiYuanTuiLiModel(in_channels=2, num_classes=19, c=c),
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
        "experiment": "guiyuan_tuili_fusion",
        "epochs": epochs,
        "channels": c,
        "device": str(device),
        "cls_results": cls_results,
        "add_results": add_results,
    }
    with open(os.path.join(out_dir, "guiyuan_tuili_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 {out_dir}/guiyuan_tuili_results.json")

    # ---- 打印汇总 ----
    print(f"\n{'='*70}")
    print(f"  归元 × TuiLi 融合实验汇总")
    print(f"{'='*70}")
    print(f"\n  {'模型':<25} {'分类准确率':>10} {'推理准确率':>10} {'参数量':>10}")
    print(f"  {'-'*60}")
    for name in cls_results:
        cls_acc = cls_results[name]["acc"]
        add_acc = add_results[name]["acc"]
        npar = cls_results[name]["params"]
        print(f"  {name:<25} {cls_acc:>10.4f} {add_acc:>10.4f} {npar:>10}")

    # 融合增益
    tuiLi_cls = cls_results["TuiLi (冠军)"]["acc"]
    fused_cls = cls_results["归元推理 (融合)"]["acc"]
    tuiLi_add = add_results["TuiLi (冠军)"]["acc"]
    fused_add = add_results["归元推理 (融合)"]["acc"]
    print(f"\n  融合增益:")
    print(f"    分类: {fused_cls:.4f} vs {tuiLi_cls:.4f} ({'+' if fused_cls >= tuiLi_cls else ''}{(fused_cls - tuiLi_cls)*100:.1f}%)")
    print(f"    推理: {fused_add:.4f} vs {tuiLi_add:.4f} ({'+' if fused_add >= tuiLi_add else ''}{(fused_add - tuiLi_add)*100:.1f}%)")

    return results


if __name__ == "__main__":
    run()
