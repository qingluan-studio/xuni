"""
12领域AI架构组件 — 每个组件忠实保留原架构核心创新，
适配为 (B,C,H,W) → (B,C,H,W) 的通用接口，供融合实验使用。

12个架构:
  1. Transformer   (NLP文本)     — 多头自注意力 + FFN
  2. ViT           (图像分类)    — Patch embedding + Transformer
  3. Conformer     (语音识别)    — 卷积+注意力并行
  4. VideoMAE      (视频理解)    — 时空分离注意力 (2D简化)
  5. DiT           (图像生成)    — ADA LayerNorm + Transformer
  6. PointTrans    (3D点云)      — 向量注意力
  7. CLIP          (多模态)      — 双编码器交叉注意力
  8. Mamba         (高效序列)    — 选择性SSM (简化)
  9. MoE           (稀疏路由)    — Top-k混合专家
 10. Graphormer    (图学习)      — 空间图注意力
 11. RWKV          (线性RNN)     — WKV时间混合
 12. Perceiver     (通用感知)    — 潜在数组交叉注意力
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Transformer (NLP文本) — 自注意力 + FFN
# ============================================================
class TransformerOp(nn.Module):
    """标准Transformer编码块: MHSA + FFN, 核心创新是全局自注意力。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        a = self.norm1(seq)
        a, _ = self.attn(a, a, a)
        seq = seq + a
        seq = seq + self.ffn(self.norm2(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 2. ViT (图像分类) — Patch embedding + Transformer
# ============================================================
class ViTOp(nn.Module):
    """ViT核心: 将特征图分patch, 做patch间注意力, 再重组。
    创新点: 图像→序列→注意力→重组 的统一管线。"""

    def __init__(self, c: int, patch_size: int = 2, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(c, c, patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 2), act_fn(), nn.Linear(c * 2, c))

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        # 分patch
        patches = self.patch_embed(x)  # (B, C, H/ps, W/ps)
        _, _, Hp, Wp = patches.shape
        seq = patches.flatten(2).transpose(1, 2)  # (B, N, C)
        # 注意力
        a = self.norm(seq)
        a, _ = self.attn(a, a, a)
        seq = seq + a
        seq = seq + self.ffn(self.norm(seq))
        # 重组
        out = seq.transpose(1, 2).reshape(B, C, Hp, Wp)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return x + out


# ============================================================
# 3. Conformer (语音识别) — 卷积+注意力并行
# ============================================================
class ConformerOp(nn.Module):
    """Conformer块: Macaron FFN → MHSA → Conv → Macaron FFN。
    创新点: 卷积(局部)与注意力(全局)在同一个块中并行融合。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        # Macaron FFN 1
        self.ffn1_norm = nn.LayerNorm(c)
        self.ffn1 = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        # 自注意力
        self.attn_norm = nn.LayerNorm(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        # 卷积
        self.conv_norm = nn.BatchNorm2d(c)
        self.conv = nn.Sequential(
            nn.Conv2d(c, c * 2, 1), act_fn(),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, groups=c * 2), act_fn(),
            nn.Conv2d(c * 2, c, 1),
        )
        # Macaron FFN 2
        self.ffn2_norm = nn.LayerNorm(c)
        self.ffn2 = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        # Macaron FFN 1 (×0.5)
        seq = seq + 0.5 * self.ffn1(self.ffn1_norm(seq))
        # MHSA
        a = self.attn_norm(seq)
        a, _ = self.attn(a, a, a)
        seq = seq + a
        # Conv
        feat = seq.transpose(1, 2).reshape(B, C, H, W)
        seq = seq + self.conv(self.conv_norm(feat)).flatten(2).transpose(1, 2)
        # Macaron FFN 2 (×0.5)
        seq = seq + 0.5 * self.ffn2(self.ffn2_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 4. VideoMAE (视频理解) — 时空分离注意力
# ============================================================
class VideoMAEOp(nn.Module):
    """VideoMAE核心: 时空分离注意力 — 先空间注意力, 再时间注意力。
    2D简化: 将空间维度分两步注意力 (行→列), 模拟时空分离。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.row_norm = nn.LayerNorm(c)
        self.col_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.col_norm = nn.LayerNorm(c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        # 行注意力 (模拟空间注意力)
        seq = x.permute(0, 3, 2, 1).reshape(B * W, H, C)  # (B*W, H, C)
        a = self.row_norm(seq)
        a, _ = self.row_attn(a, a, a)
        seq = seq + a
        # 列注意力 (模拟时间注意力)
        seq = seq.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B * H, W, C)
        a = self.col_norm(seq)
        a, _ = self.col_attn(a, a, a)
        seq = seq + a
        seq = seq + self.ffn(self.ffn_norm(seq))
        return x + seq.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ============================================================
# 5. DiT (图像生成) — ADA LayerNorm + Transformer
# ============================================================
class DiTOp(nn.Module):
    """DiT核心: adaLN-zero条件注入 — 用学习的scale/shift调制LayerNorm。
    创新点: 自适应归一化让Transformer适合生成任务。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.norm1 = nn.LayerNorm(c, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(c, elementwise_affine=False)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        # adaLN调制: 6个参数 (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(
            act_fn(), nn.Linear(c, c * 6)
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)
        # 用全局平均池化作为条件
        cond = seq.mean(dim=1, keepdim=True)  # (B, 1, C)
        params = self.adaLN(cond)  # (B, 1, 6C)
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
# 6. Point Transformer (3D点云) — 向量注意力
# ============================================================
class PointTransformerOp(nn.Module):
    """Point Transformer核心: 向量注意力 — 每个通道有独立的注意力权重。
    创新点: 向量级(而非标量级)注意力, 更细粒度的特征建模。"""

    def __init__(self, c: int, act: str = "gelu"):
        super().__init__()
        self.to_qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.GroupNorm(1, c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1), act_fn(), nn.Conv2d(c * 2, c, 1)
        )
        # 向量注意力的逐通道变换
        self.attn_scale = nn.Parameter(torch.ones(1, c, 1, 1) * (c ** -0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)  # each (B, C, H, W)
        # 向量注意力: 逐通道注意力 (B, C, HW)
        q_flat = q.flatten(2)  # (B, C, HW)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)
        # 逐通道注意力: (B, C, HW, HW)
        scale = C ** -0.5
        attn = torch.einsum('bci,bcj->bcij', q_flat, k_flat) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bcij,bcj->bci', attn, v_flat).reshape(B, C, H, W)
        x = x + self.proj(out)
        x = x + self.ffn(x)
        return x


# ============================================================
# 7. CLIP (多模态) — 双编码器交叉注意力
# ============================================================
class CLIPOp(nn.Module):
    """CLIP核心: 双流编码器 + 对比学习对齐。
    2D适配: 将输入分成"内容流"和"结构流", 交叉注意力后对齐融合。
    创新点: 跨模态对比学习对齐。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        # 流A: 空间编码器
        self.norm_a = nn.LayerNorm(c)
        self.attn_a = nn.MultiheadAttention(c, heads, batch_first=True)
        # 流B: 通道编码器
        self.norm_b = nn.LayerNorm(c)
        self.attn_b = nn.MultiheadAttention(c, heads, batch_first=True)
        # 交叉注意力对齐
        self.cross_a2b = nn.MultiheadAttention(c, heads, batch_first=True)
        self.cross_b2a = nn.MultiheadAttention(c, heads, batch_first=True)
        self.fuse = nn.Linear(c * 2, c)
        self.temp = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 流A: 空间自注意力
        a = self.norm_a(seq)
        a_out, _ = self.attn_a(a, a, a)
        stream_a = seq + a_out
        # 流B: 用通道分组模拟第二模态
        stream_b = seq.roll(shifts=C // 2, dims=-1)  # 通道偏移模拟不同模态
        b = self.norm_b(stream_b)
        b_out, _ = self.attn_b(b, b, b)
        stream_b = stream_b + b_out
        # 交叉注意力
        cross_a, _ = self.cross_a2b(stream_a, stream_b, stream_b)
        cross_b, _ = self.cross_b2a(stream_b, stream_a, stream_a)
        # 对齐融合
        fused = self.fuse(torch.cat([stream_a + cross_a, stream_b + cross_b], dim=-1))
        return x + fused.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 8. Mamba (高效序列) — 选择性SSM (简化)
# ============================================================
class MambaOp(nn.Module):
    """Mamba核心: 选择性状态空间模型 — 输入相关的门控和状态更新。
    简化实现: 用输入门控的线性递归模拟SSM的选择性机制。
    创新点: 线性时间复杂度的内容相关序列建模。"""

    def __init__(self, c: int, state_size: int = 8, act: str = "silu"):
        super().__init__()
        self.state_size = state_size
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        # 输入投影
        self.in_proj = nn.Linear(c, c * 2)  # gate + state input
        # SSM参数 (输入相关)
        self.dt_proj = nn.Linear(c, c)  # 时间步长 (选择性)
        self.A_log = nn.Parameter(torch.randn(c, state_size) * 0.01)
        self.D = nn.Parameter(torch.ones(c))
        # 输出投影
        self.out_proj = nn.Linear(c, c)
        self.norm = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.in_proj(seq)
        gate, x_in = h.chunk(2, dim=-1)
        # 选择性: dt 随输入变化
        dt = torch.sigmoid(self.dt_proj(x_in))  # (B, HW, C)
        A = -torch.exp(self.A_log)  # (C, state)
        # 简化SSM递归 (用1D卷积近似)
        state = torch.zeros(B, C, self.state_size, device=x.device)
        out = torch.zeros_like(seq)
        for t in range(seq.shape[1]):
            x_t = x_in[:, t, :]  # (B, C)
            dt_t = dt[:, t, :]  # (B, C)
            # 状态更新 (选择性)
            state = state * torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0)) + \
                    x_t.unsqueeze(-1) * dt_t.unsqueeze(-1)
            out[:, t, :] = (state * self.D.unsqueeze(0).unsqueeze(-1)).sum(-1)
        out = out * torch.sigmoid(gate)
        seq = seq + self.out_proj(out)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 9. MoE (稀疏路由) — Top-k混合专家
# ============================================================
class MoEOp(nn.Module):
    """MoE核心: Top-k路由 + 多专家FFN。
    创新点: 解耦参数量与计算量, 稀疏激活。"""

    def __init__(self, c: int, num_experts: int = 4, top_k: int = 2, act: str = "gelu"):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        # 路由网络
        self.gate = nn.Linear(c, num_experts)
        # 专家 (每个是独立的FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
            for _ in range(num_experts)
        ])
        self.norm = nn.LayerNorm(c)
        # 负载均衡损失
        self.load_balance_loss = 0.0

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        # 路由
        logits = self.gate(h)  # (B, HW, num_experts)
        weights, indices = logits.topk(self.top_k, dim=-1)  # (B, HW, k)
        weights = F.softmax(weights, dim=-1)
        # 稀疏激活: 逐专家计算, mask加权
        out = torch.zeros_like(seq)
        for e in range(self.num_experts):
            expert_out = self.experts[e](h)  # (B, HW, C)
            for k in range(self.top_k):
                mask = (indices[..., k] == e).float()  # (B, HW)
                coeff = (mask * weights[..., k]).unsqueeze(-1)  # (B, HW, 1)
                out = out + coeff * expert_out
        # 负载均衡 (简化追踪)
        with torch.no_grad():
            load = torch.zeros(self.num_experts, device=x.device)
            for e in range(self.num_experts):
                load[e] = (indices == e).float().mean()
            self.load_balance_loss = load.var().item()
        seq = seq + out
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 10. Graphormer (图学习) — 空间图注意力
# ============================================================
class GraphormerOp(nn.Module):
    """Graphormer核心: 将空间位置编码为图结构偏置, 注入注意力。
    2D适配: 用空间距离作为注意力偏置。
    创新点: 结构编码注入注意力。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, c * 3)
        self.proj = nn.Linear(c, c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)
        # 空间距离编码
        self.dist_embed = nn.Embedding(32, heads)  # 最多32种距离

    def _spatial_bias(self, H, W, device):
        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        ), dim=-1).float()  # (H, W, 2)
        coords = coords.reshape(H * W, 2)  # (HW, 2)
        dist = (coords.unsqueeze(0) - coords.unsqueeze(1)).norm(dim=-1)  # (HW, HW)
        dist = dist.clamp(max=31).long()
        return self.dist_embed(dist)  # (HW, HW, heads)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        h = self.norm(seq)
        N = H * W
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(dim=2)  # (B, N, heads, dim)
        q = q.transpose(1, 2)  # (B, heads, N, dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 注意力 + 空间偏置
        attn = (q @ k.transpose(-1, -2)) * (C // self.heads) ** -0.5
        bias = self._spatial_bias(H, W, x.device)  # (N, N, heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)  # (1, heads, N, N)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        seq = seq + self.proj(out)
        seq = seq + self.ffn(self.ffn_norm(seq))
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 11. RWKV (线性RNN) — WKV时间混合
# ============================================================
class RWKVOp(nn.Module):
    """RWKV核心: WKV机制 — 用指数加权的键值混合替代自注意力。
    简化实现: 用指数衰减加权和因果卷积模拟WKV。
    创新点: 线性复杂度, 推理恒定时间。"""

    def __init__(self, c: int, act: str = "gelu"):
        super().__init__()
        self.time_mix = nn.Linear(c, c)
        self.channel_mix = nn.Linear(c, c * 2)
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.act = act_fn()
        # WKV 参数
        self.time_decay = nn.Parameter(torch.ones(c) * 0.5)
        self.time_first = nn.Parameter(torch.ones(c) * 0.5)
        self.key = nn.Linear(c, c, bias=False)
        self.value = nn.Linear(c, c, bias=False)
        self.receptance = nn.Linear(c, c, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 时间混合 (WKV简化)
        h = self.norm1(seq)
        # 用1D因果卷积近似WKV
        w = torch.sigmoid(self.time_decay).unsqueeze(0).unsqueeze(0)
        k = self.key(h)
        v = self.value(h)
        r = torch.sigmoid(self.receptance(h))
        # 指数加权平均
        out = torch.zeros_like(seq)
        cum = torch.zeros(B, C, device=x.device)
        weight = torch.zeros(B, C, device=x.device)
        for t in range(seq.shape[1]):
            kt = k[:, t, :]
            vt = v[:, t, :]
            wt = torch.exp(-w.squeeze() * t / max(seq.shape[1], 1))
            w_sum = weight + wt
            out[:, t, :] = r[:, t, :] * (cum + wt * vt) / (w_sum + 1e-6)
            cum = cum + wt * kt * vt
            weight = w_sum
        seq = seq + self.time_mix(out)
        # 通道混合
        h2 = self.norm2(seq)
        kv = self.channel_mix(h2)
        k2, v2 = kv.chunk(2, dim=-1)
        seq = seq + self.act(k2) * v2
        return seq.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 12. Perceiver IO (通用感知) — 潜在数组交叉注意力
# ============================================================
class PerceiverOp(nn.Module):
    """Perceiver IO核心: 用少量潜在向量通过交叉注意力压缩输入。
    创新点: 将模型深度与输入规模解耦, O(N)复杂度。"""

    def __init__(self, c: int, latent_size: int = 16, heads: int = 4, act: str = "gelu"):
        super().__init__()
        self.latent_size = latent_size
        # 可学习的潜在数组
        self.latents = nn.Parameter(torch.randn(1, latent_size, c) * 0.02)
        # 输入 → 潜在 交叉注意力
        self.cross_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.cross_norm_q = nn.LayerNorm(c)
        self.cross_norm_kv = nn.LayerNorm(c)
        # 潜在空间自注意力
        self.self_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(c)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ffn = nn.Sequential(nn.Linear(c, c * 4), act_fn(), nn.Linear(c * 4, c))
        self.ffn_norm = nn.LayerNorm(c)
        # 潜在 → 输出 交叉注意力
        self.output_attn = nn.MultiheadAttention(c, heads, batch_first=True)
        self.output_norm_q = nn.LayerNorm(c)
        self.output_norm_kv = nn.LayerNorm(c)

    def forward(self, x):
        B, C, H, W = x.shape
        kv = x.flatten(2).transpose(1, 2)  # (B, HW, C)
        # 扩展潜在数组
        latents = self.latents.expand(B, -1, -1)  # (B, L, C)
        # 交叉注意力: 输入 → 潜在
        q = self.cross_norm_q(latents)
        k = self.cross_norm_kv(kv)
        cross_out, _ = self.cross_attn(q, k, k)
        latents = latents + cross_out
        # 潜在自注意力
        h = self.self_norm(latents)
        self_out, _ = self.self_attn(h, h, h)
        latents = latents + self_out
        latents = latents + self.ffn(self.ffn_norm(latents))
        # 潜在 → 输出
        q2 = self.output_norm_q(kv)
        k2 = self.output_norm_kv(latents)
        output, _ = self.output_attn(q2, k2, k2)
        return x + output.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 工具函数
# ============================================================
def get_all_ops(c: int) -> dict:
    """返回12个架构组件的字典。"""
    return {
        "Transformer": TransformerOp(c, heads=4),
        "ViT": ViTOp(c, patch_size=2, heads=4),
        "Conformer": ConformerOp(c, heads=4),
        "VideoMAE": VideoMAEOp(c, heads=4),
        "DiT": DiTOp(c, heads=4),
        "PointTrans": PointTransformerOp(c),
        "CLIP": CLIPOp(c, heads=4),
        "Mamba": MambaOp(c, state_size=8),
        "MoE": MoEOp(c, num_experts=4, top_k=2),
        "Graphormer": GraphormerOp(c, heads=4),
        "RWKV": RWKVOp(c),
        "Perceiver": PerceiverOp(c, latent_size=16, heads=4),
    }


ARCH_INFO = {
    "Transformer": {"domain": "NLP文本", "score": 9.4, "core": "多头自注意力+FFN"},
    "ViT": {"domain": "图像分类", "score": 8.4, "core": "Patch+Transformer"},
    "Conformer": {"domain": "语音识别", "score": 8.0, "core": "卷积+注意力并行"},
    "VideoMAE": {"domain": "视频理解", "score": 8.0, "core": "时空分离注意力"},
    "DiT": {"domain": "图像生成", "score": 8.4, "core": "adaLN条件注入"},
    "PointTrans": {"domain": "3D点云", "score": 8.0, "core": "向量注意力"},
    "CLIP": {"domain": "多模态", "score": 8.8, "core": "双编码器对比对齐"},
    "Mamba": {"domain": "高效序列", "score": 8.4, "core": "选择性SSM"},
    "MoE": {"domain": "稀疏路由", "score": 8.8, "core": "Top-k专家路由"},
    "Graphormer": {"domain": "图学习", "score": 8.0, "core": "空间偏置注意力"},
    "RWKV": {"domain": "线性RNN", "score": 8.2, "core": "WKV时间混合"},
    "Perceiver": {"domain": "通用感知", "score": 8.2, "core": "潜在数组交叉注意力"},
}
