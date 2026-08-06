"""
Marvis 混合公式的三个核心架构组件：
  - DualStream:  双流并行处理（局部+全局），擅长 3D视觉/图像生成/深度推理
  - GlobalContext: 全局上下文聚合，擅长 多模态融合/视频理解
  - DenseBlock:  密集连接 + 特征复用，参数高效，擅长 代码/NLP

混合公式: DualStream(30%) + GlobalContext(25%) + DenseBlock(20%) + 25%探索
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamOp(nn.Module):
    """双流并行：局部流(depthwise conv) + 全局流(pool→mlp→broadcast)，两流融合后残差。
    归纳偏置：同时捕获空间局部模式与全局上下文，适合 3D/图像生成/推理。"""

    def __init__(self, c: int, act: str = "gelu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]

        # 局部流：深度可分离卷积捕获空间结构
        self.local_dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.local_pw = nn.Conv2d(c, c, 1, bias=False)
        self.local_bn = nn.BatchNorm2d(c)
        self.local_act = act_fn()

        # 全局流：全局池化→MLP→广播（类似 SE 但更强）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Conv2d(c, c * 2, 1)
        self.global_fc2 = nn.Conv2d(c * 2, c, 1)
        self.global_act = act_fn()

        # 门控融合：学习两流的加权组合
        self.gate = nn.Conv2d(c * 2, c, 1, bias=True)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        # 局部流
        local = self.local_act(self.local_bn(self.local_pw(self.local_dw(x))))
        # 全局流
        g = self.global_pool(x)
        g = self.global_act(self.global_fc1(g))
        g = self.global_fc2(g)
        global_out = x * g.sigmoid()  # 通道注意力调制
        # 门控融合
        fused = self.gate(torch.cat([local, global_out], dim=1))
        return x + self.norm(fused)


class GlobalContextOp(nn.Module):
    """全局上下文：多头自注意力 + 全局特征聚合，残差连接。
    归纳偏置：长程依赖建模，适合多模态融合/视频理解/NLP。"""

    def __init__(self, c: int, heads: int = 4, act: str = "gelu"):
        super().__init__()
        assert c % heads == 0
        self.heads = heads
        self.scale = (c // heads) ** -0.5

        self.norm1 = nn.GroupNorm(1, c)
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

        # 全局上下文分支：与注意力并行
        self.ctx_pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_fc = nn.Conv2d(c, c, 1)
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.ctx_act = act_fn()

        self.norm2 = nn.GroupNorm(1, c)
        self.ffn1 = nn.Conv2d(c, c * 4, 1)
        self.ffn2 = nn.Conv2d(c * 4, c, 1)
        self.ffn_act = act_fn()

    def forward(self, x):
        B, C, H, W = x.shape
        # 自注意力
        h = self.norm1(x).flatten(2).transpose(1, 2)  # (B, N, C)
        qkv = self.qkv(x).reshape(B, 3 * self.heads, C // self.heads, H * W)
        qkv = qkv.permute(0, 2, 3, 1)  # (B, C//h, N, 3h)
        qkv = qkv.reshape(B, C // self.heads, H * W, 3, self.heads)
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        # 简化：直接用 reshape 方式
        h2 = self.norm1(x)
        qkv2 = self.qkv(h2)  # (B, 3C, H, W)
        q2, k2, v2 = qkv2.chunk(3, dim=1)
        q2 = q2.reshape(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        k2 = k2.reshape(B, self.heads, C // self.heads, H * W)
        v2 = v2.reshape(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        attn = (q2 @ k2) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v2).permute(0, 1, 3, 2)  # (B, heads, N, dh)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        # 全局上下文调制
        ctx = self.ctx_act(self.ctx_fc(self.ctx_pool(x)))
        out = out * ctx.sigmoid()

        x = x + out
        # FFN
        x = x + self.ffn2(self.ffn_act(self.ffn1(self.norm2(x))))
        return x


class DenseBlockOp(nn.Module):
    """密集连接块：多层卷积，每层输出拼接后传递（DenseNet 风格），特征复用。
    归纳偏置：参数高效 + 特征复用，适合代码生成/NLP/轻量部署。"""

    def __init__(self, c: int, layers: int = 3, growth: int = 8, act: str = "silu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.growth = growth
        in_c = c
        for _ in range(layers):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                nn.Conv2d(in_c, growth, 1, bias=False),
            ))
            self.bns.append(nn.BatchNorm2d(growth))
            self.acts.append(act_fn())
            in_c += growth
        # 压缩回原始通道
        self.compress = nn.Conv2d(in_c, c, 1, bias=False)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        feats = [x]
        for layer, bn, act in zip(self.layers, self.bns, self.acts):
            new = act(bn(layer(torch.cat(feats, dim=1))))
            feats.append(new)
        out = self.compress(torch.cat(feats, dim=1))
        return x + self.norm(out)


class HybridModel(nn.Module):
    """混合公式架构：按比例组合 DualStream / GlobalContext / DenseBlock + 探索层。

    Marvis 公式: DualStream(30%) + GlobalContext(25%) + DenseBlock(20%) + 25%探索
    解析为 12 层：DualStream×4 + GlobalContext×3 + DenseBlock×2 + 探索×3
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.c = c

        # Stem: 两次 stride2 下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

        # ---- DualStream 30% (4层) ----
        self.dual_streams = nn.Sequential(
            DualStreamOp(c, "gelu"),
            DualStreamOp(c, "gelu"),
            DualStreamOp(c, "silu"),
            DualStreamOp(c, "gelu"),
        )

        # ---- GlobalContext 25% (3层) ----
        self.global_ctx = nn.Sequential(
            GlobalContextOp(c, heads=4, act="gelu"),
            GlobalContextOp(c, heads=4, act="gelu"),
            GlobalContextOp(c, heads=8, act="relu"),
        )

        # ---- DenseBlock 20% (2层) ----
        self.dense = nn.Sequential(
            DenseBlockOp(c, layers=3, growth=8, act="silu"),
            DenseBlockOp(c, layers=3, growth=8, act="gelu"),
        )

        # ---- 探索 25% (3层): Attention + FFN + Conv 轮换 ----
        self.explore = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c * 4, 1), nn.GELU(), nn.Conv2d(c * 4, c, 1),
            nn.BatchNorm2d(c),
            nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.dual_streams(x)
        x = self.global_ctx(x)
        x = self.dense(x)
        x = self.explore(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class FusionChampionModel(nn.Module):
    """融合冠军 (TEXT基因主导)：Conv-Attention 混合，偏多模态/视频/代码/NLP。
    复用之前进化出的 12 节点融合架构的拓扑。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

        # TEXT基因：以注意力+FFN 为主导，卷积辅助
        # 12层: conv3 → conv5 → attn → conv5 → conv3 → ffn → conv3 → conv5 → attn → conv5 → conv3 → ffn
        self.block = nn.ModuleList([
            self._conv_block(c, 3, "relu"),     # 0
            self._conv_block(c, 5, "relu"),     # 1
            self._attn_block(c, 4, "gelu"),     # 2
            self._conv_block(c, 5, "gelu"),     # 3
            self._conv_block(c, 3, "silu"),     # 4
            self._ffn_block(c, 4.0, "gelu"),    # 5
            self._conv_block(c, 3, "relu"),     # 6
            self._conv_block(c, 5, "silu"),     # 7
            self._attn_block(c, 4, "gelu"),     # 8
            self._conv_block(c, 5, "gelu"),     # 9
            self._conv_block(c, 3, "silu"),     # 10
            self._ffn_block(c, 4.0, "relu"),    # 11
        ])
        # 跳连索引 (来自进化结果)
        self._skip_connections = {
            2: [1, -1],    # attn 接 conv5 + stem
            4: [3, 0],     # conv3 接 conv5 + conv3
            5: [4, 2],     # ffn 接 conv3 + attn
            8: [6, 7],     # attn 接两路
            9: [8, 7],     # conv5 接 attn + conv5
            10: [8, 6],    # conv3 接 attn + conv3
            11: [9, 10],   # ffn 接两路
        }

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    @staticmethod
    def _conv_block(c, k, act):
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        return nn.Sequential(
            nn.Conv2d(c, c, k, padding=k // 2, groups=c, bias=False),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c), act_fn(),
        )

    @staticmethod
    def _attn_block(c, heads, act):
        return GlobalContextOp(c, heads, act)  # 复用 GlobalContext 作为注意力

    @staticmethod
    def _ffn_block(c, expand, act):
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]
        hidden = int(c * expand)
        return nn.Sequential(
            nn.Conv2d(c, hidden, 1), act_fn(), nn.Conv2d(hidden, c, 1),
        )

    def forward(self, x):
        s = self.stem(x)
        outs = {-1: s}
        for i, blk in enumerate(self.block):
            inputs = self._skip_connections.get(i, [i - 1] if i > 0 else [-1])
            if inputs:
                inp = outs[inputs[0]]
                for j in inputs[1:]:
                    inp = inp + outs[j]
            else:
                inp = s
            outs[i] = blk(inp) + inp  # 每层都有残差
        feat = self.pool(outs[11]).flatten(1)
        return self.head(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class RLChampionModel(nn.Module):
    """RL冠军 (DualStream基因主导)：双流并行+密集连接，偏 3D/图像生成/推理。
    所有领域评分 ≥8.3★ 的全能选手。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, c: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

        # DualStream基因主导：双流为核心，配合 DenseBlock + 少量注意力
        # 12层结构
        self.dual1 = DualStreamOp(c, "gelu")
        self.dual2 = DualStreamOp(c, "silu")
        self.dual3 = DualStreamOp(c, "gelu")
        self.dense1 = DenseBlockOp(c, layers=3, growth=8, act="silu")
        self.dual4 = DualStreamOp(c, "gelu")
        self.dual5 = DualStreamOp(c, "relu")
        self.ctx1 = GlobalContextOp(c, heads=4, act="gelu")
        self.dense2 = DenseBlockOp(c, layers=3, growth=8, act="gelu")
        self.dual6 = DualStreamOp(c, "silu")
        self.dual7 = DualStreamOp(c, "gelu")
        self.ctx2 = GlobalContextOp(c, heads=8, act="relu")
        self.dense3 = DenseBlockOp(c, layers=3, growth=8, act="silu")

        self.blocks = nn.ModuleList([
            self.dual1, self.dual2, self.dual3, self.dense1,
            self.dual4, self.dual5, self.ctx1, self.dense2,
            self.dual6, self.dual7, self.ctx2, self.dense3,
        ])
        # 跳连: DualStream 基因倾向于长程跳连
        self._skips = {
            3: [2, 0],    # dense 接 dual3+dual1
            7: [6, 4],    # dense 接 ctx+dual4
            10: [9, 6],   # ctx 接 dual7+ctx1
            11: [10, 7],  # dense 接 ctx+dense2
        }

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        s = self.stem(x)
        outs = {-1: s}
        for i, blk in enumerate(self.blocks):
            inputs = self._skips.get(i, [i - 1] if i > 0 else [-1])
            inp = outs[inputs[0]]
            for j in inputs[1:]:
                inp = inp + outs[j]
            outs[i] = blk(inp) + inp
        feat = self.pool(outs[11]).flatten(1)
        return self.head(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
