"""
Marvis 冠军大乱斗终极架构 — 精确复刻版
12层变通道宽度, AUDIO+VIDEO 定向进化冠军

层 0:  Dropout(c=87, k=1, act=relu)
层 1:  MultiScale(levels=5, c=166)
层 2:  WindowAttention(win=7, heads=8)
层 3:  WindowAttention(win=7, heads=16)
层 4:  BatchNorm2d(c=60, k=5, act=silu)
层 5:  GlobalContext(ratio=0.5, c=210)
层 6:  MaxPool2d(c=46, k=3, act=gelu)
层 7:  DenseBlock(gr=32, L=6)
层 8:  LayerNorm(c=64, k=3, act=gelu)
层 9:  DualStream(branches=3, c=116)
层10:  Linear(c=254, k=4, act=gelu)
层11:  MultiScale(levels=5, c=113)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multiscale_ops import MultiScaleOp, WindowAttentionOp
from .hybrid_ops import DenseBlockOp


class DualStream3(nn.Module):
    """三流并行 DualStream: 局部(conv) + 全局(SE) + 中尺度(dilated conv)。"""

    def __init__(self, c: int, act: str = "gelu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]

        # 流1: 局部 — 小核 depthwise conv
        self.local_dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.local_pw = nn.Conv2d(c, c, 1, bias=False)
        self.local_bn = nn.BatchNorm2d(c)
        self.local_act = act_fn()

        # 流2: 全局 — SE 通道注意力
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc1 = nn.Conv2d(c, c * 2, 1)
        self.global_fc2 = nn.Conv2d(c * 2, c, 1)
        self.global_act = act_fn()

        # 流3: 中尺度 — 膨胀卷积 (dilation=2, 感受野≈5)
        self.mid_dw = nn.Conv2d(c, c, 3, padding=2, dilation=2, groups=c, bias=False)
        self.mid_pw = nn.Conv2d(c, c, 1, bias=False)
        self.mid_bn = nn.BatchNorm2d(c)
        self.mid_act = act_fn()

        # 三流门控融合
        self.gate = nn.Conv2d(c * 3, c, 1, bias=True)
        self.norm = nn.BatchNorm2d(c)

    def forward(self, x):
        local = self.local_act(self.local_bn(self.local_pw(self.local_dw(x))))
        g = self.global_act(self.global_fc1(self.global_pool(x)))
        g = self.global_fc2(g)
        global_out = x * g.sigmoid()
        mid = self.mid_act(self.mid_bn(self.mid_pw(self.mid_dw(x))))
        fused = self.gate(torch.cat([local, global_out, mid], dim=1))
        return x + self.norm(fused)


class GlobalContextRatio(nn.Module):
    """GlobalContext with ratio parameter — 控制注意力与上下文的混合比例。"""

    def __init__(self, c: int, ratio: float = 0.5, heads: int = 4, act: str = "gelu"):
        super().__init__()
        assert c % heads == 0
        self.ratio = ratio
        self.heads = heads
        self.scale = (c // heads) ** -0.5
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[act]

        self.norm1 = nn.GroupNorm(1, c)
        self.qkv = nn.Conv2d(c, c * 3, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)

        self.ctx_pool = nn.AdaptiveAvgPool2d(1)
        self.ctx_fc = nn.Conv2d(c, c, 1)
        self.ctx_act = act_fn()

        self.norm2 = nn.GroupNorm(1, c)
        self.ffn1 = nn.Conv2d(c, c * 4, 1)
        self.ffn2 = nn.Conv2d(c * 4, c, 1)
        self.ffn_act = act_fn()

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        k = k.reshape(B, self.heads, C // self.heads, H * W)
        v = v.reshape(B, self.heads, C // self.heads, H * W).permute(0, 1, 3, 2)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        attn_out = self.proj(attn_out)

        ctx = self.ctx_act(self.ctx_fc(self.ctx_pool(x)))
        ctx_out = x * ctx.sigmoid()

        # ratio 控制注意力与上下文的混合
        out = self.ratio * attn_out + (1 - self.ratio) * (ctx_out - x)
        x = x + out
        x = x + self.ffn2(self.ffn_act(self.ffn1(self.norm2(x))))
        return x


class MarvisChampionModel(nn.Module):
    """Marvis 冠军大乱斗终极架构 — 精确复刻。
    变通道宽度, 每层间用 1x1 conv 做通道投影。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10, stem_c: int = 64):
        super().__init__()

        # 层输出通道 (按 Marvis 配方)
        ch = [87, 166, 166, 166, 60, 210, 46, 46, 64, 116, 254, 113]

        # Stem: 两次下采样
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c), nn.ReLU(inplace=True),
            nn.Conv2d(stem_c, stem_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_c), nn.ReLU(inplace=True),
        )

        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

        # ---- 逐层构建 ----
        # L0: Dropout(c=87, k=1, act=relu) — 1x1 conv + dropout + relu
        self.proj0 = nn.Conv2d(stem_c, ch[0], 1, bias=False)
        self.l0 = nn.Sequential(
            nn.Dropout(0.1),
            nn.BatchNorm2d(ch[0]), nn.ReLU(),
        )

        # L1: MultiScale(levels=5, c=166)
        self.proj1 = nn.Conv2d(ch[0], ch[1], 1, bias=False)
        self.l1 = MultiScaleOp(ch[1], levels=5, act="gelu")

        # L2: WindowAttention(win=7, heads=8) — 适配通道数
        h2 = next((h for h in [8, 4, 2, 1] if ch[1] % h == 0), 2)
        self.l2 = WindowAttentionOp(ch[1], heads=h2, window_size=7, act="gelu")

        # L3: WindowAttention(win=7, heads=16) — 适配通道数
        h3 = next((h for h in [16, 8, 4, 2, 1] if ch[1] % h == 0), 2)
        self.l3 = WindowAttentionOp(ch[1], heads=h3, window_size=7, act="gelu")

        # L4: BatchNorm2d(c=60, k=5, act=silu) — conv k=5 + BN + silu
        self.proj4 = nn.Conv2d(ch[1], ch[4], 5, padding=2, bias=False)
        self.l4 = nn.Sequential(nn.BatchNorm2d(ch[4]), nn.SiLU())

        # L5: GlobalContext(ratio=0.5, c=210)
        self.proj5 = nn.Conv2d(ch[4], ch[5], 1, bias=False)
        heads5 = next((h for h in [8, 4, 2] if ch[5] % h == 0), 4)
        self.l5 = GlobalContextRatio(ch[5], ratio=0.5, heads=heads5, act="gelu")

        # L6: MaxPool2d(c=46, k=3, act=gelu) — conv k=3 + maxpool + gelu
        self.proj6 = nn.Conv2d(ch[5], ch[6], 3, stride=1, padding=1, bias=False)
        self.l6 = nn.Sequential(nn.MaxPool2d(2, 2), nn.BatchNorm2d(ch[6]), nn.GELU())

        # L7: DenseBlock(gr=32, L=6) — 在 ch[6]=46 上做密集连接
        self.l7 = DenseBlockOp(ch[6], layers=6, growth=min(32, ch[6]), act="gelu")

        # L8: LayerNorm(c=64, k=3, act=gelu) — conv k=3 + LayerNorm + gelu
        self.proj8 = nn.Conv2d(ch[6], ch[8], 3, padding=1, bias=False)
        self.l8 = nn.Sequential(nn.GroupNorm(1, ch[8]), nn.GELU())

        # L9: DualStream(branches=3, c=116)
        self.proj9 = nn.Conv2d(ch[8], ch[9], 1, bias=False)
        self.l9 = DualStream3(ch[9], act="gelu")

        # L10: Linear(c=254, k=4, act=gelu) — FFN expand
        self.proj10 = nn.Conv2d(ch[9], ch[10], 1, bias=False)
        self.l10 = nn.Sequential(nn.GELU(), nn.BatchNorm2d(ch[10]))

        # L11: MultiScale(levels=5, c=113)
        self.proj11 = nn.Conv2d(ch[10], ch[11], 1, bias=False)
        self.l11 = MultiScaleOp(ch[11], levels=5, act="gelu")

        # 输出
        self.final_proj = nn.Conv2d(ch[11], 64, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.l0(self.proj0(x))
        x = self.l1(self.proj1(x))
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(self.proj4(x))
        x = self.l5(self.proj5(x))
        x = self.l6(self.proj6(x))
        x = self.l7(x)
        x = self.l8(self.proj8(x))
        x = self.l9(self.proj9(x))
        x = self.l10(self.proj10(x))
        x = self.l11(self.proj11(x))

        x = self.final_proj(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
