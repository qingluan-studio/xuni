"""
自动生成的模型代码 (由 arch_lab 导出)
节点数: 12, 宽度: 32, 输入通道: 1, 类别: 10
可直接运行: python <this_file>.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Node0Op(nn.Module):
    """深度可分离卷积(k=3) + BN + nn.ReLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node1Op(nn.Module):
    """深度可分离卷积(k=5) + BN + nn.ReLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node2Op(nn.Module):
    """自注意力(heads=4) + 残差"""
    def __init__(self, c=32):
        super().__init__()
        assert c % 4 == 0
        self.norm = nn.GroupNorm(1, c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.heads = 4
        self.scale = (c // 4) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        qkv = self.qkv(h).reshape(B, -1, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class Node3Op(nn.Module):
    """深度可分离卷积(k=5) + BN + nn.GELU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node4Op(nn.Module):
    """深度可分离卷积(k=3) + BN + nn.SiLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node5Op(nn.Module):
    """点式前馈(expand=4.0) + nn.GELU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.fc1 = nn.Conv2d(c, 128, 1)
        self.fc2 = nn.Conv2d(128, c, 1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class Node6Op(nn.Module):
    """深度可分离卷积(k=3) + BN + nn.ReLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node7Op(nn.Module):
    """深度可分离卷积(k=5) + BN + nn.SiLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node8Op(nn.Module):
    """自注意力(heads=4) + 残差"""
    def __init__(self, c=32):
        super().__init__()
        assert c % 4 == 0
        self.norm = nn.GroupNorm(1, c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.heads = 4
        self.scale = (c // 4) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        qkv = self.qkv(h).reshape(B, -1, 3, self.heads, C // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out).transpose(1, 2).reshape(B, C, H, W)
        return x + out


class Node9Op(nn.Module):
    """深度可分离卷积(k=5) + BN + nn.GELU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, padding=2, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node10Op(nn.Module):
    """深度可分离卷积(k=3) + BN + nn.SiLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))


class Node11Op(nn.Module):
    """点式前馈(expand=4.0) + nn.ReLU + 残差"""
    def __init__(self, c=32):
        super().__init__()
        self.fc1 = nn.Conv2d(c, 128, 1)
        self.fc2 = nn.Conv2d(128, c, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


class EvolvedModel(nn.Module):
    """进化搜索发现的架构 (节点数=12)"""
    def __init__(self, in_channels=1, num_classes=10, c=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.nodes = nn.ModuleList([Node0Op(c), Node1Op(c), Node2Op(c), Node3Op(c), Node4Op(c), Node5Op(c), Node6Op(c), Node7Op(c), Node8Op(c), Node9Op(c), Node10Op(c), Node11Op(c)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(c, num_classes)

    def forward(self, x):
        s = self.stem(x)
        outs = {-1: s}
        outs[0] = self.nodes[0](outs[-1])
        outs[1] = self.nodes[1](outs[0])
        outs[2] = self.nodes[2](outs[1] + outs[-1])
        outs[3] = self.nodes[3](outs[2] + outs[1])
        outs[4] = self.nodes[4](outs[3] + outs[0])
        outs[5] = self.nodes[5](outs[4] + outs[2])
        outs[6] = self.nodes[6](outs[5])
        outs[7] = self.nodes[7](outs[5])
        outs[8] = self.nodes[8](outs[6] + outs[7])
        outs[9] = self.nodes[9](outs[8] + outs[7])
        outs[10] = self.nodes[10](outs[8] + outs[6])
        outs[11] = self.nodes[11](outs[9] + outs[10])
        feat = self.pool(outs[11]).flatten(1)
        return self.head(feat)


if __name__ == '__main__':
    model = EvolvedModel()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {n_params:,}")
