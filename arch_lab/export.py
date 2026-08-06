"""
架构导出：把基因组编译成**独立的、可直接运行的 PyTorch 代码文件**。
导出的代码不依赖 arch_lab，包含所有算子定义 + 模型类，可直接 import 使用。
"""
from __future__ import annotations
import os
from typing import Optional
from .genome import Genome


_OP_TEMPLATE = {
    "conv3": '        nn.Conv2d({c}, {c}, 3, padding=1, groups={c}, bias=False),',
    "conv5": '        nn.Conv2d({c}, {c}, 5, padding=2, groups={c}, bias=False),',
    "conv1": '        nn.Conv2d({c}, {c}, 1, bias=False),',
}

_ACT_MAP = {"relu": "nn.ReLU", "gelu": "nn.GELU", "silu": "nn.SiLU"}


def _op_class_code(node_idx: int, node, c: int) -> str:
    """生成一个节点的 nn.Module 子类代码。"""
    act = _ACT_MAP.get(node.act, "nn.ReLU")
    if node.op in ("conv3", "conv5", "conv1"):
        k = {"conv3": 3, "conv5": 5, "conv1": 1}[node.op]
        return f"""class Node{node_idx}Op(nn.Module):
    \"\"\"深度可分离卷积(k={k}) + BN + {act} + 残差\"\"\"
    def __init__(self, c={c}):
        super().__init__()
        self.dw = nn.Conv2d(c, c, {k}, padding={k//2}, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = {act}()

    def forward(self, x):
        return x + self.act(self.bn(self.pw(self.dw(x))))
"""
    if node.op == "ffn":
        hidden = max(c, int(c * node.expand))
        return f"""class Node{node_idx}Op(nn.Module):
    \"\"\"点式前馈(expand={node.expand}) + {act} + 残差\"\"\"
    def __init__(self, c={c}):
        super().__init__()
        self.fc1 = nn.Conv2d(c, {hidden}, 1)
        self.fc2 = nn.Conv2d({hidden}, c, 1)
        self.act = {act}()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))
"""
    if node.op == "attn":
        return f"""class Node{node_idx}Op(nn.Module):
    \"\"\"自注意力(heads={node.heads}) + 残差\"\"\"
    def __init__(self, c={c}):
        super().__init__()
        assert c % {node.heads} == 0
        self.norm = nn.GroupNorm(1, c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.heads = {node.heads}
        self.scale = (c // {node.heads}) ** -0.5

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
"""
    if node.op == "identity":
        return f"""class Node{node_idx}Op(nn.Module):
    def __init__(self, c={c}):
        super().__init__()
    def forward(self, x):
        return x
"""
    if node.op == "norm_act":
        return f"""class Node{node_idx}Op(nn.Module):
    \"\"\"BN + {act}\"\"\"
    def __init__(self, c={c}):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)
        self.act = {act}()
    def forward(self, x):
        return self.act(self.bn(x))
"""
    return f"""class Node{node_idx}Op(nn.Module):
    def __init__(self, c={c}):
        super().__init__()
    def forward(self, x):
        return x
"""


def export_genome(genome: Genome, filepath: str,
                  in_channels: int = 1, num_classes: int = 10,
                  channels: int = 32, model_name: str = "EvolvedModel") -> str:
    """把基因组导出为独立的 PyTorch 代码文件。返回文件路径。"""
    c = channels
    lines = []
    lines.append('"""')
    lines.append(f'自动生成的模型代码 (由 arch_lab 导出)')
    lines.append(f'节点数: {len(genome)}, 宽度: {c}, 输入通道: {in_channels}, 类别: {num_classes}')
    lines.append('可直接运行: python <this_file>.py')
    lines.append('"""')
    lines.append("import torch")
    lines.append("import torch.nn as nn")
    lines.append("import torch.nn.functional as F")
    lines.append("")
    lines.append("")

    # 生成每个节点的 Module 类
    for i, node in enumerate(genome.nodes):
        lines.append(_op_class_code(i, node, c))
        lines.append("")

    # 生成主模型类
    node_names = ", ".join(f"Node{i}Op(c)" for i in range(len(genome.nodes)))
    lines.append(f"class {model_name}(nn.Module):")
    lines.append(f'    """进化搜索发现的架构 (节点数={len(genome)})"""')
    lines.append(f"    def __init__(self, in_channels={in_channels}, num_classes={num_classes}, c={c}):")
    lines.append("        super().__init__()")

    # stem
    lines.append("        self.stem = nn.Sequential(")
    lines.append("            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),")
    lines.append("            nn.BatchNorm2d(c), nn.ReLU(inplace=True),")
    lines.append("            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),")
    lines.append("            nn.BatchNorm2d(c), nn.ReLU(inplace=True),")
    lines.append("        )")
    # nodes
    lines.append(f"        self.nodes = nn.ModuleList([{node_names}])")
    lines.append("        self.pool = nn.AdaptiveAvgPool2d(1)")
    lines.append("        self.head = nn.Linear(c, num_classes)")
    lines.append("")

    # forward
    lines.append("    def forward(self, x):")
    lines.append("        s = self.stem(x)")
    lines.append("        outs = {-1: s}")
    for i, node in enumerate(genome.nodes):
        if node.inputs:
            inp_expr = " + ".join(f"outs[{j}]" for j in node.inputs)
            lines.append(f"        outs[{i}] = self.nodes[{i}]({inp_expr})")
        else:
            lines.append(f"        outs[{i}] = self.nodes[{i}](s)")
    # terminals
    terminals = genome.terminal_nodes()
    if terminals:
        feat_expr = " + ".join(f"outs[{t}]" for t in terminals)
        lines.append(f"        feat = self.pool({feat_expr}).flatten(1)")
    else:
        lines.append(f"        feat = self.pool(outs[{len(genome.nodes)-1}]).flatten(1)")
    lines.append("        return self.head(feat)")
    lines.append("")
    lines.append("")

    # 测试代码
    lines.append("if __name__ == '__main__':")
    lines.append(f"    model = {model_name}()")
    lines.append(f"    x = torch.randn(4, {in_channels}, 28, 28)")
    lines.append("    out = model(x)")
    lines.append("    n_params = sum(p.numel() for p in model.parameters())")
    lines.append('    print(f"Output shape: {out.shape}")')
    lines.append('    print(f"Parameters: {n_params:,}")')
    lines.append("")

    code = "\n".join(lines)
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code)
    return filepath
