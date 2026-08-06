# 架构涌现实验室 (arch_lab)

> 想法：复杂系统里“属性/元素的创造与组合”会产生**涌现**。那么对 **AI 模型架构本身**做“融合 + 创造”，用进化搜索当环境压力，能不能让一个“极好的架构”自发涌现出来？本仓库用代码做这个实验。

## 核心思路

把模型架构编码成一张**有向无环图 (DAG)** 的“基因组”，在基因组空间里用进化算法搜索：

- **基因 (Gene)**：一个计算算子（深度可分离卷积、前馈 FFN、自注意力、归一化、恒等残差…），带可变异超参（激活函数、扩展倍率、注意力头数）。
- **创造 (Mutation)**：在单个基因组内部换算子、改超参、增删节点、增删跳连——制造新结构。
- **融合 (Crossover/Fusion)**：取一个亲本的一段子图嫁接进另一个亲本，或把两个网络串联并加跨支路——杂交出新拓扑。
- **环境压力 (Fitness)**：先以**免训练代理指标**（SynFlow 式梯度信号，呼应最新 NAS 高效范式）大规模筛选，再对少量精英做真实训练得到验证准确率；适应度同时惩罚参数量。
- **涌现追踪 (Emergence)**：每代统计精英群体中架构 motif（算子、相邻组合、长跨跳连、attn+ffn 邻接…）的频率，观察哪些模式从“罕见”自发变到“普及”。

所有中间张量保持常量宽度与空间尺寸，因此节点输入可直接“求和”实现残差融合，保证融合/创造后拓扑始终合法。

## 安装

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision matplotlib
```

## 用法

```bash
# 推荐：代理模式（快），自动对精英补真实训练
python -m arch_lab.main --pop 16 --gens 6 --mode proxy

# 极速冒烟测试
python -m arch_lab.main --pop 8 --gens 3 --mode proxy --full-train-top 2

# 全量训练每个个体（慢但准）
python -m arch_lab.main --pop 8 --gens 4 --mode full --epochs 2

# CIFAR-10
python -m arch_lab.main --dataset cifar10 --pop 12 --gens 6 --channels 48
```

## 输出

运行后在 `runs/<tag>/` 生成：
- `summary.json`：最优架构、每代指标、涌现 motif 列表
- `evolution.png`：适应度曲线与精英准确率
- `motifs.png`：架构 motif 频率随代数的演变
- `best_arch.png`：最优架构 DAG 可视化

## 文件结构

| 文件 | 作用 |
|------|------|
| `config.py` | 全部超参数 |
| `genome.py` | 基因/基因组(DAG)/随机生成/模型构建器 |
| `operations.py` | 创造(变异) + 融合(交叉/嫁接) |
| `evaluator.py` | 免训练代理 + 真实训练 + 多目标适应度 |
| `emergence.py` | motif 提取与涌现检测 |
| `evolution.py` | 进化引擎(选择/繁衍/精英) |
| `viz.py` | 可视化 |
| `main.py` | 命令行入口 |

## 重要说明

- 免训练代理只是**排序启发式**，与最终精度并非线性相关；它用于在 CPU 上快速探索大空间，真正结论仍以精英的真实训练为准。
- “涌现”在本实验里是**统计意义的观察**（某 motif 在精英中频率越过阈值），不是理论证明。要看到稳健的涌现，通常需要更大的种群与更多代数。
- 默认在 MNIST 子集上跑，便于在普通 CPU 上快速迭代；放大到完整数据集/更久训练可得到更有说服力的架构。
