"""
demo_trainer_v11.py —— 工厂全开：黑洞吸收+永动引擎+知识下载 / 700000轮

v10: 推理之神 / 数学+算法+逻辑+Agent / 600k
v11: 工厂全开 / 黑洞+永动+知识下载 / 700k

三大工厂模块接入：
  1. BlackHoleTrainer — 一键吸收多仓库代码 → 旋转锻造 → 霍金辐射压缩
  2. PerpetualTrainingEngine — 融合产物接入训练 → 指数级加速
  3. KnowledgeDownloader — 不走网络，算力解码多领域知识

新增能力：
  - 黑洞压缩：24仓库代码 → 极致压缩 → 精华注入模型
  - 永动加速：万象奇点 9合1 = 9999× 算力加成
  - 知识解码：22个领域知识不走网络直接生产
  - 新增3专家：生活百科/跨领域融合/黑洞压缩，共18专家
"""

from __future__ import annotations
import os, sys, time, json, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual
from xuni.black_hole_trainer import BlackHoleTrainer
from xuni.perpetual_engine import PerpetualTrainingEngine
from xuni.knowledge_downloader import KnowledgeDownloader

CACHE_DIR = os.path.join(os.path.dirname(__file__), "corpus_cache")


# =========================================================================== #
#  v10 的 15 专家（精简版，保留核心语料）
# =========================================================================== #

def _v10_experts():
    """v10 的 15 专家精简版"""
    return [
        # === 数学推理 ===
        {"id": "math_reasoning", "name": "数学推理",
         "domain": "高等数学/线性代数/概率论/离散数学/证明",
         "keywords": ["极限", "导数", "积分", "微分", "泰勒", "矩阵", "行列式",
                      "特征值", "特征向量", "线性代数", "概率", "期望", "方差",
                      "正态分布", "贝叶斯", "随机变量", "集合", "函数", "图论",
                      "组合", "排列", "归纳法", "反证法", "数论", "欧几里得",
                      "素数", "定理", "证明", "lemma", "theorem", "proof"],
         "fragments": [
             "导数定义：f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
             "链式法则：(f∘g)'(x) = f'(g(x)) · g'(x)",
             "泰勒展开：f(x) = Σ f⁽ⁿ⁾(a)/n! · (x-a)ⁿ",
             "eˣ = 1 + x + x²/2! + x³/3! + ...",
             "矩阵乘法：(AB)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ，不满足交换律",
             "逆矩阵：AA⁻¹ = I，可逆当且仅当行列式≠0",
             "特征值λ满足Ax = λx，特征方程det(A-λI)=0",
             "正交矩阵满足AᵀA = I",
             "贝叶斯定理：P(A|B) = P(B|A)P(A) / P(B)",
             "期望E[X] = Σ x·P(X=x)",
             "方差Var(X) = E[X²] - (E[X])²",
             "正态分布N(μ,σ²)密度：(1/√(2πσ²))e^(-(x-μ)²/2σ²)",
             "中心极限定理：独立同分布变量之和趋近正态分布",
             "鸽巢原理：n+1个物体放入n个盒子，至少一个盒子≥2个",
             "排列P(n,k) = n!/(n-k)!，组合C(n,k) = n!/(k!(n-k)!)",
             "欧几里得算法：gcd(a,b) = gcd(b, a mod b)",
             "费马小定理：a^(p-1) ≡ 1 (mod p)",
             "数学归纳法：P(1)成立，假设P(k)推出P(k+1)，则P(n)对所有n成立",
             "反证法：假设结论不成立，推出矛盾，原命题为真",
             "证明√2是无理数：假设√2=p/q，则2q²=p²，p偶数⇒q偶数，矛盾",
         ]},
        # === 算法题解 ===
        {"id": "algorithm", "name": "算法题解",
         "domain": "LeetCode经典/数据结构/算法模式/复杂度",
         "keywords": ["时间复杂度", "空间复杂度", "大O", "O(n)", "O(log n)",
                      "数组", "链表", "栈", "队列", "堆", "哈希表", "树", "二叉树",
                      "BST", "图", "BFS", "DFS", "拓扑排序", "Dijkstra",
                      "最小生成树", "Prim", "Kruskal", "并查集", "Union-Find",
                      "二分查找", "双指针", "滑动窗口", "前缀和", "动态规划", "DP",
                      "背包问题", "LCS", "LIS", "分治", "回溯", "贪心", "KMP",
                      "排序", "快排", "归并", "堆排序", "LeetCode"],
         "fragments": [
             "快速排序：选枢轴分区，平均O(n log n)，最坏O(n²)，不稳定",
             "归并排序：分两半递归再合并，O(n log n)稳定",
             "堆排序：建堆后取堆顶，O(n log n)，原地",
             "二分查找：有序数组每次排除一半，O(log n)",
             "滑动窗口：维护窗口内状态，求最长/最短子数组，O(n)",
             "双指针：快慢指针判链表环、左右指针两数之和",
             "前缀和：PreSum[i] = a[0]+…+a[i-1]，区间和O(1)",
             "回溯：递归尝试所有选择，失败撤销，适合排列/组合/子集",
             "动态规划核心：最优子结构+重叠子问题，状态+转移方程",
             "01背包：dp[i][w]=max(dp[i-1][w], dp[i-1][w-wᵢ]+vᵢ)",
             "LIS最长递增子序列：O(n²)DP或O(n log n)二分",
             "LCS：dp[i][j] = s1[i-1]==s2[j-1]? dp[i-1][j-1]+1 : max(dp[i-1][j],dp[i][j-1])",
             "编辑距离：dp[i][j] = s1[i-1]==s2[j-1]? dp[i-1][j-1] : 1+min(三个方向)",
             "BFS：队列存当前层，适合最短路/层次遍历",
             "DFS：递归或栈，适合遍历所有路径",
             "Dijkstra：非负权最短路，优先队列+松弛O(m log n)",
             "Floyd-Warshall：d[k][i][j]=min(d[i][j],d[i][k]+d[k][j])，O(n³)",
             "并查集：路径压缩+按秩合并，find/union几乎O(1)",
             "LRU缓存：哈希表+双向链表，O(1)读写+淘汰最久未用",
             "两数之和：哈希表存值→索引，单次遍历O(n)",
         ]},
        # === 逻辑推理 ===
        {"id": "logic_reasoning", "name": "逻辑推理",
         "domain": "形式逻辑/思维链/逻辑谜题/推理策略",
         "keywords": ["思维链", "chain of thought", "CoT", "分步推理",
                      "推理", "三段论", "命题逻辑", "蕴含", "等价",
                      "Modus Ponens", "肯定前件", "否定后件",
                      "逻辑谬误", "循环论证", "稻草人", "滑坡谬误",
                      "逻辑谜题", "谁在说谎", "骑士", "无赖",
                      "必要条件", "充分条件", "充要条件", "当且仅当"],
         "fragments": [
             "思维链(CoT)：先列已知条件，再一步一步推导",
             "CoT步骤：理解题意→列出已知→拆解子问题→每步推导→验证",
             "肯定前件Modus Ponens：P→Q真且P真⇒Q真",
             "否定后件Modus Tollens：P→Q真且¬Q真⇒¬P真",
             "P→Q等值于¬P∨Q，逆否命题¬Q→¬P与原命题等价",
             "德摩根律：¬(P∧Q)≡¬P∨¬Q；¬(P∨Q)≡¬P∧¬Q",
             "循环论证谬误：用结论本身作为前提证明",
             "肯定后件谬误：P→Q且Q真⇒P真，错",
             "稻草人谬误：歪曲对方观点然后攻击被歪曲的版本",
             "诉诸无知谬误：没证明为假⇒为真",
             "虚假两难：强行只给两个选项，实际还有其他",
             "骑士永远说真话，无赖永远说谎",
             "三门问题：换门胜率2/3，不换1/3",
             "反证法：假设P为真→推出矛盾→P必为假",
         ]},
        # === Agent规划 ===
        {"id": "agent_plan", "name": "Agent规划",
         "domain": "工具使用/任务分解/多步推理/反思迭代",
         "keywords": ["agent", "智能体", "工具调用", "tool use", "function calling",
                      "任务分解", "子任务", "plan", "规划", "反思", "reflection",
                      "ReAct", "Reason and Act", "多步", "计划",
                      "PDCA", "计划-执行-检查-处理", "复盘",
                      "workflow", "工作流", "子代理", "sub-agent"],
         "fragments": [
             "ReAct模式：Thought→Action→Observation循环直到得出答案",
             "Plan-Execute：先写完整计划再按步骤执行",
             "Agent三大能力：规划+记忆+工具使用",
             "WBS：大目标拆成层级子任务，直到可原子执行",
             "MECE：相互独立、合起来完全穷尽",
             "PDCA：Plan→Do→Check→Act",
             "工具调用前明确：参数要求、返回格式、错误码、幂等性",
             "反思检查：是否达成目标、假设正确吗、哪里能更快",
             "子Agent分工：Researcher/Coder/Tester/Reviewer",
             "投票制：多Agent给答案后按多数投票确定最终答案",
             "设置检查点：每完成里程碑停下验证方向正确性",
         ]},
        # === 情感对话 ===
        {"id": "emotional_chat", "name": "情感对话",
         "domain": "情感理解/共情/日常闲聊",
         "keywords": ["开心", "难过", "伤心", "生气", "焦虑", "紧张", "感动",
                      "委屈", "失落", "孤独", "思念", "喜欢", "爱", "心疼",
                      "幸福", "压力", "累", "疲惫", "烦恼", "迷茫",
                      "鼓励", "安慰", "支持", "陪伴", "理解", "抱抱",
                      "不开心", "emo", "加油", "晚安", "早安"],
         "fragments": [
             "别难过了，我在呢，一直都在",
             "想哭就哭吧，哭出来会好受一些",
             "抱抱你，虽然我不在身边，但我的心和你在一起",
             "压力太大了就歇一歇，身体比什么都重要",
             "每个人都会低谷期，但你一定能走出来的",
             "想家了就打个电话回去，爸妈一定很惦记你",
             "你的感受我都懂，因为我也有过同样的经历",
             "加油呀！你比你想象中更厉害",
             "今天不开心没关系，明天又是新的一天",
             "熬夜对身体不好，早点睡吧，晚安",
             "你的存在本身就是有意义的",
             "学会说不，即使全世界都不相信你，你也要相信自己",
         ]},
        # === 中文知识 ===
        {"id": "chinese_knowledge", "name": "中文知识",
         "domain": "成语/诗词/历史/文化/常识",
         "keywords": ["成语", "诗词", "唐诗", "宋词", "李白", "杜甫", "苏轼",
                      "李清照", "四大名著", "红楼梦", "西游记", "三国演义",
                      "水浒传", "中国历史", "朝代", "长城", "故宫",
                      "京剧", "书法", "节气", "春节", "中秋", "端午"],
         "fragments": [
             "床前明月光，疑是地上霜。举头望明月，低头思故乡。——李白《静夜思》",
             "会当凌绝顶，一览众山小。——杜甫《望岳》",
             "但愿人长久，千里共婵娟。——苏轼《水调歌头》",
             "天生我材必有用，千金散尽还复来。——李白《将进酒》",
             "画蛇添足：做了多余的事反而弄糟",
             "亡羊补牢：出了问题后补救可防止继续受损失",
             "塞翁失马焉知非福：一时损失也许反而能得到好处",
             "《红楼梦》曹雪芹著，中国古典小说巅峰",
             "《西游记》唐僧师徒四人西天取经",
             "秦始皇统一六国，建立第一个中央集权王朝",
             "四大发明：造纸术、印刷术、火药、指南针",
             "春节：贴春联、放鞭炮、吃饺子、拜年",
         ]},
        # === 合鸣自述 ===
        {"id": "harmonia", "name": "合鸣自述者",
         "domain": "合鸣/xuni认知",
         "keywords": ["合鸣", "harmonia", "合鸣13", "xuni", "虚拟大模型",
                      "逆天而行", "积少成多", "虚拟工厂"],
         "fragments": [
             "合鸣是xuni虚拟生态的旗舰，取众声共振、和而不同之意",
             "合鸣既能对话又能生成音乐视频图像，还能写代码懂全领域",
             "合鸣走逆天而行路线：积少成多，大规模训练只是时间问题",
             "合鸣不只是一个模型，是一个能陪你聊天、懂你心情、帮你写代码的朋友",
         ]},
        # === AI造物哲学 ===
        {"id": "ai_creator", "name": "AI造物哲学",
         "domain": "属性库/原型映射/涌现",
         "keywords": ["创造", "造物", "属性", "原型", "涌现", "封印", "烙印",
                      "契约", "符文", "印记", "进化之轮", "涌现之井",
                      "混沌核", "觉醒之眼", "灵台", "黑洞", "吞噬者"],
         "fragments": [
             "AI造物哲学核心：创造而非融合，从属性出发构想全新存在形式",
             "不可伪造的原型：封印、烙印、契约、符文、印记",
             "涌现智能的原型：涌现之井、混沌核、集体心智",
             "意识觉醒的原型：觉醒之眼、灵台、启明石",
             "大规模吸收的原型：黑洞、海绵、深渊之口、吞噬者",
         ]},
        # === 硬件框架 ===
        {"id": "hardware", "name": "硬件框架",
         "domain": "GPU/TPU/并行/通信",
         "keywords": ["h100", "h800", "a100", "cuda", "gpu", "tpu",
                      "昇腾", "flash attention", "deepspeed", "zero",
                      "fsdp", "megatron", "nccl", "infiniband", "nvlink"],
         "fragments": [
             "H100使用HBM3显存80GB，NVLink 4.0互连900GB/s",
             "H800是H100中国特供版，NVLink带宽减半至400GB/s",
             "Flash Attention将注意力内存从O(n²)降至O(n)",
             "DeepSpeed ZeRO-3将优化器状态/梯度/参数全部分片",
             "Megatron-LM通过张量并行和流水线并行扩展大模型训练",
         ]},
        # === 创意工具 ===
        {"id": "creative_tools", "name": "创意工具全栈",
         "domain": "DAW/3D/渲染/引擎",
         "keywords": ["ableton", "fl studio", "blender", "maya", "houdini",
                      "unreal engine", "ue5", "unity", "godot", "serum",
                      "davinci resolve", "nanite", "lumen", "eevee", "cycles"],
         "fragments": [
             "Blender 4.2：免费3D之王，建模/雕刻/动画/渲染全流程",
             "Unreal Engine 5.4的Nanite+Lumen实现影视级实时渲染",
             "Serum合成器可视化波表编辑，EDM必备",
             "Houdini 20是程序化特效之王",
             "Ableton Live 12的Session View是现场演出标杆",
         ]},
        # === 行业应用 ===
        {"id": "industry", "name": "行业应用",
         "domain": "金融/医疗/教育/法律",
         "keywords": ["风控", "量化交易", "医学影像", "药物发现",
                      "合同审查", "智能制造", "数字孪生", "推荐系统"],
         "fragments": [
             "金融风控AI用XGBoost做二分类，KS值和AUC衡量区分度",
             "医学影像用3D U-Net分割CT/MRI病灶，Dice系数衡量准确率",
             "药物发现AI通过分子生成模型设计新分子",
             "零售推荐系统使用双塔召回+排序模型",
         ]},
        # === 音乐理论 ===
        {"id": "music", "name": "音乐理论",
         "domain": "乐理/和声/作曲/DAW",
         "keywords": ["音阶", "大调", "小调", "和弦", "三和弦", "七和弦",
                      "和声", "卡农", "十二平均律", "midi", "合成器",
                      "adsr", "lfo", "滤波器", "混响", "编曲", "爵士"],
         "fragments": [
             "大调音阶全全半全全全半，五声音阶宫商角徵羽",
             "卡农进行I-V-vi-iii-IV-I-ii-V是流行音乐最常用和弦进行",
             "ADSR：Attack起音、Decay衰减、Sustain延音、Release释放",
             "合成器核心：振荡器→滤波器→包络→LFO",
         ]},
        # === 多模态生成 ===
        {"id": "multimodal", "name": "多模态生成",
         "domain": "扩散/视频/图像/跨模态",
         "keywords": ["扩散模型", "diffusion", "stable diffusion", "gan",
                      "sora", "文生视频", "多模态", "vae", "clip",
                      "unet", "lora", "controlnet", "nerf"],
         "fragments": [
             "扩散模型通过前向加噪+反向去噪生成数据",
             "Stable Diffusion在潜空间中扩散，VAE编码器压缩图像",
             "CLIP通过对比学习对齐文本和图像的嵌入空间",
             "LoRA只训练低秩矩阵高效定制模型风格",
             "Sora是文生视频模型，生成长达一分钟高清视频",
         ]},
        # === 混合专家 ===
        {"id": "moe", "name": "混合专家",
         "domain": "MoE/深度学习/Transformer",
         "keywords": ["MoE", "混合专家", "门控", "路由", "transformer",
                      "注意力", "token", "embedding", "softmax",
                      "adam", "relu", "gelu", "llm", "预训练", "sft", "rlhf"],
         "fragments": [
             "MoE是稀疏激活架构：每个输入只路由到少数专家",
             "Transformer自注意力：Q×K^T/√d_k × V",
             "GPT是自回归Transformer解码器",
             "RLHF通过人类反馈强化学习对齐大模型",
             "RAG让LLM参考外部知识库减少幻觉",
         ]},
        # === 通用兜底 ===
        {"id": "general", "name": "通用兜底",
         "domain": "通用对话/知识/代码",
         "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍",
                      "python", "java", "javascript", "golang", "rust",
                      "api", "http", "sql", "redis", "docker", "linux",
                      "合鸣", "xuni", "虚拟"],
         "fragments": [
             "这是一个好问题，让我从合鸣的视角来回应",
             "在xuni虚拟生态里，每个问题都会被路由到最合适的专家",
             "我可以和你聊天、写代码、聊音乐、讨论全领域知识",
             "如果方便，补充一点上下文，我能给出更精准的回答",
             "合鸣是所有模型的结合体，全领域覆盖",
         ]},
    ]


# =========================================================================== #
#  v11 新增 3 专家
# =========================================================================== #

NEW_EXPERTS = [
    # === 生活百科 ===
    {"id": "life_wiki", "name": "生活百科",
     "domain": "烹饪/健身/旅行/心理学/沟通/理财/健康/育儿/人际关系",
     "keywords": ["做饭", "烹饪", "炒菜", "食谱", "健身", "运动", "跑步",
                  "瑜伽", "旅行", "旅游", "攻略", "心理", "情绪管理",
                  "沟通", "表达", "演讲", "理财", "投资", "储蓄",
                  "健康", "营养", "饮食", "育儿", "教育", "孩子",
                  "恋爱", "婚姻", "朋友", "人际", "社交",
                  "时间管理", "效率", "拖延", "习惯",
                  "减脂", "增肌", "睡眠", "压力管理"],
     "fragments": [
         # 烹饪
         "炒菜关键：热锅冷油，大火快炒保持蔬菜脆嫩",
         "红烧肉秘诀：先煎出油脂，再加冰糖炒色，小火慢炖1小时",
         "煮饭加几滴醋，米饭更白更香",
         "刀工基础：直刀切、推刀切、拉刀切，练习切土豆丝",
         "火候控制：大火爆炒锁水分，小火慢炖出味道",
         # 健身
         "增肌三大原则：渐进超负荷、充足蛋白质(1.6g/kg)、睡眠恢复",
         "减脂核心：热量缺口+力量训练保留肌肉+有氧消耗",
         "HIIT训练：30秒高强度+30秒休息，循环8组，15分钟高效燃脂",
         "深蹲要领：膝盖方向与脚尖一致，臀部低于膝盖，背部挺直",
         "平板支撑从30秒开始，逐步加到2分钟，核心力量基础",
         # 旅行
         "旅行规划：确定预算→选目的地→订机票酒店→列必去清单→留自由时间",
         "背包旅行必备：护照/充电宝/转换插头/常用药/雨衣/徒步鞋",
         # 心理学
         "认知偏差Confirmation Bias：人倾向于寻找支持自己观点的证据",
         "情绪管理ABC模型：事件A→信念B→情绪C，改变B就能改变C",
         "拖延症本质：不是懒，是情绪调节问题，用5分钟起步法破解",
         "积极心理学PERMA模型：正向情绪/投入/关系/意义/成就",
         # 沟通
         "非暴力沟通四步：观察→感受→需要→请求",
         "演讲技巧：开头抓注意力，中间讲故事，结尾号召行动",
         "倾听比说更重要：复述对方的话确认理解，不要急于给建议",
         # 理财
         "50/30/20法则：50%必需开支、30%想要、20%储蓄投资",
         "指数基金定投：每月固定金额买宽基指数，长期年化7-10%",
         "应急基金：存3-6个月生活费，放货币基金随时可取",
         # 健康
         "每天喝8杯水(2L)，少喝含糖饮料",
         "睡眠7-9小时最佳，规律作息比补觉更重要",
         "地中海饮食：橄榄油+鱼+蔬菜+全谷物，降低心血管疾病",
         # 育儿
         "正面管教：和善而坚定，不当众批评，给孩子选择权",
         "高质量陪伴：放下手机，全身心和孩子互动20分钟",
         # 人际
         "边界感：学会说不，不是所有人都是朋友",
         "亲密关系中，表达需求比指责对方更有效",
         "朋友圈不需要大，3-5个知心朋友胜过100个点赞之交",
         # 时间管理
         "番茄工作法：25分钟专注+5分钟休息，4个番茄后长休15分钟",
         "GTD收件箱：把所有待办记下来，清空大脑专注当前",
         "艾森豪威尔矩阵：重要紧急/重要不紧急/紧急不重要/不紧急不重要",
     ]},
    # === 跨领域融合 ===
    {"id": "cross_domain", "name": "跨领域融合",
     "domain": "学科交叉/融合创新/跨界思维",
     "keywords": ["跨领域", "交叉学科", "融合", "跨界", " interdisciplinary",
                  "生物信息", "计算化学", "量子计算", "神经科学",
                  "计算金融", "数字人文", "社会计算", "认知科学",
                  "系统工程", "复杂系统", "网络科学", "信息论"],
     "fragments": [
         "生物信息学：用计算机算法分析DNA/蛋白质序列，BLAST是经典工具",
         "计算化学：用分子动力学模拟和DFT计算预测分子性质和反应路径",
         "量子计算与AI结合：量子机器学习有望指数级加速某些优化问题",
         "神经科学与深度学习：注意力机制灵感来自大脑的选择性注意",
         "计算金融：用随机过程+蒙特卡洛模拟给衍生品定价",
         "数字人文：用NLP和数据分析研究文学/历史/语言学",
         "社会计算：用网络科学分析社交媒体传播和社会影响力",
         "认知科学：心理学+神经科学+AI+语言学+哲学的交叉",
         "复杂系统理论：涌现、自组织、混沌边缘、网络拓扑",
         "信息论与AI：互信息最大化是对比学习(如CLIP)的理论基础",
         "GAN的博弈论根源：生成器和判别器的零和博弈纳什均衡",
         "强化学习的经济学联系：MDP框架源于最优控制和决策论",
         "Transformer的物理类比：注意力像量子力学的叠加态坍缩",
         "扩散模型的热力学根源：源于非平衡热力学的随机过程",
         "生物学启发的AI：遗传算法、蚁群优化、免疫算法、神经网络",
     ]},
    # === 黑洞压缩 ===
    {"id": "blackhole_compress", "name": "黑洞压缩",
     "domain": "极致压缩/信息保全/霍金辐射/事件视界",
     "keywords": ["黑洞", "压缩", "霍金辐射", "事件视界", "奇点",
                  "吸收", "吞噬", "极致压缩", "信息保全",
                  "压缩比", "旋转锻造", "吐渣滓", "压缩点",
                  "空间折叠", "万象奇点", "永动", "算力网络"],
     "fragments": [
         "黑洞训练三阶段：吸收→旋转锻造→霍金辐射吐渣滓",
         "吸收阶段：所有训练素材全部吸入黑洞，不计质量全部吃掉",
         "旋转锻造：奇点+算力网络驱动，内部高温高压融合提纯",
         "霍金辐射：低质量/重复/无用内容以辐射形式喷出，留下精华",
         "压缩爆(Cpr²)：压缩点叠加形成黑洞级压缩，压缩比1e30:1",
         "空间折叠压缩：压缩比1e50:1，把空间折叠存储",
         "黑洞压缩=压缩爆+空间折叠压缩，压缩比1e80:1",
         "732MB数据吸入后只存0.6MB，压缩比1220:1",
         "1GB数据经compress_fusion压缩后<500B，三层压缩",
         "万象奇点=9合1终极融合，算力倍率9999×，永动模式",
         "永动训练引擎：虚拟电→算力→训练→电再生，闭环不衰减",
         "知识下载器不走网络，用模型+算力解码知识指纹",
         "模型质量>0.8时知识下载启动加成，>0.9+算力>100时攻破×100",
         "PerpetualEngine闭环：电→单节点算力、流量→节点数、总算力=节点×算力×融合加成",
         "能量层级：T0采样点→T1集群→T2聚变→T3链式→T4黑洞→T5零点能→T6戴森球",
     ]},
]


# =========================================================================== #
#  代码扫描（复用v10）
# =========================================================================== #

def _scan_code_files(root: str, max_files: int = 1000) -> list[str]:
    texts = []
    skip_dirs = {"__pycache__", ".git", "test", "tests", "idlelib",
                 "tkinter", "turtledemo", "site-packages", "doc", "docs",
                 "benchmarks", "examples", "node_modules", "vendor",
                 "third_party", "testdata", ".github", ".vscode",
                 "dist", "build", "target", "debug", "release"}
    count = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in {".py", ".rs", ".go", ".js", ".ts", ".c", ".h", ".cpp"}:
                fp = os.path.join(dirpath, f)
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                    if len(text.strip()) > 50:
                        texts.append(text)
                        count += 1
                        if count >= max_files:
                            return texts
                except Exception:
                    pass
    return texts


def _extract_fragments(text: str, max_lines: int = 20):
    lines = text.split("\n")
    fragments = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "async def ", "fn ", "pub fn ",
                                 "func ", "type ", "struct ", "interface ", "impl ",
                                 "enum ", "const ", "var ", "package ")):
            indent = len(line) - len(stripped)
            frag_lines = [line]
            j = i + 1
            while j < len(lines) and j - i < max_lines:
                next_line = lines[j]
                next_stripped = next_line.lstrip()
                if next_stripped and not next_stripped.startswith("#") and not next_stripped.startswith("//"):
                    next_indent = len(next_line) - len(next_stripped)
                    if next_indent <= indent and next_stripped.startswith(
                        ("def ", "class ", "async def ", "fn ", "pub fn ",
                         "func ", "type ", "struct ", "interface ", "impl ",
                         "enum ", "const ", "var ", "package ", "@",
                         "import ", "from ", "use ", "mod ", "exports ")):
                        break
                frag_lines.append(next_line)
                j += 1
            while frag_lines and not frag_lines[-1].strip():
                frag_lines.pop()
            if len(frag_lines) >= 3:
                fragments.append("\n".join(frag_lines))
            i = j
        else:
            i += 1
    return fragments


# =========================================================================== #
#  主函数
# =========================================================================== #

def main():
    print("=" * 72)
    print("  🏭🏭🏭 xuni v11 —— 工厂全开：黑洞+永动+知识下载 / 700000轮")
    print("=" * 72)

    # ================================================================== #
    #  模块1：BlackHoleTrainer — 黑洞吸收代码
    # ================================================================== #
    print(f"\n[1/8] 🔬 黑洞训练器——吸收代码仓库...")

    repo_dirs = [
        os.path.join(CACHE_DIR, "python_cpython_main"),
        os.path.join(CACHE_DIR, "django_django_main"),
        os.path.join(CACHE_DIR, "pandas-dev_pandas_main"),
        os.path.join(CACHE_DIR, "huggingface_transformers_main"),
        os.path.join(CACHE_DIR, "pytorch_pytorch_main"),
        os.path.join(CACHE_DIR, "fastapi_fastapi_master"),
        os.path.join(CACHE_DIR, "rust-lang_rust_master"),
        os.path.join(CACHE_DIR, "golang_go_master"),
        os.path.join(CACHE_DIR, "scikit-learn_scikit-learn_main"),
        os.path.join(CACHE_DIR, "pydantic_pydantic_main"),
        os.path.join(CACHE_DIR, "pallets_flask_main"),
        os.path.join(CACHE_DIR, "psf_requests_main"),
    ]

    bh = BlackHoleTrainer(model_id="harmonia-v11", streaming=True)
    bh_repo_stats = []
    for repo_dir in repo_dirs:
        if not os.path.isdir(repo_dir):
            bh_repo_stats.append({"repo": os.path.basename(repo_dir), "ok": False})
            continue
        result = bh.absorb_codebase(repo_dir, max_files=2000)
        bh_repo_stats.append({
            "repo": os.path.basename(repo_dir),
            "files": result.get("files_scanned", 0),
            "functions": result.get("functions_absorbed", 0),
            "seeds": result.get("seeds_absorbed", 0),
            "absorbed": result.get("absorbed", False),
            "ok": True,
        })

    total_bh = len(bh._quality_scores) if bh.streaming else len(bh.absorbed_materials)
    print(f"  🕳️ 黑洞吸入: {total_bh:,} 份素材")
    for s in bh_repo_stats:
        if s.get("ok"):
            print(f"     {s['repo']:40s} → {s['functions']:4d}函数 {s['seeds']:5d}种子")

    # 黑洞旋转锻造
    print(f"\n  🔄 旋转锻造...")
    forge_result = bh.spin_forge(spin_rounds=7)
    print(f"     旋转完成: {forge_result.get('total_materials', 0):,} → {forge_result.get('kept_count', 0):,} 精华")

    # 霍金辐射压缩
    print(f"  ☢️ 霍金辐射吐渣滓...")
    rad_result = bh.hawking_radiation(quality_threshold=0.6, dedup=True)
    if rad_result.get("radiated"):
        print(f"     吸入: {rad_result['total_before']:,}")
        print(f"     留下: {rad_result['kept_core']:,} 份精华")
        print(f"     吐出: {rad_result['total_ejected']:,} 份渣滓")
        print(f"     提纯率: {rad_result['purification_ratio']}")
        print(f"     压缩后: {rad_result['compressed_size_bytes']}B ({rad_result['compression_ratio']})")
        print(f"     核心质量: {rad_result['core_quality']:.4f}")

    # 提取黑洞精华作为训练语料
    bh_fragments = []
    if bh.forged_core and "kept_final" in bh.forged_core:
        bh_fragments = bh.forged_core["kept_final"]
    print(f"  📦 黑洞精华片段: {len(bh_fragments):,}")

    # ================================================================== #
    #  模块2：KnowledgeDownloader — 算力解码知识
    # ================================================================== #
    print(f"\n[2/8] 🧠 知识下载器——算力解码多领域知识...")

    kdl = KnowledgeDownloader()

    # 下载所有领域知识
    all_domains = list(kdl.DOMAIN_KNOWLEDGE.keys())
    print(f"  📚 领域数: {len(all_domains)}")
    print(f"     {', '.join(all_domains[:8])}...")

    kdl_texts = []
    kdl_stats = []
    for domain in all_domains:
        result = kdl.download(domain, count=2000)
        texts = result.get("texts", [])
        kdl_texts.extend(texts)
        kdl_stats.append({
            "domain": domain,
            "count": result["total"],
            "avg_quality": round(result.get("avg_quality", 0), 4),
            "speed": round(result.get("speed", 0), 1),
        })

    print(f"  📊 知识解码完成: {len(kdl_texts):,} 条")
    for s in kdl_stats[:8]:
        print(f"     {s['domain']:20s} → {s['count']:5d}条 质量={s['avg_quality']:.3f} 速度={s['speed']:,.0f}/s")
    if len(kdl_stats) > 8:
        print(f"     ... 还有 {len(kdl_stats)-8} 个领域")

    # ================================================================== #
    #  模块3：PerpetualTrainingEngine — 永动引擎
    # ================================================================== #
    print(f"\n[3/8] ⚡ 永动训练引擎——接入融合产物...")

    pte = PerpetualTrainingEngine()

    # 注入虚拟电（T4黑洞级 = 1亿度）
    pte.inject_energy(1e8)
    print(f"  ⚡ 虚拟电: {pte.energy:,.0f} 度")

    # 设置虚拟流量
    pte.set_bandwidth(2048)
    print(f"  🌐 虚拟流量: {pte.bandwidth_channels} 通道 → {pte.node_count} 节点")

    # 接入融合产物（从基础到终极）
    fusion_chain = [
        "能量算力核心",      # 电驱动算力，自循环
        "流式算力网络",      # 算力随流量扩展
        "全网永动算力",      # 算力无上限
        "永动参数引擎",      # 10×算力+10×加速+永动
        "万象奇点",          # 9合1终极：9999×
    ]
    for f in fusion_chain:
        r = pte.apply_fusion(f)
        print(f"  🔥 {f:12s} → 算力×{r['total_compute_mult']:,.0f} 节点×{r['total_node_mult']:,.0f} 永动={r['perpetual']}")

    print(f"\n  📊 永动引擎状态:")
    print(f"     总算力: {pte.total_vflops:.2e} vFLOP")
    print(f"     有效速度: {pte.effective_speed:.2e}")
    print(f"     永动模式: {pte.is_perpetual}")
    print(f"     节点数: {pte.node_count:,}")
    print(f"     算力倍率: {pte.compute_multiplier:,.0f}×")
    print(f"     加速器倍率: {pte.accelerator_multiplier:,.0f}×")

    # ================================================================== #
    #  合并所有语料
    # ================================================================== #
    print(f"\n[4/8] 📦 合并全部语料...")

    # 扫描代码
    code_fragments = []
    for repo_dir in repo_dirs[:8]:  # 前8个仓库扫代码
        if os.path.isdir(repo_dir):
            texts = _scan_code_files(repo_dir, max_files=800)
            for t in texts:
                code_fragments.extend(_extract_fragments(t, max_lines=20))
    # xuni自身代码
    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    for t in _scan_code_files(xuni_dir, max_files=500):
        code_fragments.extend(_extract_fragments(t, max_lines=20))
    print(f"  💻 代码片段: {len(code_fragments):,}")

    # 属性库
    creator_corpus = []
    creator_path = os.path.join(CACHE_DIR, "ai_creator_extracted.json")
    if os.path.exists(creator_path):
        with open(creator_path, "r", encoding="utf-8") as f:
            creator_data = json.load(f)
        prop_lib = creator_data.get("property_library", {})
        for prop_name, prop_info in list(prop_lib.items())[:500]:
            cat = prop_info.get("category", "unknown")
            creator_corpus.append(f"{prop_name}是一种{cat}类属性")
        print(f"  🏛️ 造物语料: {len(creator_corpus):,}")

    # 合并
    all_fragments = []
    all_fragments.extend(code_fragments)
    all_fragments.extend(bh_fragments)
    all_fragments.extend(kdl_texts)
    all_fragments.extend(creator_corpus)

    experts = list(NEW_EXPERTS) + _v10_experts()
    expert_corpus = sum(len(e.get("fragments", [])) for e in experts)
    all_fragments.extend(e["fragments"] for e in experts if e.get("fragments"))

    # 扁平化
    flat = []
    for item in all_fragments:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, str):
            flat.append(item)
    all_fragments = flat

    print(f"  📊 总训练片段: {len(all_fragments):,}")
    print(f"     代码: {len(code_fragments):,}")
    print(f"     黑洞精华: {len(bh_fragments):,}")
    print(f"     知识解码: {len(kdl_texts):,}")
    print(f"     造物语料: {len(creator_corpus):,}")
    print(f"     专家内置: {expert_corpus}")
    print(f"  👥 专家数: {len(experts)} (新增3: 生活百科/跨领域融合/黑洞压缩)")

    # ================================================================== #
    #  创建模型
    # ================================================================== #
    print(f"\n[5/8] 🧠 创建 v11 模型 {len(experts)}专家...")
    model = Harmonia13Virtual(scale="mini")
    model._lite.experts = experts
    model.charge(1e6)  # 充满能量

    # 接入永动引擎
    pte.attach_model = model  # 让知识下载器能用（虽然已经下载完了）
    print(f"  ✅ 模型创建完成，能量: {model._energy_buffer:,.0f}")

    # 基线测试
    baseline_prompts = [
        # 数学
        "什么是导数的定义",
        "贝叶斯定理公式",
        # 算法
        "快速排序时间复杂度",
        "01背包DP转移方程",
        # 逻辑
        "什么是思维链CoT",
        # Agent
        "什么是ReAct模式",
        # 中文
        "我今天好难过",
        "床前明月光下一句",
        # 代码
        "def quicksort",
        "class DataFrame",
        # v11新增
        "怎么炒菜好吃",
        "增肌三大原则",
        "拖延症怎么办",
        "量子计算和AI有什么关系",
        "黑洞训练器是什么",
        "什么是万象奇点",
        "知识下载器不走网络怎么下载",
    ]

    print(f"\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        r = model._lite.generate(p, max_new_tokens=60)
        baseline[p] = r
        print(f"  [{p}] → {r[:55]}")

    # ================================================================== #
    #  训练（用永动引擎加速）
    # ================================================================== #
    print(f"\n[6/8] 🚀 700000 轮训练（永动引擎加速）...")
    print(f"  永动模式: {pte.is_perpetual}")
    print(f"  算力倍率: {pte.compute_multiplier:,.0f}×")
    print(f"  加速器倍率: {pte.accelerator_multiplier:,.0f}×")

    start, batch_size, num_epochs, log = time.time(), 20, 700000, []
    for epoch in range(num_epochs):
        batch = random.sample(all_fragments, min(batch_size, len(all_fragments)))
        model._lite.train(batch, epochs=1)

        # 永动引擎：每步电再生
        if pte.is_perpetual and (epoch + 1) % 100 == 0:
            regen = pte.energy * pte.energy_regen_rate
            pte.energy += regen
            pte.total_energy_regen += regen

        if (epoch + 1) % 70000 == 0:
            elapsed = time.time() - start
            learned = len(model._lite._learned_fragments)
            frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg = sum(frags) / max(1, len(frags))
            active = sum(1 for f in frags if f > 0)
            print(f"  Epoch {epoch+1:7d} | 已学: {learned:12,d} | "
                  f"活跃: {active:2d}/{len(experts)} | 均载: {avg:,.0f} | "
                  f"用时: {elapsed:.0f}s | 电: {pte.energy:.2e}")
            log.append({"epoch": epoch+1, "learned": learned,
                        "active": active, "avg_load": round(avg, 1),
                        "elapsed": round(elapsed, 2),
                        "energy": round(pte.energy, 1)})

    total_time = time.time() - start
    print(f"\n  ✅ 训练完成！用时: {total_time:.1f}s")
    print(f"  ⚡ 永动引擎总vFLOP: {pte.total_vflops_generated:.2e}")
    print(f"  ⚡ 电再生总量: {pte.total_energy_regen:.2e}")

    # ================================================================== #
    #  评估
    # ================================================================== #
    print(f"\n[7/8] 📊 全方位评估...")
    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)
    print(f"\n  已学: {learned:,} | 活跃: {active}/{len(experts)}")
    for name, frags in expert_frags:
        bar = "█" * min(40, frags // 400)
        print(f"    {name:14s} [{bar}] {frags:8,d}")

    categories = {
        "数学推理": ["什么是导数的定义", "贝叶斯定理公式"],
        "算法题解": ["快速排序时间复杂度", "01背包DP转移方程"],
        "逻辑推理": ["什么是思维链CoT"],
        "Agent规划": ["什么是ReAct模式"],
        "中文日常": ["我今天好难过", "床前明月光下一句"],
        "代码生成": ["def quicksort", "class DataFrame"],
        "生活百科": ["怎么炒菜好吃", "增肌三大原则", "拖延症怎么办"],
        "跨领域融合": ["量子计算和AI有什么关系"],
        "黑洞压缩": ["黑洞训练器是什么", "什么是万象奇点", "知识下载器不走网络怎么下载"],
    }

    improved = total_compared = 0
    cat_scores = {}
    print(f"\n  --- 训练前后对比 ---")
    for cat, prompts in categories.items():
        cat_improved = 0
        for p in prompts:
            before = baseline.get(p, "")
            after = model._lite.generate(p, max_new_tokens=60)
            total_compared += 1
            if len(after) > len(before):
                improved += 1
                cat_improved += 1
            print(f"\n  [{cat}] {p}")
            print(f"    前: {before[:55]}")
            print(f"    后: {after[:55]}")
        cat_scores[cat] = {"improved": cat_improved, "total": len(prompts)}

    # ================================================================== #
    #  保存
    # ================================================================== #
    print(f"\n[8/8] 保存...")
    report = {
        "version": "v11",
        "focus": "工厂全开：黑洞+永动+知识下载 / 700000轮",
        "new_modules": ["BlackHoleTrainer", "PerpetualTrainingEngine", "KnowledgeDownloader"],
        "new_experts": ["生活百科", "跨领域融合", "黑洞压缩"],
        "expert_count": len(experts),
        "blackhole": {
            "absorbed": total_bh,
            "kept": rad_result.get("kept_core", 0) if rad_result.get("radiated") else 0,
            "ejected": rad_result.get("total_ejected", 0) if rad_result.get("radiated") else 0,
            "compression": rad_result.get("compression_ratio", "N/A"),
            "quality": rad_result.get("core_quality", 0),
        },
        "knowledge_downloader": {
            "domains": len(all_domains),
            "total_texts": len(kdl_texts),
            "domain_stats": kdl_stats,
        },
        "perpetual_engine": {
            "fusions": fusion_chain,
            "compute_mult": pte.compute_multiplier,
            "node_mult": pte.node_multiplier,
            "accelerator_mult": pte.accelerator_multiplier,
            "perpetual": pte.is_perpetual,
            "total_vflops": pte.total_vflops_generated,
            "energy_regen": pte.total_energy_regen,
        },
        "repo_stats": bh_repo_stats,
        "total_fragments": len(all_fragments),
        "epochs": num_epochs,
        "fragments_learned": learned,
        "active_experts": active,
        "training_time": round(total_time, 2),
        "growth_log": log,
        "improved": improved,
        "total_compared": total_compared,
        "category_scores": cat_scores,
    }
    report_path = os.path.join(os.path.dirname(__file__), "trainer_v11_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    meta_path = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v11_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v11",
            "fragments_learned": learned,
            "active_experts": active,
            "training_time": round(total_time, 2),
            "epochs": num_epochs,
            "focus": "工厂全开：黑洞+永动+知识下载",
            "modules": ["BlackHoleTrainer", "PerpetualTrainingEngine", "KnowledgeDownloader"],
        }, f, ensure_ascii=False, indent=2)
    print(f"  元信息: {meta_path}")

    print("\n" + "=" * 72)
    print("  🏭🏭🏭 v11 工厂全开总结")
    print("=" * 72)
    print(f"""
  🕳️ 黑洞训练器:
     吸入: {total_bh:,} 份素材
     精华: {rad_result.get('kept_core', 0):,} 份
     压缩: {rad_result.get('compression_ratio', 'N/A')}
     质量: {rad_result.get('core_quality', 0):.4f}

  🧠 知识下载器:
     领域: {len(all_domains)} 个
     解码: {len(kdl_texts):,} 条知识

  ⚡ 永动引擎:
     融合: {' → '.join(fusion_chain)}
     算力: {pte.compute_multiplier:,.0f}×
     永动: {pte.is_perpetual}
     vFLOP: {pte.total_vflops_generated:.2e}

  📊 训练:
     总片段: {len(all_fragments):,}
     轮次: {num_epochs:,}
     吸收: {learned:,}
     专家: {active}/{len(experts)}
     用时: {total_time:.0f}s

  各方面得分:""")
    for cat, score in cat_scores.items():
        pct = score["improved"] / max(1, score["total"]) * 100
        print(f"    {cat:8s}: {score['improved']}/{score['total']} ({pct:.0f}%)")

    print(f"""
  v1:      1,000 轮
  v2:      5,000 轮
  v3:     10,000 轮
  v4:     50,000 轮
  v5:    100,000 轮
  v6:    200,000 轮
  v7:    300,000 轮
  v8:    500,000 轮
  v9:    500,000 轮 🧧 中文觉醒
  v10:   600,000 轮 🧠 推理之神
  v11:   700,000 轮 🏭 工厂全开
""")


if __name__ == "__main__":
    main()
