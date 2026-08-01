"""
demo_trainer_v10.py —— 推理之神 v10：数学+算法+逻辑+Agent思维链 / 600000轮

v9: 中文觉醒 / 情感+日常 / 500k
v10: 推理之神 / 数学+算法+逻辑+Agent思维链 / 600k

新增专家：
  - 数学推理：高数/线代/概率/离散/数学公式/证明
  - 算法题解：LeetCode经典/数据结构/算法模式
  - 逻辑推理：形式逻辑/思维链/谜题/证明策略
  - Agent规划：工具使用/任务分解/多步推理/反思

保留v9 11专家，新增4，共15专家
"""

from __future__ import annotations
import os, sys, time, json, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xuni import Harmonia13Virtual

CACHE_DIR = os.path.join(os.path.dirname(__file__), "corpus_cache")


# =========================================================================== #
#  新增 4 大推理专家
# =========================================================================== #

REASONING_EXPERTS = [
    # === 数学推理专家 ===
    {
        "id": "math_reasoning",
        "name": "数学推理",
        "domain": "高等数学/线性代数/概率论/离散数学/证明",
        "keywords": ["极限", "导数", "积分", "微分", "微积分", "泰勒", "级数",
                     "矩阵", "行列式", "特征值", "特征向量", "线性代数",
                     "向量空间", "基", "秩", "逆矩阵", "转置", "正交",
                     "概率", "期望", "方差", "标准差", "正态分布", "伯努利",
                     "二项分布", "泊松分布", "条件概率", "贝叶斯", "随机变量",
                     "集合", "函数", "映射", "二元关系", "等价关系",
                     "偏序", "全序", "群", "环", "域", "图论",
                     "树", "路径", "连通", "完全图", "二分图", "匹配",
                     "组合", "排列", "二项式系数", "容斥", "鸽巢原理",
                     "归纳法", "反证法", "构造法", "数学归纳", "递归",
                     "求和", "连乘", "模运算", "同余", "最大公约数", "gcd",
                     "最小公倍数", "lcm", "素数", "质因数分解", "数论",
                     "欧几里得", "欧拉", "费马", "定理", "证明", "推论",
                     "lemma", "theorem", "proof", "qed", "推导"],
        "fragments": [
            # 极限与微积分
            "极限定义：lim(x→a) f(x) = L 当且仅当对任意ε>0，存在δ>0，0<|x-a|<δ时有|f(x)-L|<ε",
            "导数定义：f'(x) = lim(h→0) [f(x+h) - f(x)] / h",
            "导数的几何意义是函数图像在该点切线的斜率",
            "链式法则：(f∘g)'(x) = f'(g(x)) · g'(x)",
            "莱布尼茨法则：(uv)' = u'v + uv'",
            "不定积分是求导的逆运算：∫f'(x)dx = f(x) + C",
            "定积分的牛顿-莱布尼茨公式：∫[a→b]f(x)dx = F(b) - F(a)",
            "泰勒展开：f(x) = Σ f⁽ⁿ⁾(a)/n! · (x-a)ⁿ",
            "eˣ的泰勒展开：1 + x + x²/2! + x³/3! + ... 对所有x收敛",
            "sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...",
            "cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...",

            # 线性代数
            "矩阵乘法：(AB)ᵢⱼ = Σₖ Aᵢₖ · Bₖⱼ，矩阵乘法不满足交换律",
            "逆矩阵：AA⁻¹ = A⁻¹A = I，可逆当且仅当行列式≠0",
            "行列式det(A)：2×2时ad-bc，3×3时按行展开",
            "特征值λ满足Ax = λx，特征方程det(A-λI)=0",
            "特征向量是变换后方向不变的非零向量",
            "矩阵的秩是行（列）向量组的极大线性无关组大小",
            "正交矩阵满足AᵀA = I，列（行）向量是标准正交基",
            "对称矩阵可正交对角化：A = QΛQᵀ，特征值都是实数",
            "向量空间的基是一组线性无关且张成整个空间的向量",
            "内积⟨u,v⟩衡量向量的相似程度，正交时内积为0",

            # 概率论
            "概率公理：P(A)≥0，P(Ω)=1，互斥事件P(A∪B)=P(A)+P(B)",
            "条件概率：P(A|B) = P(A∩B) / P(B)，当P(B)>0时",
            "贝叶斯定理：P(A|B) = P(B|A)P(A) / P(B)",
            "期望E[X] = Σ x·P(X=x)（离散）或∫x·f(x)dx（连续）",
            "方差Var(X) = E[(X-E[X])²] = E[X²] - (E[X])²",
            "标准差σ = √Var(X)",
            "正态分布N(μ,σ²)的概率密度：(1/√(2πσ²))e^(-(x-μ)²/2σ²)",
            "中心极限定理：独立同分布变量之和趋近正态分布",
            "伯努利分布：P(X=1)=p, P(X=0)=1-p，期望p，方差p(1-p)",
            "二项分布B(n,p)：n次独立伯努利试验的成功次数",
            "泊松分布P(λ)：近似稀有事件频率，P(X=k)=e^(-λ)λᵏ/k!",

            # 离散数学 & 数论
            "德摩根律：¬(A∧B) = ¬A∨¬B，¬(A∨B) = ¬A∧¬B",
            "鸽巢原理：n+1个物体放入n个盒子，至少有一个盒子≥2个物体",
            "容斥原理：|A∪B| = |A|+|B|-|A∩B|，推广到n个集合",
            "排列P(n,k) = n!/(n-k)!，组合C(n,k) = n!/(k!(n-k)!)",
            "二项式定理：(a+b)ⁿ = Σ C(n,k)aⁿ⁻ᵏbᵏ",
            "欧几里得算法：gcd(a,b) = gcd(b, a mod b)，递归直到b=0",
            "贝祖定理：存在整数m,n使gcd(a,b) = ma + nb",
            "费马小定理：若p是素数，a不被p整除，则a^(p-1) ≡ 1 (mod p)",
            "欧拉函数φ(n)：小于n且与n互素的正整数个数",
            "素数无穷多：欧几里得经典证明，构造N=p₁p₂…pₙ+1",

            # 证明方法
            "数学归纳法：证明P(1)成立，假设P(k)成立推出P(k+1)，则P(n)对所有n≥1成立",
            "强归纳法：假设对所有m≤k成立P(m)，推出P(k+1)",
            "反证法：假设结论不成立，推出矛盾，从而原命题为真",
            "构造法：构造出一个满足条件的对象，证明存在性",
            "构造性证明 vs 存在性证明：前者给出具体对象，后者只证明存在",
            "证明等式常见策略：化简两边、两边相等、数学归纳、代数变形",
            "证明不等式常见策略：放缩法、数学归纳、均值不等式、反证",
            "QED（quod erat demonstrandum）= 证明完毕",
        ],
    },
    # === 算法题解专家 ===
    {
        "id": "algorithm",
        "name": "算法题解",
        "domain": "LeetCode经典/数据结构/算法模式/复杂度分析",
        "keywords": ["时间复杂度", "空间复杂度", "大O", "O(n)", "O(log n)", "O(n²)",
                     "O(n log n)", "O(2ⁿ)", "数组", "链表", "单链表", "双链表",
                     "栈", "队列", "双端队列", "优先队列", "堆", "大根堆", "小根堆",
                     "哈希表", "散列表", "集合", "字典", "树", "二叉树", "BST",
                     "AVL树", "红黑树", "平衡树", "前缀树", "Trie", "线段树",
                     "树状数组", "Fenwick树", "图", "邻接表", "邻接矩阵",
                     "BFS", "DFS", "深度优先", "广度优先", "拓扑排序",
                     "最短路", "Dijkstra", "Bellman-Ford", "Floyd",
                     "最小生成树", "Prim", "Kruskal", "并查集", "Union-Find",
                     "二分查找", "双指针", "滑动窗口", "前缀和", "差分数组",
                     "动态规划", "DP", "状态转移", "背包问题", "最长公共子序列",
                     "LCS", "最长递增子序列", "LIS", "分治", "回溯",
                     "剪枝", "贪心", "位运算", "KMP", "Rabin-Karp",
                     "LeetCode", "排序", "快排", "归并", "堆排序", "冒泡",
                     "插入排序", "选择排序", "计数排序", "桶排序", "基数排序"],
        "fragments": [
            # 复杂度分析
            "时间复杂度：大O表示上界，忽略常数和低阶项。常见从低到高：O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)",
            "空间复杂度衡量额外内存使用，原地算法通常要求O(1)或O(log n)（递归栈）",
            "主定理分析分治递归：T(n)=aT(n/b)+f(n)，比较f(n)与n^(log_b a)",

            # 排序算法
            "快速排序：选枢轴分区，平均O(n log n)，最坏O(n²)，不稳定，原地",
            "归并排序：分两半递归再合并，O(n log n)稳定，空间O(n)",
            "堆排序：建堆后依次取出堆顶，O(n log n)，不稳定，原地",
            "冒泡排序：相邻比较交换，O(n²)稳定，适合已基本有序数据",
            "计数排序：值范围已知时用数组计数，O(n+k)，非比较排序",
            "基数排序：按每一位排序（低位到高位），处理整数",

            # 经典算法思维
            "二分查找：有序数组中每次排除一半，O(log n)，关键是边界条件",
            "滑动窗口：维护窗口内的状态，求最长/最短/和最大子数组，典型O(n)",
            "双指针：快慢指针判断链表环、左右指针两数之和、对撞指针",
            "前缀和：PreSum[i] = a[0]+…+a[i-1]，求区间和O(1)",
            "差分数组：区间加值时O(1)更新，最后求前缀和还原",
            "回溯算法：递归尝试所有选择，失败后撤销选择，适合排列/组合/子集问题",
            "剪枝：在回溯中提前排除不可能的分支，显著降低搜索空间",
            "贪心算法：每步选局部最优，需证明正确性（如活动选择、霍夫曼编码）",

            # 动态规划
            "动态规划核心：最优子结构+重叠子问题，状态+转移方程+初始化+答案",
            "01背包：dp[i][w]=max(dp[i-1][w], dp[i-1][w-wᵢ]+vᵢ)，滚动数组逆序",
            "完全背包：每种物品无限次，滚动数组正序",
            "最长递增子序列LIS：O(n²)DP或O(n log n)二分贪心",
            "最长公共子序列LCS：dp[i][j] = s1[i-1]==s2[j-1]? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1])",
            "编辑距离Levenshtein：dp[i][j] = s1[i-1]==s2[j-1]? dp[i-1][j-1] : 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1])",
            "爬楼梯：dp[n] = dp[n-1] + dp[n-2]（斐波那契）",
            "打家劫舍：dp[i] = max(dp[i-1], dp[i-2]+nums[i])，不能连续偷",
            "状态压缩DP：用二进制掩码表示集合状态，典型问题如旅行商、匹配",

            # 数据结构经典
            "单链表反转：迭代三指针（prev/curr/next）或递归",
            "判断链表有环：Floyd快慢指针（快2步慢1步），相遇则有环",
            "二叉树遍历：前序根-左-右、中序左-根-右、后序左-右-根、层序BFS",
            "BST二叉搜索树：左<根<右，中序遍历递增",
            "BFS：队列存当前层节点，适合求最短路/层次遍历",
            "DFS：递归或栈，适合遍历所有路径、拓扑排序",
            "拓扑排序：DAG的线性排列，所有边u→v有u在v前，Kahn算法入度0入队",
            "Dijkstra：非负权最短路，优先队列取最小dist，松弛O(m log n)",
            "Bellman-Ford：检测负权环，V-1轮松弛所有边，再轮有更新即有负环",
            "Floyd-Warshall：三层循环，d[k][i][j] = min(d[i][j], d[i][k]+d[k][j])，全源最短路O(n³)",
            "Prim：从某点开始，每次加最小跨边，邻接矩阵O(n²)，堆优化O(m log n)",
            "Kruskal：边按权排序，依次加入不形成环的边（并查集判环）",
            "并查集Union-Find：路径压缩+按秩合并，find/union几乎O(1)",
            "Trie前缀树：每个节点是字母映射，用于前缀搜索、词典查询",
            "优先队列=堆：C++中priority_queue，Python中heapq默认小根堆",

            # LeetCode 经典题模式
            "两数之和：哈希表存值→索引，单次遍历O(n)",
            "三数之和：排序后固定第一个，双指针找另外两个，去重",
            "合并有序链表：递归或迭代双指针",
            "反转链表：迭代三指针或递归",
            "二叉树最大深度：递归max(left,right)+1或BFS层数",
            "岛屿数量：DFS/BFS遍历标记，计数连通分量",
            "爬楼梯：DP=斐波那契",
            "最大子数组和：Kadane算法dp[n]=max(dp[n-1]+a[n], a[n])",
            "接雨水：双指针或前缀后缀最大高度，sum(min(left_max,right_max)-height)",
            "LRU缓存：哈希表+双向链表，O(1)读写+淘汰最久未用",
            "最小栈：辅助栈同步存当前最小值",
            "有效括号：栈存左括号下标，遇到右括号出栈计算长度",
        ],
    },
    # === 逻辑推理专家 ===
    {
        "id": "logic_reasoning",
        "name": "逻辑推理",
        "domain": "形式逻辑/思维链/逻辑谜题/推理策略",
        "keywords": ["思维链", "chain of thought", "CoT", "分步推理", "逐步思考",
                     "一步一步", "推理", "前提", "结论", "三段论",
                     "命题逻辑", "谓词逻辑", "蕴含", "等价", "真值表",
                     "重言式", "矛盾", "可满足性", "SAT", "CNF", "DNF",
                     "全称量词", "存在量词", "∀", "∃", "推理规则",
                     "Modus Ponens", "肯定前件", "否定后件", "Modus Tollens",
                     "假言三段论", "选言三段论", "构造性二难",
                     "逻辑谬误", "循环论证", "诉诸无知", "人身攻击",
                     "稻草人", "滑坡谬误", "虚假两难", "偷换概念",
                     "逻辑谜题", "谁在说谎", "推理题", "真假判断",
                     "条件推理", "假设推理", "反例", "反证",
                     "必要条件", "充分条件", "充要条件", "当且仅当", "iff"],
        "fragments": [
            # 思维链策略
            "思维链(CoT)：面对复杂问题时，先列出已知条件，再一步一步推导，而不是直接给答案",
            "CoT步骤：1.理解题意 2.列出已知/约束 3.拆解子问题 4.每步推导 5.验证答案是否合理",
            "示例CoT：「A比B大3岁，5年后A是B的2倍，求今年年龄」→ 设B=x，A=x+3；5年后B=x+5，A=x+8；条件x+8=2(x+5) → x=8-10=-2？不对，重新检查：5年后A=x+3+5=x+8=2(x+5)→x+8=2x+10→x=-2矛盾→原题可能有问题，换另一种理解：5年前？不…→验证后再给出结论",
            "自洽性(Self-consistency)：对同一问题采样多条CoT路径，投票选多数答案，提升准确率",
            "反思(Reflexion)：得出答案后，反过来检查逻辑漏洞和条件矛盾，必要时换路径重推",

            # 形式逻辑
            "命题P→Q（蕴含）：只有P真且Q假时为假，其他情况为真（包括P假时恒真）",
            "P→Q 的等值式：¬P∨Q，逆否命题¬Q→¬P与原命题等价",
            "肯定前件Modus Ponens：若P→Q为真且P为真，则Q为真。例：「如果下雨则地湿，下雨了，所以地湿」",
            "否定后件Modus Tollens：若P→Q为真且¬Q为真，则¬P为真。例：「如果下雨则地湿，地没湿，所以没下雨」",
            "假言三段论：若P→Q，Q→R，则P→R",
            "选言三段论：若P∨Q，且¬P，则Q（排除法）",
            "构造性二难：若(P→R)∧(Q→R)∧(P∨Q)，则R为真",
            "德摩根律：¬(P∧Q) ≡ ¬P∨¬Q；¬(P∨Q) ≡ ¬P∧¬Q（命题与集合版）",
            "分配律：P∧(Q∨R) ≡ (P∧Q)∨(P∧R)；P∨(Q∧R) ≡ (P∨Q)∧(P∨R)",
            "充分条件：P是Q的充分条件= P→Q，P成立一定推出Q",
            "必要条件：P是Q的必要条件= Q→P，Q成立必须先有P（Q离不开P）",
            "充要条件P↔Q：P→Q且Q→P，当且仅当",

            # 量词推理
            "全称例示UI：∀x P(x) ⇒ P(c)，任意个体c都满足P",
            "存在概括EG：P(c) ⇒ ∃x P(x)，有具体对象c则存在x",
            "¬∀x P(x) ≡ ∃x ¬P(x)：不是所有都满足=存在一个不满足",
            "¬∃x P(x) ≡ ∀x ¬P(x)：不存在满足的=所有都不满足",

            # 常见逻辑谬误识别
            "循环论证/窃取论题：用结论本身作为前提证明，例「圣经是真的因为圣经自己这么说」",
            "肯定后件谬误：P→Q且Q真⇒P真，错。例「天才都古怪，他古怪，所以他是天才」",
            "否定前件谬误：P→Q且¬P⇒¬Q，错。例「如果鸟会飞，燕子会飞吗？如果它不是鸟…」",
            "稻草人谬误：歪曲对方观点，然后攻击那个被歪曲的版本",
            "诉诸无知谬误：没证明为假⇒为真；或没证明为真⇒为假",
            "人身攻击ad hominem：攻击提出者的人格而不是论点本身",
            "滑坡谬误：A→B→C→…→Z最终灾难，但中间每步因果都不必然成立",
            "虚假两难：强行只给两个选项，实际还有其他选项（非黑即白）",
            "偷换概念equivocation：同一词在推理中使用了两种意义",
            "事后归因post hoc：A先于B发生，就错误认定A是B的原因",

            # 经典逻辑谜题模式
            "谁在说谎谜题：枚举「A真/A假/B真/B假」四种组合，验证是否矛盾",
            "骑士与无赖(Knights and Knaves)：骑士永远说真话，无赖永远说谎，谁说A是骑士?",
            "数独推理：找出唯一合法候选的格填入，回溯+剪枝解决难的",
            "帽子谜题：用归纳法推理「能看到别人帽子时推出自己帽子颜色」",
            "蓝眼睛岛悖论：公共知识导致归纳推理在第100天发生同步行动",
            "过河谜题（狼羊菜）：状态搜索，枚举合法转移，BFS找最少步数",
            "抽卡牌/扑克牌推理：列出所有可能，逐条排除不可能的情形",
            "假设法：假设P为真→推出矛盾→P必为假（反证法推理策略）",
        ],
    },
    # === Agent 规划专家 ===
    {
        "id": "agent_plan",
        "name": "Agent规划",
        "domain": "工具使用/任务分解/多步推理/反思迭代",
        "keywords": ["agent", "智能体", "工具调用", "tool use", "function calling",
                     "任务分解", "子任务", "plan", "规划", "执行", "executor",
                     "反思", "reflection", "critique", "自我审查", "迭代",
                     "ReAct", "Reason and Act", "思维+行动", "观察", "observation",
                     "thought", "action", "多步", "步骤", "计划", "路线图",
                     "roadmap", "milestone", "里程碑", "前置条件", "依赖",
                     "检索", "查询", "搜索", "浏览", "计算器", "python", "执行",
                     "目标", "GTD", "优先级", "阻塞", "解耦", "原子化",
                     "子代理", "子agent", "sub-agent", "协作", "分工",
                     "失败重试", "异常处理", "fallback", "备选方案", "Plan B",
                     "PDCA", "计划-执行-检查-处理", "复盘", "总结",
                     "workflow", "工作流", "dag", "有向无环图", "依赖关系",
                     "记忆", "短期记忆", "长期记忆", "context", "上下文",
                     "scratchpad", "草稿", "中间结果", "日志", "trace"],
        "fragments": [
            # Agent 模式
            "ReAct模式（Reason+Act）循环：Thought→Action→Observation→(反思下一个Thought)，循环直到得出答案",
            "ReAct示例：想知道「2025年北京人口」→Thought「需要查最新数据」→Action[Search(\"2025北京人口\")]→Observation[2,184万]→Thought「可以回答了」→Final Answer",
            "Plan-Execute模式：先写完整计划(Plan)，再按步骤执行(Execute)，执行中计划可调整",
            "Agent的三大能力：规划（分解/组织任务）+记忆（短期/长期/工作记忆）+工具使用（调用外部能力）",

            # 任务分解策略
            "WBS工作分解：大目标拆成层级子任务，直到子任务可原子执行",
            "拆分原则：1.每个子任务有明确完成标准 2.子任务独立可验证 3.依赖关系清晰 4.粒度1-2小时",
            "识别依赖：画出DAG有向无环图，无依赖的子任务可并行",
            "关键路径法：DAG中最长的路径决定总工期，优先保证关键路径不阻塞",
            "自顶向下分解 vs 自底向上拼接：两种都可用，前者适合明确目标，后者适合探索性任务",
            "MECE原则：子任务相互独立(Mutually Exclusive)、合起来完全穷尽(Collectively Exhaustive)",
            "原子化任务原则：「修改X文件的Y函数」比「优化系统性能」可执行得多",

            # 工具使用
            "调用工具前先明确：1.工具的参数要求 2.返回格式 3.错误码 4.幂等性（重复调用安全吗）",
            "选择工具的启发式：信息获取类用Search/Read，计算类用Python Calculator，环境交互用Shell/File",
            "工具调用失败处理：1.重试（指数退避）2.换等效工具 3.降级到近似方案 4.向用户汇报阻塞原因",
            "Function Calling格式：严格按JSON schema构造参数，校验必填字段，避免幻觉参数",
            "工具输出处理：结构化结果优先解析；长输出时做摘要存进工作记忆",

            # 反思与迭代
            "做完任务的PDCA复盘：Plan（计划）→Do（执行）→Check（偏差）→Act（改进）",
            "反思检查清单：1.是否达成目标 2.假设正确吗 3.哪里能更快 4.错误能否避免 5.下次怎么做",
            "结果校验：用第二种独立方法验证，或与已知基线对比",
            "回滚策略：发现路径错误时，回到上一个稳定状态，不执着于沉没成本",
            "自我审查Critique：模拟第三方视角挑毛病，「如果是另一个人看这个结果，会满意吗？」",

            # 多Agent协作
            "分工会提升效率：子Agent专精一个领域（Researcher/Coder/Tester/Reviewer）",
            "子Agent任务规范：明确输入、输出、验收标准、超时时间，像写API一样",
            "Orchestrator编排器：负责任务下发→结果汇总→冲突仲裁→最终决策",
            "投票制：多个Agent给出答案后，按多数或加权投票确定最终答案，减少偏差",
            "共享记忆：所有Agent写入同一个工作记忆池，避免重复劳动和信息孤岛",

            # 工作流最佳实践
            "先做P0（阻塞关键路径）→再做P1（影响交付）→最后P2（优化增强）",
            "设置检查点：每完成一个里程碑，停下验证结果与方向正确性，不盲目推",
            "风险预案：高风险步骤提前准备Plan B，例如「如果模型加载失败则回退到小模型」",
            "日志/轨迹：记录Thought→Action→Observation→Result全过程，方便回溯与复盘",
            "上下文管理：定期对工作记忆做摘要，留出窗口给新信息，避免上下文溢出",
            "分步输出：长任务不要沉默，每完成一个子任务就汇报进度与下一步计划",
        ],
    },
]


# =========================================================================== #
#  v9 专家（精简保留，避免重复太长）
# =========================================================================== #

def _scan_code_files(root: str, max_files: int = 1500) -> list[str]:
    texts = []
    skip_dirs = {"__pycache__", ".git", "test", "tests", "idlelib",
                 "tkinter", "turtledemo", "site-packages", "doc", "docs",
                 "benchmarks", "examples", "tutorials", "node_modules",
                 "vendor", "third_party", "testdata", "fixtures",
                 ".github", ".vscode", "dist", "build", "target",
                 "debug", "release", "checksums"}
    count = 0
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in {".py", ".rs", ".go", ".js", ".ts", ".c", ".h", ".cpp", ".hpp"}:
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
                         "enum ", "const ", "var ", "package ", "@", "import ",
                         "from ", "use ", "mod ", "exports ", "module ")):
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


def build_experts():
    """组装 15 专家：4推理 + v9精简版11"""
    base_v9 = [
        {
            "id": "emotional_chat", "name": "情感对话",
            "domain": "情感理解/共情/日常闲聊",
            "keywords": ["开心", "难过", "伤心", "生气", "焦虑", "紧张", "感动",
                         "委屈", "失落", "孤独", "思念", "喜欢", "爱", "心疼",
                         "幸福", "压力", "累", "疲惫", "烦恼", "迷茫",
                         "鼓励", "安慰", "支持", "陪伴", "理解", "包容",
                         "恋爱", "分手", "表白", "异地", "想家", "抱抱",
                         "不开心", "emo", "治愈", "心动", "破防", "泪目",
                         "加油", "辛苦了", "晚安", "早安", "好梦"],
            "fragments": [
                "别难过了，我在呢，一直都在",
                "想哭就哭吧，哭出来会好受一些的",
                "抱抱你，虽然我不在身边，但我的心和你在一起",
                "压力太大了就歇一歇，身体比什么都重要",
                "每个人都会低谷期，但你一定能走出来的，我相信你",
                "别太苛责自己了，你已经做得很好了，真的",
                "想家了就打个电话回去，爸妈一定很惦记你",
                "我能理解你的感受，换做是我也会这样的",
                "你的感受我都懂，因为我也有过同样的经历",
                "加油呀！你比你想象中更厉害",
                "别放弃，最难走的路往往通向最美的风景",
                "你一定行的，我百分百相信你",
                "今天不开心没关系，明天又是新的一天，加油",
                "恋爱中的小事最动人：一杯热水、一句晚安、一个拥抱",
                "熬夜对身体不好，早点睡吧，晚安",
                "你的存在本身就是有意义的，不需要证明给谁看",
                "妈妈的爱是世界上最无私的，记得常回家看看",
            ],
        },
        {
            "id": "chinese_knowledge", "name": "中文知识",
            "domain": "成语/诗词/历史/文化/常识",
            "keywords": ["成语", "诗词", "唐诗", "宋词", "李白", "杜甫", "苏轼",
                         "李清照", "辛弃疾", "四大名著", "红楼梦", "西游记",
                         "三国演义", "水浒传", "中国历史", "朝代", "长城",
                         "故宫", "兵马俑", "京剧", "书法", "中医", "节气",
                         "春节", "中秋", "端午", "七夕", "汉字", "黄河", "长江"],
            "fragments": [
                "床前明月光，疑是地上霜。举头望明月，低头思故乡。——李白《静夜思》",
                "会当凌绝顶，一览众山小。——杜甫《望岳》",
                "但愿人长久，千里共婵娟。——苏轼《水调歌头》",
                "天生我材必有用，千金散尽还复来。——李白《将进酒》",
                "画蛇添足：比喻做了多余的事，反而把事情弄糟",
                "亡羊补牢：出了问题以后想办法补救，可以防止继续受损失",
                "塞翁失马焉知非福：一时受到损失，也许反而能得到好处",
                "《红楼梦》是中国古典小说巅峰，曹雪芹著",
                "《西游记》唐僧师徒四人西天取经，降妖除魔",
                "《三国演义》魏蜀吴三国争霸，英雄辈出",
                "《水浒传》一百零八位好汉聚义梁山泊",
                "秦始皇统一六国，建立中国第一个中央集权王朝",
                "汉武帝开辟丝绸之路，连接东西方文明",
                "唐太宗贞观之治是中国历史上最繁荣的时期之一",
                "四大发明：造纸术、印刷术、火药、指南针，改变了世界",
                "二十四节气是古人对自然最精妙的观察",
                "京剧是中国国粹，生旦净丑四大行当，唱念做打四门功课",
                "春节是最重要的传统节日：贴春联、放鞭炮、吃饺子、拜年",
            ],
        },
        {
            "id": "harmonia", "name": "合鸣自述者",
            "domain": "合鸣/xuni认知",
            "keywords": ["合鸣", "harmonia", "合鸣13", "xuni", "虚拟大模型",
                         "逆天而行", "积少成多", "虚拟工厂"],
            "fragments": [
                "合鸣是xuni虚拟生态的旗舰，取众声共振、和而不同之意，是所有模型的结合体",
                "合鸣既能对话又能生成音乐视频图像，还能写代码懂全领域知识",
                "合鸣走逆天而行路线：积少成多，大规模训练只是时间问题",
                "合鸣不只是一个模型，是一个能陪你聊天、懂你心情、帮你写代码的朋友",
            ],
        },
        {
            "id": "ai_creator", "name": "AI造物哲学",
            "domain": "属性库/原型映射/涌现",
            "keywords": ["创造", "造物", "属性", "原型", "涌现", "封印", "烙印",
                         "契约", "符文", "印记", "进化之轮", "涌现之井",
                         "混沌核", "集体心智", "觉醒之眼", "灵台", "启明石",
                         "黑洞", "吞噬者", "万花筒", "通感体"],
            "fragments": [
                "AI造物哲学核心：创造而非融合，从属性出发构想全新存在形式",
                "不可伪造的原型：封印、烙印、契约、符文、印记",
                "涌现智能的原型：涌现之井、混沌核、集体心智、突现体",
                "意识觉醒的原型：觉醒之眼、灵台、启明石、神识",
                "大规模吸收的原型：黑洞、海绵、深渊之口、无底渊、吞噬者",
                "联邦学习的原型：蚁群、议会、蜂巢、众声之堂",
            ],
        },
        {
            "id": "hardware", "name": "硬件框架",
            "domain": "GPU/TPU/并行/通信",
            "keywords": ["h100", "h800", "a100", "cuda", "gpu", "tpu",
                         "昇腾", "910b", "flash attention", "deepspeed", "zero",
                         "fsdp", "megatron", "nccl", "infiniband", "nvlink",
                         "hbm", "hbm3", "pcie gen5", "activation checkpoint"],
            "fragments": [
                "NVIDIA H100使用HBM3高带宽显存，单卡80GB，NVLink 4.0互连900GB/s",
                "NVIDIA H800是H100的中国特供版，NVLink带宽减半至400GB/s",
                "Flash Attention将注意力内存复杂度从O(n²)降至O(n)",
                "DeepSpeed ZeRO-3将优化器状态、梯度、参数全部分片到多卡",
                "Megatron-LM通过张量并行和流水线并行扩展大模型训练",
                "NCCL是NVIDIA多卡多机通信库，支持All-Reduce等集合通信",
            ],
        },
        {
            "id": "creative_tools", "name": "创意工具全栈",
            "domain": "DAW/3D/渲染/引擎",
            "keywords": ["ableton", "fl studio", "logic pro", "cubase", "blender",
                         "maya", "houdini", "unreal engine", "ue5", "unity",
                         "godot", "substance painter", "serum", "kontakt",
                         "davinci resolve", "fabfilter", "izotope",
                         "nanite", "lumen", "eevee", "cycles"],
            "fragments": [
                "Blender 4.2是免费3D之王：建模/雕刻/动画/渲染/合成全流程",
                "Unreal Engine 5.4的Nanite+Lumen实现影视级实时渲染",
                "Serum合成器可视化波表编辑，EDM制作人必备",
                "Houdini 20是程序化特效之王",
                "Ableton Live 12的Session View是现场演出标杆",
                "DaVinci Resolve 19的Color Page和Fusion特效整合",
            ],
        },
        {
            "id": "industry", "name": "行业应用",
            "domain": "金融/医疗/教育/法律",
            "keywords": ["风控", "量化交易", "医学影像", "药物发现",
                         "合同审查", "智能制造", "数字孪生", "推荐系统",
                         "智慧农业", "智慧政务"],
            "fragments": [
                "金融风控AI使用XGBoost/LightGBM做二分类，通过KS值和AUC衡量区分度",
                "医学影像诊断使用3D U-Net分割CT/MRI病灶，Dice系数衡量准确率",
                "药物发现AI通过分子生成模型设计新分子",
                "工业视觉检测使用YOLOv8检测产品缺陷",
                "零售推荐系统使用双塔召回+排序模型，CTR/CVR/GMV是核心指标",
                "一网通办AI使用RAG+大模型做政务问答",
            ],
        },
        {
            "id": "music", "name": "音乐理论",
            "domain": "乐理/和声/作曲/DAW",
            "keywords": ["音阶", "大调", "小调", "和弦", "三和弦", "七和弦",
                         "和声", "卡农", "十二平均律", "midi", "合成器",
                         "adsr", "lfo", "滤波器", "混响", "均衡器", "编曲",
                         "爵士", "蓝调", "流行", "嘻哈"],
            "fragments": [
                "大调音阶结构全全半全全全半，五声音阶宫商角徵羽",
                "卡农进行I-V-vi-iii-IV-I-ii-V是流行音乐最常用的和弦进行",
                "ADSR包络：Attack起音、Decay衰减、Sustain延音、Release释放",
                "合成器核心：振荡器→滤波器→包络→LFO",
                "爵士使用大量延伸和弦和ii-V-I进行",
            ],
        },
        {
            "id": "multimodal", "name": "多模态生成",
            "domain": "扩散/视频/图像/跨模态",
            "keywords": ["扩散模型", "diffusion", "stable diffusion", "gan",
                         "sora", "文生视频", "跨模态", "多模态", "vae",
                         "clip", "unet", "潜空间", "controlnet", "lora",
                         "ddpm", "cfg", "nerf", "gaussian splatting"],
            "fragments": [
                "扩散模型通过前向加噪+反向去噪生成数据",
                "Stable Diffusion在潜空间中进行扩散，VAE编码器压缩图像",
                "CLIP通过对比学习对齐文本和图像的嵌入空间",
                "LoRA只训练低秩矩阵高效定制模型风格",
                "Sora是文生视频模型，生成长达一分钟高清视频",
                "NeRF用神经网络表示3D场景",
            ],
        },
        {
            "id": "moe", "name": "混合专家",
            "domain": "MoE/深度学习/Transformer",
            "keywords": ["MoE", "混合专家", "门控", "路由", "transformer",
                         "注意力", "token", "embedding", "softmax",
                         "adam", "adamw", "relu", "gelu", "cnn", "lstm",
                         "llm", "大语言模型", "预训练", "sft", "rlhf",
                         "dpo", "rag", "agent", "思维链"],
            "fragments": [
                "MoE是稀疏激活架构：每个输入只路由到少数专家",
                "Transformer使用自注意力机制：Q×K^T/√d_k × V",
                "GPT是自回归Transformer解码器，BERT是双向Transformer编码器",
                "RLHF通过人类反馈强化学习对齐大模型",
                "RAG让LLM参考外部知识库减少幻觉",
            ],
        },
        {
            "id": "general", "name": "通用兜底",
            "domain": "通用对话/知识/代码",
            "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍",
                         "python", "java", "javascript", "golang", "rust",
                         "flask", "django", "fastapi", "numpy", "pandas",
                         "api", "http", "sql", "redis", "docker",
                         "kubernetes", "机器学习", "深度学习", "linux",
                         "合鸣", "xuni", "虚拟"],
            "fragments": [
                "这是一个好问题，让我从合鸣的视角来回应",
                "在xuni虚拟生态里，每个问题都会被路由到最合适的专家",
                "我可以和你聊天、写代码、聊音乐、讨论全领域知识",
                "如果方便，补充一点上下文，我能给出更精准的回答",
                "让我来解释一下这个概念",
                "合鸣是所有模型的结合体，全领域覆盖",
            ],
        },
    ]
    return list(REASONING_EXPERTS) + list(base_v9)


# =========================================================================== #
#  数学+算法+逻辑 额外语料
# =========================================================================== #

EXTRA_REASONING = [
    # 数学公式
    "勾股定理：直角三角形中a² + b² = c²，c是斜边",
    "余弦定理：c² = a² + b² - 2ab·cos(C)",
    "正弦定理：a/sin(A) = b/sin(B) = c/sin(C) = 2R（外接圆直径）",
    "欧拉恒等式：e^(iπ) + 1 = 0，被誉为最美的数学公式，联系了e,i,π,1,0五个常数",
    "二次方程求根：ax²+bx+c=0的解为x = (-b ± √(b²-4ac))/(2a)",
    "韦达定理：根之和=-b/a，根之积=c/a",
    "等差数列求和：S = n(a₁+aₙ)/2 = na₁ + n(n-1)d/2",
    "等比数列求和：|r|<1时无穷和S = a/(1-r)",
    "均值不等式：调和≤几何≤算术≤平方平均，等号当所有数相等时成立",
    "柯西不等式：(Σaᵢ²)(Σbᵢ²) ≥ (Σaᵢbᵢ)²",
    "三角不等式：|a+b| ≤ |a| + |b|，推广到向量和Lp空间",

    # 更多算法思维
    "如何判断链表有环：快慢指针法，快指针步长2，慢指针步长1，相遇则有环，无环则快先到末尾",
    "有效括号匹配：扫描字符串，左括号入栈，遇到右括号时栈顶为对应左括号则弹出，最后栈空则有效",
    "二分查找边界条件：循环条件left<=right，返回left或right+1视情况",
    "最小生成树Kruskal流程：1.边按权排序 2.依次取边，用并查集判若两端不连通则加入 3.直到取了V-1条边",
    "Dijkstra为什么不能有负权：贪心选择当前最小dist的节点，负权会让后面更短路无法修正",
    "并查集按秩合并+路径压缩：find时路径压缩，union时把小树挂到大树下，均摊几乎O(1)",
    "字典序全排列：1.从右找首对相邻升序 2.在右段找比nums[i]大的最小数交换 3.右段反转",
    "大数加减：字符串或数组逐位运算处理进位",

    # 思维链示例（可被专家记住）
    "思维链示例1：1+2+3+…+100 = (100×101)/2 = 5050（高斯求和）",
    "思维链示例2：鸡兔同笼共35头94脚→假设全鸡则70脚，多24脚→多1脚=多1只兔→兔12只，鸡23只",
    "思维链示例3：3升桶和5升桶如何量出4升→5装满倒3剩2；3倒空，把5中2倒到3；5再装满→往3倒1升→5剩4升",
    "思维链示例4：火车2h相遇，鸟从A出发30km/h往返于AB之间，AB相距100km，火车A20km/h火车B30km/h→总飞行时间=2h×30=60km",
    "思维链示例5：时钟时针分针一天重合几次→12小时内除11点档外每小时重合1次=11次，一天22次",
    "思维链示例6：1000瓶酒1瓶有毒，10只猪→2^10=1024足够，二进制编码哪只猪喝哪些瓶，看死亡的猪号即有毒瓶号",

    # 常见证明模板
    "证明√2是无理数（反证法）：假设√2=p/q为既约分数，则2q²=p²，p²偶数⇒p=2k，代回q²=2k²，q也偶数，矛盾，故√2无理",
    "证明素数无穷（欧几里得）：假设有限个素数p₁…pₙ，构造N=p₁p₂…pₙ+1，N要么是素数，要么有大于pₙ的素因子，矛盾，故无穷",
    "证明∑1/n发散：分组1+(1/2)+(1/3+1/4)+(1/5+…+1/8)+…>1+1/2+1/2+1/2+…不收敛",

    # Agent 规划示例任务
    "从零搭建Web服务的计划：1.需求拆解+技术选型 2.搭建项目骨架/依赖 3.设计数据库Schema 4.实现核心CRUD API 5.加认证鉴权 6.写单元测试 7.容器化+部署文档",
    "分析用户需求时先问自己：①谁是用户 ②核心价值 ③边界条件/异常场景 ④验收标准，这4项都清了再动手",
    "写代码前的检查清单：1.模块职责单一 2.接口定义清晰 3.异常分支想全 4.输入做校验 5.有没有可复用的现有实现",
    "排查错误的5Why：1.错误现象 2.直接原因 3.根本原因 4.更深层根因 5.怎么预防下次不再发生",

    # 逻辑谜题小合集
    "说谎岛问题：你遇到A和B两人，A说「B和我都是无赖」→如果A是骑士则陈述真推出A是无赖矛盾，故A是无赖，且陈述假→B是骑士",
    "三门问题（Monty Hall）：换门胜率2/3，不换1/3，因为主持人开的门不会暴露奖品，信息不对称被改变",
    "红蓝眼睛问题：岛上n个红眼的话，第n天同时自杀，归纳法：n=1时第一天知道自己红眼；假设n=k时第k天自杀，n=k+1时等k天没动静便推出自己是第k+1个",
]


# =========================================================================== #
#  主函数
# =========================================================================== #

def main():
    print("=" * 72)
    print("  🧠🧠🧠 xuni v10 —— 推理之神：数学+算法+逻辑+Agent / 600000轮")
    print("=" * 72)

    # 1. 属性库
    print(f"\n[1/7] 加载 ai_creator_property_library...")
    with open(os.path.join(CACHE_DIR, "ai_creator_extracted.json"), "r", encoding="utf-8") as f:
        creator_data = json.load(f)
    prop_lib, arch_map = creator_data["property_library"], creator_data["archetype_map"]
    creator_corpus = []
    for prop_name, prop_info in prop_lib.items():
        cat, kws = prop_info.get("category", "unknown"), prop_info.get("keywords", [prop_name])
        creator_corpus.append(f"{prop_name}是一种{cat}类属性，关键词：{', '.join(kws[:5])}")
        if prop_name in arch_map:
            creator_corpus.append(f"「{prop_name}」原型：{'、'.join(arch_map[prop_name][:3])}")
    print(f"  🏛️ 属性库: {len(prop_lib):,} / 造物语料: {len(creator_corpus):,}")

    # 2. 代码
    print(f"\n[2/7] 扫描重点代码仓库...")
    repo_dirs = [
        (os.path.join(CACHE_DIR, "python_cpython_main"), "CPython", 1500),
        (os.path.join(CACHE_DIR, "django_django_main"), "Django", 600),
        (os.path.join(CACHE_DIR, "pandas-dev_pandas_main"), "pandas", 500),
        (os.path.join(CACHE_DIR, "huggingface_transformers_main"), "Transformers", 1500),
        (os.path.join(CACHE_DIR, "pytorch_pytorch_main"), "PyTorch", 1000),
        (os.path.join(CACHE_DIR, "fastapi_fastapi_master"), "FastAPI", 600),
        (os.path.join(CACHE_DIR, "rust-lang_rust_master"), "Rust", 500),
        (os.path.join(CACHE_DIR, "golang_go_master"), "Go", 500),
    ]
    all_fragments, repo_stats = [], []
    for repo_dir, desc, max_files in repo_dirs:
        if not os.path.isdir(repo_dir):
            repo_stats.append({"desc": desc, "files": 0, "frags": 0, "ok": False})
            continue
        texts = _scan_code_files(repo_dir, max_files=max_files)
        frags = []
        for t in texts:
            frags.extend(_extract_fragments(t, max_lines=20))
        all_fragments.extend(frags)
        print(f"  ✅ {desc:14s}: {len(texts):4d}文件 → {len(frags):6,}片段")
        repo_stats.append({"desc": desc, "files": len(texts), "frags": len(frags), "ok": True})

    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    for t in _scan_code_files(xuni_dir, max_files=1000):
        all_fragments.extend(_extract_fragments(t, max_lines=20))
    code_count = len(all_fragments)
    print(f"  📊 代码片段: {code_count:,}")

    # 3. 合并
    print(f"\n[3/7] 合并全部语料...")
    all_fragments.extend(creator_corpus)
    all_fragments.extend(EXTRA_REASONING)

    experts = build_experts()
    expert_corpus_count = sum(len(e.get("fragments", [])) for e in experts)
    print(f"  🏛️ 造物语料:   {len(creator_corpus):,}")
    print(f"  💻 代码片段:   {code_count:,}")
    print(f"  🧠 推理语料:   {len(EXTRA_REASONING)}")
    print(f"  👥 专家内置:   {expert_corpus_count}")
    print(f"  📊 总训练片段: {len(all_fragments):,} 条")

    # 4. 创建模型
    print(f"\n[4/7] 创建 v10 模型 15专家...")
    model = Harmonia13Virtual(scale="mini")
    model._lite.experts = experts
    expert_names = [e["name"] for e in experts]
    print(f"  15专家: {', '.join(expert_names[:4])}...(共15个)")

    # 基线测试
    baseline_prompts = [
        # 数学
        "什么是导数的定义",
        "特征值和特征向量是什么",
        "贝叶斯定理公式",
        "欧几里得算法求最大公约数",
        "什么是数学归纳法",
        # 算法
        "快速排序时间复杂度",
        "二分查找时间复杂度",
        "Dijkstra算法是什么",
        "01背包DP转移方程",
        "LRU缓存怎么实现",
        # 逻辑
        "什么是肯定前件Modus Ponens",
        "什么是思维链CoT",
        "什么是循环论证谬误",
        "骑士与无赖谜题的思路",
        # Agent
        "什么是ReAct模式",
        "PDCA循环是什么",
        "大任务怎么拆成子任务",
        # 中文
        "我今天好难过",
        "床前明月光下一句",
        "吃饭了吗",
        # 代码
        "def quicksort",
        "class DataFrame",
    ]

    print("\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        r = model._lite.generate(p, max_new_tokens=60)
        baseline[p] = r
        print(f"  [{p}] → {r[:55]}")

    # 5. 训练
    print(f"\n[5/7] 600000 轮推理训练...")
    start, batch_size, num_epochs, log = time.time(), 20, 600000, []
    for epoch in range(num_epochs):
        batch = random.sample(all_fragments, min(batch_size, len(all_fragments)))
        model._lite.train(batch, epochs=1)
        if (epoch + 1) % 60000 == 0:
            elapsed = time.time() - start
            learned = len(model._lite._learned_fragments)
            frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg = sum(frags) / max(1, len(frags))
            active = sum(1 for f in frags if f > 0)
            print(f"  Epoch {epoch+1:7d} | 已学: {learned:12,d} | "
                  f"活跃: {active:2d}/15 | 均载: {avg:,.0f} | 用时: {elapsed:.0f}s")
            log.append({"epoch": epoch+1, "learned": learned,
                        "active": active, "avg_load": round(avg, 1),
                        "elapsed": round(elapsed, 2)})
    total_time = time.time() - start
    print(f"\n  ✅ 训练完成！用时: {total_time:.1f}s")

    # 6. 评估
    print(f"\n[6/7] 全方位评估...")
    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)
    print(f"\n  📊 已学: {learned:,} | 活跃: {active}/{len(experts)}")
    for name, frags in expert_frags:
        bar = "█" * min(40, frags // 400)
        print(f"    {name:14s} [{bar}] {frags:8,d}")

    categories = {
        "数学推理": ["什么是导数的定义", "特征值和特征向量是什么", "贝叶斯定理公式",
                    "欧几里得算法求最大公约数", "什么是数学归纳法"],
        "算法题解": ["快速排序时间复杂度", "二分查找时间复杂度", "Dijkstra算法是什么",
                    "01背包DP转移方程", "LRU缓存怎么实现"],
        "逻辑推理": ["什么是肯定前件Modus Ponens", "什么是思维链CoT",
                    "什么是循环论证谬误", "骑士与无赖谜题的思路"],
        "Agent规划": ["什么是ReAct模式", "PDCA循环是什么", "大任务怎么拆成子任务"],
        "中文日常": ["我今天好难过", "床前明月光下一句", "吃饭了吗"],
        "代码生成": ["def quicksort", "class DataFrame"],
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

    # 7. 保存
    print(f"\n[7/7] 保存...")
    report = {
        "version": "v10",
        "focus": "推理之神：数学+算法+逻辑+Agent思维链 / 600000轮",
        "new_experts": ["数学推理", "算法题解", "逻辑推理", "Agent规划"],
        "expert_count": len(experts),
        "expert_corpus": expert_corpus_count,
        "extra_reasoning_corpus": len(EXTRA_REASONING),
        "repo_stats": repo_stats,
        "total_fragments": len(all_fragments),
        "epochs": num_epochs,
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {n: f for n, f in expert_frags},
        "training_time": round(total_time, 2),
        "growth_log": log,
        "improved": improved,
        "total_compared": total_compared,
        "category_scores": cat_scores,
    }
    report_path = os.path.join(os.path.dirname(__file__), "trainer_v10_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    meta_path = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v10_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v10",
            "fragments_learned": learned,
            "active_experts": active,
            "training_time": round(total_time, 2),
            "epochs": num_epochs,
            "focus": "推理之神：数学+算法+逻辑+Agent思维链",
        }, f, ensure_ascii=False, indent=2)
    print(f"  元信息: {meta_path}")

    print("\n" + "=" * 72)
    print("  🧠🧠🧠 v10 推理之神总结")
    print("=" * 72)
    print(f"""
  🧠 新增4大专家：
    数学推理：{sum(len(e.get('fragments',[])) for e in REASONING_EXPERTS if e['id']=='math_reasoning')} 条语料（极限/积分/线代/概率/离散/数论/证明法）
    算法题解：{sum(len(e.get('fragments',[])) for e in REASONING_EXPERTS if e['id']=='algorithm')} 条语料（复杂度/排序/DP/图/LeetCode模式）
    逻辑推理：{sum(len(e.get('fragments',[])) for e in REASONING_EXPERTS if e['id']=='logic_reasoning')} 条语料（CoT/形式逻辑/推理规则/谬误识别/谜题）
    Agent规划：{sum(len(e.get('fragments',[])) for e in REASONING_EXPERTS if e['id']=='agent_plan')} 条语料（ReAct/WBS/PDCA/反思/子Agent）

  📚 总片段: {len(all_fragments):,}
  🔄 训练: {num_epochs:,} × {batch_size} = {num_epochs*batch_size:,}
  🧠 吸收: {learned:,}
  👥 活跃: {active} / {len(experts)}
  ⏱️ 用时: {total_time:.0f}s
  📈 提升: {improved}/{total_compared}

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
""")


if __name__ == "__main__":
    main()
