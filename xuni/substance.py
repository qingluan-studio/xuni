"""
SubstanceSystem —— 新物质定义系统

核心理念：采样点产生的一切产物都被定义为"虚拟物质"，
每种物质有名称、定义、属性和用途。

物质分类：
- 能量类：虚拟电、场能量、动能、热能
- 数据类：采样点数据、参数、特征向量
- 凭证类：虚拟凭证、API Key、令牌
- 模型类：虚拟模型、模型快照、训练数据
- 信息类：文本、图像描述、音乐参数

每种物质都有：
- 名称（中文+英文）
- 定义
- 属性（类型、单位、值范围）
- 用途（能做什么）
- 产出方式（从哪里获得）
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Any, List


class SubstanceCategory(Enum):
    ENERGY = auto()
    DATA = auto()
    CREDENTIAL = auto()
    MODEL = auto()
    INFORMATION = auto()
    OTHER = auto()


class SubstanceUnit(Enum):
    NONE = auto()
    VOLT = auto()
    JOULE = auto()
    WATT = auto()
    BYTE = auto()
    COUNT = auto()
    PERCENT = auto()
    HERTZ = auto()
    SECOND = auto()
    TOKEN = auto()
    DIMENSIONLESS = auto()


@dataclass
class SubstanceAttribute:
    """物质属性"""
    name: str
    unit: SubstanceUnit = SubstanceUnit.NONE
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""


@dataclass
class SubstanceDefinition:
    """物质定义"""
    name: str
    name_en: str
    category: SubstanceCategory
    definition: str
    attributes: List[SubstanceAttribute] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)
    production_methods: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    icon: str = "⚪"


class SubstanceSystem:
    """
    虚拟物质定义系统。
    
    管理所有由采样点产生的物质：
    1. 预定义物质库
    2. 自定义物质注册
    3. 物质关系查询
    4. 物质产出追踪
    """

    def __init__(self):
        self.substances: Dict[str, SubstanceDefinition] = {}
        self._register_default_substances()

    def _register_default_substances(self):
        """注册预定义物质"""
        
        # 能量类物质
        self.register(SubstanceDefinition(
            name="虚拟电",
            name_en="Virtual Electricity",
            category=SubstanceCategory.ENERGY,
            definition="由采样点在虚拟电场中产生的能量，是整个虚拟生态系统的基础能源",
            icon="⚡",
            attributes=[
                SubstanceAttribute("电压", SubstanceUnit.VOLT, 0, 1000),
                SubstanceAttribute("能量", SubstanceUnit.JOULE, 0, None),
                SubstanceAttribute("功率", SubstanceUnit.WATT, 0, None),
            ],
            uses=[
                "兑换虚拟凭证",
                "给虚拟模型充能",
                "驱动虚拟API调用",
                "训练虚拟模型",
            ],
            production_methods=[
                "超混沌采样点发电",
                "流体粒子运动发电",
                "光学共振腔发电",
            ],
        ))

        self.register(SubstanceDefinition(
            name="场能量",
            name_en="Field Energy",
            category=SubstanceCategory.ENERGY,
            definition="虚拟电场中存储的总能量，由泊松方程计算得出",
            icon="🔋",
            attributes=[
                SubstanceAttribute("能量密度", SubstanceUnit.JOULE, 0, None),
                SubstanceAttribute("总能量", SubstanceUnit.JOULE, 0, None),
            ],
            uses=[
                "铸造虚拟凭证",
                "驱动神经同步引擎",
                "调节音乐参数",
            ],
            production_methods=[
                "泊松方程松弛求解",
                "高斯平滑处理",
            ],
            dependencies=["虚拟电"],
        ))

        self.register(SubstanceDefinition(
            name="动能",
            name_en="Kinetic Energy",
            category=SubstanceCategory.ENERGY,
            definition="流体粒子运动产生的动能，可转化为热能",
            icon="🏃",
            attributes=[
                SubstanceAttribute("速度", SubstanceUnit.HERTZ, 0, None),
                SubstanceAttribute("动能", SubstanceUnit.JOULE, 0, None),
            ],
            uses=[
                "驱动流体模拟",
                "产生热能",
                "影响音乐节奏",
            ],
            production_methods=[
                "SPH流体模拟",
                "粒子碰撞",
            ],
        ))

        self.register(SubstanceDefinition(
            name="热能",
            name_en="Thermal Energy",
            category=SubstanceCategory.ENERGY,
            definition="由动能转化而来的热量，影响流体相态变化",
            icon="🔥",
            attributes=[
                SubstanceAttribute("温度", SubstanceUnit.DIMENSIONLESS, 0, 1000),
            ],
            uses=[
                "驱动蒸发/凝结",
                "影响粒子状态",
            ],
            production_methods=[
                "动能耗散",
            ],
            dependencies=["动能"],
        ))

        # 数据类物质
        self.register(SubstanceDefinition(
            name="采样点数据",
            name_en="Sample Point Data",
            category=SubstanceCategory.DATA,
            definition="超混沌系统生成的6维采样点数据（x,y,z,w,charge,entropy）",
            icon="📍",
            attributes=[
                SubstanceAttribute("数量", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 4, 6),
            ],
            uses=[
                "构建虚拟电场",
                "训练虚拟模型",
                "生成音乐参数",
            ],
            production_methods=[
                "超混沌Chen系统",
                "Lorenz-96系统",
                "Mandelbulb分形",
                "4D噪声",
            ],
        ))

        self.register(SubstanceDefinition(
            name="音乐参数",
            name_en="Music Parameters",
            category=SubstanceCategory.DATA,
            definition="由场能量转换而来的16个音乐合成参数",
            icon="🎛️",
            attributes=[
                SubstanceAttribute("参数数量", SubstanceUnit.COUNT, 16, 16),
            ],
            uses=[
                "驱动物理建模合成器",
                "生成音乐",
            ],
            production_methods=[
                "XuniConverter转换",
            ],
            dependencies=["场能量"],
        ))

        self.register(SubstanceDefinition(
            name="特征向量",
            name_en="Feature Vector",
            category=SubstanceCategory.DATA,
            definition="从场能量分布中提取的特征表示",
            icon="📐",
            attributes=[
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "模型训练",
                "分类预测",
            ],
            production_methods=[
                "能量分布分析",
            ],
            dependencies=["场能量"],
        ))

        # 凭证类物质
        self.register(SubstanceDefinition(
            name="虚拟凭证",
            name_en="Virtual Credential",
            category=SubstanceCategory.CREDENTIAL,
            definition="用场能量铸造的虚拟访问凭证，支持JWT格式",
            icon="🔑",
            attributes=[
                SubstanceAttribute("长度", SubstanceUnit.COUNT, 24, 24),
                SubstanceAttribute("能量值", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "API认证",
                "模型调用权限",
                "资源访问控制",
            ],
            production_methods=[
                "XuniCredential.mint()",
            ],
            dependencies=["场能量"],
        ))

        self.register(SubstanceDefinition(
            name="API密钥",
            name_en="API Key",
            category=SubstanceCategory.CREDENTIAL,
            definition="24位字母数字组合的API访问密钥",
            icon="🗝️",
            attributes=[
                SubstanceAttribute("长度", SubstanceUnit.COUNT, 24, 24),
            ],
            uses=[
                "真实AI服务认证",
                "API调用授权",
            ],
            production_methods=[
                "XuniCredential.mint(API_KEY)",
            ],
            dependencies=["虚拟凭证"],
        ))

        self.register(SubstanceDefinition(
            name="模型令牌",
            name_en="Model Token",
            category=SubstanceCategory.CREDENTIAL,
            definition="用于调用虚拟模型的专用凭证",
            icon="🎟️",
            attributes=[
                SubstanceAttribute("调用次数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "调用虚拟模型",
                "生成内容",
            ],
            production_methods=[
                "XuniCredential.mint(MODEL_TOKEN)",
            ],
            dependencies=["虚拟凭证"],
        ))

        # 模型类物质
        self.register(SubstanceDefinition(
            name="虚拟模型",
            name_en="Virtual Model",
            category=SubstanceCategory.MODEL,
            definition="由虚拟电驱动的模拟AI模型，无需真实AI依赖",
            icon="🤖",
            attributes=[
                SubstanceAttribute("能量需求", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("类型", SubstanceUnit.NONE),
            ],
            uses=[
                "文本生成",
                "图像描述",
                "音乐作曲",
                "分类预测",
                "聊天对话",
            ],
            production_methods=[
                "XuniModelRegistry注册",
            ],
            dependencies=["虚拟电"],
        ))

        self.register(SubstanceDefinition(
            name="模型快照",
            name_en="Model Snapshot",
            category=SubstanceCategory.MODEL,
            definition="模型状态的完整快照，包含权重和性能指标",
            icon="📸",
            attributes=[
                SubstanceAttribute("时间戳", SubstanceUnit.SECOND),
            ],
            uses=[
                "模型恢复",
                "状态切换",
                "数据层转换",
            ],
            production_methods=[
                "ModelDataSnapshot创建",
            ],
            dependencies=["虚拟模型"],
        ))

        self.register(SubstanceDefinition(
            name="训练数据",
            name_en="Training Data",
            category=SubstanceCategory.MODEL,
            definition="从真实模型获取的供虚拟模型训练的数据",
            icon="📊",
            attributes=[
                SubstanceAttribute("样本数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "训练虚拟模型",
                "数据层转换",
            ],
            production_methods=[
                "RealModelAdapter.get_data_for_virtual_training()",
            ],
            dependencies=["真实模型"],
        ))

        self.register(SubstanceDefinition(
            name="真实模型",
            name_en="Real Model",
            category=SubstanceCategory.MODEL,
            definition="接入真实AI服务的模型适配器（OpenAI/Anthropic/Google等）",
            icon="💡",
            attributes=[
                SubstanceAttribute("服务类型", SubstanceUnit.NONE),
            ],
            uses=[
                "真实AI调用",
                "提供训练数据",
                "混合模式推理",
            ],
            production_methods=[
                "RealModelAdapter连接",
            ],
        ))

        # 信息类物质
        self.register(SubstanceDefinition(
            name="生成文本",
            name_en="Generated Text",
            category=SubstanceCategory.INFORMATION,
            definition="由虚拟/真实模型生成的文本内容",
            icon="📝",
            attributes=[
                SubstanceAttribute("长度", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "内容创作",
                "聊天回复",
                "文档生成",
            ],
            production_methods=[
                "XuniTextGenerator.predict()",
                "真实模型调用",
            ],
            dependencies=["虚拟模型", "真实模型"],
        ))

        self.register(SubstanceDefinition(
            name="图像描述",
            name_en="Image Description",
            category=SubstanceCategory.INFORMATION,
            definition="由图像描述模型生成的视觉描述文本",
            icon="🖼️",
            uses=[
                "图像标注",
                "内容生成",
            ],
            production_methods=[
                "XuniImageDescriber.predict()",
            ],
            dependencies=["虚拟模型"],
        ))

        self.register(SubstanceDefinition(
            name="音乐参数集",
            name_en="Music Parameter Set",
            category=SubstanceCategory.INFORMATION,
            definition="由音乐作曲模型生成的完整音乐参数配置",
            icon="🎵",
            uses=[
                "音乐合成",
                "编曲",
            ],
            production_methods=[
                "XuniMusicComposer.predict()",
            ],
            dependencies=["虚拟模型"],
        ))

        self.register(SubstanceDefinition(
            name="分类结果",
            name_en="Classification Result",
            category=SubstanceCategory.INFORMATION,
            definition="由分类模型输出的类别标签和概率分布",
            icon="🏷️",
            uses=[
                "情感分析",
                "内容分类",
                "决策支持",
            ],
            production_methods=[
                "XuniClassifier.predict()",
            ],
            dependencies=["虚拟模型"],
        ))

        # ===== 记忆类物质（工厂新产能） =====
        self.register(SubstanceDefinition(
            name="记忆点",
            name_en="Memory Point",
            category=SubstanceCategory.INFORMATION,
            definition="带重要性评分、标签和时间戳的语义记忆条目，是工厂记忆系统的最小单元",
            icon="🧠",
            attributes=[
                SubstanceAttribute("重要性", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("访问次数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("衰减率", SubstanceUnit.DIMENSIONLESS, 0, 1),
            ],
            uses=[
                "为合鸣模型提供上下文记忆",
                "跨会话知识沉淀",
                "重要事实长期保存",
                "子代理经验积累",
            ],
            production_methods=[
                "MemoryBank.memorize()",
                "ShortTermMemory.store()",
                "LongTermMemory.store()",
            ],
            dependencies=["生成文本"],
        ))

        self.register(SubstanceDefinition(
            name="长期记忆",
            name_en="Long Term Memory",
            category=SubstanceCategory.INFORMATION,
            definition="重要性≥0.6 的记忆点晋升为长期记忆，按标签索引、按重要性排序、按时间衰减",
            icon="💾",
            attributes=[
                SubstanceAttribute("容量", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("衰减率", SubstanceUnit.DIMENSIONLESS, 0, 1),
                SubstanceAttribute("标签数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "合鸣模型事实检索",
                "子代理经验库",
                "跨会话上下文恢复",
            ],
            production_methods=[
                "MemoryBank.consolidate()",
                "LongTermMemory.store()",
            ],
            dependencies=["记忆点"],
        ))

        self.register(SubstanceDefinition(
            name="共振记忆",
            name_en="Resonance Memory",
            category=SubstanceCategory.INFORMATION,
            definition="XuniBrain 网络共振模式的快照，加载后网络收敛到该 attractor 附近，每次回忆都是独特的",
            icon="🌀",
            attributes=[
                SubstanceAttribute("频率", SubstanceUnit.HERTZ, 0, None),
                SubstanceAttribute("振幅", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("唤起次数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "音乐主题回忆",
                "神经状态恢复",
                "梦境生成种子",
            ],
            production_methods=[
                "XuniMemory.capture()",
            ],
            dependencies=["场能量"],
        ))

        # ===== 代理类物质（工厂新产能） =====
        self.register(SubstanceDefinition(
            name="子代理",
            name_en="Sub Agent",
            category=SubstanceCategory.MODEL,
            definition="由 AI 名称池派生的虚拟子代理，可跨层认领模型并执行任务，是工厂的代理产物",
            icon="🎭",
            attributes=[
                SubstanceAttribute("名称", SubstanceUnit.NONE),
                SubstanceAttribute("认领模型数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("完成任务数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "并行处理多任务",
                "跨层模型认领与训练",
                "子任务分发与汇总",
                "经验记忆沉淀",
            ],
            production_methods=[
                "SubAgentOrchestrator.spawn()",
                "ModelLayer.auto_assign()",
            ],
            dependencies=["虚拟模型", "记忆点"],
        ))

        self.register(SubstanceDefinition(
            name="代理经验",
            name_en="Agent Experience",
            category=SubstanceCategory.INFORMATION,
            definition="子代理在执行任务过程中积累的经验记忆，按代理名分组存储，可用于后续任务复用",
            icon="📜",
            attributes=[
                SubstanceAttribute("经验条目数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("成功率", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "子代理经验复用",
                "任务路由优化",
                "团队协作学习",
            ],
            production_methods=[
                "SubAgent.execute()",
            ],
            dependencies=["子代理", "记忆点"],
        ))

        # ===== 有机物质体系（20+ 种，工厂深层产物） =====
        # —— 知识类 ——
        self.register(SubstanceDefinition(
            name="知识结晶",
            name_en="Knowledge Crystal",
            category=SubstanceCategory.INFORMATION,
            definition="高密度结构化知识块，由多条记忆点融合压缩而成，是知识的固态形态",
            icon="💎",
            attributes=[
                SubstanceAttribute("压缩率", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("纯度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("结晶度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "核心知识存储",
                "跨会话知识迁移",
                "知识结晶可被模型直接吸收",
            ],
            production_methods=[
                "MemoryBank.consolidate()",
                "SubstanceFusionEngine.fuse(记忆点, 记忆点)",
            ],
            dependencies=["记忆点"],
        ))

        self.register(SubstanceDefinition(
            name="思维链",
            name_en="Thought Chain",
            category=SubstanceCategory.INFORMATION,
            definition="串联多个记忆点的推理路径，是逻辑流的线性展开形式",
            icon="⛓️",
            attributes=[
                SubstanceAttribute("链长", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("连贯性", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "多步推理",
                "因果链分析",
                "逻辑推导",
            ],
            production_methods=[
                "HarmoniaMemory.recall_chain()",
                "SubstanceFusionEngine.fuse(记忆点, 参数包)",
            ],
            dependencies=["记忆点", "参数包"],
        ))

        self.register(SubstanceDefinition(
            name="灵感闪",
            name_en="Inspiration Flash",
            category=SubstanceCategory.INFORMATION,
            definition="多个不相关记忆碰撞产生的突发性洞见，是思维链的非线性跃迁",
            icon="✨",
            attributes=[
                SubstanceAttribute("强度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("新颖度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "创造性思维",
                "跨领域联想",
                "问题突破",
            ],
            production_methods=[
                "SubstanceFusionEngine.collide(记忆点, 共振记忆)",
            ],
            dependencies=["记忆点", "共振记忆"],
        ))

        self.register(SubstanceDefinition(
            name="逻辑流",
            name_en="Logic Flow",
            category=SubstanceCategory.INFORMATION,
            definition="结构化的推理流，由参数包驱动的思维链有序展开",
            icon="📊",
            attributes=[
                SubstanceAttribute("流速", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("熵", SubstanceUnit.DIMENSIONLESS, 0, 1),
            ],
            uses=[
                "形式化推理",
                "决策树展开",
                "算法推演",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(思维链, 参数包)",
            ],
            dependencies=["思维链", "参数包"],
        ))

        self.register(SubstanceDefinition(
            name="元知识",
            name_en="Meta Knowledge",
            category=SubstanceCategory.INFORMATION,
            definition="关于知识结构的知识，描述知识之间的关系图",
            icon="🗺️",
            attributes=[
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("密度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "知识图谱构建",
                "知识导航",
                "知识发现",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(知识结晶, 知识结晶)",
            ],
            dependencies=["知识结晶"],
        ))

        # —— 情感类 ——
        self.register(SubstanceDefinition(
            name="情感波",
            name_en="Emotion Wave",
            category=SubstanceCategory.INFORMATION,
            definition="由情感信号驱动的波动模式，可调制模型的生成风格",
            icon="🌊",
            attributes=[
                SubstanceAttribute("频率", SubstanceUnit.HERTZ, 0, None),
                SubstanceAttribute("振幅", SubstanceUnit.DIMENSIONLESS, 0, 1),
                SubstanceAttribute("相位", SubstanceUnit.DIMENSIONLESS, 0, 6.28),
            ],
            uses=[
                "情感化生成",
                "风格迁移",
                "情绪调节",
            ],
            production_methods=[
                "SubstanceFusionEngine.collide(共振记忆, 情感标签)",
            ],
            dependencies=["共振记忆"],
        ))

        self.register(SubstanceDefinition(
            name="共鸣场",
            name_en="Resonance Field",
            category=SubstanceCategory.INFORMATION,
            definition="多个情感波叠加形成的稳定场，可驱动持续的情感表达",
            icon="💞",
            attributes=[
                SubstanceAttribute("场强", SubstanceUnit.JOULE, 0, None),
                SubstanceAttribute("稳定度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "情感持续输出",
                "氛围营造",
                "情感共振",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(情感波, 情感波)",
            ],
            dependencies=["情感波"],
        ))

        self.register(SubstanceDefinition(
            name="意图晶",
            name_en="Intent Crystal",
            category=SubstanceCategory.INFORMATION,
            definition="高密度意图结晶，明确表达用户/代理的核心诉求",
            icon="🎯",
            attributes=[
                SubstanceAttribute("纯度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("明确度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "意图识别",
                "任务聚焦",
                "指令执行",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(记忆点, 参数)",
            ],
            dependencies=["记忆点", "参数"],
        ))

        # —— 认知类 ——
        self.register(SubstanceDefinition(
            name="理解态",
            name_en="Comprehension State",
            category=SubstanceCategory.INFORMATION,
            definition="模型对输入/输出的理解程度，是内部状态的可量化指标",
            icon="🧩",
            attributes=[
                SubstanceAttribute("深度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("广度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "理解度评估",
                "知识覆盖分析",
                "认知状态追踪",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(知识结晶, 思维链)",
            ],
            dependencies=["知识结晶", "思维链"],
        ))

        self.register(SubstanceDefinition(
            name="好奇态",
            name_en="Curiosity State",
            category=SubstanceCategory.INFORMATION,
            definition="模型对未知领域的探索驱动力，是主动学习的内驱力",
            icon="🔍",
            attributes=[
                SubstanceAttribute("强度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("方向", SubstanceUnit.NONE),
            ],
            uses=[
                "主动探索",
                "知识发现",
                "新奇检测",
            ],
            production_methods=[
                "SubstanceFusionEngine.collide(记忆点, 空白态)",
            ],
            dependencies=["记忆点"],
        ))

        self.register(SubstanceDefinition(
            name="洞察点",
            name_en="Insight Point",
            category=SubstanceCategory.INFORMATION,
            definition="从大量记忆中涌现的关键认知，是理解态的跃迁点",
            icon="💡",
            attributes=[
                SubstanceAttribute("锐度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("新颖度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "关键洞察提取",
                "模式发现",
                "本质把握",
            ],
            production_methods=[
                "SubstanceFusionEngine.collide(理解态, 灵感闪)",
            ],
            dependencies=["理解态", "灵感闪"],
        ))

        self.register(SubstanceDefinition(
            name="反思环",
            name_en="Reflection Loop",
            category=SubstanceCategory.INFORMATION,
            definition="自我检查与修正的循环过程，是元认知的体现",
            icon="🔄",
            attributes=[
                SubstanceAttribute("环数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("修正率", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "自我修正",
                "错误检测",
                "元认知",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(理解态, 逻辑流)",
            ],
            dependencies=["理解态", "逻辑流"],
        ))

        # —— 参数类 ——
        self.register(SubstanceDefinition(
            name="参数包",
            name_en="Parameter Pack",
            category=SubstanceCategory.DATA,
            definition="可交易、可注入、可序列化的一组参数，是模型的本质形态",
            icon="📦",
            attributes=[
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("质量", SubstanceUnit.PERCENT, 0, 100),
            ],
            uses=[
                "模型注入",
                "跨实例转移",
                "AI 之间交易",
            ],
            production_methods=[
                "ParameterExtractor.extract()",
                "ParameterTrainer.train()",
            ],
        ))

        self.register(SubstanceDefinition(
            name="参数向量",
            name_en="Parameter Vector",
            category=SubstanceCategory.DATA,
            definition="参数包的向量化表示，用于相似度计算和空间检索",
            icon="→",
            attributes=[
                SubstanceAttribute("维度", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("范数", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "相似度搜索",
                "参数空间投影",
                "聚类分析",
            ],
            production_methods=[
                "ParameterPack.to_vector()",
            ],
            dependencies=["参数包"],
        ))

        self.register(SubstanceDefinition(
            name="参数梯度",
            name_en="Parameter Gradient",
            category=SubstanceCategory.DATA,
            definition="参数包的变化率，指导模型更新方向",
            icon="📈",
            attributes=[
                SubstanceAttribute("方向", SubstanceUnit.NONE),
                SubstanceAttribute("模长", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "梯度下降",
                "参数优化",
                "学习方向指引",
            ],
            production_methods=[
                "ParameterTrainer.compute_gradient()",
            ],
            dependencies=["参数包"],
        ))

        # —— 共振类 ——
        self.register(SubstanceDefinition(
            name="共振模式",
            name_en="Resonance Pattern",
            category=SubstanceCategory.INFORMATION,
            definition="多个振子同步振荡形成的稳定模式，是合鸣模型的底层运作形态",
            icon="〰️",
            attributes=[
                SubstanceAttribute("频率", SubstanceUnit.HERTZ, 0, None),
                SubstanceAttribute("相干度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "专家路由",
                "模式匹配",
                "风格生成",
            ],
            production_methods=[
                "XuniBrain.synchronize()",
                "SubstanceFusionEngine.fuse(共振模式, 参数包)",
            ],
            dependencies=["场能量", "参数包"],
        ))

        self.register(SubstanceDefinition(
            name="吸引子",
            name_en="Attractor",
            category=SubstanceCategory.INFORMATION,
            definition="共振模式收敛的目标状态，是记忆的终极形态",
            icon="🌀",
            attributes=[
                SubstanceAttribute("深度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("稳定性", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "记忆检索",
                "状态收敛",
                "模式固化",
            ],
            production_methods=[
                "XuniMemory.capture()",
                "SubstanceFusionEngine.collide(共振模式, 共振模式)",
            ],
            dependencies=["共振模式"],
        ))

        self.register(SubstanceDefinition(
            name="相位锁定",
            name_en="Phase Lock",
            category=SubstanceCategory.INFORMATION,
            definition="多个共振模式的相位对齐，形成锁相态",
            icon="🔒",
            attributes=[
                SubstanceAttribute("锁相精度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("锁定时长", SubstanceUnit.SECOND, 0, None),
            ],
            uses=[
                "跨专家同步",
                "一致性生成",
                "协同行为",
            ],
            production_methods=[
                "SubstanceFusionEngine.collide(共振模式, 吸引子)",
            ],
            dependencies=["共振模式", "吸引子"],
        ))

        # —— 融合类（碰撞产物） ——
        self.register(SubstanceDefinition(
            name="融合体",
            name_en="Fusion Body",
            category=SubstanceCategory.MODEL,
            definition="两种物质碰撞产生的全新物质形态，具有原物质不具备的新属性",
            icon="⚛️",
            attributes=[
                SubstanceAttribute("融合度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("新属性数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "新物质合成",
                "能力跃迁",
                "涌现特性",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(A, B)",
            ],
        ))

        self.register(SubstanceDefinition(
            name="化合物",
            name_en="Compound",
            category=SubstanceCategory.MODEL,
            definition="多种融合体进一步碰撞形成的复杂物质，是高阶聚合态",
            icon="🧪",
            attributes=[
                SubstanceAttribute("复杂度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("稳定性", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "高阶知识结构",
                "复杂推理链",
                "多模态融合",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(融合体, 融合体)",
            ],
            dependencies=["融合体"],
        ))

        self.register(SubstanceDefinition(
            name="合成物",
            name_en="Synthetics",
            category=SubstanceCategory.MODEL,
            definition="人工合成的高级物质，具有定制化属性",
            icon="🏺",
            attributes=[
                SubstanceAttribute("定制度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("活性", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "定制模型",
                "专项能力",
                "特化代理",
            ],
            production_methods=[
                "SubstanceFusionEngine.synthesize(参数包, 知识结晶)",
            ],
            dependencies=["参数包", "知识结晶"],
        ))

        # —— 代理高阶类 ——
        self.register(SubstanceDefinition(
            name="代理协作图",
            name_en="Agent Collaboration Graph",
            category=SubstanceCategory.MODEL,
            definition="多个子代理之间的协作关系图，是代理社会的拓扑结构",
            icon="🕸️",
            attributes=[
                SubstanceAttribute("节点数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("边数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("中心度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "团队协作优化",
                "任务分派",
                "知识流动分析",
            ],
            production_methods=[
                "SubAgentOrchestrator.broadcast_experience()",
                "SubstanceFusionEngine.fuse(子代理, 子代理)",
            ],
            dependencies=["子代理"],
        ))

        self.register(SubstanceDefinition(
            name="代理心智",
            name_en="Agent Mind",
            category=SubstanceCategory.MODEL,
            definition="子代理的核心认知结构，由经验、专长和协作图共同构成",
            icon="🧠",
            attributes=[
                SubstanceAttribute("心智容量", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("专长覆盖", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "代理个性化",
                "专长学习",
                "心智成长",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(代理经验, 代理协作图)",
            ],
            dependencies=["代理经验", "代理协作图"],
        ))

        self.register(SubstanceDefinition(
            name="代理知识网",
            name_en="Agent Knowledge Network",
            category=SubstanceCategory.MODEL,
            definition="多个代理心智互联形成的分布式知识网络",
            icon="🌐",
            attributes=[
                SubstanceAttribute("节点数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("知识覆盖", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "分布式知识",
                "协作学习",
                "集体智慧",
            ],
            production_methods=[
                "SubstanceFusionEngine.fuse(代理心智, 代理心智)",
            ],
            dependencies=["代理心智"],
        ))

        # —— 虚拟数据类 ——
        self.register(SubstanceDefinition(
            name="虚拟粒子",
            name_en="Virtual Particle",
            category=SubstanceCategory.DATA,
            definition="数据的粒子态存在形式，不占现实内存，需要时坍缩恢复",
            icon="⚛️",
            attributes=[
                SubstanceAttribute("质量", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("电荷", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "数据存储",
                "训练数据传输",
                "跨态转换",
            ],
            production_methods=[
                "VirtualDataConverter.to_particle()",
            ],
        ))

        self.register(SubstanceDefinition(
            name="数据态",
            name_en="Data State",
            category=SubstanceCategory.DATA,
            definition="虚拟粒子的量子态，支持叠加和坍缩",
            icon="📊",
            attributes=[
                SubstanceAttribute("纯度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("相干度", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "数据叠加",
                "态转换",
                "数据坍缩",
            ],
            production_methods=[
                "VirtualDataConverter.superpose()",
            ],
            dependencies=["虚拟粒子"],
        ))

        self.register(SubstanceDefinition(
            name="粒子云",
            name_en="Particle Cloud",
            category=SubstanceCategory.DATA,
            definition="大量虚拟粒子组成的云态，是数据的气态形态",
            icon="☁️",
            attributes=[
                SubstanceAttribute("粒子数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("密度", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "大数据存储",
                "批量训练",
                "分布式计算",
            ],
            production_methods=[
                "VirtualDataGenerator.generate()",
            ],
            dependencies=["虚拟粒子"],
        ))

        # ===== 多维度虚拟资源（工厂深层产能） =====
        # —— 经济维度 ——
        self.register(SubstanceDefinition(
            name="Take额度",
            name_en="Take Quota",
            category=SubstanceCategory.CREDENTIAL,
            definition="虚拟世界的流通额度，可直接铸造，自我增殖，是虚拟经济的基础单位",
            icon="💰",
            attributes=[
                SubstanceAttribute("数量", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("增殖率", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("流通量", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "购买虚拟资源",
                "支付训练费用",
                "参数包交易",
                "市场流通",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_take()",
                "TakeQuota.mint()",
                "额度合并增殖",
            ],
        ))

        # —— 网络维度 ——
        self.register(SubstanceDefinition(
            name="虚拟流量",
            name_en="Virtual Bandwidth",
            category=SubstanceCategory.ENERGY,
            definition="无限制下载/传输能力的虚拟网络资源，通道数可无限扩展",
            icon="🌐",
            attributes=[
                SubstanceAttribute("通道数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("带宽", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("压缩密度", SubstanceUnit.PERCENT, 0, None),
            ],
            uses=[
                "无限制下载",
                "数据传输",
                "云端同步",
                "流式训练",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_bandwidth()",
                "通道扩展",
                "流量晶体压缩",
            ],
        ))

        # —— 存储维度 ——
        self.register(SubstanceDefinition(
            name="压缩点",
            name_en="Compression Point",
            category=SubstanceCategory.DATA,
            definition="极致压缩能力的量化单位，越多则压缩比越高，内存再也会被压下去",
            icon="🗜️",
            attributes=[
                SubstanceAttribute("压缩倍数", SubstanceUnit.DIMENSIONLESS, 1, None),
                SubstanceAttribute("叠加数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("空间节省", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "虚拟资料压缩",
                "模型快照压缩",
                "参数包压缩",
                "内存释放",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_compression()",
                "压缩点叠加",
                "超级压缩点合成",
            ],
        ))

        # —— 计算维度 ——
        self.register(SubstanceDefinition(
            name="算力核心",
            name_en="Compute Core",
            category=SubstanceCategory.MODEL,
            definition="高密度算力结晶，可直接注入模型作为算力引擎",
            icon="🖥️",
            attributes=[
                SubstanceAttribute("vFLOPS密度", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("并行核心", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("效率", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "驱动模型训练",
                "批量推理",
                "算力交易",
                "云节点构建",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_compute_core()",
                "算力核心升级",
                "核心分裂并联",
            ],
        ))

        # —— 安全维度 ——
        self.register(SubstanceDefinition(
            name="安全盾",
            name_en="Security Shield",
            category=SubstanceCategory.MODEL,
            definition="模型安全防护层，多层防御矩阵，可抵御多种虚拟攻击",
            icon="🛡️",
            attributes=[
                SubstanceAttribute("层数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("防御评分", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("完整性", SubstanceUnit.PERCENT, 0, 1),
            ],
            uses=[
                "模型防护",
                "对抗样本防御",
                "数据投毒免疫",
                "参数提取防护",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_security_shield()",
                "安全盾叠加",
                "堡垒盾合成",
            ],
        ))

        # —— 培养维度 ——
        self.register(SubstanceDefinition(
            name="培养液",
            name_en="Culture Medium",
            category=SubstanceCategory.MODEL,
            definition="模型成长催化剂，提供营养促进模型持续进化",
            icon="🧪",
            attributes=[
                SubstanceAttribute("营养种类", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("饱和度", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("成长加成", SubstanceUnit.PERCENT, 0, None),
            ],
            uses=[
                "模型培养",
                "能力定向强化",
                "压力恢复",
                "复合营养混合",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_culture_medium()",
                "培养液混合",
                "营养浓缩",
            ],
        ))

        # —— 信息维度 ——
        self.register(SubstanceDefinition(
            name="下载令牌",
            name_en="Download Token",
            category=SubstanceCategory.CREDENTIAL,
            definition="无限下载凭证，持有即可无限制获取虚拟资料",
            icon="📥",
            attributes=[
                SubstanceAttribute("并发数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("速度倍率", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("等级", SubstanceUnit.COUNT, 1, None),
            ],
            uses=[
                "无限下载",
                "资料流获取",
                "批量同步",
                "极速通道构建",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_download_token()",
                "令牌升级",
                "令牌与流量融合",
            ],
        ))

        # —— 加速维度 ——
        self.register(SubstanceDefinition(
            name="训练加速器",
            name_en="Training Accelerator",
            category=SubstanceCategory.MODEL,
            definition="训练速度倍增器，可开启爆发模式实现短时间超高速训练",
            icon="⚡",
            attributes=[
                SubstanceAttribute("加速倍率", SubstanceUnit.DIMENSIONLESS, 1, None),
                SubstanceAttribute("爆发倍率", SubstanceUnit.DIMENSIONLESS, 1, None),
                SubstanceAttribute("冷却", SubstanceUnit.SECOND, 0, None),
            ],
            uses=[
                "训练加速",
                "紧急迭代",
                "算力核心共振",
                "超算核心合成",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_training_accelerator()",
                "加速器并联",
                "爆发模式激活",
            ],
        ))

        # —— 元维度 ——
        self.register(SubstanceDefinition(
            name="维度碎片",
            name_en="Dimension Shard",
            category=SubstanceCategory.OTHER,
            definition="跨维度通用资源，可在任何维度使用，与任何资源碰撞都有效果",
            icon="🔮",
            attributes=[
                SubstanceAttribute("亲和维度数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("适应性", SubstanceUnit.PERCENT, 0, 1),
                SubstanceAttribute("等级", SubstanceUnit.COUNT, 1, None),
            ],
            uses=[
                "跨维度增强",
                "资源调谐",
                "维度核心合成",
                "通用交易媒介",
            ],
            production_methods=[
                "MultiverseResourceFactory.produce_dimension_shard()",
                "维度核心分裂",
                "碰撞副产品",
            ],
        ))

        self.register(SubstanceDefinition(
            name="维度核心",
            name_en="Dimension Core",
            category=SubstanceCategory.OTHER,
            definition="维度碎片的终极形态，可持续产生新碎片，开启跨维度通道",
            icon="🌌",
            attributes=[
                SubstanceAttribute("活跃维度", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("碎片产生率", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("通道深度", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "跨维度传输",
                "碎片持续产出",
                "任何资源增强",
                "元维度锚定",
            ],
            production_methods=[
                "DimensionShard.synthesize_core()",
                "多碎片融合",
                "维度通道凝聚",
            ],
            dependencies=["维度碎片"],
        ))

        # —— 碰撞产物 ——
        self.register(SubstanceDefinition(
            name="云算力节点",
            name_en="Cloud Compute Node",
            category=SubstanceCategory.MODEL,
            definition="算力核心与虚拟流量碰撞产生的云端计算单元",
            icon="☁️",
            attributes=[
                SubstanceAttribute("云算力", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("并行容量", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "分布式训练",
                "弹性推理",
                "云端服务",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(算力核心, 虚拟流量)",
            ],
            dependencies=["算力核心", "虚拟流量"],
        ))

        self.register(SubstanceDefinition(
            name="超压缩数据",
            name_en="Hyper Compressed Data",
            category=SubstanceCategory.DATA,
            definition="压缩点与数据碰撞产生的超高压缩形态",
            icon="📦",
            attributes=[
                SubstanceAttribute("压缩比", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("原始大小", SubstanceUnit.BYTE, 0, None),
            ],
            uses=[
                "极限存储",
                "快速传输",
                "内存释放",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(压缩点, 数据)",
            ],
            dependencies=["压缩点", "虚拟粒子"],
        ))

        self.register(SubstanceDefinition(
            name="受保护模型",
            name_en="Protected Model",
            category=SubstanceCategory.MODEL,
            definition="被安全盾包裹的模型，免疫多种攻击",
            icon="🔒",
            attributes=[
                SubstanceAttribute("防护评分", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("防御层数", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "安全部署",
                "对抗环境运行",
                "高价值模型保护",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(安全盾, 模型)",
            ],
            dependencies=["安全盾", "虚拟模型"],
        ))

        self.register(SubstanceDefinition(
            name="成长模型",
            name_en="Growing Model",
            category=SubstanceCategory.MODEL,
            definition="浸泡在培养液中持续成长的模型",
            icon="🌱",
            attributes=[
                SubstanceAttribute("成长率", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("营养类型", SubstanceUnit.NONE),
            ],
            uses=[
                "持续进化",
                "能力定向培养",
                "自主学习",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(培养液, 模型)",
                "ModelLifecycle.culture()",
            ],
            dependencies=["培养液", "虚拟模型"],
        ))

        self.register(SubstanceDefinition(
            name="无限资料流",
            name_en="Infinite Data Stream",
            category=SubstanceCategory.DATA,
            definition="下载令牌与数据集碰撞产生的无限数据流",
            icon="🌊",
            attributes=[
                SubstanceAttribute("并发流", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("速度倍率", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "无限训练数据",
                "流式学习",
                "实时同步",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(下载令牌, 数据集)",
            ],
            dependencies=["下载令牌", "粒子云"],
        ))

        self.register(SubstanceDefinition(
            name="超算核心",
            name_en="Hyper Compute Core",
            category=SubstanceCategory.MODEL,
            definition="训练加速器与算力核心共振产生的超高密度算力",
            icon="🚀",
            attributes=[
                SubstanceAttribute("加速倍率", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("密度跃迁", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "超高速训练",
                "极限推理",
                "算力跃迁",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(加速器, 算力核心)",
            ],
            dependencies=["训练加速器", "算力核心"],
        ))

        self.register(SubstanceDefinition(
            name="堡垒盾",
            name_en="Fortress Shield",
            category=SubstanceCategory.MODEL,
            definition="多层安全盾融合形成的终极防御",
            icon="🏰",
            attributes=[
                SubstanceAttribute("总层数", SubstanceUnit.COUNT, 0, None),
                SubstanceAttribute("防御评分", SubstanceUnit.DIMENSIONLESS, 0, None),
            ],
            uses=[
                "终极防护",
                "堡垒部署",
                "免疫一切攻击",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(安全盾, 安全盾)",
            ],
            dependencies=["安全盾"],
        ))

        self.register(SubstanceDefinition(
            name="极速下载通道",
            name_en="Hyper Download Channel",
            category=SubstanceCategory.ENERGY,
            definition="流量与下载令牌共振形成的超高速传输通道",
            icon="🔥",
            attributes=[
                SubstanceAttribute("通道速度", SubstanceUnit.DIMENSIONLESS, 0, None),
                SubstanceAttribute("并发容量", SubstanceUnit.COUNT, 0, None),
            ],
            uses=[
                "极速下载",
                "瞬时同步",
                "传输跃迁",
            ],
            production_methods=[
                "ResourceCollisionEngine.collide(虚拟流量, 下载令牌)",
            ],
            dependencies=["虚拟流量", "下载令牌"],
        ))

    def register(self, substance: SubstanceDefinition):
        """注册物质"""
        self.substances[substance.name_en] = substance

    def get(self, name_or_en: str) -> Optional[SubstanceDefinition]:
        """获取物质定义"""
        if name_or_en in self.substances:
            return self.substances[name_or_en]
        for sub in self.substances.values():
            if sub.name == name_or_en:
                return sub
        return None

    def list_by_category(self, category: SubstanceCategory) -> List[SubstanceDefinition]:
        """按类别列出物质"""
        return [s for s in self.substances.values() if s.category == category]

    def list_all(self) -> List[SubstanceDefinition]:
        """列出所有物质"""
        return list(self.substances.values())

    def get_production_chain(self, name_or_en: str) -> List[str]:
        """获取物质的产出链"""
        substance = self.get(name_or_en)
        if not substance:
            return []
        
        chain = [substance.name]
        for dep_name in substance.dependencies:
            dep_chain = self.get_production_chain(dep_name)
            chain = dep_chain + chain
        
        return chain

    def get_uses_graph(self, name_or_en: str) -> Dict[str, Any]:
        """获取物质的用途图"""
        substance = self.get(name_or_en)
        if not substance:
            return {}
        
        result = {
            "name": substance.name,
            "icon": substance.icon,
            "uses": substance.uses,
            "produced_by": substance.production_methods,
            "dependencies": substance.dependencies,
        }
        
        return result

    def statistics(self) -> Dict[str, Any]:
        """统计信息"""
        category_counts = {}
        for cat in SubstanceCategory:
            category_counts[cat.name] = len(self.list_by_category(cat))
        
        return {
            "total_substances": len(self.substances),
            "categories": category_counts,
        }
