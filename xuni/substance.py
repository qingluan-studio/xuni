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
