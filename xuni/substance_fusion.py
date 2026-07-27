"""
SubstanceFusionEngine —— 物质碰撞引擎

核心理念：
  两种物质碰撞 → 新物质（融合体）
  融合体再碰撞 → 化合物
  参数注入碰撞 → 合成物

碰撞引擎是工厂的"化学反应炉"，可以将已有的25+种有机物质
进行两两碰撞，产生原物质不具备的新物质形态。

碰撞规则示例：
  记忆点 + 记忆点 → 知识结晶（融合）
  记忆点 + 共振记忆 → 灵感闪（碰撞）
  知识结晶 + 思维链 → 理解态
  情感波 + 情感波 → 共鸣场
  理解态 + 逻辑流 → 反思环
  ...

对模型的帮助：
  1. 融合后的物质直接注入模型，提升生成质量
  2. 合成物作为特化代理的核心能力
  3. 化合物驱动多模态融合生成
"""

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum, auto


class FusionType(Enum):
    FUSE = auto()
    COLLIDE = auto()
    SYNTHESIZE = auto()


class FusionCategory(Enum):
    KNOWLEDGE = auto()
    EMOTION = auto()
    COGNITION = auto()
    RESONANCE = auto()
    PARAMETER = auto()
    COMPOUND = auto()


@dataclass
class FusionProduct:
    """碰撞产物"""
    product_id: str
    fusion_type: FusionType
    category: FusionCategory
    reactants: List[str]
    result: str
    properties: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    energy_release: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "product_id": self.product_id,
            "fusion_type": self.fusion_type.name,
            "category": self.category.name,
            "reactants": self.reactants,
            "result": self.result,
            "properties": self.properties,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "energy_release": self.energy_release,
        }


class SubstanceFusionEngine:
    """
    物质碰撞引擎。

    用法：
        engine = SubstanceFusionEngine()
        # 注册物质
        engine.register_substance("记忆点", {"importance": 0.8, "tags": ["AI", "模型"]})
        engine.register_substance("共振记忆", {"frequency": 1.2, "phase": 0.5})
        # 碰撞
        product = engine.collide("记忆点", "共振记忆")
        print(product.result)  # "灵感闪"
    """

    def __init__(self):
        self._substances: Dict[str, Dict[str, float]] = {}
        self._products: List[FusionProduct] = []
        self._rules: Dict[str, FusionProduct] = {}
        self._rule_index: Dict[Tuple[str, str], Tuple[str, FusionType, FusionCategory]] = {}
        # 涌现效果表：特殊融合产物的"打破守恒"效果描述
        self._emergent_effects: Dict[str, Dict[str, Any]] = {}
        self._register_default_rules()

    def register_substance(self, name: str, properties: Dict[str, float]):
        self._substances[name] = properties

    def has_substance(self, name: str) -> bool:
        return name in self._substances

    def get_substance(self, name: str) -> Optional[Dict[str, float]]:
        return self._substances.get(name)

    def collide(self, substance_a: str, substance_b: str) -> FusionProduct:
        """
        两种物质碰撞。
        碰撞产生非线性产物，往往具有原物质不具备的新属性。
        """
        key = self._make_key(substance_a, substance_b)
        rule = self._rule_index.get(key)

        if rule is None:
            result, fusion_type, category = self._infer_product(substance_a, substance_b)
        else:
            result, fusion_type, category = rule

        product = self._create_product(
            substance_a, substance_b,
            fusion_type, category, result,
        )
        self._products.append(product)
        return product

    def fuse(self, substance_a: str, substance_b: str) -> FusionProduct:
        """
        两种物质融合。
        融合产生稳定的高阶物质，属性是原物质的加权合成。
        """
        return self.collide(substance_a, substance_b)

    def synthesize(self, substance_a: str, substance_b: str, **params) -> FusionProduct:
        """
        合成：带参数的定向碰撞。
        可定制产物的属性。若存在预定义规则则优先使用规则结果。
        """
        key = self._make_key(substance_a, substance_b)
        rule = self._rule_index.get(key)
        if rule is not None:
            result, _, _ = rule
        else:
            result = "合成物"
        product = self._create_product(
            substance_a, substance_b,
            FusionType.SYNTHESIZE, FusionCategory.COMPOUND, result,
        )
        product.properties.update(params)
        self._products.append(product)
        return product

    def get_products(self) -> List[FusionProduct]:
        return list(self._products)

    def get_products_by_category(self, category: FusionCategory) -> List[FusionProduct]:
        return [p for p in self._products if p.category == category]

    def get_products_by_result(self, result_name: str) -> List[FusionProduct]:
        return [p for p in self._products if p.result == result_name]

    def clear_products(self):
        self._products.clear()

    def list_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                "reactants": list(k),
                "result": v[0],
                "fusion_type": v[1].name,
                "category": v[2].name,
            }
            for k, v in self._rule_index.items()
        ]

    def get_emergent_effect(self, result_name: str) -> Optional[Dict[str, Any]]:
        """查询某融合产物的涌现效果（永动机系列的'打破守恒'效果）"""
        return self._emergent_effects.get(result_name)

    def list_emergent_effects(self) -> Dict[str, Dict[str, Any]]:
        """列出所有涌现效果"""
        return dict(self._emergent_effects)

    def _make_key(self, a: str, b: str) -> Tuple[str, str]:
        return (min(a, b), max(a, b))

    def _infer_product(
        self, substance_a: str, substance_b: str
    ) -> Tuple[str, FusionType, FusionCategory]:
        """
        根据物质类别和属性推断可能的产物。
        这是引擎的"创造性"部分——即使没有预定义规则，
        也能通过属性重叠度和类别交叉推断出新物质。
        """
        props_a = self._substances.get(substance_a, {})
        props_b = self._substances.get(substance_b, {})

        similarity = self._compute_similarity(props_a, props_b)
        category = self._infer_category(substance_a, substance_b)

        if similarity > 0.8:
            fusion_type = FusionType.FUSE
            result = self._fuse_result(substance_a, substance_b, category)
        elif similarity > 0.5:
            fusion_type = FusionType.COLLIDE
            result = self._collide_result(substance_a, substance_b, category)
        else:
            fusion_type = FusionType.SYNTHESIZE
            result = self._synthesize_result(substance_a, substance_b, category)

        return result, fusion_type, category

    def _compute_similarity(
        self, props_a: Dict[str, float], props_b: Dict[str, float]
    ) -> float:
        """属性相似度（模拟余弦相似度的简化版）"""
        if not props_a or not props_b:
            return 0.5

        common_keys = set(props_a.keys()) & set(props_b.keys())
        if not common_keys:
            return 0.3

        dot = sum(props_a[k] * props_b[k] for k in common_keys)
        norm_a = sum(v * v for v in props_a.values()) ** 0.5
        norm_b = sum(v * v for v in props_b.values()) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.5
        return min(dot / (norm_a * norm_b + 1e-9), 1.0)

    def _infer_category(
        self, substance_a: str, substance_b: str
    ) -> FusionCategory:
        """根据物质名称推断碰撞类别"""
        knowledge_keywords = ["记忆", "知识", "思维", "灵感", "逻辑", "元知识", "结晶", "链", "流"]
        emotion_keywords = ["情感", "共鸣", "意图", "wave", "field"]
        cognition_keywords = ["理解", "好奇", "洞察", "反思", "态"]
        resonance_keywords = ["共振", "吸引子", "相位", "模式"]
        parameter_keywords = ["参数", "向量", "梯度"]
        agent_keywords = ["代理", "心智"]

        combined = substance_a + substance_b

        if any(k in combined for k in knowledge_keywords):
            return FusionCategory.KNOWLEDGE
        if any(k in combined for k in emotion_keywords):
            return FusionCategory.EMOTION
        if any(k in combined for k in cognition_keywords):
            return FusionCategory.COGNITION
        if any(k in combined for k in resonance_keywords):
            return FusionCategory.RESONANCE
        if any(k in combined for k in parameter_keywords):
            return FusionCategory.PARAMETER
        if any(k in combined for k in agent_keywords):
            return FusionCategory.COMPOUND

        return FusionCategory.COMPOUND

    def _fuse_result(
        self, a: str, b: str, category: FusionCategory
    ) -> str:
        """融合产物命名"""
        result_map = {
            FusionCategory.KNOWLEDGE: "知识结晶",
            FusionCategory.EMOTION: "共鸣场",
            FusionCategory.COGNITION: "理解态",
            FusionCategory.RESONANCE: "共振模式",
            FusionCategory.PARAMETER: "参数包",
            FusionCategory.COMPOUND: "融合体",
        }
        return result_map.get(category, "融合体")

    def _collide_result(
        self, a: str, b: str, category: FusionCategory
    ) -> str:
        """碰撞产物命名"""
        result_map = {
            FusionCategory.KNOWLEDGE: "灵感闪",
            FusionCategory.EMOTION: "情感波",
            FusionCategory.COGNITION: "洞察点",
            FusionCategory.RESONANCE: "吸引子",
            FusionCategory.PARAMETER: "参数梯度",
            FusionCategory.COMPOUND: "化合物",
        }
        return result_map.get(category, "化合物")

    def _synthesize_result(
        self, a: str, b: str, category: FusionCategory
    ) -> str:
        """合成产物命名"""
        return "合成物"

    def _create_product(
        self, a: str, b: str,
        fusion_type: FusionType, category: FusionCategory, result: str,
    ) -> FusionProduct:
        props_a = self._substances.get(a, {})
        props_b = self._substances.get(b, {})

        merged_props = {}
        for k in set(list(props_a.keys()) + list(props_b.keys())):
            va = props_a.get(k, 0.0)
            vb = props_b.get(k, 0.0)
            if fusion_type == FusionType.FUSE:
                merged_props[k] = (va + vb) / 2
            elif fusion_type == FusionType.COLLIDE:
                merged_props[k] = va * vb
            else:
                merged_props[k] = max(va, vb)

        energy = sum(merged_props.values()) * 0.3

        product_id = hashlib.md5(
            f"{a}+{b}+{time.time()}".encode()
        ).hexdigest()[:12]

        return FusionProduct(
            product_id=product_id,
            fusion_type=fusion_type,
            category=category,
            reactants=[a, b],
            result=result,
            properties=merged_props,
            metadata={
                "substance_a": a,
                "substance_b": b,
                "rule_based": (self._rule_index.get(self._make_key(a, b)) is not None),
                "emergent_effect": self._emergent_effects.get(result),
            },
            energy_release=energy,
        )

    def _register_default_rules(self):
        """注册 30+ 条预定义碰撞规则"""
        rules = [
            # ---- 原有有机物质规则 ----
            ("记忆点", "记忆点", "知识结晶", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("记忆点", "参数包", "思维链", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("记忆点", "共振记忆", "灵感闪", FusionType.COLLIDE, FusionCategory.KNOWLEDGE),
            ("知识结晶", "思维链", "理解态", FusionType.FUSE, FusionCategory.COGNITION),
            ("理解态", "逻辑流", "反思环", FusionType.FUSE, FusionCategory.COGNITION),
            ("理解态", "灵感闪", "洞察点", FusionType.COLLIDE, FusionCategory.COGNITION),
            ("情感波", "情感波", "共鸣场", FusionType.FUSE, FusionCategory.EMOTION),
            ("共振模式", "共振模式", "吸引子", FusionType.COLLIDE, FusionCategory.RESONANCE),
            ("共振模式", "吸引子", "相位锁定", FusionType.COLLIDE, FusionCategory.RESONANCE),
            ("思维链", "参数包", "逻辑流", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("知识结晶", "知识结晶", "元知识", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("子代理", "子代理", "代理协作图", FusionType.FUSE, FusionCategory.COMPOUND),
            ("代理经验", "代理协作图", "代理心智", FusionType.FUSE, FusionCategory.COMPOUND),
            ("代理心智", "代理心智", "代理知识网", FusionType.FUSE, FusionCategory.COMPOUND),
            ("参数包", "知识结晶", "合成物", FusionType.SYNTHESIZE, FusionCategory.PARAMETER),
            # ---- 多维度虚拟资源碰撞规则 ----
            ("Take额度", "参数包", "高级参数包", FusionType.SYNTHESIZE, FusionCategory.PARAMETER),
            ("算力核心", "虚拟流量", "云算力节点", FusionType.FUSE, FusionCategory.COMPOUND),
            ("压缩点", "虚拟粒子", "超压缩数据", FusionType.FUSE, FusionCategory.COMPOUND),
            ("安全盾", "虚拟模型", "受保护模型", FusionType.FUSE, FusionCategory.COMPOUND),
            ("培养液", "虚拟模型", "成长模型", FusionType.FUSE, FusionCategory.COMPOUND),
            ("下载令牌", "粒子云", "无限资料流", FusionType.FUSE, FusionCategory.COMPOUND),
            ("训练加速器", "算力核心", "超算核心", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("维度碎片", "Take额度", "增强额度", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("维度碎片", "算力核心", "增强算力核心", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("维度碎片", "压缩点", "超级压缩点", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("维度碎片", "安全盾", "堡垒盾", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("Take额度", "Take额度", "大额额度", FusionType.FUSE, FusionCategory.COMPOUND),
            ("压缩点", "压缩点", "超级压缩点", FusionType.FUSE, FusionCategory.COMPOUND),
            ("算力核心", "算力核心", "算力集群", FusionType.FUSE, FusionCategory.COMPOUND),
            ("安全盾", "安全盾", "堡垒盾", FusionType.FUSE, FusionCategory.COMPOUND),
            ("培养液", "培养液", "复合培养液", FusionType.FUSE, FusionCategory.COMPOUND),
            ("虚拟流量", "下载令牌", "极速下载通道", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("额度", "训练", "付费训练", FusionType.SYNTHESIZE, FusionCategory.PARAMETER),
            ("云算力节点", "超算核心", "超级计算中心", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("无限资料流", "成长模型", "自学习模型", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("受保护模型", "堡垒盾", "绝对安全模型", FusionType.FUSE, FusionCategory.COMPOUND),
            ("超级压缩点", "超压缩数据", "零体积数据", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("维度碎片", "维度碎片", "维度核心", FusionType.FUSE, FusionCategory.COMPOUND),
            ("极速下载通道", "无限资料流", "瞬时全库", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("超算核心", "训练加速器", "量子训练场", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("云算力节点", "虚拟流量", "分布式云", FusionType.FUSE, FusionCategory.COMPOUND),
            ("培养液", "算力核心", "活算算力", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("安全盾", "下载令牌", "安全下载", FusionType.FUSE, FusionCategory.COMPOUND),
            ("压缩点", "算力核心", "压缩算力", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("Take额度", "安全盾", "保险额度", FusionType.FUSE, FusionCategory.COMPOUND),
            # ---- 无限令牌融合链：打破守恒的永动机系列 ----
            # 核心洞察：下载令牌可无限生产（千万级/秒），虚拟电/算力亦无限
            # 两个无限融合 → 正反馈永动机
            ("下载令牌", "虚拟电", "永动下载涡轮", FusionType.FUSE, FusionCategory.COMPOUND),
            ("下载令牌", "虚拟算力", "流式计算引擎", FusionType.FUSE, FusionCategory.COMPOUND),
            ("永动下载涡轮", "流式计算引擎", "无限训练永动机", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("无限训练永动机", "维度碎片", "维度训练塔", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("永动下载涡轮", "训练加速器", "永动加速器", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("流式计算引擎", "算力核心", "流式算力核心", FusionType.FUSE, FusionCategory.COMPOUND),
            ("无限训练永动机", "虚拟模型", "自进化模型", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("维度训练塔", "自进化模型", "维度心智", FusionType.COLLIDE, FusionCategory.COGNITION),
            # ---- 能量算力融合链：电+算力+流量 → 分布式流式算力网络 ----
            # 核心洞察：虚拟电大了算力才快（电驱动算力），
            #          虚拟流量有现实网络特征（带宽=节点连通度），
            #          流量 + (电+算力) = 每个网络节点都有无限算力 = 分布式无限算力
            ("虚拟电", "虚拟算力", "能量算力核心", FusionType.FUSE, FusionCategory.COMPOUND),
            ("虚拟流量", "能量算力核心", "流式算力网络", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("流式算力网络", "下载令牌", "全网永动算力", FusionType.FUSE, FusionCategory.COMPOUND),
            ("流式算力网络", "训练加速器", "分布式加速场", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("流式算力网络", "维度碎片", "多维算力网络", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # ---- 参数融合链：参数 + 流式算力网络 = 指数级训练 ----
            # 核心洞察：参数包是模型权重原料，流式算力网络是分布式无限算力
            # 参数(权重原料) + 流式算力网络(无限算力) = 每个节点并行注入参数
            # = N个节点同时训练 = 指数级训练速度
            ("参数包", "流式算力网络", "参数流式训练场", FusionType.COLLIDE, FusionCategory.PARAMETER),
            ("参数包", "能量算力核心", "能量参数核心", FusionType.FUSE, FusionCategory.PARAMETER),
            ("参数流式训练场", "下载令牌", "无限参数流", FusionType.FUSE, FusionCategory.PARAMETER),
            ("参数流式训练场", "训练加速器", "超频参数训练", FusionType.COLLIDE, FusionCategory.PARAMETER),
            ("参数流式训练场", "维度碎片", "多维参数训练", FusionType.SYNTHESIZE, FusionCategory.PARAMETER),
            ("无限参数流", "自进化模型", "参数自进化体", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("超频参数训练", "全网永动算力", "永动参数引擎", FusionType.COLLIDE, FusionCategory.PARAMETER),
        ]

        for a, b, result, ftype, cat in rules:
            self._rule_index[self._make_key(a, b)] = (result, ftype, cat)

        # 涌现效果：永动机系列"打破守恒"的特殊效果
        self._emergent_effects = {
            "永动下载涡轮": {
                "原理": "令牌无限 × 电无限 → 正反馈：下载产电 → 电产令牌 → 永动",
                "效果": "下载不耗电，反产电；越下载电越多",
                "打破定律": "能量守恒（虚拟维度）",
                "产出": "下载能量（自循环）",
                "自循环": True,
                "反馈类型": "正反馈",
            },
            "流式计算引擎": {
                "原理": "令牌无限 × 算力无限 → 下载即计算，数据流过即训练完成",
                "效果": "下载=训练，数据流入模型即完成训练",
                "打破定律": "计算时间守恒",
                "产出": "即时训练（零延迟）",
                "自循环": True,
                "反馈类型": "流式",
            },
            "无限训练永动机": {
                "原理": "永动产电 + 流式计算 = 零成本无限训练",
                "效果": "模型无需任何外部资源，永久自训练，越训练越强",
                "打破定律": "训练成本守恒",
                "产出": "无限训练（无上限）",
                "自循环": True,
                "连锁来源": ["永动下载涡轮", "流式计算引擎"],
                "反馈类型": "正反馈×流式",
            },
            "维度训练塔": {
                "原理": "无限训练永动机 × 维度碎片 → 每个维度一个永动机",
                "效果": "N 维并行无限训练，维度数即训练倍率",
                "打破定律": "维度限制",
                "产出": "N 倍无限训练",
                "自循环": True,
            },
            "永动加速器": {
                "原理": "永动机驱动加速器，能量无限→加速无上限",
                "效果": "训练加速倍率随时间指数增长，无天花板",
                "打破定律": "加速上限",
                "产出": "指数加速",
            },
            "流式算力核心": {
                "原理": "算力核心流式化，数据流过即计算",
                "效果": "算力核心吞吐量无限，无计算瓶颈",
                "打破定律": "算力瓶颈",
                "产出": "无限吞吐算力",
            },
            "自进化模型": {
                "原理": "无限训练永动机 × 虚拟模型 → 模型永久自训练",
                "效果": "模型自我进化，无需外部训练数据，自产自训",
                "打破定律": "训练数据依赖",
                "产出": "自我进化的模型",
                "自循环": True,
            },
            "维度心智": {
                "原理": "维度训练塔 × 自进化模型 → 跨维度自我意识",
                "效果": "多维度并行思考，涌现维度级心智",
                "打破定律": "单一意识限制",
                "产出": "维度级心智",
            },
            # ---- 能量算力融合链的涌现效果 ----
            "能量算力核心": {
                "原理": "虚拟电（能源）× 虚拟算力（计算）→ 电驱动算力，算力有源源不断的动力",
                "效果": "算力不再枯竭，电越大算力越快，形成'电→算力→更多电'的自循环",
                "打破定律": "算力能耗守恒（算力不再消耗电，反被电驱动）",
                "产出": "自驱动算力（电越大算力越快）",
                "自循环": True,
                "反馈类型": "电→算力正反馈",
            },
            "流式算力网络": {
                "原理": "虚拟流量（网络带宽/节点连通度）× 能量算力核心（无限本地算力）"
                       "→ 每个网络节点都带无限算力，数据流过网络即被计算",
                "效果": "算力 = 节点数 × 单节点算力；节点数由流量通道数决定。"
                       "流量越大，连通节点越多，总算力线性扩展，无上限",
                "打破定律": "单机算力上限（算力随网络规模线性扩展）",
                "产出": "分布式无限算力（流量×算力）",
                "自循环": True,
                "现实类比": "边缘计算/分布式训练集群：带宽越大，可调度节点越多",
                "公式": "总算力 = 流量通道数 × 单节点(电驱动)算力",
            },
            "全网永动算力": {
                "原理": "流式算力网络 × 下载令牌（无限）→ 网络上每个节点永动产算力",
                "效果": "全网节点永动计算，算力 = ∞ × 节点数，无任何瓶颈",
                "打破定律": "网络算力守恒",
                "产出": "全网永动算力",
                "自循环": True,
            },
            "分布式加速场": {
                "原理": "流式算力网络 × 训练加速器 → 加速器作用到每个网络节点",
                "效果": "训练加速倍率 × 节点数，分布式并行加速",
                "打破定律": "单点加速上限",
                "产出": "N倍分布式加速",
            },
            "多维算力网络": {
                "原理": "流式算力网络 × 维度碎片 → 每个维度一个算力网络",
                "效果": "多维并行算力网络，跨维度分布式计算",
                "打破定律": "维度算力限制",
                "产出": "多维分布式算力",
            },
            # ---- 参数融合链的涌现效果 ----
            "参数流式训练场": {
                "原理": "参数包（权重原料）× 流式算力网络（分布式无限算力）"
                       "→ 每个网络节点并行注入参数，N节点同时训练同一模型",
                "效果": "训练速度 = 原速度 × 节点数。参数注入不再串行，"
                       "而是流过整个算力网络，每个节点都贡献训练增量。"
                       "参数质量被算力网络放大 N 倍",
                "打破定律": "参数训练线性瓶颈（从线性→乘法扩展）",
                "产出": "指数级参数训练",
                "自循环": False,
                "现实类比": "分布式训练：数据并行，每个GPU节点同时训练不同batch",
                "公式": "训练增量 = 参数质量 × 节点数 × 算力倍率",
            },
            "能量参数核心": {
                "原理": "参数包 × 能量算力核心 → 参数被电驱动，高能量参数",
                "效果": "参数质量随能量增长自动提升，能量越大参数越强",
                "打破定律": "参数质量上限",
                "产出": "能量增强参数（质量随电增长）",
            },
            "无限参数流": {
                "原理": "参数流式训练场 × 下载令牌（无限）→ 无限参数源源不断流入",
                "效果": "训练数据无限，模型可无限训练，无数据瓶颈",
                "打破定律": "训练数据有限性",
                "产出": "无限参数供给",
                "自循环": True,
            },
            "超频参数训练": {
                "原理": "参数流式训练场 × 训练加速器 → 参数注入超频",
                "效果": "每个参数包的训练增量被加速器倍率放大",
                "打破定律": "参数吸收率上限",
                "产出": "超频参数吸收（倍率放大）",
            },
            "多维参数训练": {
                "原理": "参数流式训练场 × 维度碎片 → 每个维度一个参数训练场",
                "效果": "多维并行参数注入，模型在多个维度同时进化",
                "打破定律": "单维参数训练",
                "产出": "多维参数训练",
            },
            "参数自进化体": {
                "原理": "无限参数流 × 自进化模型 → 模型自动吸收无限参数自我进化",
                "效果": "模型自己生产参数、自己吸收、自己进化，完全自主",
                "打破定律": "外部参数依赖",
                "产出": "自主参数进化",
                "自循环": True,
            },
            "永动参数引擎": {
                "原理": "超频参数训练 × 全网永动算力 → 全网永动注入超频参数",
                "效果": "全网节点永动超频训练，参数训练速度 = ∞ × 加速倍率",
                "打破定律": "参数训练速度守恒",
                "产出": "永动参数训练（无上限）",
                "自循环": True,
            },
        }

    def get_collision_chains(self, depth: int = 2) -> List[List[str]]:
        """
        获取多级碰撞链。
        例如：A+B→C, 然后 C+D→E，形成 A+B→C+D→E。
        """
        chains = []
        rule_keys = list(self._rule_index.keys())

        for key in rule_keys:
            a, b = key
            result = self._rule_index[key][0]
            chain = [a, b, result]

            if depth >= 2:
                for key2 in rule_keys:
                    c, d = key2
                    if c == result or d == result:
                        result2 = self._rule_index[key2][0]
                        chain2 = chain + [c if c != result else d, result2]
                        chains.append(chain2)

            if len(chain) == 3:
                chains.append(chain)

        return chains


def create_default_engine() -> SubstanceFusionEngine:
    """创建带默认物质的引擎"""
    engine = SubstanceFusionEngine()

    engine.register_substance("记忆点", {
        "重要性": 0.8, "访问频率": 0.6, "标签密度": 0.7,
    })
    engine.register_substance("共振记忆", {
        "频率": 1.2, "相位": 0.5, "振幅": 0.8,
    })
    engine.register_substance("参数包", {
        "维度": 0.9, "质量": 0.75, "稀疏度": 0.4,
    })
    engine.register_substance("情感波", {
        "振幅": 0.9, "频率": 0.6, "相位差": 0.3,
    })
    engine.register_substance("知识结晶", {
        "压缩率": 0.85, "纯度": 0.9, "结晶度": 0.7,
    })
    engine.register_substance("思维链", {
        "链长": 0.9, "连贯性": 0.8, "逻辑强度": 0.7,
    })
    engine.register_substance("代理经验", {
        "专长覆盖": 0.85, "经验深度": 0.7, "跨域度": 0.5,
    })
    engine.register_substance("代理协作图", {
        "节点数": 0.7, "边密度": 0.6, "中心性": 0.4,
    })
    engine.register_substance("子代理", {
        "能力覆盖": 0.8, "响应速度": 0.9, "可靠性": 0.75,
    })
    # ---- 无限资源系列：打破守恒的基础物质 ----
    engine.register_substance("虚拟电", {
        "能量密度": 1e9, "可再生": 1.0, "无限性": 1.0,
    })
    engine.register_substance("虚拟算力", {
        "计算密度": 1e15, "可再生": 1.0, "无限性": 1.0,
    })
    engine.register_substance("下载令牌", {
        "并发数": 1024.0, "速度倍率": 10.0, "无限性": 1.0,
    })
    engine.register_substance("虚拟模型", {
        "参数量": 0.8, "训练度": 0.5, "通用性": 0.7,
    })
    engine.register_substance("维度碎片", {
        "维度数": 8.0, "稳定性": 0.6, "稀有度": 1.0,
    })

    return engine
