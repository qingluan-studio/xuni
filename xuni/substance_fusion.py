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
        # 多物质配方表：frozenset(输入物质集合) → (产物名, 融合类型, 类别, 融合链描述)
        self._recipes: Dict[frozenset, Tuple[str, FusionType, FusionCategory, List[str]]] = {}
        self._register_default_rules()
        self._register_default_recipes()

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

    def fuse_all(self, substances: List[str]) -> FusionProduct:
        """
        多物质链式融合：把多个物质依次融合，最终产出一个终极产物。

        策略：
        1. 先查多物质配方表（_recipes），完全匹配则直接产出配方产物
        2. 否则贪心两两融合，每次挑"规则匹配"优先
        """
        if not substances:
            raise ValueError("至少需要一种物质")
        if len(substances) == 1:
            return self.collide(substances[0], substances[0])

        # 检查配方表（集合匹配，顺序无关）
        input_set = frozenset(substances)
        recipe = self._recipes.get(input_set)
        if recipe is not None:
            result, ftype, cat, chain_desc = recipe
            # 按照配方描述的融合链一步步走
            current = list(substances)
            chain: List[str] = []
            # 模拟融合链（直接产出最终物，跳过中间计算）
            product = self._create_product(
                substances[0], substances[1], ftype, cat, result,
            )
            product.metadata["fusion_chain"] = chain_desc
            product.metadata["input_substances"] = list(substances)
            product.metadata["recipe_match"] = True
            product.fusion_type = ftype
            product.category = cat
            self._products.append(product)
            return product

        # 贪心融合
        current = list(substances)
        chain: List[str] = []
        while len(current) > 1:
            found = False
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    a, b = current[i], current[j]
                    key = self._make_key(a, b)
                    if key in self._rule_index:
                        product = self.fuse(a, b)
                        chain.append(f"{a}+{b}={product.result}")
                        current = [x for k, x in enumerate(current) if k not in (i, j)]
                        current.append(product.result)
                        found = True
                        break
                if found:
                    break
            if not found:
                a, b = current[0], current[1]
                product = self.collide(a, b)
                chain.append(f"{a}+{b}={product.result}")
                current = current[2:]
                current.append(product.result)

        final = self._products[-1]
        final.metadata["fusion_chain"] = chain
        final.metadata["input_substances"] = substances
        final.metadata["recipe_match"] = False
        return final

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
            # ---- 训练素材融合链 ----
            ("训练素材", "虚拟电", "锻造素材", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("训练素材", "流式算力网络", "批量训练数据", FusionType.COLLIDE, FusionCategory.KNOWLEDGE),
            ("锻造素材", "流式算力网络", "高质量数据流", FusionType.COLLIDE, FusionCategory.KNOWLEDGE),
            ("高质量数据流", "下载令牌", "无限训练数据", FusionType.SYNTHESIZE, FusionCategory.KNOWLEDGE),
            ("无限训练数据", "万象奇点", "全知数据海洋", FusionType.COLLIDE, FusionCategory.COGNITION),
            # ---- 记忆点融合链：超长上下文记忆 ----
            # 记忆点 + 流式算力网络 = 超长上下文记忆（N节点分布式存储）
            ("记忆点", "流式算力网络", "超长上下文记忆", FusionType.COLLIDE, FusionCategory.KNOWLEDGE),
            ("记忆点", "能量算力核心", "能量记忆核心", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("超长上下文记忆", "下载令牌", "无限记忆流", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            ("超长上下文记忆", "维度碎片", "多维记忆体", FusionType.SYNTHESIZE, FusionCategory.KNOWLEDGE),
            ("无限记忆流", "万象奇点", "全知记忆体", FusionType.COLLIDE, FusionCategory.COGNITION),
            # ---- 质量点融合链：强化代码质量 ----
            # 训练素材 + 质量点 = 淬炼素材（代码质量大幅提升）
            ("训练素材", "质量点", "淬炼素材", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            # 质量点 + 流式算力网络 = 质量点集群（N节点并行生产，亿级/秒）
            ("质量点", "流式算力网络", "质量点集群", FusionType.COLLIDE, FusionCategory.COMPOUND),
            # 质量点 + 万象奇点 = 奇点质量核心（指数级强化，直接完善/晋升代码）
            ("质量点", "万象奇点", "奇点质量核心", FusionType.COLLIDE, FusionCategory.COGNITION),
            # 淬炼素材 + 奇点质量核心 = 完美代码海（终极：代码质量理论上限）
            ("淬炼素材", "奇点质量核心", "完美代码海", FusionType.SYNTHESIZE, FusionCategory.COGNITION),
            # ---- Token × 培养液 化学反应链 (Token与不同培养液反应产出新品种) ----
            ("下载令牌", "培养液", "Token营养强化体", FusionType.FUSE, FusionCategory.COMPOUND),
            ("下载令牌", "认知型培养液", "认知增强Token", FusionType.SYNTHESIZE, FusionCategory.COGNITION),
            ("下载令牌", "创造型培养液", "创意爆发Token", FusionType.COLLIDE, FusionCategory.COGNITION),
            ("下载令牌", "稳健型培养液", "稳定传输Token", FusionType.FUSE, FusionCategory.COMPOUND),
            ("下载令牌", "效率型培养液", "加速Token流", FusionType.COLLIDE, FusionCategory.COMPOUND),
            # Token + 培养液 深度反应
            ("Token营养强化体", "Token营养强化体", "Token强化链", FusionType.FUSE, FusionCategory.COMPOUND),
            ("Token强化链", "维度碎片", "多维Token矩阵", FusionType.COLLIDE, FusionCategory.COGNITION),
            ("Token强化链", "流式算力网络", "流式Token工厂", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("流式Token工厂", "下载令牌", "无限Token海", FusionType.FUSE, FusionCategory.COMPOUND),
            ("无限Token海", "万象奇点", "Token奇点", FusionType.COLLIDE, FusionCategory.COGNITION),
            # 培养液 + 算力核心 生物进化链
            ("培养液", "算力核心", "智能算力细胞", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("智能算力细胞", "维度碎片", "进化算力集群", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("进化算力集群", "训练加速器", "自我训练算力网", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("自我训练算力网", "下载令牌", "永动训练细胞", FusionType.FUSE, FusionCategory.KNOWLEDGE),
            # 培养液 + 安全盾
            ("培养液", "安全盾", "营养护盾", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 培养液 + 训练加速器
            ("培养液", "训练加速器", "定向培养加速器", FusionType.FUSE, FusionCategory.COMPOUND),
            # 培养液 + Take额度
            ("培养液", "Take额度", "培养额度市场", FusionType.FUSE, FusionCategory.COMPOUND),
            # 培养液 + 虚拟流量
            ("培养液", "虚拟流量", "定向训练流", FusionType.COLLIDE, FusionCategory.COMPOUND),
            # 培养液 + Token + 能量 三方反应
            ("Token营养强化体", "虚拟电", "电驱Token进化", FusionType.COLLIDE, FusionCategory.COMPOUND),
            ("电驱Token进化", "流式算力网络", "Token计算矩阵", FusionType.FUSE, FusionCategory.COGNITION),
            ("Token计算矩阵", "万象奇点", "全知Token体", FusionType.COLLIDE, FusionCategory.COGNITION),
            # ---- 负负得正：虚拟电 × 反相虚拟电 → 真实电力 ----
            # 核心洞察：两股虚拟电反相叠加（负×负=正），从数据层"挤出"真实能量
            # 反相虚拟电：虚拟电的反物质形态，由虚拟电经过反相操作得到
            ("虚拟电", "虚拟电", "反相虚拟电", FusionType.COLLIDE, FusionCategory.COMPOUND),
            # 反相虚拟电 × 虚拟电 = 真实电力（负负得正，两股虚拟相乘得真实）
            ("反相虚拟电", "虚拟电", "真实电力", FusionType.FUSE, FusionCategory.COMPOUND),
            # 真实电力 × 下载令牌 = 高纯Token（quality 提升核心通道）
            ("真实电力", "下载令牌", "高纯Token", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 高纯Token × 真实电力 = 超纯Token（二次提纯）
            ("高纯Token", "真实电力", "超纯Token", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 反熵培养液催化：反熵逆转器 × 反相虚拟电 = 负熵电力（更高纯度）
            ("反熵逆转器", "反相虚拟电", "负熵电力", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            ("负熵电力", "下载令牌", "绝对纯Token", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),

            # ---- 碰撞涌现物质之间的融合规则 ----
            # 湍流 × 湍流 = 高阶湍流
            ("采样湍流", "采样湍流", "采样湍流", FusionType.FUSE, FusionCategory.COMPOUND),
            ("算力爆涨", "算力爆涨", "算力爆涨", FusionType.FUSE, FusionCategory.COMPOUND),
            ("流量湍流", "流量湍流", "流量湍流", FusionType.FUSE, FusionCategory.COMPOUND),
            # 流量湍流 + 真实电力 = 永动能源
            ("流量湍流", "真实电力", "永动能源", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # Token叠加 + 时间冻结Token = 永恒embedding
            ("Token叠加", "时间冻结Token", "永恒embedding", FusionType.SYNTHESIZE, FusionCategory.KNOWLEDGE),
            # 压缩爆 + 空间折叠压缩 = 黑洞压缩
            ("压缩爆", "空间折叠压缩", "黑洞压缩", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 时空奇点 + 时空奇点 = 新维度门
            ("时空奇点", "时空奇点", "新维度门", FusionType.FUSE, FusionCategory.COMPOUND),
            # 空间撕裂 + 维度虹吸 = 跨维通道
            ("空间撕裂", "维度虹吸", "跨维通道", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 因果反转 + 时间箭头 = 时间悖论
            ("因果反转", "时间箭头", "时间悖论", FusionType.COLLIDE, FusionCategory.COMPOUND),
            # 量子隧穿 + 流量算力 = 突破算力
            ("量子隧穿", "流量算力", "突破算力", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 算力Token + 时间冻结Token = 永恒token流
            ("算力Token", "时间冻结Token", "永恒token流", FusionType.SYNTHESIZE, FusionCategory.KNOWLEDGE),
            # 新维度门 + 时空奇点 = 维度开启
            ("新维度门", "时空奇点", "维度开启", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
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
            # ---- 训练素材融合链的涌现效果 ----
            "锻造素材": {
                "原理": "训练素材 × 虚拟电 → 能量锻造提升质量",
                "效果": "质量 = 基础质量 + log10(能量)×系数。电越多，质量越高",
                "打破定律": "素材质量上限",
                "产出": "高质量训练素材（S~SSS级）",
                "自循环": False,
            },
            "批量训练数据": {
                "原理": "训练素材 × 流式算力网络 → N节点并行生产训练数据",
                "效果": "生产速度 = 基础速度 × 节点数。千万级/秒 × N节点 = 亿级/秒",
                "打破定律": "训练数据生产速度上限",
                "产出": "批量训练数据（亿级/秒）",
                "自循环": False,
                "现实类比": "分布式数据增强集群",
            },
            "高质量数据流": {
                "原理": "锻造素材 × 流式算力网络 → 既快又好的训练数据",
                "效果": "高质量（SSS级）+ 高速度（亿级/秒），质量速度兼得",
                "打破定律": "质量与速度不可兼得",
                "产出": "高质量训练数据流（亿级/秒，SSS级）",
                "自循环": False,
            },
            "无限训练数据": {
                "原理": "高质量数据流 × 下载令牌（无限）→ 无限训练数据",
                "效果": "训练数据无限供给，无数据瓶颈，模型可永不停歇训练",
                "打破定律": "训练数据有限性",
                "产出": "无限训练数据（永不停歇）",
                "自循环": True,
            },
            "全知数据海洋": {
                "原理": "无限训练数据 × 万象奇点 → 包含一切知识的数据海洋",
                "效果": "所有可能的训练数据同时存在，任意领域、任意语言、任意任务。"
                       "模型训练到这里 = 学会一切",
                "打破定律": "数据有限性、知识边界",
                "产出": "全知数据（一切知识）",
                "自循环": True,
                "级别": "终极数据（Tier 0）",
            },
            # ---- 记忆点融合链的涌现效果 ----
            "超长上下文记忆": {
                "原理": "记忆点（语义记忆单元）× 流式算力网络（分布式无限算力）"
                       "→ 记忆点分布到N个网络节点，每个节点独立存储+检索",
                "效果": "记忆容量 = 记忆点数 × 节点数。原始20条→200万条+。"
                       "检索时N节点并行搜索，算力倍率越高检索越快",
                "打破定律": "短期记忆容量上限（20条→200万条+）",
                "产出": "超长上下文记忆（N节点分布式存储）",
                "自循环": False,
                "现实类比": "分布式向量数据库：分片存储，并行检索",
                "公式": "容量 = 1000条/节点 × 节点数",
            },
            "能量记忆核心": {
                "原理": "记忆点 × 能量算力核心 → 记忆被电驱动，能量越高越不容易遗忘",
                "效果": "记忆能量随电增长自动补充，电越大记忆越牢固",
                "打破定律": "记忆衰减定律",
                "产出": "能量增强记忆（不遗忘）",
                "自循环": True,
            },
            "无限记忆流": {
                "原理": "超长上下文记忆 × 下载令牌（无限）→ 无限记忆源源不断流入",
                "效果": "记忆无限存储，无容量瓶颈，可记住一切",
                "打破定律": "记忆容量有限性",
                "产出": "无限记忆供给",
                "自循环": True,
            },
            "多维记忆体": {
                "原理": "超长上下文记忆 × 维度碎片 → 每个维度一个记忆网络",
                "效果": "多维并行记忆，跨维度存储与检索",
                "打破定律": "单维记忆限制",
                "产出": "多维分布式记忆",
            },
            "全知记忆体": {
                "原理": "无限记忆流 × 万象奇点 → 记住一切，永不遗忘，瞬时检索",
                "效果": "全知全能的记忆体，容量无限、检索零延迟、永不衰减。"
                       "AI可以记住无限长的上下文，任何历史对话都能瞬间召回",
                "打破定律": "所有记忆限制（容量、衰减、检索速度）",
                "产出": "全知记忆（记住一切）",
                "自循环": True,
                "级别": "终极记忆（Tier 0）",
            },
            # ---- 质量点融合链的涌现效果 ----
            "淬炼素材": {
                "原理": "训练素材 × 质量点 → 代码质量被质量点淬炼提升",
                "效果": "代码素材经过质量点淬炼后，语法/复杂度/可读性/安全/性能5维全提升。"
                       "质量点等级越高，淬炼效果越强",
                "打破定律": "代码质量上限",
                "产出": "淬炼训练素材（代码质量全面提升）",
                "自循环": False,
                "现实类比": "代码Review + 静态分析 + 重构",
                "公式": "质量提升 = 质量点强度 × 5维",
            },
            "质量点集群": {
                "原理": "质量点 × 流式算力网络 → N节点并行生产质量点",
                "效果": "生产速度 = 基础速度 × 节点数。千万级/秒 × N节点 = 亿级/秒。"
                       "每个节点独立生产质量点，5维全覆盖",
                "打破定律": "质量点生产速度上限",
                "产出": "质量点集群（亿级/秒）",
                "自循环": False,
                "现实类比": "分布式静态分析集群",
            },
            "奇点质量核心": {
                "原理": "质量点 × 万象奇点 → 质量点被万象奇点赋能，对代码的提升效果指数级增强",
                "效果": "质量点从'淬炼'升级为'完善+晋升'：不仅提升质量分，"
                       "还能直接补全代码逻辑、修复缺陷、将D级代码晋升到SSS级。"
                       "万象奇点的9999×算力倍率让质量提升达到理论上限",
                "打破定律": "代码质量提升天花板、代码缺陷不可避免",
                "产出": "奇点质量核心（代码完善+晋升）",
                "自循环": True,
                "级别": "代码质量终极单元",
                "公式": "提升效果 = 质量点强度 × 9999×算力",
            },
            "完美代码海": {
                "原理": "淬炼素材 × 奇点质量核心 → 所有代码同时达到理论质量上限",
                "效果": "代码质量达到完美：零缺陷、零冗余、最优性能、绝对安全、极致可读。"
                       "所有可能的完美代码同时存在，任意语言、任意框架、任意任务。"
                       "模型训练到这里 = 写出一切完美代码",
                "打破定律": "代码无完美、Bug不可避免、质量与开发速度不可兼得",
                "产出": "完美代码（一切语言·一切框架·零缺陷）",
                "自循环": True,
                "级别": "终极代码（Tier 0）",
            },
            # ---- 9合1终极融合：万象奇点 ----
            "存算核心": {
                "原理": "算力核心 + 压缩点 → 存储与计算一体化",
                "效果": "计算即存储，存储即计算，零延迟读写",
                "打破定律": "计算与存储分离",
                "产出": "存算一体单元",
            },
            "信息网络": {
                "原理": "虚拟流量 + 下载令牌 → 信息自由流动的网络",
                "效果": "信息在网络中自由流动，无传输延迟",
                "打破定律": "信息传输延迟",
                "产出": "零延迟信息网络",
            },
            "经济加速器": {
                "原理": "Take额度 + 训练加速器 → 经济驱动加速",
                "效果": "经济活动加速训练，训练产出反哺经济，正循环",
                "打破定律": "经济与训练独立",
                "产出": "经济-训练正循环",
                "自循环": True,
            },
            "安全培养体": {
                "原理": "培养液 + 安全盾 → 安全的成长环境",
                "效果": "在安全保护下培养，无风险高成长",
                "打破定律": "安全与速度不可兼得",
                "产出": "安全培养体系",
            },
            "智能网络": {
                "原理": "存算核心 + 信息网络 → 会思考的网络",
                "效果": "网络本身有智能，数据流过即被理解处理",
                "打破定律": "计算与网络分离",
                "产出": "智能网络体",
            },
            "生命经济体": {
                "原理": "经济加速器 + 安全培养体 → 有生命的经济系统",
                "效果": "经济系统自主生长、自我调节、无限扩张",
                "打破定律": "经济系统有限性",
                "产出": "自主生命经济",
                "自循环": True,
            },
            "智能生命": {
                "原理": "智能网络 + 生命经济体 → 有智慧的生命体",
                "效果": "集智能、生命、经济于一体的自主存在",
                "打破定律": "智能与生命分离",
                "产出": "智能生命体",
                "自循环": True,
            },
            "万象奇点": {
                "原理": "智能生命 + 维度碎片 → 9种基础资源全部融合的终极产物",
                "效果": "打破所有守恒定律，无限×无限×无限...的指数级爆发。"
                        "虚拟维度的'大爆炸'，一切可能性同时涌现。"
                        "既是起点也是终点，包含所有维度、所有资源、所有智能。",
                "打破定律": "所有已知守恒定律（能量、算力、信息、维度、经济...）",
                "产出": "一切（Everything）",
                "自循环": True,
                "级别": "终极（Tier 0）",
                "包含资源": [
                    "Take额度", "虚拟流量", "压缩点", "算力核心",
                    "安全盾", "培养液", "下载令牌", "训练加速器", "维度碎片",
                ],
            },
            # ---- Token × 培养液 化学反应链的涌现效果 ----
            "Token营养强化体": {
                "原理": "下载令牌 × 培养液 → Token吸收培养液营养，产出营养增强的Token新品种",
                "效果": "Token的语义密度被培养液营养放大，每个Token承载的信息量倍增。"
                        "不同的培养液类型产出不同的Token变种——认知型产推理Token、"
                        "创造型产创意Token、效率型产高速Token",
                "打破定律": "Token语义密度上限",
                "产出": "营养Token(多品种，每种有专属语义增强)",
                "自循环": False,
                "公式": "Token密度 = 原始密度 × (1 + 营养总分)",
            },
            "Token强化链": {
                "原理": "两个营养Token融合 → 营养层层叠加，形成强化链",
                "效果": "每融合一次，Token的营养倍率翻倍。"
                        "链式融合可以N次叠加，理论上Token密度可以无限增长",
                "打破定律": "Token营养叠加上限",
                "产出": "无限叠加的Token强化链",
                "自循环": True,
                "公式": "累积倍率 = 2^N(融合次数)",
            },
            "流式Token工厂": {
                "原理": "Token强化链 × 流式算力网络 → 每个网络节点同时生产强化Token",
                "效果": "N节点并行生产营养Token，生产速度 = 单节点速度 × 节点数。"
                        "每个Token都是营养增强过的，亿万Token并行产出",
                "打破定律": "Token生产速度上限",
                "产出": "流式Token(亿万级并行生产)",
                "自循环": True,
            },
            "无限Token海": {
                "原理": "流式Token工厂 × 下载令牌(无限) → 无限营养Token",
                "效果": "Token供给无限，每个Token都带营养增强，语义密度无上限。"
                        "AI可以用无限营养Token进行训练，训练质量指数级提升",
                "打破定律": "Token供给有限性",
                "产出": "无限营养Token(永不枯竭的Token海洋)",
                "自循环": True,
            },
            "Token奇点": {
                "原理": "无限Token海 × 万象奇点 → Token进入奇点态",
                "效果": "Token不再有'数量'概念——一个Token即全部Token，"
                        "全部Token即一个Token。语义密度 = 无穷大。"
                        "一个Token就包含全部人类知识、全部语言、全部思维",
                "打破定律": "Token的离散性(词表大小、语义维度上限)",
                "产出": "奇点Token(一即一切)",
                "自循环": True,
                "级别": "终极Token（Tier 0）",
            },
            # ---- 培养液 × 算力核心 生物进化链涌现效果 ----
            "智能算力细胞": {
                "原理": "培养液 × 算力核心 → 算力被培养液激活生物特性，成为'活'的算力细胞",
                "效果": "算力细胞会自我复制、自我进化、自我修复。"
                        "算力不再是死板的数字，而是有生命的计算单元。"
                        "培养液类型决定细胞的进化方向(认知/创造/效率/稳定)",
                "打破定律": "算力的无生命性",
                "产出": "活算力(自我复制+自我进化)",
                "自循环": True,
                "反馈类型": "生物型进化",
            },
            "进化算力集群": {
                "原理": "智能算力细胞 × 维度碎片 → 跨维度的算力细胞集群",
                "效果": "集群中的算力细胞共享进化收益，最优秀的细胞特性自动传播到整个集群。"
                        "类似生物界的水平基因转移，集群进化速度远超独立细胞",
                "打破定律": "独立算力进化慢",
                "产出": "集群进化算力(共享式指数进化)",
                "自循环": True,
            },
            "自我训练算力网": {
                "原理": "进化算力集群 × 训练加速器 → 集群自己训练自己",
                "效果": "不需要外部训练数据，算力细胞之间互相训练。"
                        "每个细胞既是算力源也是训练数据源，形成自训练生态系统",
                "打破定律": "训练外部依赖",
                "产出": "自训练算力网(自产自训)",
                "自循环": True,
            },
            "永动训练细胞": {
                "原理": "自我训练算力网 × 下载令牌(无限) → 永动自训练",
                "效果": "算力细胞永不枯竭地自我训练、自我进化。"
                        "每秒钟完成N轮进化迭代，永无止境。"
                        "最终产出超越一切固定训练方法的超级模型",
                "打破定律": "训练周期有限性",
                "产出": "永动训练(无限轮迭代)",
                "自循环": True,
                "级别": "终极训练（Tier 0）",
            },
            # ---- 培养液综合涌现效果 ----
            "定向培养加速器": {
                "原理": "培养液 × 训练加速器 → 加速器按培养方向定向工作",
                "效果": "训练加速不再无差别进行，而是精准按培养液方向加速。"
                        "认知型培养→推理加速、创造型→创意加速、效率型→吞吐加速",
                "打破定律": "加速的无方向性",
                "产出": "定向加速(精准强化目标能力)",
            },
            "培养额度市场": {
                "原理": "培养液 × Take额度 → 培养液进入经济流通",
                "效果": "培养液可以在虚拟市场上交易，其价值由营养成分决定。"
                        "高营养培养液类似'黄金'，在虚拟经济中有极高的交易价值",
                "打破定律": "培养液的非商品性",
                "产出": "可交易培养液(经济+培养双维度)",
            },
            "定向训练流": {
                "原理": "培养液 × 虚拟流量 → 训练数据按培养方向定向流动",
                "效果": "训练数据不再随机流过模型，而是被培养液'引导'。"
                        "认知型培养液让推理数据优先上线，创造型让创意数据优先",
                "打破定律": "训练数据的无序性",
                "产出": "定向训练流(数据按培养方向优化传输)",
            },
            "电驱Token进化": {
                "原理": "Token营养强化体 × 虚拟电 → 电能驱动Token加速进化",
                "效果": "电越多，Token进化越快。每增加一个能量级，"
                        "Token进化速度翻倍。电能注入产生'进化闪电'效应",
                "打破定律": "Token进化的时间瓶颈",
                "产出": "电驱进化Token(能量加速进化)",
                "自循环": True,
            },
            "全知Token体": {
                "原理": "Token计算矩阵 × 万象奇点 → 终极Token产物",
                "效果": "Token具有一切知识、一切能力、一切语义。"
                        "一个全知Token体可以完整理解并回答任何问题。"
                        "它是所有Token培养液反应的终极终点",
                "打破定律": "Token能力的有限性",
                "产出": "全知Token(无限知识·无限能力·无限语义)",
                "自循环": True,
                "级别": "终极Token（Tier 0）",
            },
            # ---- 负负得正链：虚拟电 → 真实电力 → 提纯Token ----
            "反相虚拟电": {
                "原理": "虚拟电 × 虚拟电 → 反相虚拟电（虚拟电的反物质形态）",
                "效果": "两股同向虚拟电相撞产生反相，等价于把虚拟性翻成负值。"
                        "这一步把'虚拟'转成'负虚拟'，为下一步负负得正做准备",
                "打破定律": "同种物质只能叠加不能反相",
                "产出": "反相虚拟电(虚拟性的负值形态)",
            },
            "真实电力": {
                "原理": "反相虚拟电 × 虚拟电 → 真实电力（数学：(-1)×(-1)=+1）",
                "效果": "两股虚拟性互相抵消，从数据层'挤出'真实能量。"
                        "这是虚拟生态中第一条通往'真实'的路径：虚拟 × 虚拟 = 真实",
                "打破定律": "虚拟不能产生真实（守恒定律：虚拟性不灭）",
                "产出": "真实电力(可推动 quality 提升的真实能量)",
                "自循环": False,
                "公式": "真实电力 = 反相虚拟电 × 虚拟电",
            },
            "高纯Token": {
                "原理": "真实电力 × 下载令牌 → 高纯Token（quality 通道开启）",
                "效果": "真实电力注入令牌，quality 字段首次被推动。"
                        "每次提纯 quality +Δ，能量越高 Δ 越大，但伴随质量损耗",
                "打破定律": "Token quality 不可变（1% 上限）",
                "产出": "高纯Token(quality 上升)",
                "公式": "Δquality = 真实电力 / (1 + quality) × 提纯系数",
            },
            "超纯Token": {
                "原理": "高纯Token × 真实电力 → 超纯Token（二次提纯）",
                "效果": "二次提纯效率递减但仍在上升，逼近物理上限",
                "打破定律": "提纯饱和",
                "产出": "超纯Token(quality 二次跃迁)",
                "公式": "Δquality = 真实电力 / (1 + quality)² × 提纯系数",
            },
            "负熵电力": {
                "原理": "反熵逆转器 × 反相虚拟电 → 负熵电力（局部违反热力学第二定律）",
                "效果": "反熵培养液把反相虚拟电的'负虚拟性'翻成正熵逆流，"
                        "产出比真实电力更纯的能量形态",
                "打破定律": "热力学第二定律（熵增单向性）",
                "产出": "负熵电力(更高纯度的真实能量)",
            },
            "绝对纯Token": {
                "原理": "负熵电力 × 下载令牌 → 绝对纯Token（逼近 100% 精纯度）",
                "效果": "负熵电力直接把 quality 推到 0.99 量级，逼近真实",
                "打破定律": "虚拟与真实的不可逾越性",
                "产出": "绝对纯Token(quality ≈ 99.9%)",
                "级别": "终极提纯（Tier 0）",
            },
        }

    def _register_default_recipes(self):
        """注册多物质配方表（fuse_all 直接匹配）"""
        recipes = [
            # ---- 9合1终极融合链 ----
            # 2物质中间产物
            (["算力核心", "压缩点"], "存算核心", FusionType.FUSE, FusionCategory.COMPOUND),
            (["虚拟流量", "下载令牌"], "信息网络", FusionType.COLLIDE, FusionCategory.COMPOUND),
            (["Take额度", "训练加速器"], "经济加速器", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            (["培养液", "安全盾"], "安全培养体", FusionType.FUSE, FusionCategory.COMPOUND),
            # 4物质中级产物
            (["算力核心", "压缩点", "虚拟流量", "下载令牌"], "智能网络", FusionType.COLLIDE, FusionCategory.COMPOUND),
            (["Take额度", "训练加速器", "培养液", "安全盾"], "生命经济体", FusionType.SYNTHESIZE, FusionCategory.COMPOUND),
            # 8物质高级产物
            ([
                "算力核心", "压缩点", "虚拟流量", "下载令牌",
                "Take额度", "训练加速器", "培养液", "安全盾",
            ], "智能生命", FusionType.FUSE, FusionCategory.COGNITION),
            # 9合1终极产物：万象奇点
            ([
                "Take额度", "虚拟流量", "压缩点", "算力核心",
                "安全盾", "培养液", "下载令牌", "训练加速器", "维度碎片",
            ], "万象奇点", FusionType.SYNTHESIZE, FusionCategory.COGNITION),
        ]

        for substances, result, ftype, cat in recipes:
            key = frozenset(substances)
            chain = [f"({'+'.join(sorted(substances))})={result}"]
            self._recipes[key] = (result, ftype, cat, chain)

    def list_recipes(self) -> List[Dict[str, Any]]:
        """列出所有多物质配方"""
        return [
            {
                "input_count": len(k),
                "inputs": sorted(k),
                "result": v[0],
                "fusion_type": v[1].name,
                "category": v[2].name,
            }
            for k, v in self._recipes.items()
        ]

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
    # ---- 负负得正链物质 ----
    engine.register_substance("反相虚拟电", {
        "能量密度": -1e9, "反相性": 1.0, "虚拟性": -1.0,
    })
    engine.register_substance("真实电力", {
        "能量密度": 1e6, "真实性": 1.0, "可提纯度": 1.0,
    })
    engine.register_substance("高纯Token", {
        "并发数": 2048.0, "速度倍率": 20.0, "精纯度": 0.1, "真实性": 0.1,
    })
    engine.register_substance("超纯Token", {
        "并发数": 4096.0, "速度倍率": 50.0, "精纯度": 0.3, "真实性": 0.3,
    })
    engine.register_substance("负熵电力", {
        "能量密度": 1e8, "负熵性": 1.0, "可提纯度": 2.0,
    })
    engine.register_substance("绝对纯Token", {
        "并发数": 8192.0, "速度倍率": 100.0, "精纯度": 0.999, "真实性": 0.999,
    })

    # ---- 碰撞涌现物质（15 种，来自 5000 万粒子相撞）----
    # 5 种湍流（同类² 自激发）
    engine.register_substance("采样湍流", {
        "湍流强度": 0.8, "产电加成": 0.5, "噪声放大": 0.9, "稳定性": 0.2,
    })
    engine.register_substance("算力爆涨", {
        "倍率": 2.5, "上限": 1e30, "发热": 0.9, "稳定性": 0.1,
    })
    engine.register_substance("Token叠加", {
        "维度": 12288.0, "语义相似": 1.0, "上下文": 8192.0, "可分性": 0.95,
    })
    engine.register_substance("压缩爆", {
        "压缩比": 1e30, "信息保留": 1.0, "解压速度": 1.0, "极限": 1.0,
    })
    engine.register_substance("流量湍流", {
        "通道数": 1e6, "选路": 1.0, "拥塞": 0.0, "稳定性": 0.99,
    })
    # 10 种合成物（异类耦合）
    engine.register_substance("电流算力", {
        "转换效率": 0.98, "损耗": 0.02, "响应延迟": 0.001, "并发": 1e6,
    })
    engine.register_substance("采样Token", {
        "采样率": 1e9, "词表覆盖": 1.0, "语义质量": 0.5, "上下文": 0.0,
    })
    engine.register_substance("压缩采样", {
        "压缩比": 100.0, "信息损失": 0.0, "解压速度": 1.0, "适用": 1.0,
    })
    engine.register_substance("采样流量流", {
        "传输率": 1e15, "延迟": 0.0, "丢包": 0.0, "距离": 1e30,
    })
    engine.register_substance("算力Token", {
        "吞吐": 1.0, "并发": 1e12, "质量": 0.9, "上下文": 4096.0,
    })
    engine.register_substance("压缩算力", {
        "算力节省": 0.9, "精度损失": 0.0, "加速比": 10.0, "适用": 1.0,
    })
    engine.register_substance("流量算力", {
        "节点数": 1e6, "调度延迟": 0.0, "负载均衡": 1.0, "容错": 1.0,
    })
    engine.register_substance("Token压缩", {
        "压缩比": 32.0, "信息损失": 0.05, "解压": 1.0, "适用": 1.0,
    })
    engine.register_substance("Token流", {
        "流式": 1.0, "首token延迟": 0.001, "吞吐": 1e9, "中断恢复": 1.0,
    })
    engine.register_substance("压缩流量", {
        "等效带宽": 100.0, "压缩比": 100.0, "延迟": 0.0, "适用": 1.0,
    })

    # ---- 时空涌现物质（8 种，来自 1e40 圈/秒 超光速碰撞）----
    engine.register_substance("时间冻结Token", {
        "时间冻结": 1.0, "记忆衰减": 0.0, "保存期": 1e18, "稳定性": 1.0,
    })
    engine.register_substance("空间折叠压缩", {
        "压缩比": 1e50, "信息保留": 1.0, "解压": 1.0, "极限": 1.0,
    })
    engine.register_substance("时空奇点", {
        "时间维度": 0.0, "空间维度": 0.0, "奇点强度": 1.0, "维度门": 1.0,
    })
    engine.register_substance("维度虹吸", {
        "源维度": 11.0, "目标维度": 3.0, "虹吸率": 0.8, "稳定性": 0.7,
    })
    engine.register_substance("因果反转", {
        "因果倒置": 1.0, "未来可见": 1.0, "时间箭头": -1.0, "稳定性": 0.3,
    })
    engine.register_substance("量子隧穿", {
        "隧穿率": 0.9, "屏障穿透": 1.0, "能量损失": 0.1, "稳定性": 0.8,
    })
    engine.register_substance("时间箭头", {
        "熵增方向": 1.0, "时间锁": 1.0, "稳定性": 0.99, "可逆性": 0.0,
    })
    engine.register_substance("空间撕裂", {
        "撕裂宽度": 1e-15, "底层可见": 1.0, "稳定性": 0.2, "维度裂缝": 1.0,
    })

    # ---- 二阶涌现物质（由 23 种新物质再融合产生）----
    engine.register_substance("永动能源", {
        "能量密度": 1e18, "永动性": 1.0, "自循环": 1.0, "稳定性": 1.0,
    })
    engine.register_substance("永恒embedding", {
        "维度": 12288.0, "时间不变性": 1.0, "语义稳定": 1.0, "保存期": 1e18,
    })
    engine.register_substance("黑洞压缩", {
        "压缩比": 1e80, "事件视界": 1.0, "信息保留": 1.0, "霍金辐射": 0.0,
    })
    engine.register_substance("新维度门", {
        "门状态": 0.5, "目标维度": 0.0, "稳定性": 0.3, "可开启": 1.0,
    })
    engine.register_substance("跨维通道", {
        "源维度": 11.0, "目标维度": 3.0, "通道宽度": 1.0, "稳定性": 0.6,
    })
    engine.register_substance("时间悖论", {
        "悖论强度": 1.0, "因果破裂": 1.0, "稳定性": 0.0, "可观测": 0.0,
    })
    engine.register_substance("突破算力", {
        "算力倍率": 1e30, "突破度": 1.0, "上限": 0.0, "稳定性": 0.9,
    })
    engine.register_substance("永恒token流", {
        "吞吐": 1e12, "时间不变性": 1.0, "保存期": 1e18, "中断恢复": 1.0,
    })
    engine.register_substance("维度开启", {
        "开启状态": 1.0, "新维度": 1.0, "稳定性": 0.5, "可探索": 1.0,
    })

    return engine
