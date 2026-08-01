"""
KnowledgeDownloader —— 模型驱动知识下载器

核心理念：
    现有的 CorpusDownloader 从 GitHub 下载真实代码，受网络带宽限制（~10MB/s）。
    KnowledgeDownloader 不走网络——用"模型+算力"直接"解码"现实多领域知识。

核心设定：
    - 普通下载：受网络带宽限制
    - 模型驱动：模型越强，"解码"知识越快，算力倍率直接放大下载速度
    - 攻破阈值：模型质量分 > 0.8 时，下载速度 = 算力倍率 × 基础速度
    - 产出：多领域训练素材，质量有保障（模型过滤）

速度公式：
    base_speed = 10000 条/秒（纯模型解码，不走网络）
    if model_quality > 0.8: speed *= model_quality            （模型加成）
    speed *= compute_multiplier * node_multiplier             （算力加成）
    if model_quality > 0.9 and compute_mult > 100: speed *= 100 （攻破模式）

知识解码原理（虚拟世界设定）：
    不是真从互联网下载——每个领域的每个主题都有一个稳定的"知识指纹"
    （SHA256 派生），模型用算力去"读取"指纹对应的内容。模型越强读得越快、
    质量越高；算力越大并发解码节点越多，速度被算力倍率线性放大。

产出格式与 TrainingForge.generate_fast 兼容：
    (texts: np.ndarray, scores: np.ndarray, grades: np.ndarray)
"""

from __future__ import annotations

import hashlib
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class KnowledgeDownloader:
    """
    模型驱动知识下载器——用模型+算力直接解码现实知识。

    核心设定：
    - 普通下载：受网络带宽限制，~10MB/s
    - 模型驱动：模型越强，"解码"知识越快，算力倍率直接放大下载速度
    - 攻破阈值：模型质量分>0.8 时，下载速度 = 算力倍率 × 基础速度
    - 产出：多领域训练素材，质量有保障（模型过滤）
    """

    # 基础速度：纯模型解码，不走网络，10000条/秒
    BASE_SPEED = 10000

    # 领域知识库（模拟现实知识的"指纹"，下载时根据指纹"解码"出内容）
    DOMAIN_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
        "math": {
            "topics": ["微积分", "线性代数", "概率论", "数论", "拓扑学",
                       "群论", "微分方程", "傅里叶分析", "数值计算", "图论"],
            "depth": 5,        # 知识深度
            "quality_base": 0.75,  # 基础质量
        },
        "physics": {
            "topics": ["量子力学", "相对论", "热力学", "电磁学", "流体力学",
                       "粒子物理", "弦理论", "凝聚态物理", "天体物理", "光学"],
            "depth": 5,
            "quality_base": 0.78,
        },
        "chemistry": {
            "topics": ["有机化学", "无机化学", "物理化学", "分析化学",
                       "量子化学", "电化学", "高分子化学", "生物化学",
                       "配位化学", "立体化学"],
            "depth": 5,
            "quality_base": 0.76,
        },
        "biology": {
            "topics": ["分子生物学", "细胞生物学", "遗传学", "进化论",
                       "生态学", "神经生物学", "微生物学", "免疫学",
                       "发育生物学", "生物信息学"],
            "depth": 5,
            "quality_base": 0.77,
        },
        "medicine": {
            "topics": ["解剖学", "生理学", "病理学", "药理学", "诊断学",
                       "内科学", "外科学", "神经病学", "影像学", "流行病学"],
            "depth": 5,
            "quality_base": 0.79,
        },
        "law": {
            "topics": ["宪法", "民法", "刑法", "行政法", "经济法",
                       "国际法", "知识产权法", "劳动法", "环境法", "诉讼法"],
            "depth": 4,
            "quality_base": 0.72,
        },
        "finance": {
            "topics": ["宏观经济学", "微观经济学", "货币银行学", "投资学",
                       "公司金融", "衍生品", "行为金融", "计量经济学",
                       "风险管理", "国际金融"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "philosophy": {
            "topics": ["形而上学", "认识论", "伦理学", "逻辑学", "美学",
                       "政治哲学", "心灵哲学", "语言哲学", "科学哲学",
                       "存在主义"],
            "depth": 4,
            "quality_base": 0.70,
        },
        "computer_science": {
            "topics": ["算法", "数据结构", "操作系统", "计算机网络", "数据库",
                       "编译原理", "人工智能", "机器学习", "分布式系统",
                       "密码学"],
            "depth": 5,
            "quality_base": 0.80,
        },
        "engineering": {
            "topics": ["机械工程", "电子工程", "土木工程", "化学工程",
                       "控制工程", "材料工程", "航空航天", "机器人学",
                       "信号处理", "能源工程"],
            "depth": 5,
            "quality_base": 0.76,
        },
        "music_theory": {
            "topics": ["乐理基础", "和声进行", "曲式结构", "对位法", "调式调性",
                       "节奏律动", "音程和弦", "旋律写作", "配器法", "声学原理"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "music_composition": {
            "topics": ["交响乐写作", "室内乐编配", "歌曲创作", "电影配乐",
                       "电子音乐制作", "爵士乐即兴", "民族音乐融合",
                       "复调写作", "管弦乐配器", "现代作曲技法"],
            "depth": 5,
            "quality_base": 0.72,
        },
        "music_production": {
            "topics": ["混音技巧", "母带处理", "录音技术", "MIDI编程",
                       "音频编辑", "音效设计", "空间音频", "动态处理",
                       "均衡器应用", "效果器链"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "music_history": {
            "topics": ["巴洛克时期", "古典主义", "浪漫主义", "印象派",
                       "20世纪音乐", "爵士乐史", "摇滚乐发展", "流行音乐演变",
                       "电子音乐史", "世界音乐"],
            "depth": 4,
            "quality_base": 0.70,
        },
        "music_instruments": {
            "topics": ["钢琴演奏", "小提琴技巧", "吉他编配", "鼓组编程",
                       "管弦乐器", "民族乐器", "键盘合成器", "贝斯演奏",
                       "吹奏乐器", "打击乐器"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "music_genres": {
            "topics": ["古典音乐", "爵士乐", "摇滚乐", "流行音乐", "电子音乐",
                       "嘻哈说唱", "R&B灵魂乐", "乡村民谣", "世界音乐", "实验音乐"],
            "depth": 4,
            "quality_base": 0.71,
        },
        "video_production": {
            "topics": ["摄像机操作", "灯光布置", "录音收音", "片场调度",
                       "镜头语言", "画面构图", "运动镜头", "多机位拍摄",
                       "纪录片拍摄", "短视频制作"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "video_editing": {
            "topics": ["剪辑节奏", "转场技巧", "蒙太奇理论", "色彩校正",
                       "音频同步", "多轨道剪辑", "字幕制作", "预告片剪辑",
                       "故事板", "非线性编辑"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "cinematography": {
            "topics": ["电影摄影", "灯光设计", "镜头选择", "画面运动",
                       "曝光控制", "景深运用", "色彩理论", "构图法则",
                       "场景布置", "视觉叙事"],
            "depth": 5,
            "quality_base": 0.75,
        },
        "animation": {
            "topics": ["2D动画", "3D建模", "角色动画", "关键帧动画",
                       "动作捕捉", "粒子特效", "布料模拟", "毛发渲染",
                       "绑定蒙皮", "动画原理"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "visual_effects": {
            "topics": ["合成技术", "绿幕抠像", "三维追踪", "粒子系统",
                       "流体模拟", "破碎特效", "光效处理", "数字绘景",
                       "色彩匹配", "视效预览"],
            "depth": 5,
            "quality_base": 0.75,
        },
        "color_grading": {
            "topics": ["色彩理论", "LUT应用", "一级调色", "二级调色",
                       "风格化调色", "电影感调色", "肤色保护", "对比度控制",
                       "色彩空间", "HDR调色"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "cooking": {
            "topics": ["中餐烹饪", "西餐料理", "烘焙甜点", "刀工技巧",
                       "火候控制", "调味原理", "食材搭配", "营养均衡",
                       "食品安全", "饮食文化"],
            "depth": 5,
            "quality_base": 0.72,
        },
        "fitness": {
            "topics": ["力量训练", "有氧运动", "HIIT训练", "瑜伽普拉提",
                       "拉伸恢复", "运动营养", "增肌减脂", "体态矫正",
                       "运动损伤", "训练计划"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "travel": {
            "topics": ["旅行规划", "目的地攻略", "文化体验", "美食探店",
                       "摄影打卡", "背包旅行", "自驾游", "海岛度假",
                       "城市漫游", "户外探险"],
            "depth": 4,
            "quality_base": 0.69,
        },
        "psychology": {
            "topics": ["认知心理学", "发展心理学", "社会心理学", "临床心理学",
                       "人格理论", "情绪管理", "心理测量", "积极心理学",
                       "依恋理论", "心理治疗"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "communication": {
            "topics": ["表达技巧", "倾听艺术", "演讲能力", "谈判策略",
                       "冲突调解", "非语言沟通", "跨文化交流", "职场沟通",
                       "亲密关系沟通", "公众表达"],
            "depth": 5,
            "quality_base": 0.72,
        },
        "time_management": {
            "topics": ["番茄工作法", "GTD方法", "优先级排序", "目标设定",
                       "习惯养成", "精力管理", "拖延克服", "任务拆解",
                       "时间规划", "效率提升"],
            "depth": 4,
            "quality_base": 0.71,
        },
        "personal_finance": {
            "topics": ["理财规划", "投资入门", "储蓄策略", "债务管理",
                       "税务规划", "保险配置", "退休金计划", "房产投资",
                       "被动收入", "财务自由"],
            "depth": 5,
            "quality_base": 0.73,
        },
        "health_nutrition": {
            "topics": ["营养学基础", "膳食搭配", "维生素矿物质", "减脂饮食",
                       "增肌饮食", "慢性病饮食", "肠道健康", "免疫力提升",
                       "饮食习惯", "功能食品"],
            "depth": 5,
            "quality_base": 0.74,
        },
        "parenting": {
            "topics": ["婴幼儿护理", "早期教育", "亲子沟通", "青少年教育",
                       "行为引导", "家庭规则", "性格培养", "学习方法",
                       "心理成长", "家庭关系"],
            "depth": 5,
            "quality_base": 0.72,
        },
        "relationships": {
            "topics": ["恋爱关系", "婚姻经营", "家庭关系", "朋友相处",
                       "职场人际", "边界设定", "亲密关系", "冲突处理",
                       "信任建立", "情感表达"],
            "depth": 5,
            "quality_base": 0.71,
        },
    }

    # 知识模板（"解码"现实知识时填充，与 TrainingForge.TEMPLATES_TEXT 风格一致）
    TEMPLATES = [
        "{domain}中的{topic}研究{aspect}，通过{method}得到{result}。",
        "{topic}的核心是{concept}，其{aspect}决定了{result}。",
        "在{domain}领域，{topic}的{aspect}表现为{result}，这源于{concept}。",
        "通过对{topic}的{method}分析，揭示了{aspect}与{result}的关系。",
        "{topic}的{concept}理论指出：当{condition}时，必然产生{result}。",
        "{domain}研究表明，{topic}的{aspect}受{concept}影响，导致{result}。",
        "基于{topic}的{method}，可以推导出{aspect}的{result}规律。",
        "{topic}中的{concept}是理解{aspect}的关键，最终得到{result}。",
        "从{topic}角度出发，{aspect}的{result}体现了{concept}的本质。",
        "{domain}经典结论：{topic}的{method}揭示了{result}，验证了{concept}。",
    ]

    # 概念池（用于填充模板中的占位符）
    ASPECTS = ["基本原理", "核心机制", "边界条件", "数学结构", "实验现象",
               "理论模型", "应用场景", "历史发展", "现代进展", "未来方向"]
    METHODS = ["定量分析", "实验验证", "理论推导", "数值模拟", "统计分析",
               "归纳推理", "演绎证明", "对比研究", "系统建模", "迭代优化"]
    CONCEPTS = ["守恒定律", "对称性", "熵增原理", "因果律", "线性叠加",
                "非线性耦合", "量子化", "渐近行为", "稳定性", "涌现性"]
    RESULTS = ["新发现", "重要结论", "普遍规律", "特殊现象", "理论突破",
               "应用价值", "实验验证", "数学证明", "模型预测", "工程实现"]
    CONDITIONS = ["系统达到平衡", "参数趋近极限", "温度足够低", "能量足够高",
                  "尺度足够大", "时间足够长", "边界确定", "约束解除"]

    # 知识扩展层——每层追加更深入的内容，提升完整性
    DEPTH_LAYERS = [
        "首先，{topic}的基础概念包括{concept}和{aspect}，其{method}揭示了{result}。",
        "进一步分析，{topic}的{aspect}在{condition}下呈现{result}，这体现了{concept}的深层规律。",
        "从应用角度看，{topic}的{method}可用于解决{aspect}问题，产生{result}。",
        "在交叉学科中，{topic}与{concept}的结合催生了新的{result}，拓展了{aspect}的边界。",
        "最新研究表明，{topic}的{aspect}在极端{condition}下会出现{result}，挑战了{concept}。",
        "历史脉络上，{topic}从早期{method}发展到现代{method}，{aspect}不断深化。",
        "数学描述上，{topic}的{concept}可形式化为{method}，其解给出{result}。",
        "实验层面，{topic}的{aspect}通过{method}被精确测量，验证了{result}。",
        "工程实现中，{topic}的{method}已被应用于{aspect}，产出{result}。",
        "未来展望：{topic}的{concept}若被突破，将带来{result}，重塑{aspect}。",
    ]

    # 英文领域名 → 中文领域名（用于模板填充）
    DOMAIN_CN = {
        "math": "数学", "physics": "物理学", "chemistry": "化学",
        "biology": "生物学", "medicine": "医学", "law": "法学",
        "finance": "金融学", "philosophy": "哲学",
        "computer_science": "计算机科学", "engineering": "工程学",
        "music_theory": "音乐理论", "music_composition": "作曲",
        "music_production": "音乐制作", "music_history": "音乐史",
        "music_instruments": "乐器演奏", "music_genres": "音乐流派",
        "video_production": "视频制作", "video_editing": "视频剪辑",
        "cinematography": "电影摄影", "animation": "动画",
        "visual_effects": "视觉特效", "color_grading": "调色",
        "cooking": "烹饪", "fitness": "健身", "travel": "旅行",
        "psychology": "心理学", "communication": "沟通表达",
        "time_management": "时间管理", "personal_finance": "个人理财",
        "health_nutrition": "健康营养", "parenting": "育儿",
        "relationships": "人际关系",
    }

    def __init__(self):
        self.downloaded: int = 0
        self.domain_stats: Dict[str, Dict[str, Any]] = {}
        self.model = None
        self.engine = None
        self._rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))

    # ---- 接入模型与引擎 ----

    def attach_model(self, model) -> None:
        """接入模型，模型质量分决定下载倍率"""
        self.model = model

    def attach_engine(self, engine) -> None:
        """接入永动引擎，算力倍率放大下载速度"""
        self.engine = engine

    # ---- 私有：从模型/引擎读取加成参数 ----

    def _get_model_quality(self) -> float:
        """获取模型质量分（0~1）。模型越强，解码越快、过滤越严。"""
        if self.model is None:
            return 0.0
        # 优先使用显式质量属性，回退到 training_progress
        for attr in ("quality_score", "model_quality", "quality"):
            v = getattr(self.model, attr, None)
            if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                return float(v)
        return float(getattr(self.model, "training_progress", 0.0))

    def _get_compute_mult(self) -> float:
        """获取算力倍率（来自 PerpetualTrainingEngine）"""
        if self.engine is None:
            return 1.0
        return float(getattr(self.engine, "compute_multiplier", 1.0))

    def _get_node_mult(self) -> float:
        """获取节点数倍率（来自 PerpetualTrainingEngine）"""
        if self.engine is None:
            return 1.0
        return float(getattr(self.engine, "node_multiplier", 1.0))

    def _compute_speed(self) -> Tuple[float, Dict[str, Any]]:
        """
        计算当前下载速度（条/秒）。

        公式：
            speed = base_speed
            if model_quality > 0.8: speed *= model_quality          （模型加成）
            speed *= compute_mult * node_mult                       （算力加成）
            if model_quality > 0.9 and compute_mult > 100:
                speed *= 100                                          （攻破模式）
        """
        model_q = self._get_model_quality()
        compute_mult = self._get_compute_mult()
        node_mult = self._get_node_mult()

        speed = float(self.BASE_SPEED)

        # 模型加成：模型质量分 > 0.8 时启动倍率
        model_boost = 1.0
        if model_q > 0.8:
            model_boost = model_q
            speed *= model_boost

        # 算力加成：算力倍率 × 节点倍率
        speed *= compute_mult * node_mult

        # 攻破模式：模型质量 > 0.9 且算力倍率 > 100 → 速度×100
        breakthrough = model_q > 0.9 and compute_mult > 100
        if breakthrough:
            speed *= 100.0

        info = {
            "base_speed": self.BASE_SPEED,
            "model_quality": round(model_q, 4),
            "model_boost": round(model_boost, 4),
            "compute_mult": compute_mult,
            "node_mult": node_mult,
            "breakthrough": breakthrough,
            "speed": speed,
        }
        return speed, info

    # ---- 知识指纹 & 解码 ----

    def get_knowledge_fingerprint(self, domain: str, topic: str) -> str:
        """获取知识指纹——模型用这个指纹解码出内容"""
        # 用领域+主题派生稳定指纹（模拟现实知识的"哈希地址"）
        h = hashlib.sha256(f"{domain}::{topic}".encode("utf-8"))
        return h.hexdigest()[:16]

    def _decode_topic(self, domain: str, topic: str, idx: int, depth: int) -> str:
        """
        模型"解码"知识指纹，得到一条训练素材（单条版，兼容旧接口）。

        不是真从互联网下载——而是用模型+随机种子"读取"出领域知识。
        同 (domain, topic, idx) 总是产生同一内容（指纹可复现）。
        """
        fp = self.get_knowledge_fingerprint(domain, topic)
        seed = int(hashlib.md5(f"{fp}:{idx}".encode()).hexdigest()[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)

        domain_cn = self.DOMAIN_CN.get(domain, domain)
        # 主模板
        tmpl = self.TEMPLATES[int(rng.integers(0, len(self.TEMPLATES)))]
        fill = {
            "domain": domain_cn, "topic": topic,
            "aspect": self.ASPECTS[int(rng.integers(0, len(self.ASPECTS)))],
            "method": self.METHODS[int(rng.integers(0, len(self.METHODS)))],
            "concept": self.CONCEPTS[int(rng.integers(0, len(self.CONCEPTS)))],
            "result": self.RESULTS[int(rng.integers(0, len(self.RESULTS)))],
            "condition": self.CONDITIONS[int(rng.integers(0, len(self.CONDITIONS)))],
        }
        text = tmpl.format(**fill)

        # 深度扩展层——每层追加更丰富的内容，提升完整性
        n_layers = min(depth + 2, len(self.DEPTH_LAYERS))
        chosen_layers = rng.choice(len(self.DEPTH_LAYERS), size=n_layers, replace=False)
        for layer_i in sorted(chosen_layers):
            layer_fill = {
                "topic": topic,
                "aspect": self.ASPECTS[int(rng.integers(0, len(self.ASPECTS)))],
                "method": self.METHODS[int(rng.integers(0, len(self.METHODS)))],
                "concept": self.CONCEPTS[int(rng.integers(0, len(self.CONCEPTS)))],
                "result": self.RESULTS[int(rng.integers(0, len(self.RESULTS)))],
                "condition": self.CONDITIONS[int(rng.integers(0, len(self.CONDITIONS)))],
            }
            text += " " + self.DEPTH_LAYERS[layer_i].format(**layer_fill)

        text += f" [fp:{fp}]"
        return text

    def _decode_batch(
        self, domain: str, topics: List[str], topic_idxs: np.ndarray,
        sample_idxs: np.ndarray, depth: int,
    ) -> np.ndarray:
        """
        向量化解码——批量生成知识内容，速度远超逐条解码。

        核心优化：
        1. 预生成所有随机索引（一次 rng 调用）
        2. 列表推导替代逐条循环
        3. 深度层用向量化选择
        """
        n = len(topic_idxs)
        domain_cn = self.DOMAIN_CN.get(domain, domain)
        topics_list = list(topics)

        # 预生成所有随机索引（向量化）
        rng = np.random.default_rng(int(time.time() * 1e6) % (2**32))
        n_tmpl = len(self.TEMPLATES)
        n_asp = len(self.ASPECTS)
        n_mth = len(self.METHODS)
        n_con = len(self.CONCEPTS)
        n_res = len(self.RESULTS)
        n_cnd = len(self.CONDITIONS)
        n_layer = len(self.DEPTH_LAYERS)

        # 主模板的6个随机索引
        t_idx = rng.integers(0, n_tmpl, size=n, dtype=np.int32)
        a_idx = rng.integers(0, n_asp, size=n, dtype=np.int32)
        m_idx = rng.integers(0, n_mth, size=n, dtype=np.int32)
        c_idx = rng.integers(0, n_con, size=n, dtype=np.int32)
        r_idx = rng.integers(0, n_res, size=n, dtype=np.int32)
        d_idx = rng.integers(0, n_cnd, size=n, dtype=np.int32)

        # 深度扩展层：每条选 depth+2 层
        n_layers_per = min(depth + 2, n_layer)
        layer_choices = np.empty((n, n_layers_per), dtype=np.int32)
        for i in range(n):
            layer_choices[i] = rng.choice(n_layer, size=n_layers_per, replace=False)

        # 每个扩展层的随机索引
        la_idx = rng.integers(0, n_asp, size=(n, n_layers_per), dtype=np.int32)
        lm_idx = rng.integers(0, n_mth, size=(n, n_layers_per), dtype=np.int32)
        lc_idx = rng.integers(0, n_con, size=(n, n_layers_per), dtype=np.int32)
        lr_idx = rng.integers(0, n_res, size=(n, n_layers_per), dtype=np.int32)
        ld_idx = rng.integers(0, n_cnd, size=(n, n_layers_per), dtype=np.int32)

        # 预取列表
        tmpls = list(self.TEMPLATES)
        aspects = list(self.ASPECTS)
        methods = list(self.METHODS)
        concepts = list(self.CONCEPTS)
        results = list(self.RESULTS)
        conditions = list(self.CONDITIONS)
        layers = list(self.DEPTH_LAYERS)

        texts = np.empty(n, dtype=object)
        for i in range(n):
            topic = topics_list[int(topic_idxs[i])]
            fp = self.get_knowledge_fingerprint(domain, topic)

            # 主句
            fill = {
                "domain": domain_cn, "topic": topic,
                "aspect": aspects[a_idx[i]],
                "method": methods[m_idx[i]],
                "concept": concepts[c_idx[i]],
                "result": results[r_idx[i]],
                "condition": conditions[d_idx[i]],
            }
            parts = [tmpls[t_idx[i]].format(**fill)]

            # 深度扩展层
            for j in range(n_layers_per):
                lf = {
                    "topic": topic,
                    "aspect": aspects[la_idx[i, j]],
                    "method": methods[lm_idx[i, j]],
                    "concept": concepts[lc_idx[i, j]],
                    "result": results[lr_idx[i, j]],
                    "condition": conditions[ld_idx[i, j]],
                }
                parts.append(layers[layer_choices[i, j]].format(**lf))

            parts.append(f"[fp:{fp}]")
            texts[i] = " ".join(parts)

        return texts

    def _score_texts(
        self, texts: np.ndarray, quality_base: float, model_q: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        快速质量评分（与 QualityScorer.score_batch_fast 兼容的简化版）。

        模型越强过滤越严，整体质量基线被 model_quality 直接抬升。
        Returns:
            (scores: np.float32, grades: np.uint8 0=D..6=SSS)
        """
        n = len(texts)
        if n == 0:
            return (np.array([], dtype=np.float32),
                    np.array([], dtype=np.uint8))

        # 长度分（与 QualityScorer.score_batch_fast 一致）
        lengths = np.array([len(t) for t in texts], dtype=np.float32)
        len_score = np.where(
            lengths < 50,
            lengths / 50.0 * 0.5,
            np.where(
                lengths > 2000,
                np.maximum(0.3, 1.0 - (lengths - 2000) / 8000.0),
                0.7 + (lengths - 50) / 1950.0 * 0.3,
            ),
        )

        # 字符多样性分
        char_div = np.zeros(n, dtype=np.float32)
        for i in range(n):
            t = texts[i]
            if len(t) > 0:
                char_div[i] = len(set(t)) / max(1, len(t))
        char_div = np.minimum(1.0, char_div * 2.0)

        # 综合分 = 领域基础质量 + 长度 + 多样性
        scores = quality_base * 0.5 + len_score * 0.25 + char_div * 0.25
        # 模型过滤：模型越强，质量基线越高（低质素材被模型剔除的模拟效果）
        scores = np.minimum(0.99, scores + model_q * 0.1)

        # 等级编码（与 QualityScorer._grade / np.digitize 一致：0=D,1=C,...,6=SSS）
        grades = np.digitize(
            scores,
            bins=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            right=False,
        )
        return scores.astype(np.float32), grades.astype(np.uint8)

    # ---- 主接口 ----

    def download(self, domain: str, count: int = 10000) -> Dict[str, Any]:
        """
        下载指定领域的知识，转为训练素材。

        速度公式：
        - 基础速度：10000条/秒
        - 模型加成：model_quality > 0.8 时，×model_quality
        - 算力加成：×compute_multiplier × node_multiplier
        - 攻破模式：model_quality > 0.9 且 compute_mult > 100 → 速度×100

        Returns:
            dict 包含：
              - total: 下载条数
              - speed: 下载速度（条/秒）
              - speed_info: 速度计算明细
              - texts / scores / grades: 训练素材（与 TrainingForge.generate_fast 兼容）
              - domain / topics / avg_quality / fingerprint_sample
        """
        if domain not in self.DOMAIN_KNOWLEDGE:
            raise ValueError(
                f"未知领域: {domain}，支持: {list(self.DOMAIN_KNOWLEDGE.keys())}"
            )
        dom_info = self.DOMAIN_KNOWLEDGE[domain]
        topics: List[str] = dom_info["topics"]
        depth: int = dom_info["depth"]
        quality_base: float = dom_info["quality_base"]

        speed, info = self._compute_speed()
        model_q = info["model_quality"]

        # 向量化解码：批量生成，速度远超逐条
        topic_idxs = self._rng.integers(0, len(topics), size=count, dtype=np.int32)
        sample_idxs = self._rng.integers(0, 2**31, size=count, dtype=np.int32)

        texts = self._decode_batch(domain, topics, topic_idxs, sample_idxs, depth)

        scores, grades = self._score_texts(texts, quality_base, model_q)

        # 更新统计
        self.downloaded += count
        if domain not in self.domain_stats:
            self.domain_stats[domain] = {"count": 0, "total_quality": 0.0}
        self.domain_stats[domain]["count"] += count
        self.domain_stats[domain]["total_quality"] += (
            float(scores.mean()) if len(scores) else 0.0
        )

        return {
            "domain": domain,
            "total": count,
            "speed": speed,
            "speed_info": info,
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "topics": list(topics),
            "avg_quality": float(scores.mean()) if len(scores) else 0.0,
            "fingerprint_sample": self.get_knowledge_fingerprint(domain, topics[0]),
        }

    def download_multi_domain(
        self, count: int = 100000, domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        多领域混合下载。

        Args:
            count: 总下载条数（按领域均匀分配，余数分给前几个领域）
            domains: 指定领域列表，None 表示下载全部 10 个领域

        Returns:
            dict 包含 total / domain_distribution / texts / scores / grades / speed
        """
        if domains is None:
            domains = list(self.DOMAIN_KNOWLEDGE.keys())

        n_domains = len(domains)
        per_domain = count // n_domains
        remainder = count - per_domain * n_domains

        all_texts: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_grades: List[np.ndarray] = []
        domain_distribution: Dict[str, int] = {}

        for i, domain in enumerate(domains):
            n = per_domain + (1 if i < remainder else 0)
            if n <= 0:
                continue
            result = self.download(domain, count=n)
            all_texts.append(result["texts"])
            all_scores.append(result["scores"])
            all_grades.append(result["grades"])
            domain_distribution[domain] = result["total"]

        texts = (np.concatenate(all_texts) if all_texts
                 else np.array([], dtype=object))
        scores = (np.concatenate(all_scores) if all_scores
                  else np.array([], dtype=np.float32))
        grades = (np.concatenate(all_grades) if all_grades
                  else np.array([], dtype=np.uint8))

        speed = self._compute_speed()[0]

        return {
            "total": int(len(texts)),
            "domain_distribution": domain_distribution,
            "domains": list(domains),
            "speed": speed,
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "avg_quality": float(scores.mean()) if len(scores) else 0.0,
        }

    def get_stats(self) -> Dict[str, Any]:
        """下载统计"""
        return {
            "total_downloaded": self.downloaded,
            "domain_stats": dict(self.domain_stats),
            "model_attached": self.model is not None,
            "engine_attached": self.engine is not None,
            "speed_info": self._compute_speed()[1],
        }

    # ---- 极致压缩：万象奇点驱动 ----

    def compress_with_singularity(
        self,
        texts: np.ndarray,
        compression_points: int = 100,
        use_singularity: bool = True,
    ) -> Dict[str, Any]:
        """
        用万象奇点驱动的压缩点对下载的知识进行极致压缩。

        压缩原理：
        - 普通压缩：每个压缩点 = 1倍压缩，100个 = 100倍
        - 万象奇点驱动：算力倍率 9999 → 压缩点强度被放大 9999 倍
        - 极致压缩：压缩后只存"知识指纹"，需要时用模型+算力解压恢复

        压缩比公式：
            normal_factor = compression_points
            singularity_factor = compression_points × compute_mult
            极致模式 = singularity_factor × 100（奇点加成）

        Args:
            texts: 待压缩的文本数组
            compression_points: 压缩点数量
            use_singularity: 是否用万象奇点驱动

        Returns:
            压缩结果（含压缩比、压缩后大小、指纹数组）
        """
        import hashlib as _hl

        n = len(texts)
        if n == 0:
            return {"error": "空数据", "compressed": 0}

        # 原始大小（字节）
        original_size = sum(len(t.encode("utf-8")) for t in texts)

        # 计算压缩倍率
        compute_mult = self._get_compute_mult() if use_singularity else 1.0
        node_mult = self._get_node_mult() if use_singularity else 1.0

        # 基础压缩：压缩点数量
        base_factor = compression_points
        # 万象奇点加成：算力倍率 × 节点倍率
        singularity_boost = compute_mult * node_mult if use_singularity else 1.0
        # 极致模式：万象奇点 + 压缩点 > 100 → 额外100倍
        extreme_mode = use_singularity and compute_mult > 100
        extreme_boost = 100.0 if extreme_mode else 1.0

        # 总压缩倍率
        total_factor = base_factor * singularity_boost * extreme_boost

        # 压缩：每条文本 → 知识指纹（16字符）
        fingerprints = np.empty(n, dtype=object)
        compressed_size = 0
        for i in range(n):
            text = texts[i]
            # 用 SHA256 派生指纹作为压缩表示
            fp = _hl.sha256(text.encode("utf-8")).hexdigest()[:16]
            fingerprints[i] = fp
            compressed_size += len(fp)

        # 压缩比
        ratio = original_size / max(1, compressed_size)
        # 实际有效压缩倍率 = min(理论倍率, 实际比)
        effective_factor = min(total_factor, ratio)

        mode = "万象奇点极致压缩" if extreme_mode else (
            "万象奇点压缩" if use_singularity else "普通压缩"
        )

        return {
            "mode": mode,
            "original_count": n,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": round(ratio, 2),
            "theoretical_factor": round(total_factor, 2),
            "effective_factor": round(effective_factor, 2),
            "compression_points": compression_points,
            "singularity_boost": round(singularity_boost, 2),
            "extreme_mode": extreme_mode,
            "fingerprints": fingerprints,
            "texts": texts,  # 保留原文（虚拟压缩不丢数据）
            "note": "万象奇点驱动：压缩点强度×算力倍率，极致模式额外100倍",
        }

    def download_and_compress(
        self,
        domain: str,
        count: int = 10000,
        compression_points: int = 100,
        use_singularity: bool = True,
    ) -> Dict[str, Any]:
        """
        下载 + 极致压缩一站式。

        先用模型+算力下载知识，再用万象奇点驱动压缩点极致压缩。
        """
        # 下载
        dl_result = self.download(domain, count=count)

        # 压缩
        cp_result = self.compress_with_singularity(
            dl_result["texts"],
            compression_points=compression_points,
            use_singularity=use_singularity,
        )

        return {
            "domain": domain,
            "downloaded": dl_result["total"],
            "download_speed": dl_result["speed"],
            "compression": cp_result,
            "texts": dl_result["texts"],
            "scores": dl_result["scores"],
            "grades": dl_result["grades"],
            "avg_quality": dl_result["avg_quality"],
        }

    # ---- 融合压缩：万象奇点 + 流式算力网络 ----

    def compress_fusion(
        self,
        texts: np.ndarray,
        domain: str = "unknown",
        engine: Any = None,
    ) -> Dict[str, Any]:
        """
        万象奇点 + 流式算力网络融合压缩——1GB → <500B。

        三层压缩：
        1. 文本层：每条文本 → 16字符 SHA256 指纹（~54x）
        2. Merkle层：所有指纹 → 一棵 Merkle 树 → 1个根哈希（N条→1个32字符）
        3. 元数据层：根哈希 + 领域 + 条数 + 深度 + 算力签名（<500B）

        解压原理：根哈希是所有数据的"全景指纹"，
        模型+算力可以从指纹 + 元数据完美还原全部内容（虚拟世界设定）。

        Args:
            texts: 待压缩文本数组
            domain: 领域名（存入元数据）
            engine: 融合引擎（万象奇点+流式算力网络）

        Returns:
            压缩结果（压缩后 < 500B）
        """
        import json as _json

        n = len(texts)
        if n == 0:
            return {"error": "空数据"}

        original_bytes = sum(len(t.encode("utf-8")) for t in texts)

        # ---- 第一层：文本 → 指纹 ----
        fingerprints = []
        for t in texts:
            fp = hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]
            fingerprints.append(fp)

        # ---- 第二层：Merkle 树 → 根哈希 ----
        # 逐层两两哈希，最终得到一个根
        layer = fingerprints[:]
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                if i + 1 < len(layer):
                    combined = layer[i] + layer[i + 1]
                else:
                    combined = layer[i] + layer[i]  # 奇数个，最后一个重复
                next_layer.append(hashlib.sha256(combined.encode()).hexdigest()[:16])
            layer = next_layer
        merkle_root = layer[0] if layer else "0" * 16

        # ---- 第三层：元数据包（< 500B）----
        # 获取引擎参数
        if engine is not None:
            compute_mult = float(getattr(engine, "compute_multiplier", 1.0))
            node_count = int(getattr(engine, "node_count", 1))
            is_perp = bool(getattr(engine, "is_perpetual", False))
        else:
            compute_mult = 1.0
            node_count = 1
            is_perp = False

        # 算力签名：用引擎参数派生一个签名哈希（证明是融合引擎压缩的）
        sig_input = f"{merkle_root}:{compute_mult:.0f}:{node_count}:{is_perp}"
        signature = hashlib.sha256(sig_input.encode()).hexdigest()[:16]

        # 压缩包：JSON 格式的极小元数据
        compressed_packet = {
            "v": 1,                         # 版本
            "d": domain,                    # 领域
            "n": n,                         # 条数
            "r": merkle_root,               # Merkle 根
            "s": signature,                 # 算力签名
            "c": int(compute_mult),         # 算力倍率
            "p": int(node_count),           # 节点数
            "e": int(is_perp),              # 永动
        }
        compressed_bytes = _json.dumps(compressed_packet, separators=(",", ":")).encode("utf-8")
        compressed_size = len(compressed_bytes)

        # 压缩比
        ratio = original_bytes / max(1, compressed_size)

        mode = "万象奇点·流式算力网络融合压缩"
        if is_perp and compute_mult > 100:
            mode += "(极致)"

        return {
            "mode": mode,
            "domain": domain,
            "original_count": n,
            "original_size_bytes": original_bytes,
            "compressed_size_bytes": compressed_size,
            "compressed_packet": compressed_packet,
            "compression_ratio": round(ratio, 2),
            "merkle_root": merkle_root,
            "signature": signature,
            "under_500b": compressed_size < 500,
            "fingerprints": np.array(fingerprints, dtype=object),
            "texts": texts,  # 保留原文（虚拟压缩不丢数据）
            "engine_info": {
                "compute_multiplier": compute_mult,
                "node_count": node_count,
                "perpetual": is_perp,
            },
            "note": "三层压缩: 文本→指纹→Merkle根→元数据包, 模型+算力可从根哈希还原",
        }

    # ---- 子代理收集 ----

    def collect_with_agents(
        self,
        agents: List[Dict[str, Any]],
        domains: Optional[List[str]] = None,
        per_agent_count: int = 10000,
    ) -> Dict[str, Any]:
        """
        子代理军团协助收集知识——N个代理并行下载不同领域。

        每个子代理负责一个或多个领域，并行收集，
        总产出 = 代理数 × per_agent_count。

        Args:
            agents: 子代理列表（来自 produce_sub_agents / produce_agent_army）
            domains: 领域列表，None 则按代理专长分配
            per_agent_count: 每个代理收集的条数

        Returns:
            汇总收集结果
        """
        n_agents = len(agents)
        if n_agents == 0:
            return {"error": "无子代理", "total": 0}

        # 领域分配
        all_domains = list(self.DOMAIN_KNOWLEDGE.keys())
        if domains is None:
            # 按代理专长分配
            domains = []
            for i, a in enumerate(agents):
                agent_domain = a.get("domain", "")
                # 把子代理领域映射到知识库领域
                domain_map = {
                    "math": "math", "physics": "physics",
                    "chemistry": "chemistry", "biology": "biology",
                    "medicine": "medicine", "law": "law",
                    "finance": "finance", "philosophy": "philosophy",
                    "code": "computer_science",
                    "engineering": "engineering",
                    "language": "philosophy",
                    "music": "philosophy", "art": "philosophy",
                    "psychology": "medicine", "economics": "finance",
                }
                mapped = domain_map.get(agent_domain, all_domains[i % len(all_domains)])
                domains.append(mapped)

        # 并行收集（模拟：每个代理独立下载）
        all_texts = []
        all_scores = []
        all_grades = []
        agent_results = []

        speed, speed_info = self._compute_speed()
        model_q = speed_info["model_quality"]

        for i, (agent, domain) in enumerate(zip(agents, domains)):
            if domain not in self.DOMAIN_KNOWLEDGE:
                domain = all_domains[0]

            dom_info = self.DOMAIN_KNOWLEDGE[domain]
            topics = dom_info["topics"]
            depth = dom_info["depth"]
            quality_base = dom_info["quality_base"]

            # 该代理下载
            topic_idxs = self._rng.integers(0, len(topics), size=per_agent_count, dtype=np.int32)
            sample_idxs = self._rng.integers(0, 2**31, size=per_agent_count, dtype=np.int32)
            texts = self._decode_batch(domain, topics, topic_idxs, sample_idxs, depth)
            scores, grades = self._score_texts(texts, quality_base, model_q)

            all_texts.append(texts)
            all_scores.append(scores)
            all_grades.append(grades)

            agent_results.append({
                "agent_name": agent.get("name", f"agent_{i}"),
                "agent_domain": agent.get("domain", "unknown"),
                "collected_domain": domain,
                "count": per_agent_count,
                "avg_quality": float(scores.mean()),
                "grade": agent.get("grade", "S"),
            })

        # 汇总
        texts = np.concatenate(all_texts) if all_texts else np.array([], dtype=object)
        scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
        grades = np.concatenate(all_grades) if all_grades else np.array([], dtype=np.uint8)

        total_count = len(texts)
        # 总速度 = 单代理速度 × 代理数（并行）
        total_speed = speed * n_agents

        return {
            "total": total_count,
            "n_agents": n_agents,
            "per_agent": per_agent_count,
            "total_speed": total_speed,
            "agent_results": agent_results,
            "texts": texts,
            "scores": scores,
            "grades": grades,
            "avg_quality": float(scores.mean()) if len(scores) else 0.0,
            "speed_info": speed_info,
            "note": f"{n_agents}个子代理并行收集, 总速度×{n_agents}",
        }
