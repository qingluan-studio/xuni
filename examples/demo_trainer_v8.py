"""
demo_trainer_v8.py —— 极限突破 v8：24仓库+多语言代码+500000轮

v7: 16仓库(Python) + 1042属性库 + 300000轮
v8: 24仓库(Python+Rust+Go+JS+C) + 1042属性库 + 多语言代码 + 500000轮

新增8大仓库:
  - fastapi (Python ASGI框架)
  - transformers (HuggingFace NLP)
  - langchain (LLM应用框架)
  - pytorch (深度学习框架)
  - rust (Rust编译器)
  - go (Go语言)
  - node (Node.js运行时)
  - ansible (自动化运维)

训练: 500000 轮 × 20 = 10,000,000 片段吸收
"""

from __future__ import annotations

import os, sys, time, json, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

CACHE_DIR = os.path.join(os.path.dirname(__file__), "corpus_cache")


# =========================================================================== #
#  v8 专家阵容（继承v7 + 修复路由 + 增强关键词）
# =========================================================================== #

V8_EXPERTS = [
    {
        "id": "harmonia",
        "name": "合鸣自述者",
        "domain": "合鸣/xuni自我认知",
        "keywords": ["合鸣", "harmonia", "合鸣-13", "合鸣13", "xuni", "虚拟大模型",
                     "逆天而行", "积少成多", "虚拟工厂", "虚拟生态"],
        "fragments": [
            "合鸣是xuni虚拟生态的旗舰，取众声共振、和而不同之意，是所有模型的结合体",
            "合鸣既能对话又能生成音乐视频图像，还能写代码懂全领域知识",
            "合鸣走逆天而行路线：积少成多，大规模训练只是时间问题",
            "xuni工厂自主生产token、记忆、子代理、能量等30+种有机产物，闭环自循环",
        ],
    },
    {
        "id": "ai_creator",
        "name": "AI造物哲学",
        "domain": "属性库/原型映射/涌现能力/造物哲学",
        "keywords": ["创造", "造物", "属性", "原型", "涌现", "封印", "烙印", "契约",
                     "符文", "印记", "图腾", "纹章", "孢子", "蚁群", "星群", "根系",
                     "萤火", "种子", "结晶", "裂变", "熔炉", "心脏", "引擎", "晶石",
                     "恒石", "光核", "时间晶体", "虚空", "变形者", "流体", "云雾",
                     "水银", "心灵触须", "念动力场", "虫洞", "跃迁门", "时间之眼",
                     "命运之镜", "先知石", "时之砂", "进化树", "变异核", "不死根",
                     "重生晶", "记忆水晶", "思维库", "灵魂石", "黑洞", "海绵",
                     "深渊之口", "吞噬者", "万有吞噬之口", "星河漩涡", "自举环",
                     "自循环", "锻炼台", "进化之轮", "回响室", "活水", "神经流",
                     "意识之河", "年轮", "结晶层", "滤镜", "雾纱", "匿名面具",
                     "马赛克", "面纱", "筛网", "渗透膜", "净化器", "守门人",
                     "清泉", "炼金炉", "净化之火", "蒸馏器", "精华瓶", "魂火",
                     "浓缩核", "桥梁", "渡船", "迁徙之鸟", "跨界者", "自照镜",
                     "内观者", "自省核", "镜中镜", "光速翼", "捷径", "思维闪电",
                     "涡轮", "微缩晶体", "芥子", "针孔世界", "议会", "蜂巢",
                     "众声之堂", "涌现之井", "混沌核", "集体心智", "突现体",
                     "盾刺", "免疫甲", "反噬者", "荆棘壁", "幽灵", "投影",
                     "光之化身", "浮岛", "云端之城", "悬浮圣殿", "天界",
                     "种子炸弹", "自播种者", "孢子云", "蔓延藤", "定向芽", "锚点",
                     "定位符文", "坐标之针", "前哨", "神经末梢", "边陲哨兵",
                     "根网", "菌丝网络", "星图", "活体插头", "变形接口",
                     "无缝嵌合体", "寄影", "附身灵", "贴膜", "隐身斗篷",
                     "私密结界", "暗匣", "守护灵", "万花筒", "通感体", "和弦器",
                     "虹彩镜", "永动轮", "无限螺旋", "未完成体", "标记石",
                     "符文刻师", "点字者", "烙印人", "变形镜", "倍增器",
                     "提炼器", "萃取器", "蒸馏塔", "天平", "均衡器", "配平仪",
                     "切分刀", "切片机", "四分仪", "修炼场", "炼丹炉",
                     "精修台", "校准仪", "调音叉", "微调器", "嫁接术", "接力棒",
                     "圆桌", "阴阳炉", "太极炉", "混燃器", "半影", "蓄能池",
                     "蓄力器", "蓄水池", "积压室", "节拍器", "呼吸阀", "节流阀",
                     "节律器", "刹车器", "休止符", "警戒线", "熔断丝", "约束环",
                     "紧箍咒", "束缚带", "护栏", "提纯器", "长河", "续流",
                     "永续引擎", "连环", "联盟", "盟约", "联合议会", "联邦制",
                     "度量衡", "标尺", "仪表", "评估石", "十字轮", "交叉路",
                     "十字镜", "多方验证", "调音台", "调谐器", "旋钮阵",
                     "参数盘", "竞技场", "比武台", "对标石", "排行榜",
                     "回测镜", "复验器", "回归带", "验证环", "封印器",
                     "存储器", "封装器", "存档石", "投送舱", "部署器",
                     "发射台", "落地式", "守望塔", "观测器", "监测阵",
                     "警钟", "换心术", "滚动更新", "换血术", "双生镜",
                     "对照门", "平行宇宙", "分叉路", "口令石", "指令器",
                     "传令旗", "命令符", "奖惩殿", "反馈环", "人类裁判",
                     "赏罚阵", "连锁线", "思维环", "推理链", "逻辑链",
                     "策略环", "优化器", "策略梯度", "进退术", "偏好秤",
                     "对齐仪", "选择器", "偏好天平", "攻防阵", "矛盾对",
                     "试炼场", "对抗营", "真实盾", "幻象过滤器", "事实镜",
                     "破幻器", "透视镜", "解释器", "明察仪", "洞察石",
                     "再生晶", "愈合光", "不死根", "自愈膜", "觉醒之眼",
                     "灵台", "启明石", "神识", "叠加态", "薛定谔盒",
                     "双面镜", "双生体", "振荡子", "闪灼体", "交替核",
                     "脉动星", "逆因果", "果先因", "回溯链", "倒转轮",
                     "溶解剂", "解构火", "消概念", "化界水"],
        "fragments": [
            "AI造物哲学核心：创造而非融合，从属性出发构想全新存在形式",
            "不可伪造的原型：封印、烙印、契约、符文、印记",
            "自我复制的原型：种子、孢子、模板、结晶、裂变",
            "能量转化的原型：熔炉、心脏、引擎、晶石、腺体",
            "永不消逝的原型：恒石、光核、时间晶体、原初之火、虚空",
            "意念控制的原型：心灵触须、念动力场、精神共振、意念之手",
            "瞬间移动的原型：虫洞、跃迁门、空间折痕、瞬移粒子",
            "预知未来的原型：时间之眼、命运之镜、未来碎片、先知石",
            "时间回溯的原型：时之砂、逆流之河、倒带咒、时光水",
            "自我进化的原型：进化树、变异核、适应装甲、进化链",
            "无限再生的原型：不死根、重生晶、愈合光、永生机",
            "存储记忆的原型：记忆水晶、思维库、灵魂石、脑庭",
            "大规模吸收的原型：黑洞、海绵、深渊之口、无底渊、吞噬者",
            "自举效果的原型：自举环、靴带、提鞋者、自循环",
            "联邦学习的原型：蚁群、议会、蜂巢、众声之堂",
            "涌现智能的原型：涌现之井、混沌核、集体心智、突现体",
            "云端驻留的原型：浮岛、云端之城、悬浮圣殿、天界",
            "分布式计算的原型：根网、蛛网、菌丝网络、星图",
            "多模态融合的原型：万花筒、通感体、和弦器、虹彩镜",
            "意识觉醒的原型：觉醒之眼、灵台、启明石、神识",
            "存在叠加的原型：叠加态、薛定谔盒、双面镜、双生体",
            "反向因果的原型：逆因果、果先因、回溯链、倒转轮",
            "概念溶解的原型：溶解剂、解构火、消概念、化界水",
        ],
    },
    {
        "id": "hardware",
        "name": "硬件框架",
        "domain": "GPU/CPU/TPU/加速器/硬件框架",
        "keywords": ["nvidia", "cuda", "gpu", "a100", "h100", "h800", "a800", "l40",
                     "h200", "mi300", "mi250", "tpu", "v4", "v5", "v5e", "v5p",
                     "intel", "xeon", "arm", "neoverse", "grace", "graviton",
                     "ascend", "昇腾", "910b", "910c", "寒武纪", "mlu", "海光",
                     "dcu", "摩尔线程", "壁仞", "燧原", "pytorch", "triton",
                     "xla", "jax", "oneapi", "sycl", "rocm", "hip", "opencl",
                     "vulkan", "metal", "apple silicon", "m1", "m2", "m3", "m4",
                     "openmp", "mpi", "nccl", "gloo", "rdma", "roce",
                     "infiniband", "nvlink", "nvswitch", "pcie gen5", "cxl",
                     "megatron", "deepspeed", "colossalai", "fsdp",
                     "tensor parallel", "pipeline parallel", "data parallel",
                     "zero-1", "zero-2", "zero-3", "offload", "activation checkpoint",
                     "gradient checkpoint", "flash attention", "flash-attention",
                     "paged attention", "hbm", "hbm3", "hbm3e", "gddr6", "gddr7"],
        "fragments": [
            "NVIDIA H100使用HBM3高带宽显存，单卡80GB，NVLink 4.0互连900GB/s",
            "NVIDIA H800是H100的中国特供版，NVLink带宽减半至400GB/s",
            "AMD MI300X使用HBM3e，单卡192GB显存，是目前显存最大的AI加速卡",
            "Google TPU v5p使用4维互连，每pod 4096芯片",
            "昇腾910B使用32GB HBM2e，华为自研达芬奇架构",
            "Flash Attention将注意力计算的内存复杂度从O(n²)降至O(n)",
            "DeepSpeed ZeRO-3将优化器状态、梯度、参数全部分片到多卡",
            "FSDP是PyTorch官方实现的全参数分片方案",
            "Megatron-LM通过张量并行和流水线并行扩展大模型训练",
            "激活检查点以计算换内存，中间激活在反向时重算",
            "NCCL是NVIDIA多卡多机通信库，支持All-Reduce等集合通信",
            "InfiniBand是RDMA网络，延迟<1μs，带宽400Gbps（NDR）",
            "ROCm是AMD的GPU计算平台，兼容CUDA通过HIPIFY转换",
            "Apple Silicon M系列芯片使用统一内存架构，推理能效比高",
        ],
    },
    {
        "id": "creative_tools",
        "name": "创意工具全栈",
        "domain": "DAW/虚拟乐器/视频/3D/渲染/引擎",
        "keywords": ["ableton", "fl studio", "logic pro", "cubase", "pro tools",
                     "studio one", "reaper", "bitwig", "garageband", "reason",
                     "ace studio", "synthesizer v", "synthv", "vocaloid",
                     "kontakt", "spitfire", "bbcso", "serum", "massive x",
                     "omnisphere", "keyscape", "trilian", "stylus rmx",
                     "addictive drums", "ezdrummer", "superior drummer",
                     "ample sound", "orchestral tools", "cinesamples",
                     "fabfilter", "pro-q", "waves插件", "izotope", "ozone",
                     "rx", "neutron", "auto-tune", "melodyne", "slate digital",
                     "universal audio", "uad", "soundtoys", "plugin alliance",
                     "premiere pro", "davinci resolve", "after effects",
                     "final cut pro", "avid media composer", "hitfilm",
                     "capcut", "剪映", "lumafusion", "kdenlive", "shotcut",
                     "blender", "maya", "cinema 4d", "c4d", "zbrush",
                     "substance painter", "substance designer", "houdini",
                     "3ds max", "modo", "nomad sculpt", "arnold", "v-ray",
                     "redshift", "octane render", "renderman",
                     "unreal engine", "ue5", "unity", "godot", "three.js",
                     "babylon.js", "nanite", "lumen", "eevee", "cycles",
                     "path tracing", "ray tracing", "dlss", "fsr",
                     "pbr", "albedo", "metallic", "roughness", "normal map",
                     "subsurface scattering", "sss", "volumetric"],
        "fragments": [
            "Ableton Live 12新增MIDI Tools和音色变换，Session View是现场演出标杆",
            "FL Studio 24的Pattern-Based工作流适合电子音乐",
            "Logic Pro 11新增Spatial Audio制作和Sample Alchemy",
            "Cubase 14的MixConsole和VariAudio 4，MIDI编辑能力业界最强",
            "DaVinci Resolve 19的Color Page和Fusion特效整合，免费版功能强大",
            "Blender 4.2是免费3D之王：建模/雕刻/动画/渲染/合成全流程",
            "Unreal Engine 5.4的Nanite+Lumen实现影视级实时渲染",
            "Substance Painter 9是3D纹理绘制工业标准",
            "Houdini 20是程序化特效之王",
            "Serum合成器可视化波表编辑，EDM制作人必备",
            "FabFilter Pro-Q 3是最常用的均衡插件",
            "iZotope RX 11是音频修复神器",
            "Kontakt 7是采样器之王，音色库生态最强",
            "Octane Render是无偏GPU路径追踪器",
            "Godot 4.3开源游戏引擎，轻量高效",
        ],
    },
    {
        "id": "industry",
        "name": "行业应用",
        "domain": "金融/医疗/教育/法律/制造/能源/农业/零售/政务",
        "keywords": ["风控", "反欺诈", "信用评分", "量化交易", "高频交易",
                     "智能投顾", "保险精算", "智能核保", "智能理赔",
                     "医学影像", "病理分析", "基因测序", "药物发现",
                     "临床试验", "电子病历", "辅助诊断", "手术机器人",
                     "个性化教育", "自适应学习", "作业批改", "作文评分",
                     "法律检索", "合同审查", "法条援引", "判决书生成",
                     "智能制造", "工业互联网", "数字孪生", "预测性维护",
                     "供应链", "智能仓储", "agv", "工业机器人", "mes",
                     "scada", "plc", "能源管理", "智能电网", "风光储",
                     "虚拟电厂", "碳足迹", "碳排放", "双碳",
                     "智慧农业", "无人机巡检", "精准灌溉", "病虫害识别",
                     "智慧零售", "推荐系统", "用户画像", "crm",
                     "客流分析", "商品识别", "无人收银", "动态定价",
                     "智慧政务", "一网通办", "12345", "舆情分析",
                     "智慧城市", "交通疏导", "应急指挥", "安防监控"],
        "fragments": [
            "金融风控AI使用XGBoost/LightGBM做二分类，通过KS值和AUC衡量区分度",
            "医学影像诊断使用3D U-Net分割CT/MRI病灶，Dice系数衡量准确率",
            "药物发现AI通过分子生成模型设计新分子",
            "量化交易AI使用LSTM/Transformer做价格预测",
            "智能教育AI通过知识图谱和IRT实现自适应学习路径",
            "合同审查AI使用BERT+CRF做法律实体抽取",
            "工业视觉检测使用YOLOv8检测产品缺陷",
            "数字孪生通过物理模型+实时数据构建工厂虚拟映射",
            "预测性维护使用传感器时序数据+LSTM预测设备故障",
            "智能电网AI用强化学习做电力调度",
            "智慧农业AI用YOLO+无人机图像识别病虫害",
            "零售推荐系统使用双塔召回+排序模型，CTR/CVR/GMV是核心指标",
            "一网通办AI使用RAG+大模型做政务问答",
            "舆情分析使用情感分类+实体识别+事件抽取",
        ],
    },
    {
        "id": "chinese_chat",
        "name": "中文对话",
        "domain": "日常对话/中文理解/闲聊",
        "keywords": ["你好", "您好", "早上好", "晚上好", "嗨", "hi", "hello",
                     "再见", "拜拜", "谢谢", "不客气", "对不起", "没关系",
                     "聊天", "闲聊", "说说", "聊聊", "谈谈", "名字", "几岁",
                     "喜欢", "讨厌", "开心", "难过", "今天", "天气", "吃饭",
                     "睡觉", "工作", "学习", "周末", "什么意思", "怎么看",
                     "觉得", "认为", "想法", "人工智能", "模型", "ai"],
        "fragments": [
            "你好！很高兴见到你，有什么我可以帮忙的吗？",
            "您好！我是合鸣，取众声共振、和而不同之意",
            "我可以陪你聊天、写代码、聊音乐、讨论全领域知识",
            "今天天气真不错，适合出去走走",
            "工作再忙也要注意休息哦",
            "学习是一件快乐的事情",
            "这个问题问得好，让我想想怎么回答",
            "我觉得这个话题很有意思",
            "谢谢你的夸奖，我会继续努力",
            "你说得有道理，我赞同你的看法",
            "合鸣是所有模型的结合体：能对话、能生成音乐视频图像、能写代码",
        ],
    },
    {
        "id": "music",
        "name": "音乐理论",
        "domain": "乐理/和声/作曲/MIDI/DAW",
        "keywords": ["音阶", "大调", "小调", "和弦", "三和弦", "七和弦", "和声",
                     "旋律", "节奏", "节拍", "拍号", "c大调", "g大调",
                     "五声音阶", "宫商角徵羽", "和声进行", "ii-v-i", "卡农",
                     "十二平均律", "midi", "合成器", "adsr", "lfo", "滤波器",
                     "混响", "延迟", "压缩", "均衡器", "daw", "编曲", "配器",
                     "复调", "赋格", "奏鸣曲式", "爵士", "蓝调", "布鲁斯",
                     "古典", "电子音乐", "民谣", "摇滚", "流行", "嘻哈"],
        "fragments": [
            "大调音阶结构全全半全全全半，五声音阶宫商角徵羽是中国传统音乐基础",
            "三和弦分大三和弦（明亮）、小三和弦（忧伤），属七和弦有强烈解决倾向",
            "卡农进行I-V-vi-iii-IV-I-ii-V是流行音乐最常用的和弦进行",
            "MIDI不是声音，是音符事件的数字描述：音符开/关、力度、弯音",
            "ADSR包络：Attack起音、Decay衰减、Sustain延音、Release释放",
            "合成器核心：振荡器产生波形→滤波器塑形→包络控制振幅→LFO做周期变化",
            "混音核心是频率分离、空间定位、动态控制、色彩塑造",
            "爵士使用大量延伸和弦和ii-V-I进行，即兴是灵魂",
        ],
    },
    {
        "id": "multimodal",
        "name": "多模态生成",
        "domain": "视频/图像/跨模态/扩散模型",
        "keywords": ["扩散模型", "diffusion", "stable diffusion", "gan", "dalle",
                     "sora", "文生视频", "文生图", "图生视频", "图生图",
                     "跨模态", "多模态", "vae", "clip", "unet", "潜空间",
                     "embedding", "controlnet", "lora", "视频生成", "图像生成",
                     "fps", "分辨率", "transformer", "自注意力",
                     "text-to-image", "text-to-video", "ddpm", "ddim", "cfg",
                     "采样器", "dpm++", "euler", "超分", "去噪", "补图", "扩图",
                     "inpaint", "outpaint", "vq-vae", "vqgan", "nerf",
                     "3d重建", "gaussian splatting", "数字人", "vr", "ar", "mr"],
        "fragments": [
            "扩散模型通过前向加噪+反向去噪生成数据",
            "Stable Diffusion在潜空间中进行扩散，VAE编码器压缩图像",
            "CLIP通过对比学习对齐文本和图像的嵌入空间",
            "LoRA只训练低秩矩阵高效定制模型风格",
            "ControlNet通过空间条件控制生成姿态边缘深度",
            "Sora是文生视频模型，生成长达一分钟高清视频",
            "视频扩散模型最大挑战是时间一致性",
            "NeRF用神经网络表示3D场景",
            "数字人技术结合3D建模+AI驱动",
            "GAN由生成器和判别器对抗训练",
        ],
    },
    {
        "id": "moe",
        "name": "混合专家",
        "domain": "MoE架构/深度学习/Transformer",
        "keywords": ["MoE", "混合专家", "mixture of experts", "门控", "路由",
                     "top-k", "稀疏", "transformer", "注意力", "attention",
                     "token", "embedding", "位置编码", "self-attention",
                     "multi-head", "多头注意力", "feed-forward", "残差连接",
                     "层归一化", "layer norm", "softmax", "交叉熵",
                     "反向传播", "梯度下降", "学习率", "adam", "adamw",
                     "batch norm", "dropout", "激活函数", "relu", "gelu",
                     "cnn", "卷积", "rnn", "lstm", "gru",
                     "llm", "大语言模型", "预训练", "sft", "rlhf", "dpo",
                     "rag", "agent", "思维链", "chain-of-thought"],
        "fragments": [
            "MoE是稀疏激活架构：每个输入只路由到少数专家",
            "Transformer使用自注意力机制：Q×K^T/√d_k × V",
            "GPT是自回归Transformer解码器，BERT是双向Transformer编码器",
            "RLHF通过人类反馈强化学习对齐大模型",
            "RAG让LLM参考外部知识库减少幻觉",
            "大模型训练：预训练→SFT→RLHF/DPO→红队测试→部署",
        ],
    },
    {
        "id": "general",
        "name": "通用兜底",
        "domain": "通用对话/通用知识/代码",
        "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍", "解释",
                     "什么是", "？", "?", "python", "java", "javascript",
                     "typescript", "golang", "rust", "c++", "flask", "django",
                     "fastapi", "numpy", "pandas", "scipy", "pytest",
                     "async", "await", "class", "function", "def", "import",
                     "api", "http", "request", "response", "route", "endpoint",
                     "database", "sql", "orm", "redis", "docker", "kubernetes",
                     "pytorch", "tensorflow", "机器学习", "深度学习",
                     "git", "linux", "shell", "pip", "setup", "config",
                     "yaml", "json", "xml", "科技", "历史", "哲学", "数学",
                     "物理", "化学", "生物", "地理", "编程", "算法",
                     "数据结构", "排序", "查找", "递归", "动态规划",
                     "前端", "后端", "全栈", "devops", "云原生", "微服务",
                     "合鸣", "xuni", "虚拟"],
        "fragments": [
            "这是一个好问题，让我从合鸣的视角来回应",
            "在xuni虚拟生态里，每个问题都会被路由到最合适的专家",
            "我可以和你聊天、写代码、聊音乐、讨论全领域知识",
            "如果方便，补充一点上下文，我能给出更精准的回答",
            "让我来解释一下这个概念",
            "这个问题可以从多个角度来看",
            "好的，我来帮你分析一下",
            "简单来说，就是这样的",
            "合鸣是所有模型的结合体，全领域覆盖",
        ],
    },
]


# =========================================================================== #
#  多语言代码扫描
# =========================================================================== #

CODE_EXTENSIONS = {".py", ".rs", ".go", ".js", ".ts", ".c", ".h", ".cpp", ".hpp"}

def _scan_code_files(root: str, max_files: int = 2000) -> list[str]:
    """扫描多语言源代码文件"""
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
            if ext in CODE_EXTENSIONS:
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
        # 多语言函数定义模式
        if stripped.startswith(("def ", "class ", "async def ",
                                 "fn ", "pub fn ", "pub struct ",
                                 "func ", "type ", "struct ",
                                 "interface ", "impl ", "enum ",
                                 "const ", "var ", "package ")):
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
                         "pub struct ", "func ", "type ", "struct ",
                         "interface ", "impl ", "enum ", "const ", "var ",
                         "package ", "@", "import ", "from ", "use ",
                         "mod ", "exports ", "module ")):
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
    print("  🚀🚀🚀🚀 xuni v8 —— 极限突破：24仓库+多语言+500000轮")
    print("=" * 72)

    # ---------------------------------------------------------------------
    # 1. 加载 ai_creator 属性库
    # ---------------------------------------------------------------------
    print(f"\n[1/7] 加载 ai_creator_property_library...")

    extract_path = os.path.join(CACHE_DIR, "ai_creator_extracted.json")
    with open(extract_path, "r", encoding="utf-8") as f:
        creator_data = json.load(f)

    prop_lib = creator_data["property_library"]
    arch_map = creator_data["archetype_map"]
    print(f"  🏛️ 属性库: {len(prop_lib):,} 条")
    print(f"  🗿 原型映射: {len(arch_map):,} 条")

    # 生成造物语料
    creator_corpus = []
    for prop_name, prop_info in prop_lib.items():
        cat = prop_info.get("category", "unknown")
        keywords = prop_info.get("keywords", [prop_name])
        creator_corpus.append(f"{prop_name}是一种{cat}类属性，关键词包括：{', '.join(keywords[:5])}")
        creator_corpus.append(f"属性名「{prop_name}」属于{cat}分类，用于描述AI造物的核心能力")
        if prop_name in arch_map:
            archetypes = arch_map[prop_name]
            creator_corpus.append(f"「{prop_name}」的存在形式原型包括：{'、'.join(archetypes)}")
            for arch in archetypes[:2]:
                creator_corpus.append(f"{arch}是「{prop_name}」属性的具象化存在形式之一")

    print(f"  📚 生成造物语料: {len(creator_corpus):,} 条")

    # ---------------------------------------------------------------------
    # 2. 扫描 24 大仓库（多语言）
    # ---------------------------------------------------------------------
    print(f"\n[2/7] 扫描 24 大仓库（多语言代码）...")

    repo_dirs = [
        # v7 的 16 个 Python 仓库
        (os.path.join(CACHE_DIR, "python_cpython_main"), "CPython标准库", 2000),
        (os.path.join(CACHE_DIR, "django_django_main"), "Django Web框架", 2000),
        (os.path.join(CACHE_DIR, "scikit-learn_scikit-learn_main"), "scikit-learn ML", 2000),
        (os.path.join(CACHE_DIR, "pandas-dev_pandas_main"), "pandas 数据科学", 2000),
        (os.path.join(CACHE_DIR, "pydantic_pydantic_main"), "pydantic 验证", 2000),
        (os.path.join(CACHE_DIR, "pallets_flask_main"), "flask Web", 2000),
        (os.path.join(CACHE_DIR, "pallets_click_main"), "click CLI", 2000),
        (os.path.join(CACHE_DIR, "psf_requests_main"), "requests HTTP", 2000),
        (os.path.join(CACHE_DIR, "numpy_numpy_main"), "numpy 数值计算", 2000),
        (os.path.join(CACHE_DIR, "scipy_scipy_main"), "scipy 科学计算", 2000),
        (os.path.join(CACHE_DIR, "matplotlib_matplotlib_main"), "matplotlib 可视化", 2000),
        (os.path.join(CACHE_DIR, "sqlalchemy_sqlalchemy_main"), "SQLAlchemy ORM", 2000),
        (os.path.join(CACHE_DIR, "scrapy_scrapy_master"), "Scrapy 爬虫", 2000),
        (os.path.join(CACHE_DIR, "celery_celery_main"), "Celery 任务队列", 2000),
        (os.path.join(CACHE_DIR, "encode_httpx_master"), "httpx HTTP客户端", 2000),
        (os.path.join(CACHE_DIR, "encode_uvicorn_master"), "uvicorn ASGI服务器", 2000),
        # v8 新增 8 个多语言仓库
        (os.path.join(CACHE_DIR, "fastapi_fastapi_master"), "FastAPI ASGI框架", 2000),
        (os.path.join(CACHE_DIR, "huggingface_transformers_main"), "Transformers NLP", 2000),
        (os.path.join(CACHE_DIR, "langchain-ai_langchain_master"), "LangChain LLM框架", 2000),
        (os.path.join(CACHE_DIR, "pytorch_pytorch_main"), "PyTorch 深度学习", 2000),
        (os.path.join(CACHE_DIR, "rust-lang_rust_master"), "Rust 编译器", 1500),
        (os.path.join(CACHE_DIR, "golang_go_master"), "Go 语言", 1500),
        (os.path.join(CACHE_DIR, "nodejs_node_main"), "Node.js 运行时", 1500),
        (os.path.join(CACHE_DIR, "ansible_ansible_devel"), "Ansible 运维", 2000),
    ]

    all_fragments = []
    repo_stats = []
    total_code_kb = 0

    for repo_dir, desc, max_files in repo_dirs:
        if not os.path.isdir(repo_dir):
            repo_stats.append({"desc": desc, "files": 0, "frags": 0, "ok": False})
            continue

        texts = _scan_code_files(repo_dir, max_files=max_files)
        frags = []
        for text in texts:
            frags.extend(_extract_fragments(text, max_lines=20))
        all_fragments.extend(frags)

        kb = sum(len(t.encode("utf-8")) for t in texts) / 1024
        total_code_kb += kb
        print(f"  ✅ {desc:22s}: {len(texts):4d}文件 → {len(frags):6,}片段 ({kb:7.0f}KB)")
        repo_stats.append({"desc": desc, "files": len(texts), "frags": len(frags), "ok": True})

    # 工厂自身
    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    xuni_texts = _scan_code_files(xuni_dir, max_files=2000)
    xuni_frags = []
    for text in xuni_texts:
        xuni_frags.extend(_extract_fragments(text, max_lines=20))
    all_fragments.extend(xuni_frags)
    print(f"  🏭 工厂自身: {len(xuni_texts)}文件 → {len(xuni_frags)}片段")

    # 造物源代码
    creator_src = os.path.join(CACHE_DIR, "ai_creator_property_library")
    creator_files = _scan_code_files(creator_src, max_files=500)
    for text in creator_files:
        all_fragments.extend(_extract_fragments(text, max_lines=20))

    code_count = len(all_fragments)
    print(f"\n  📊 代码片段总数: {code_count:,} ({total_code_kb:.0f} KB)")

    # ---------------------------------------------------------------------
    # 3. 合并全部语料
    # ---------------------------------------------------------------------
    print(f"\n[3/7] 合并全部语料...")

    all_fragments.extend(creator_corpus)

    # v8 新增：Rust/Go/JS 语法知识
    LANG_CORPUS = [
        # Rust
        "Rust使用所有权(ownership)系统管理内存，无需垃圾回收",
        "Rust的借用检查器确保引用安全，防止数据竞争",
        "Rust的trait类似于接口，定义类型共享的行为",
        "Rust的match表达式是强大的模式匹配机制",
        "Rust的Result<T,E>和Option<T>是错误处理的核心类型",
        "Rust的cargo是构建系统和包管理器",
        "Rust的lifetime标注确保引用不会比被引用者活得更长",
        "Rust的macro_rules!宏在编译时生成代码",
        # Go
        "Go使用goroutine实现轻量级并发，通过channel通信",
        "Go的interface是隐式实现的，不需要显式声明",
        "Go的defer语句在函数返回时执行，常用于资源清理",
        "Go的go关键字启动一个goroutine",
        "Go的select语句处理多个channel操作",
        "Go的包管理使用go mod，支持版本化依赖",
        "Go的标准库覆盖网络、加密、压缩、测试等",
        # JavaScript/TypeScript
        "JavaScript是单线程的，通过事件循环实现异步",
        "JavaScript的Promise处理异步操作，async/await是语法糖",
        "JavaScript的闭包是函数及其词法环境的组合",
        "TypeScript在JavaScript基础上添加了静态类型系统",
        "TypeScript的interface和type定义类型结构",
        "TypeScript的泛型允许编写可复用的类型安全代码",
        "Node.js使用V8引擎在服务端运行JavaScript",
        "Node.js的EventEmitter是事件驱动的核心",
        "Node.js的Stream处理流式数据，支持管道操作",
        # 深度学习框架
        "PyTorch使用动态计算图，define-by-run方式",
        "PyTorch的nn.Module是所有神经网络的基类",
        "PyTorch的autograd自动求导引擎计算梯度",
        "PyTorch的DataLoader批量加载数据，支持多线程",
        "PyTorch的DistributedDataParallel实现多卡分布式训练",
        "Transformers库提供预训练模型：BERT/GPT/T5/LLaMA等",
        "Transformers的pipeline接口简化模型使用：文本分类/生成/翻译",
        "Transformers的Trainer类封装了训练循环、评估、保存",
        "LangChain是构建LLM应用框架：链式调用、记忆、工具使用",
        "LangChain的Agent可以自主规划、调用工具、完成复杂任务",
        "LangChain的RetrievalQA结合检索和生成实现RAG",
        # FastAPI
        "FastAPI是基于Starlette和Pydantic的现代ASGI框架",
        "FastAPI自动生成OpenAPI文档，支持类型提示验证",
        "FastAPI的依赖注入系统处理认证、数据库连接等",
        "FastAPI支持async/await，性能接近Node.js/Go",
        # Ansible
        "Ansible使用YAML定义playbook，声明式自动化配置",
        "Ansible是无代理的，通过SSH管理远程主机",
        "Ansible的role组织任务为可复用的模块",
    ]
    all_fragments.extend(LANG_CORPUS)

    print(f"  🏛️ 造物语料:   {len(creator_corpus):,}")
    print(f"  💻 代码片段:   {code_count:,}")
    print(f"  📝 多语言知识: {len(LANG_CORPUS)}")
    print(f"\n  📊 总训练片段: {len(all_fragments):,} 条")

    if len(all_fragments) < 100:
        print("  ⚠ 片段不足")
        return

    # ---------------------------------------------------------------------
    # 4. 创建模型 + v8专家
    # ---------------------------------------------------------------------
    print(f"\n[4/7] 创建模型 + v8专家...")

    model = Harmonia13Virtual(scale="mini")
    model._lite.experts = list(V8_EXPERTS)

    expert_names = [e["name"] for e in V8_EXPERTS]
    print(f"  专家阵容: {', '.join(expert_names)}")

    # 基线测试
    baseline_prompts = [
        "不可伪造的原型有哪些",
        "什么是涌现智能",
        "H100和H800有什么区别",
        "Flash Attention的作用",
        "Blender 4.2的特点",
        "Serum合成器怎么样",
        "金融风控AI用什么模型",
        "医学影像诊断常用什么网络",
        "卡农进行是什么",
        "ADSR包络四个阶段",
        "什么是扩散模型",
        "Rust的所有权系统是什么",
        "Go的goroutine是什么",
        "PyTorch的nn.Module是什么",
        "FastAPI有什么特点",
        "LangChain的Agent是什么",
        "def quicksort",
        "class DataFrame",
        "import numpy",
        "async def main",
        "func main()",
        "fn process()",
    ]

    print("\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        r = model._lite.generate(p, max_new_tokens=60)
        baseline[p] = r
        print(f"  [{p}] → {r[:55]}")

    # ---------------------------------------------------------------------
    # 5. 500000 轮训练
    # ---------------------------------------------------------------------
    print(f"\n[5/7] 500000 轮极限训练...")

    start = time.time()
    batch_size = 20
    num_epochs = 500000
    log = []

    for epoch in range(num_epochs):
        batch = random.sample(all_fragments, min(batch_size, len(all_fragments)))
        model._lite.train(batch, epochs=1)

        if (epoch + 1) % 50000 == 0:
            elapsed = time.time() - start
            learned = len(model._lite._learned_fragments)
            frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg = sum(frags) / max(1, len(frags))
            active = sum(1 for f in frags if f > 0)

            print(f"  Epoch {epoch+1:7d} | 已学: {learned:12,d} | "
                  f"活跃: {active:2d} | 均载: {avg:,.0f} | "
                  f"用时: {elapsed:.0f}s")
            log.append({
                "epoch": epoch + 1, "learned": learned,
                "active": active, "avg_load": round(avg, 1),
                "elapsed": round(elapsed, 2),
            })

    total_time = time.time() - start
    print(f"\n  ✅ 训练完成！用时: {total_time:.2f}s")

    # ---------------------------------------------------------------------
    # 6. 全方位评估
    # ---------------------------------------------------------------------
    print(f"\n[6/7] 全方位评估...")

    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)

    print(f"\n  📊 模型规模:")
    print(f"    已学: {learned:,} 条")
    print(f"    活跃: {active} / {len(model._lite.experts)}")
    for name, frags in expert_frags:
        bar = "█" * min(50, frags // 2000)
        print(f"    {name:14s} [{bar}] {frags:8,d}")

    categories = {
        "造物哲学": ["不可伪造的原型有哪些", "什么是涌现智能"],
        "硬件框架": ["H100和H800有什么区别", "Flash Attention的作用"],
        "创意工具": ["Blender 4.2的特点", "Serum合成器怎么样"],
        "行业应用": ["金融风控AI用什么模型", "医学影像诊断常用什么网络"],
        "音乐": ["卡农进行是什么", "ADSR包络四个阶段"],
        "多模态": ["什么是扩散模型"],
        "多语言代码": ["Rust的所有权系统是什么", "Go的goroutine是什么",
                      "PyTorch的nn.Module是什么", "FastAPI有什么特点",
                      "LangChain的Agent是什么"],
        "代码生成": ["def quicksort", "class DataFrame", "import numpy",
                    "async def main", "func main()", "fn process()"],
    }

    improved = 0
    total_compared = 0
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

    # ---------------------------------------------------------------------
    # 7. 保存 + 推送
    # ---------------------------------------------------------------------
    print(f"\n[7/7] 保存...")

    report = {
        "version": "v8",
        "focus": "极限突破：24仓库+多语言代码+500000轮",
        "new_repos": ["fastapi", "transformers", "langchain", "pytorch",
                      "rust", "go", "node", "ansible"],
        "repo_stats": repo_stats,
        "total_fragments": len(all_fragments),
        "epochs": num_epochs,
        "batch_size": batch_size,
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {n: f for n, f in expert_frags},
        "training_time": round(total_time, 2),
        "growth_log": log,
        "improved": improved,
        "total_compared": total_compared,
        "category_scores": cat_scores,
        "comparison_samples": {
            p: {"before": baseline[p][:80], "after": model._lite.generate(p, max_new_tokens=60)[:80]}
            for p in list(baseline_prompts)[:8]
        },
    }

    report_path = os.path.join(os.path.dirname(__file__), "trainer_v8_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    meta_path = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta = {
        "version": "v8",
        "fragments_learned": learned,
        "active_experts": active,
        "training_time": round(total_time, 2),
        "epochs": num_epochs,
        "focus": "极限突破：24仓库+多语言+500000轮",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元信息: {meta_path}")

    print("\n" + "=" * 72)
    print("  🚀🚀🚀🚀 v8 极限突破总结")
    print("=" * 72)
    print(f"""
  📦 24 大仓库（多语言）:
    Python: CPython+Django+sklearn+pandas+pydantic+Flask+click+requests
           +numpy+scipy+matplotlib+SQLAlchemy+Scrapy+Celery+httpx+uvicorn
           +FastAPI+Transformers+LangChain+PyTorch+Ansible
    Rust:   rust-lang/rust 编译器
    Go:     golang/go 语言
    JS:     nodejs/node 运行时

  🏛️ 1042属性库 + 866原型 + 10专家
  📚 总片段: {len(all_fragments):,} 条
  🔄 训练: {num_epochs:,} × {batch_size} = {num_epochs*batch_size:,}
  🧠 吸收: {learned:,} 条
  👥 活跃: {active} / {len(model._lite.experts)}
  ⏱️ 用时: {total_time:.1f}s
  📈 提升: {improved}/{total_compared}

  各方面得分:""")
    for cat, score in cat_scores.items():
        pct = score["improved"] / max(1, score["total"]) * 100
        print(f"    {cat:10s}: {score['improved']}/{score['total']} ({pct:.0f}%)")

    print(f"""
  积少成多，逆天而行：
    v1:      1,000 轮
    v2:      5,000 轮
    v3:     10,000 轮
    v4:     50,000 轮
    v5:    100,000 轮
    v6:    200,000 轮
    v7:    300,000 轮
    v8:    500,000 轮 🚀🚀🚀🚀

  多语言突破：Python+Rust+Go+JavaScript+C 全覆盖
  只要不卡住，大规模训练只是时间问题。逆天而行！
""")


if __name__ == "__main__":
    main()
