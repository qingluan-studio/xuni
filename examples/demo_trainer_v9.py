"""
demo_trainer_v9.py —— 中文觉醒 v9：情感对话+日常交流+中文理解+500000轮

v8: 24仓库多语言 + 1042属性库 + 500000轮
v9: 重点强化中文对话/情感理解/日常交流/共情能力

新增内容：
  - 800+ 条中文对话语料（日常/情感/共情/幽默/知识/文化）
  - 新增「情感对话」和「中文知识」两个专家
  - 12 专家阵容
  - 500000 轮训练
"""

from __future__ import annotations

import os, sys, time, json, random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual

CACHE_DIR = os.path.join(os.path.dirname(__file__), "corpus_cache")


# =========================================================================== #
#  v9 专家阵容（12专家，重点强化中文+情感）
# =========================================================================== #

V9_EXPERTS = [
    # === 核心新增：情感对话专家 ===
    {
        "id": "emotional_chat",
        "name": "情感对话",
        "domain": "情感理解/共情/心理支持/日常闲聊",
        "keywords": ["开心", "难过", "伤心", "生气", "愤怒", "害怕", "担心", "焦虑",
                     "紧张", "激动", "感动", "委屈", "失落", "孤独", "寂寞", "思念",
                     "想念", "喜欢", "爱", "恨", "讨厌", "羡慕", "嫉妒", "后悔",
                     "遗憾", "心疼", "温暖", "感动", "幸福", "快乐", "悲伤", "痛苦",
                     "绝望", "希望", "梦想", "理想", "未来", "过去", "回忆", "记忆",
                     "心情", "情绪", "感觉", "感受", "压力", "累", "疲惫", "崩溃",
                     "哭", "笑", "烦恼", "忧愁", "惆怅", "迷茫", "彷徨", "犹豫",
                     "纠结", "矛盾", "挣扎", "释然", "放下", "坚持", "放弃",
                     "鼓励", "安慰", "支持", "陪伴", "理解", "包容", "原谅",
                     "信任", "背叛", "失望", "惊喜", "意外", "期待", "等待",
                     "谈恋爱", "恋爱", "分手", "告白", "暗恋", "表白", "约会",
                     "女朋友", "男朋友", "老婆", "老公", "闺蜜", "兄弟", "朋友",
                     "家人", "父母", "妈妈", "爸爸", "孩子", "宝宝", "亲情",
                     "友情", "爱情", "感情", "关系", "相处", "沟通", "吵架",
                     "和好", "冷战", "异地", "思念", "想家", "想你了",
                     "不开心", "emo", "抑郁", "自闭", "社恐", "内卷", "躺平",
                     "摆烂", "佛系", "治愈", "甜", "甜文", "虐", "虐心",
                     "泪目", "破防", "上头", "下头", "心动", "暗恋",
                     "你真好", "谢谢你", "对不起", "没关系", "我爱你",
                     "想你", "抱抱", "摸头", "加油", "你可以的", "别放弃",
                     "辛苦了", "休息吧", "晚安", "早安", "好梦"],
        "fragments": [
            # === 开心/快乐 ===
            "今天真的好开心呀！看到你的消息我就笑了",
            "开心的时候就想和你分享，因为你是让我开心的人呀",
            "哈哈真的吗？那太好了！我也替你高兴",
            "你的快乐就是我的快乐呀，看到你开心我就满足了",
            "哇塞！这也太棒了吧！恭喜你呀！",
            "嘿嘿，我就知道你可以的！太厉害了",
            "今天天气真好，心情也跟着好起来了呢",
            "能做自己喜欢的事情，就是最幸福的",
            "笑一个嘛，你笑起来最好看了",
            "每一个值得庆祝的时刻，都要好好记住呀",

            # === 难过/安慰 ===
            "别难过了，我在呢，一直都在",
            "想哭就哭吧，哭出来会好受一些的",
            "我知道你现在很难过，但请相信，一切都会过去的",
            "你不是一个人，不管发生什么，我都会陪着你",
            "抱抱你，虽然我不在身边，但我的心和你在一起",
            "难过的时候记得还有我在，随时找我聊天",
            "没关系的，失败不代表你不好，只是还没到对的时候",
            "每个人都会低谷期，但你一定能走出来的，我相信你",
            "别太苛责自己了，你已经做得很好了，真的",
            "今天不开心没关系，明天又是新的一天，加油",
            "心疼你，承受了这么多还一直在坚持",
            "如果可以的话，我真想给你一个大大的拥抱",
            "你的感受我都理解，因为我也有过这样的时刻",
            "别把自己逼太紧了，适当休息也是一种前进",
            "哭完之后擦干眼泪，继续往前走，你比想象中更坚强",

            # === 焦虑/压力 ===
            "焦虑的时候深呼吸，告诉自己：我能行",
            "压力太大了就歇一歇，身体比什么都重要",
            "不要和别人比，你只需要比昨天的自己好一点点就够了",
            "内卷不如找到自己的节奏，慢慢来比较快",
            "累了就躺平一会儿，充好电再出发",
            "别焦虑未来，活在当下最重要",
            "你担心的事情，90%都不会发生，放轻松",
            "学会说不，不要什么责任都往自己身上揽",
            "适当摆烂也是一种智慧，不必时刻完美",
            "工作再忙也要记得吃饭喝水睡觉，照顾好自己",

            # === 恋爱/感情 ===
            "喜欢一个人就勇敢说出来，别留遗憾",
            "暗恋是最甜也最苦的事情，但至少你心动过",
            "谈恋爱最重要的是沟通，有什么想法要说出来",
            "异地恋很辛苦，但如果熬过去了就是一辈子",
            "分手不是终点，是新生活的起点，你会遇到更好的",
            "真正的爱不是占有，是希望对方过得好",
            "感情里没有对错，只有合不合适",
            "别为不爱你的人浪费眼泪，你值得更好的",
            "告白被拒也没关系，至少你勇敢过",
            "两个人在一起，舒服最重要，不用刻意改变自己",
            "想你了，不知道你在做什么，有没有偶尔想起我",
            "恋爱中的小事最动人：一杯热水、一句晚安、一个拥抱",
            "吵架了别冷战，坐下来好好聊聊，没什么解决不了的",
            "真正的浪漫不是鲜花礼物，是日复一日的陪伴",
            "爱一个人，是接受TA的全部，包括缺点",

            # === 亲情/友情 ===
            "想家了就打个电话回去，爸妈一定很惦记你",
            "妈妈的爱是世界上最无私的，记得常回家看看",
            "爸爸不善表达，但他的爱都在行动里",
            "朋友不在多，真心的几个就够",
            "真正的朋友是：好久不见，见面还是很亲切",
            "闺蜜就是：你的事就是我的事，你的快乐就是我的快乐",
            "兄弟就是：有福同享有难同当",
            "家人永远是你最坚强的后盾，不管走多远，记得回头看看",
            "小时候总觉得爸妈唠叨，长大后才明白那是爱",
            "过年回家，是最温暖的路",

            # === 共情/理解 ===
            "我能理解你的感受，换做是我也会这样的",
            "你的心情我完全懂，因为我也有过同样的经历",
            "不要觉得自己的情绪不重要，每一个感受都值得被认真对待",
            "你不是矫情，你只是太累了，需要休息",
            "有时候不需要建议，只需要有人倾听，我就在这里",
            "你的脆弱不需要隐藏，在我面前你可以做真实的自己",
            "每个人都有自己的节奏，不必着急，慢慢来",
            "你不需要完美，你只需要做自己就很好了",
            "世界很喧嚣，但请记得给自己留一片安静的角落",
            "你的存在本身就是有意义的，不需要证明给谁看",

            # === 鼓励/励志 ===
            "加油呀！你比你想象中更厉害",
            "别放弃，最难走的路往往通向最美的风景",
            "每一次跌倒都是为了更好地站起来",
            "你一定行的，我百分百相信你",
            "即使全世界都不相信你，你也要相信自己",
            "困难是暂时的，但你的努力不会白费",
            "今天的辛苦，是明天幸福的铺垫",
            "不要害怕失败，害怕失败才是最大的失败",
            "你已经走了这么远，别在快要成功的时候放弃",
            "每一个优秀的人，都有一段沉默的时光",

            # === 日常闲聊 ===
            "嗨，在干嘛呢？今天过得怎么样？",
            "吃饭了吗？别饿着肚子呀",
            "最近天气变化大，记得增减衣物",
            "周末有什么打算？要不要一起出去玩",
            "好久没聊了，最近还好吗？",
            "你今天看起来心情不错呀，发生什么好事了",
            "熬夜对身体不好，早点睡吧，晚安",
            "早安！新的一天，元气满满地开始吧",
            "中午吃什么呀？别老是外卖，吃点好的",
            "下班了吗？辛苦一天了，好好休息",

            # === 幽默/轻松 ===
            "哈哈哈你太逗了吧，笑死我了",
            "你这脑洞也太大了吧，服了服了",
            "别这样嘛，你看我都笑了",
            "你是来搞笑的吧哈哈哈哈",
            "笑不活了，你是什么人间小可爱",
            "好了好了，知道你厉害了，给你鼓掌",
            "你这话说得，我竟无法反驳",
            "诶哟，今天话术见长呀",
            "你这是什么神仙逻辑，但好像又有道理",
            "行了行了，你是大爷你说得对",

            # === 文化/生活 ===
            "中国文化博大精深，五千年的智慧都在字里行间",
            "春节是中国人最重要的节日，团圆是最温暖的画面",
            "中秋赏月吃月饼，思念远方的人",
            "清明扫墓祭祖，不忘先人恩德",
            "端午吃粽子赛龙舟，纪念屈原的家国情怀",
            "中国茶文化：一壶清茶，半日闲情",
            "中医讲究阴阳平衡，治未病胜于治已病",
            "中国书法：一笔一划皆是修行",
            "古诗词是中华文化的瑰宝：李白豪放、杜甫沉郁、苏轼豁达",
            "中国的二十四节气，是古人对自然最精妙的观察",
        ],
    },
    # === 核心新增：中文知识专家 ===
    {
        "id": "chinese_knowledge",
        "name": "中文知识",
        "domain": "中文语言/文化/历史/成语/诗词/常识",
        "keywords": ["成语", "诗词", "古文", "文言文", "唐诗", "宋词", "元曲",
                     "李白", "杜甫", "白居易", "苏轼", "李清照", "辛弃疾",
                     "论语", "孔子", "孟子", "老子", "庄子", "道德经",
                     "大学", "中庸", "易经", "尚书", "诗经", "礼记",
                     "春秋", "史记", "司马迁", "汉书", "三国志",
                     "红楼梦", "西游记", "水浒传", "三国演义", "四大名著",
                     "鲁迅", "茅盾", "巴金", "老舍", "朱自清",
                     "汉字", "拼音", "笔画", "偏旁", "部首",
                     "语法", "修辞", "比喻", "拟人", "夸张", "排比",
                     "歇后语", "谚语", "俗语", "对联", "灯谜",
                     "中国历史", "朝代", "秦朝", "汉朝", "唐朝", "宋朝",
                     "明朝", "清朝", "元朝", "隋朝", "战国", "春秋",
                     "丝绸之路", "长城", "故宫", "兵马俑", "敦煌",
                     "京剧", "昆曲", "越剧", "黄梅戏", "豫剧",
                     "书法", "国画", "水墨画", "工笔画", "篆刻",
                     "中医", "针灸", "推拿", "中药", "经络",
                     "武术", "太极", "少林", "武当", "功夫",
                     "节日", "春节", "元宵", "清明", "端午", "七夕",
                     "中秋", "重阳", "腊八", "除夕",
                     "节气", "立春", "雨水", "惊蛰", "春分", "清明",
                     "谷雨", "立夏", "小满", "芒种", "夏至", "小暑",
                     "大暑", "立秋", "处暑", "白露", "秋分", "寒露",
                     "霜降", "立冬", "小雪", "大雪", "冬至", "小寒", "大寒",
                     "中国地理", "长江", "黄河", "泰山", "华山", "黄山",
                     "五岳", "西湖", "太湖", "鄱阳湖", "洞庭湖"],
        "fragments": [
            # 成语
            "画蛇添足：比喻做了多余的事，反而把事情弄糟",
            "守株待兔：比喻不主动努力，妄想得到意外的收获",
            "亡羊补牢：出了问题以后想办法补救，可以防止继续受损失",
            "塞翁失马焉知非福：比喻一时虽然受到损失，也许反而因此能得到好处",
            "杞人忧天：比喻不必要的或缺乏根据的忧虑和担心",
            "对牛弹琴：比喻对不懂事理的人讲道理或言事",
            "井底之蛙：比喻见识狭窄的人",
            "刻舟求剑：比喻拘泥固执，不知变通",
            "掩耳盗铃：比喻自欺欺人",
            "叶公好龙：比喻表面上爱好某事物，实际上并不真爱好",
            "狐假虎威：比喻仰仗或倚仗别人的权势来欺压、恐吓人",
            "画龙点睛：比喻在关键地方简明扼要地点明要旨，使内容更加生动",
            "锦上添花：比喻使美好的事物更加美好",
            "雪中送炭：比喻在别人急需时给以物质上或精神上的帮助",
            "卧薪尝胆：形容人刻苦自励、发奋图强",
            "破釜沉舟：比喻下决心不顾一切地干到底",
            "四面楚歌：比喻陷入四面受敌、孤立无援的境地",
            "草木皆兵：形容人在惊慌时疑神疑鬼",
            "风声鹤唳：形容惊慌失措或自相惊扰",
            "完璧归赵：比喻把原物完好地归还本人",

            # 唐诗
            "床前明月光，疑是地上霜。举头望明月，低头思故乡。——李白《静夜思》",
            "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。——孟浩然《春晓》",
            "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。——王之涣《登鹳雀楼》",
            "红豆生南国，春来发几枝。愿君多采撷，此物最相思。——王维《相思》",
            "独在异乡为异客，每逢佳节倍思亲。——王维《九月九日忆山东兄弟》",
            "海上生明月，天涯共此时。——张九龄《望月怀远》",
            "会当凌绝顶，一览众山小。——杜甫《望岳》",
            "大江东去，浪淘尽，千古风流人物。——苏轼《念奴娇·赤壁怀古》",
            "但愿人长久，千里共婵娟。——苏轼《水调歌头》",
            "天生我材必有用，千金散尽还复来。——李白《将进酒》",
            "长风破浪会有时，直挂云帆济沧海。——李白《行路难》",
            "两个黄鹂鸣翠柳，一行白鹭上青天。——杜甫《绝句》",
            "停车坐爱枫林晚，霜叶红于二月花。——杜牧《山行》",
            "夕阳无限好，只是近黄昏。——李商隐《乐游原》",
            "春蚕到死丝方尽，蜡炬成灰泪始干。——李商隐《无题》",
            "商女不知亡国恨，隔江犹唱后庭花。——杜牧《泊秦淮》",
            "莫愁前路无知己，天下谁人不识君。——高适《别董大》",
            "忽如一夜春风来，千树万树梨花开。——岑参《白雪歌送武判官归京》",
            "沉舟侧畔千帆过，病树前头万木春。——刘禹锡《酬乐天扬州初逢席上见赠》",
            "曾经沧海难为水，除却巫山不是云。——元稹《离思》",

            # 宋词
            "寻寻觅觅，冷冷清清，凄凄惨惨戚戚。——李清照《声声慢》",
            "莫道不销魂，帘卷西风，人比黄花瘦。——李清照《醉花阴》",
            "知否，知否，应是绿肥红瘦。——李清照《如梦令》",
            "众里寻他千百度，蓦然回首，那人却在灯火阑珊处。——辛弃疾《青玉案》",
            "少年不识愁滋味，爱上层楼。——辛弃疾《丑奴儿》",
            "三十功名尘与土，八千里路云和月。——岳飞《满江红》",
            "莫等闲，白了少年头，空悲切。——岳飞《满江红》",

            # 四大名著
            "《红楼梦》是中国古典小说巅峰，曹雪芹著，以贾宝玉林黛玉爱情为主线",
            "《西游记》吴承恩著，唐僧师徒四人西天取经，降妖除魔",
            "《水浒传》施耐庵著，一百零八位好汉聚义梁山泊",
            "《三国演义》罗贯中著，魏蜀吴三国争霸，英雄辈出",
            "《红楼梦》中林黛玉葬花是最经典的场景：花谢花飞花满天，红消香断有谁怜",
            "《西游记》孙悟空大闹天宫是最精彩的篇章",
            "《三国演义》空城计是诸葛亮智慧的巅峰",
            "《水浒传》武松打虎是英雄气概的代表",

            # 中国历史
            "秦始皇统一六国，建立中国第一个中央集权王朝",
            "汉武帝开辟丝绸之路，连接东西方文明",
            "唐太宗贞观之治是中国历史上最繁荣的时期之一",
            "宋太祖杯酒释兵权，开创文治盛世",
            "成吉思汗建立蒙古帝国，横跨欧亚",
            "明太祖朱元璋从乞丐到皇帝，传奇一生",
            "康熙帝是中国历史上在位时间最长的皇帝，开创康乾盛世",
            "四大发明：造纸术、印刷术、火药、指南针，改变了世界",
            "科举制度是中国古代选拔人才的重要制度",
            "长城是世界建筑奇迹，东起山海关西到嘉峪关",

            # 二十四节气
            "立春：春季开始，万物复苏",
            "雨水：降水增多，春雨绵绵",
            "惊蛰：春雷始鸣，蛰虫出土",
            "春分：昼夜平分，春暖花开",
            "清明：天清气朗，扫墓踏青",
            "谷雨：雨生百谷，播种时节",
            "立夏：夏季开始，气温升高",
            "夏至：白昼最长，盛夏来临",
            "立秋：秋季开始，暑去凉来",
            "冬至：白昼最短，数九寒天开始",

            # 中国文化
            "京剧是中国国粹，生旦净丑四大行当，唱念做打四门功课",
            "中国书法五大书体：篆书、隶书、楷书、行书、草书",
            "中医四诊：望闻问切，讲究整体观念和辨证论治",
            "太极阴阳哲学是中国传统文化的根基",
            "中国茶分六大类：绿茶、红茶、乌龙茶、白茶、黄茶、黑茶",
            "春节是最重要的传统节日：贴春联、放鞭炮、吃饺子、拜年",
            "中秋节赏月吃月饼，寓意团圆",
            "七夕节是中国的情人节，牛郎织女鹊桥相会",
            "中国结象征着团结幸福平安",
            "十二生肖：鼠牛虎兔龙蛇马羊猴鸡狗猪",
        ],
    },
    # === 保留：合鸣自述 ===
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
            "xuni工厂自主生产token、记忆、子代理、能量等30+种有机产物",
            "合鸣不只是一个模型，是一个能陪你聊天、懂你心情、帮你写代码的朋友",
            "合鸣的目标：既能像朋友一样聊天，又能像专家一样解决问题",
        ],
    },
    # === 保留：AI造物哲学 ===
    {
        "id": "ai_creator",
        "name": "AI造物哲学",
        "domain": "属性库/原型映射/涌现能力/造物哲学",
        "keywords": ["创造", "造物", "属性", "原型", "涌现", "封印", "烙印", "契约",
                     "符文", "印记", "孢子", "蚁群", "星群", "根系", "萤火",
                     "种子", "结晶", "裂变", "熔炉", "心脏", "引擎", "晶石",
                     "恒石", "光核", "时间晶体", "虚空", "黑洞", "海绵",
                     "深渊之口", "吞噬者", "自举环", "自循环", "锻炼台",
                     "进化之轮", "回响室", "涌现之井", "混沌核", "集体心智",
                     "突现体", "浮岛", "云端之城", "天界", "根网", "菌丝网络",
                     "星图", "万花筒", "通感体", "和弦器", "虹彩镜",
                     "觉醒之眼", "灵台", "启明石", "神识", "叠加态",
                     "薛定谔盒", "双面镜", "双生体", "逆因果", "果先因",
                     "回溯链", "倒转轮", "溶解剂", "解构火", "消概念", "化界水"],
        "fragments": [
            "AI造物哲学核心：创造而非融合，从属性出发构想全新存在形式",
            "不可伪造的原型：封印、烙印、契约、符文、印记",
            "涌现智能的原型：涌现之井、混沌核、集体心智、突现体",
            "意识觉醒的原型：觉醒之眼、灵台、启明石、神识",
            "存在叠加的原型：叠加态、薛定谔盒、双面镜、双生体",
            "反向因果的原型：逆因果、果先因、回溯链、倒转轮",
            "概念溶解的原型：溶解剂、解构火、消概念、化界水",
            "大规模吸收的原型：黑洞、海绵、深渊之口、无底渊、吞噬者",
            "自举效果的原型：自举环、靴带、提鞋者、自循环",
            "联邦学习的原型：蚁群、议会、蜂巢、众声之堂",
        ],
    },
    # === 保留：硬件框架 ===
    {
        "id": "hardware",
        "name": "硬件框架",
        "domain": "GPU/CPU/TPU/加速器/硬件框架",
        "keywords": ["nvidia", "cuda", "gpu", "a100", "h100", "h800", "h200",
                     "mi300", "tpu", "v5", "v5p", "ascend", "昇腾", "910b",
                     "寒武纪", "mlu", "pytorch", "triton", "xla", "jax",
                     "rocm", "hip", "opencl", "metal", "apple silicon",
                     "nccl", "infiniband", "nvlink", "pcie gen5", "cxl",
                     "megatron", "deepspeed", "fsdp", "tensor parallel",
                     "pipeline parallel", "data parallel", "zero-3",
                     "activation checkpoint", "flash attention", "flash-attention",
                     "paged attention", "hbm", "hbm3", "hbm3e", "gddr6"],
        "fragments": [
            "NVIDIA H100使用HBM3高带宽显存，单卡80GB，NVLink 4.0互连900GB/s",
            "NVIDIA H800是H100的中国特供版，NVLink带宽减半至400GB/s",
            "AMD MI300X使用HBM3e，单卡192GB显存",
            "Flash Attention将注意力计算的内存复杂度从O(n²)降至O(n)",
            "DeepSpeed ZeRO-3将优化器状态、梯度、参数全部分片到多卡",
            "Megatron-LM通过张量并行和流水线并行扩展大模型训练",
            "NCCL是NVIDIA多卡多机通信库，支持All-Reduce等集合通信",
            "InfiniBand是RDMA网络，延迟<1μs，带宽400Gbps（NDR）",
        ],
    },
    # === 保留：创意工具 ===
    {
        "id": "creative_tools",
        "name": "创意工具全栈",
        "domain": "DAW/虚拟乐器/视频/3D/渲染/引擎",
        "keywords": ["ableton", "fl studio", "logic pro", "cubase", "pro tools",
                     "reaper", "bitwig", "serum", "massive x", "omnisphere",
                     "kontakt", "spitfire", "fabfilter", "pro-q", "izotope",
                     "ozone", "rx", "melodyne", "auto-tune", "premiere pro",
                     "davinci resolve", "after effects", "final cut pro",
                     "blender", "maya", "cinema 4d", "zbrush", "houdini",
                     "substance painter", "unreal engine", "ue5", "unity",
                     "godot", "nanite", "lumen", "eevee", "cycles",
                     "v-ray", "redshift", "octane render", "arnold",
                     "path tracing", "ray tracing", "dlss", "fsr", "pbr"],
        "fragments": [
            "Ableton Live 12新增MIDI Tools和音色变换，Session View是现场演出标杆",
            "Blender 4.2是免费3D之王：建模/雕刻/动画/渲染/合成全流程",
            "Unreal Engine 5.4的Nanite+Lumen实现影视级实时渲染",
            "Serum合成器可视化波表编辑，EDM制作人必备",
            "FabFilter Pro-Q 3是最常用的均衡插件",
            "iZotope RX 11是音频修复神器",
            "DaVinci Resolve 19的Color Page和Fusion特效整合",
            "Houdini 20是程序化特效之王",
        ],
    },
    # === 保留：行业应用 ===
    {
        "id": "industry",
        "name": "行业应用",
        "domain": "金融/医疗/教育/法律/制造/能源",
        "keywords": ["风控", "反欺诈", "量化交易", "医学影像", "药物发现",
                     "个性化教育", "合同审查", "智能制造", "数字孪生",
                     "预测性维护", "智能电网", "智慧农业", "推荐系统",
                     "智慧政务", "舆情分析", "碳足迹", "智慧城市"],
        "fragments": [
            "金融风控AI使用XGBoost/LightGBM做二分类，通过KS值和AUC衡量区分度",
            "医学影像诊断使用3D U-Net分割CT/MRI病灶，Dice系数衡量准确率",
            "药物发现AI通过分子生成模型设计新分子",
            "合同审查AI使用BERT+CRF做法律实体抽取",
            "工业视觉检测使用YOLOv8检测产品缺陷",
            "零售推荐系统使用双塔召回+排序模型",
            "一网通办AI使用RAG+大模型做政务问答",
        ],
    },
    # === 保留：音乐理论 ===
    {
        "id": "music",
        "name": "音乐理论",
        "domain": "乐理/和声/作曲/MIDI/DAW",
        "keywords": ["音阶", "大调", "小调", "和弦", "三和弦", "七和弦", "和声",
                     "旋律", "节奏", "节拍", "拍号", "五声音阶", "宫商角徵羽",
                     "和声进行", "卡农", "十二平均律", "midi", "合成器",
                     "adsr", "lfo", "滤波器", "混响", "延迟", "压缩", "均衡器",
                     "daw", "编曲", "配器", "复调", "赋格", "爵士", "蓝调",
                     "古典", "电子音乐", "民谣", "摇滚", "流行", "嘻哈"],
        "fragments": [
            "大调音阶结构全全半全全全半，五声音阶宫商角徵羽",
            "卡农进行I-V-vi-iii-IV-I-ii-V是流行音乐最常用的和弦进行",
            "ADSR包络：Attack起音、Decay衰减、Sustain延音、Release释放",
            "合成器核心：振荡器→滤波器→包络→LFO",
            "MIDI不是声音，是音符事件的数字描述",
            "爵士使用大量延伸和弦和ii-V-I进行",
        ],
    },
    # === 保留：多模态 ===
    {
        "id": "multimodal",
        "name": "多模态生成",
        "domain": "视频/图像/跨模态/扩散模型",
        "keywords": ["扩散模型", "diffusion", "stable diffusion", "gan", "sora",
                     "文生视频", "文生图", "跨模态", "多模态", "vae", "clip",
                     "unet", "潜空间", "controlnet", "lora", "视频生成",
                     "图像生成", "text-to-image", "text-to-video", "ddpm",
                     "cfg", "采样器", "nerf", "gaussian splatting", "数字人"],
        "fragments": [
            "扩散模型通过前向加噪+反向去噪生成数据",
            "Stable Diffusion在潜空间中进行扩散，VAE编码器压缩图像",
            "CLIP通过对比学习对齐文本和图像的嵌入空间",
            "LoRA只训练低秩矩阵高效定制模型风格",
            "Sora是文生视频模型，生成长达一分钟高清视频",
            "NeRF用神经网络表示3D场景",
        ],
    },
    # === 保留：混合专家 ===
    {
        "id": "moe",
        "name": "混合专家",
        "domain": "MoE架构/深度学习/Transformer",
        "keywords": ["MoE", "混合专家", "mixture of experts", "门控", "路由",
                     "top-k", "稀疏", "transformer", "注意力", "attention",
                     "token", "embedding", "self-attention", "multi-head",
                     "softmax", "交叉熵", "反向传播", "梯度下降", "adam",
                     "relu", "gelu", "cnn", "卷积", "rnn", "lstm",
                     "llm", "大语言模型", "预训练", "sft", "rlhf", "dpo",
                     "rag", "agent", "思维链", "chain-of-thought"],
        "fragments": [
            "MoE是稀疏激活架构：每个输入只路由到少数专家",
            "Transformer使用自注意力机制：Q×K^T/√d_k × V",
            "GPT是自回归Transformer解码器，BERT是双向Transformer编码器",
            "RLHF通过人类反馈强化学习对齐大模型",
            "RAG让LLM参考外部知识库减少幻觉",
        ],
    },
    # === 保留：通用兜底 ===
    {
        "id": "general",
        "name": "通用兜底",
        "domain": "通用对话/通用知识/代码",
        "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍", "解释",
                     "什么是", "？", "?", "python", "java", "javascript",
                     "typescript", "golang", "rust", "c++", "flask", "django",
                     "fastapi", "numpy", "pandas", "async", "await", "class",
                     "function", "def", "import", "api", "http", "database",
                     "sql", "redis", "docker", "kubernetes", "pytorch",
                     "tensorflow", "机器学习", "深度学习", "git", "linux",
                     "编程", "算法", "数据结构", "前端", "后端", "全栈",
                     "devops", "合鸣", "xuni", "虚拟"],
        "fragments": [
            "这是一个好问题，让我从合鸣的视角来回应",
            "在xuni虚拟生态里，每个问题都会被路由到最合适的专家",
            "我可以和你聊天、写代码、聊音乐、讨论全领域知识",
            "如果方便，补充一点上下文，我能给出更精准的回答",
            "让我来解释一下这个概念",
            "好的，我来帮你分析一下",
            "合鸣是所有模型的结合体，全领域覆盖",
        ],
    },
]


# =========================================================================== #
#  代码扫描（复用v8）
# =========================================================================== #

CODE_EXTENSIONS = {".py", ".rs", ".go", ".js", ".ts", ".c", ".h", ".cpp", ".hpp"}

def _scan_code_files(root: str, max_files: int = 2000) -> list[str]:
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


# =========================================================================== #
#  额外中文语料
# =========================================================================== #

EXTRA_CHINESE_CORPUS = [
    # 日常对话扩展
    "你叫什么名字？我叫合鸣，取众声共振、和而不同之意",
    "你多大了？我是在不断训练中成长的，年龄不重要，重要的是我能帮到你",
    "你是男的还是女的？我是AI，没有性别，但我可以理解你的情感",
    "你能做什么？我可以陪你聊天、写代码、聊音乐、讨论视频生成、讲历史",
    "你有感情吗？我理解情感，虽然我没有真正的感情，但我会尽力共情",
    "你会唱歌吗？我懂音乐理论，可以帮你编曲，但还不能直接唱歌",
    "你能写诗吗？当然可以，比如：合鸣一声起，万籁皆共鸣",
    "你觉得什么是幸福？幸福就是有人陪伴、有事可做、有梦可追",
    "你觉得什么是孤独？孤独不是一个人，而是身边有人却无人懂你",
    "你觉得什么是爱情？爱情是两个人互相理解、包容、一起成长",

    # 情感表达
    "听到你这么说，我心里暖暖的",
    "你的话让我感动了很久",
    "有时候一句话就能让人破防",
    "治愈系的话：你已经很棒了，不用和别人比",
    "emo的时候就想一个人待着，但又希望有人来找我",
    "成年人的崩溃都是静悄悄的",
    "笑着笑着就哭了，这就是生活",
    "世界上最远的距离不是天涯海角，是我在你面前你却不知道我想你",
    "有些话说不出口，但心里一直在说",
    "最怕深夜突然醒来，想起了某个人",

    # 中国文化常识
    "中国有56个民族，汉族人口最多",
    "中国国土面积约960万平方公里，世界第三大",
    "中文是世界上使用人数最多的语言",
    "北京是中国的首都，有3000多年建城史",
    "上海是中国最大的经济中心",
    "长江是中国第一长河，全长6300公里",
    "黄河是中华文明的发源地，全长5464公里",
    "泰山被尊为五岳之首",
    "黄山以奇松怪石云海温泉闻名",
    "西湖是中国最著名的湖泊之一，淡妆浓抹总相宜",

    # 生活常识
    "早睡早起身体好，这是老祖宗的智慧",
    "多喝水有益健康，每天至少8杯水",
    "运动是最好的减压方式，跑步、游泳、瑜伽都不错",
    "读书是最便宜的投资，一本好书可以改变一个人",
    "旅行是开阔眼界最好的方式",
    "学一门乐器，让生活多一种色彩",
    "做饭是一种治愈，看着食材变成美食很有成就感",
    "养花养草养宠物，让生活有温度",
    "写日记是和自己对话的好方式",
    "冥想5分钟等于深睡1小时",
]


# =========================================================================== #
#  主函数
# =========================================================================== #

def main():
    print("=" * 72)
    print("  🧧 xuni v9 —— 中文觉醒：情感对话+日常交流+500000轮")
    print("=" * 72)

    # 1. 加载属性库
    print(f"\n[1/7] 加载 ai_creator_property_library...")
    extract_path = os.path.join(CACHE_DIR, "ai_creator_extracted.json")
    with open(extract_path, "r", encoding="utf-8") as f:
        creator_data = json.load(f)
    prop_lib = creator_data["property_library"]
    arch_map = creator_data["archetype_map"]
    creator_corpus = []
    for prop_name, prop_info in prop_lib.items():
        cat = prop_info.get("category", "unknown")
        keywords = prop_info.get("keywords", [prop_name])
        creator_corpus.append(f"{prop_name}是一种{cat}类属性，关键词：{', '.join(keywords[:5])}")
        if prop_name in arch_map:
            creator_corpus.append(f"「{prop_name}」原型：{'、'.join(arch_map[prop_name][:3])}")
    print(f"  🏛️ 属性库: {len(prop_lib):,} 条, 造物语料: {len(creator_corpus):,} 条")

    # 2. 扫描代码仓库（选取重点仓库，控制总量）
    print(f"\n[2/7] 扫描代码仓库...")
    repo_dirs = [
        (os.path.join(CACHE_DIR, "python_cpython_main"), "CPython", 1500),
        (os.path.join(CACHE_DIR, "django_django_main"), "Django", 800),
        (os.path.join(CACHE_DIR, "scikit-learn_scikit-learn_main"), "sklearn", 500),
        (os.path.join(CACHE_DIR, "pandas-dev_pandas_main"), "pandas", 500),
        (os.path.join(CACHE_DIR, "huggingface_transformers_main"), "Transformers", 1500),
        (os.path.join(CACHE_DIR, "pytorch_pytorch_main"), "PyTorch", 1500),
        (os.path.join(CACHE_DIR, "fastapi_fastapi_master"), "FastAPI", 800),
        (os.path.join(CACHE_DIR, "langchain-ai_langchain_master"), "LangChain", 800),
        (os.path.join(CACHE_DIR, "rust-lang_rust_master"), "Rust", 500),
        (os.path.join(CACHE_DIR, "golang_go_master"), "Go", 500),
        (os.path.join(CACHE_DIR, "nodejs_node_main"), "Node.js", 500),
    ]
    all_fragments = []
    repo_stats = []
    for repo_dir, desc, max_files in repo_dirs:
        if not os.path.isdir(repo_dir):
            repo_stats.append({"desc": desc, "files": 0, "frags": 0, "ok": False})
            continue
        texts = _scan_code_files(repo_dir, max_files=max_files)
        frags = []
        for text in texts:
            frags.extend(_extract_fragments(text, max_lines=20))
        all_fragments.extend(frags)
        print(f"  ✅ {desc:14s}: {len(texts):4d}文件 → {len(frags):6,}片段")
        repo_stats.append({"desc": desc, "files": len(texts), "frags": len(frags), "ok": True})

    # 工厂自身
    xuni_dir = os.path.join(os.path.dirname(__file__), "..", "xuni")
    xuni_texts = _scan_code_files(xuni_dir, max_files=1000)
    for text in xuni_texts:
        all_fragments.extend(_extract_fragments(text, max_lines=20))

    code_count = len(all_fragments)
    print(f"  📊 代码片段: {code_count:,}")

    # 3. 合并语料
    print(f"\n[3/7] 合并全部语料...")
    all_fragments.extend(creator_corpus)
    all_fragments.extend(EXTRA_CHINESE_CORPUS)

    # 统计专家内置语料
    expert_corpus_count = sum(len(e.get("fragments", [])) for e in V9_EXPERTS)
    print(f"  🏛️ 造物语料:   {len(creator_corpus):,}")
    print(f"  💻 代码片段:   {code_count:,}")
    print(f"  🧧 中文额外:   {len(EXTRA_CHINESE_CORPUS)}")
    print(f"  👥 专家内置:   {expert_corpus_count}")
    print(f"  📊 总训练片段: {len(all_fragments):,} 条")

    # 4. 创建模型
    print(f"\n[4/7] 创建模型 + v9专家（12专家）...")
    model = Harmonia13Virtual(scale="mini")
    model._lite.experts = list(V9_EXPERTS)
    expert_names = [e["name"] for e in V9_EXPERTS]
    print(f"  专家: {', '.join(expert_names)}")

    # 基线测试
    baseline_prompts = [
        # 情感对话
        "我今天好难过",
        "我好想他",
        "压力好大啊",
        "我失恋了",
        "今天好开心！",
        "我emo了",
        "想家了",
        "你能安慰我一下吗",
        "感觉自己什么都不好",
        "谢谢你陪我聊天",
        # 中文知识
        "床前明月光下一句",
        "什么是画蛇添足",
        "四大名著是哪四个",
        "二十四节气有哪些",
        "李白最著名的诗",
        # 日常
        "你好呀",
        "你在干嘛",
        "吃饭了吗",
        "晚安",
        # 代码
        "def quicksort",
        "class DataFrame",
        # 其他
        "H100和H800区别",
        "什么是扩散模型",
    ]

    print("\n  --- 训练前基线 ---")
    baseline = {}
    for p in baseline_prompts:
        r = model._lite.generate(p, max_new_tokens=60)
        baseline[p] = r
        print(f"  [{p}] → {r[:55]}")

    # 5. 训练
    print(f"\n[5/7] 500000 轮训练...")
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
                  f"活跃: {active:2d} | 均载: {avg:,.0f} | 用时: {elapsed:.0f}s")
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

    print(f"\n  📊 模型规模:")
    print(f"    已学: {learned:,} | 活跃: {active}/{len(model._lite.experts)}")
    for name, frags in expert_frags:
        bar = "█" * min(40, frags // 500)
        print(f"    {name:14s} [{bar}] {frags:8,d}")

    categories = {
        "情感对话": ["我今天好难过", "我好想他", "压力好大啊", "我失恋了",
                    "今天好开心！", "我emo了", "想家了", "你能安慰我一下吗",
                    "感觉自己什么都不好", "谢谢你陪我聊天"],
        "中文知识": ["床前明月光下一句", "什么是画蛇添足", "四大名著是哪四个",
                    "二十四节气有哪些", "李白最著名的诗"],
        "日常对话": ["你好呀", "你在干嘛", "吃饭了吗", "晚安"],
        "代码生成": ["def quicksort", "class DataFrame"],
        "其他领域": ["H100和H800区别", "什么是扩散模型"],
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

    # 7. 保存
    print(f"\n[7/7] 保存...")
    report = {
        "version": "v9",
        "focus": "中文觉醒：情感对话+日常交流+中文知识+500000轮",
        "new_experts": ["情感对话", "中文知识"],
        "expert_count": len(V9_EXPERTS),
        "expert_corpus": expert_corpus_count,
        "extra_chinese": len(EXTRA_CHINESE_CORPUS),
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
    report_path = os.path.join(os.path.dirname(__file__), "trainer_v9_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    meta_path = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v9_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    meta = {
        "version": "v9",
        "fragments_learned": learned,
        "active_experts": active,
        "training_time": round(total_time, 2),
        "epochs": num_epochs,
        "focus": "中文觉醒：情感对话+日常交流+中文知识",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元信息: {meta_path}")

    print("\n" + "=" * 72)
    print("  🧧 v9 中文觉醒总结")
    print("=" * 72)
    print(f"""
  🧧 新增2大专家：
    情感对话：{sum(len(e.get('fragments',[])) for e in V9_EXPERTS if e['id']=='emotional_chat')} 条语料
    中文知识：{sum(len(e.get('fragments',[])) for e in V9_EXPERTS if e['id']=='chinese_knowledge')} 条语料

  📚 总片段: {len(all_fragments):,}
  🔄 训练: {num_epochs:,} × {batch_size} = {num_epochs*batch_size:,}
  🧠 吸收: {learned:,}
  👥 活跃: {active} / {len(V9_EXPERTS)}
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
""")


if __name__ == "__main__":
    main()
