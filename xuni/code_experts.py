"""
代码领域专家配置 + 语料分配器

把 13 位专家从"合鸣生态"分化为"代码领域"：
- 保留 harmonia/moe/dualstate/compute/general (核心概念不变)
- 替换 field/music/chaos/hydro/glass/credential/brain/philosophy 为代码领域专家

每个代码专家有专属关键词 + 专属语料仓库映射。
"""

from typing import Dict, List, Any


CODE_EXPERTS: List[Dict[str, Any]] = [
    {
        "id": "harmonia",
        "name": "合鸣自述者",
        "domain": "合鸣 / xuni 自我认知",
        "keywords": ["合鸣", "harmonia", "合鸣-13", "合鸣13", "xuni", "虚拟", "虚拟大模型"],
        "fragments": [
            "合鸣（Harmonia）是 xuni 虚拟生态中的旗舰对话模型，名取「众声共振、和而不同」之意",
            "合鸣-13 是一个由 13 位专家组成的混合专家（MoE）虚拟大模型，由虚拟电场能量驱动，不依赖任何外部真实算力",
            "合鸣lite 是合鸣-13 的轻量替代物：在粒子态训练时作为脚手架，让合鸣-13 能像真实模型一样被真正训练",
            "合鸣走非传统路线：不用 transformer，而是用检索 + n-gram 共振 + 场调制，完全免费、可在手机上运行",
        ],
        "repo_map": [],
    },
    {
        "id": "moe",
        "name": "混合专家",
        "domain": "MoE 架构",
        "keywords": ["MoE", "moe", "混合专家", "mixture of experts", "专家", "门控", "路由", "top-k", "topk", "稀疏"],
        "fragments": [
            "MoE（Mixture of Experts，混合专家）是一种稀疏激活架构：每个输入只路由到少数专家，从而以更少算力获得更大容量",
            "MoE 的关键两步是门控（gate）给每个专家打分，路由（routing）选出 top-k 专家并合并它们的输出",
            "合鸣-13 的门控不是神经网络，而是关键词共振：用提示词与每个专家的关键词集合求重叠，重叠越多得分越高",
            "MoE 的好处是容量大、计算省；难点是负载均衡与专家崩塌，合鸣用共振评分天然分散负载",
        ],
        "repo_map": [],
    },
    {
        "id": "web",
        "name": "Web 框架",
        "domain": "Web 框架 / API / 路由 / 中间件",
        "keywords": ["fastapi", "flask", "django", "starlette", "sanic", "falcon", "tornado",
                     "web 框架", "路由", "中间件", "request", "response", "api", "rest",
                     "endpoint", "view", "app", "应用", "框架"],
        "fragments": [],
        "repo_map": ["fastapi", "flask", "django", "starlette", "sanic", "falcon",
                      "tornado", "bottle", "werkzeug", "aiohttp"],
    },
    {
        "id": "data",
        "name": "数据科学",
        "domain": "数据处理 / 数值计算 / 数据分析",
        "keywords": ["numpy", "pandas", "scipy", "scikit", "sklearn", "scikit-learn",
                     "数组", "dataframe", "矩阵", "数值", "统计", "分析", "plot",
                     "图表", "可视化", "matplotlib", "seaborn", "bokeh", "plotly"],
        "fragments": [],
        "repo_map": ["numpy", "pandas", "scipy", "scikit-learn", "scikit-image",
                      "matplotlib", "seaborn", "bokeh", "plotly.py", "xarray",
                      "dask", "modin", "polars", "pyarrow", "numba", "sympy"],
    },
    {
        "id": "http",
        "name": "网络通信",
        "domain": "HTTP / 客户端 / 请求 / 响应",
        "keywords": ["requests", "httpx", "urllib3", "http", "https", "httpie",
                     "请求", "响应", "客户端", "client", "session", "get", "post",
                     "http client", "网络", "aiohttp", "urllib"],
        "fragments": [],
        "repo_map": ["requests", "requests-html", "httpx", "urllib3", "aiohttp", "httpie"],
    },
    {
        "id": "db",
        "name": "数据库",
        "domain": "数据库 / ORM / SQL / 缓存",
        "keywords": ["sqlalchemy", "sql", "orm", "数据库", "redis", "mongodb", "mongo",
                     "query", "查询", "表", "model", "迁移", "migration", "session",
                     "engine", "连接池", "缓存", "cache"],
        "fragments": [],
        "repo_map": ["sqlalchemy", "redis-py", "mongo-python-driver", "databases", "orm"],
    },
    {
        "id": "ml",
        "name": "机器学习",
        "domain": "机器学习 / 深度学习 / 模型 / 训练",
        "keywords": ["transformers", "pytorch", "tensorflow", "keras", "模型", "训练",
                     "深度学习", "神经网络", "llm", "大模型", "huggingface", "hugging face",
                     "dataset", "tokenizer", "加速", "accelerate", "pipeline",
                     "inference", "推理", "langchain"],
        "fragments": [],
        "repo_map": ["transformers", "datasets", "tokenizers", "accelerate",
                      "pytorch", "tensorflow", "keras", "ray", "spaCy", "rasa",
                      "scikit-learn", "scikit-image", "langchain", "openai-python"],
    },
    {
        "id": "tool",
        "name": "开发工具",
        "domain": "CLI / 测试 / 代码质量 / 打包",
        "keywords": ["click", "rich", "pytest", "black", "flake8", "tox", "coverage",
                     "测试", "单元测试", "cli", "命令行", "终端", "terminal",
                     "打包", "pip", "setuptools", "wheel", "build", "lint"],
        "fragments": [],
        "repo_map": ["click", "rich", "python-prompt-toolkit", "textual",
                      "pytest", "tox", "coverage", "black", "flake8",
                      "pip", "setuptools", "wheel", "build", "cpython"],
    },
    {
        "id": "async_task",
        "name": "异步任务",
        "domain": "异步 / 任务队列 / 并发",
        "keywords": ["celery", "rq", "任务队列", "异步", "async", "await", "并发",
                     "worker", "broker", "redis", "rabbitmq", "任务", "调度",
                     "scheduler", "定时", "uvicorn", "asgi", "dramatiq"],
        "fragments": [],
        "repo_map": ["celery", "rq", "dramatiq", "uvicorn", "aiohttp", "aiomysql", "aiopg"],
    },
    {
        "id": "crawler",
        "name": "爬虫自动化",
        "domain": "爬虫 / 自动化 / 解析",
        "keywords": ["scrapy", "爬虫", "crawler", "spider", "抓取", "解析", "beautifulsoup",
                     "bs4", "html", "selenium", "自动化", "scrape", "网页", "mechanicalsoup"],
        "fragments": [],
        "repo_map": ["scrapy", "requests-html", "beautifulsoup4", "mechanicalsoup", "selenium"],
    },
    {
        "id": "config",
        "name": "配置序列化",
        "domain": "配置 / 序列化 / 验证",
        "keywords": ["pydantic", "配置", "config", "yaml", "toml", "json", "序列化",
                     "验证", "校验", "schema", "模型", "类型", "type",
                     "pyyaml", "cloudpickle", "simplejson"],
        "fragments": [],
        "repo_map": ["pydantic", "pyyaml", "toml", "simplejson", "cloudpickle", "pyjwt"],
    },
    {
        "id": "sys",
        "name": "系统底层",
        "domain": "系统 / 安全 / 加密 / 底层",
        "keywords": ["cpython", "python", "解释器", "cryptography", "加密", "安全",
                     "paramiko", "ssh", "认证", "证书", "certificate",
                     "pynacl", "底层", "字节码", "c extension", "c扩展"],
        "fragments": [],
        "repo_map": ["cpython", "cryptography", "paramiko", "pynacl", "certifi",
                      "pycrypto", "requests-oauthlib", "jwt"],
    },
    {
        "id": "general",
        "name": "通用兜底",
        "domain": "通用对话",
        "keywords": ["你好", "是什么", "为什么", "怎么", "如何", "介绍", "解释", "什么是", "？", "?"],
        "fragments": [
            "这是一个好问题，让我从合鸣的视角来回应",
            "在 xuni 虚拟生态里，每个问题都会被路由到最合适的专家",
            "我可以聊聊编程框架、数据科学、机器学习、网络通信，或者 xuni 的设计哲学",
            "如果方便，补充一点上下文，我能给出更精准的共振回答",
        ],
        "repo_map": ["others"],
    },
]


def get_repo_to_expert_map() -> Dict[str, str]:
    """获取 仓库名 → 专家ID 的映射"""
    mapping = {}
    for exp in CODE_EXPERTS:
        for repo in exp.get("repo_map", []):
            mapping[repo] = exp["id"]
    return mapping


def get_expert_by_id(expert_id: str) -> Dict[str, Any]:
    for exp in CODE_EXPERTS:
        if exp["id"] == expert_id:
            return exp
    return CODE_EXPERTS[-1]  # general 兜底
