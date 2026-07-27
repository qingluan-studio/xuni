"""
demo_trainer_v2.py —— 大规模训练 v2

用工厂自己产的 358 条真实代码片段 + 扩展合成语料，
进行 5000 轮增量训练，对比训练前后生成质量。

运行：
  cd /workspace/xuni
  python examples/demo_trainer_v2.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xuni import Harmonia13Virtual


CORPUS_PATH = os.path.join(os.path.dirname(__file__), "real_corpus.json")


def main():
    print("=" * 60)
    print("  🚀 xuni 大规模训练 v2 —— 5000 轮")
    print("  语料：工厂自身 358 条真实代码 + 扩展合成语料")
    print("=" * 60)

    # ---------------------------------------------------------------------
    # 1. 加载语料
    # ---------------------------------------------------------------------
    print("\n[1/6] 加载真实语料...")
    real_corpus = _load_real_corpus()
    print(f"  真实代码片段: {len(real_corpus)} 条")

    print("  生成扩展语料...")
    synth_corpus = _generate_extended_corpus()
    print(f"  扩展合成语料: {len(synth_corpus)} 条")

    corpus = real_corpus + synth_corpus
    print(f"  总语料: {len(corpus)} 条")

    # ---------------------------------------------------------------------
    # 2. 创建模型 + 训练前基线
    # ---------------------------------------------------------------------
    print("\n[2/6] 创建模型 + 训练前基线测试...")
    model = Harmonia13Virtual(scale="mini")
    print(f"  初始专家: {len(model._lite.experts)} 位")
    print(f"  初始语料: {len(model._lite._learned_fragments)} 条")

    baseline_prompts = [
        "def quicksort",
        "class XuniSampler",
        "def generate",
        "import numpy",
        "class MemoryBank",
        "def train",
        "class Harmonia",
        "def collide",
    ]

    print("\n  --- 训练前生成基线 ---")
    baseline_results = {}
    for p in baseline_prompts:
        result = model._lite.generate(p, max_new_tokens=30)
        baseline_results[p] = result
        print(f"  [{p}] → {result[:60]}")

    # ---------------------------------------------------------------------
    # 3. 5000 轮训练
    # ---------------------------------------------------------------------
    print(f"\n[3/6] 开始 5000 轮增量训练...")
    print("  (每 500 轮报告进度)")

    start_time = time.time()
    batch_size = 8
    num_epochs = 5000
    growth_log = []

    for epoch in range(num_epochs):
        batch = random.sample(corpus, min(batch_size, len(corpus)))
        model._lite.train(batch, epochs=1)

        if (epoch + 1) % 500 == 0:
            elapsed = time.time() - start_time
            learned = len(model._lite._learned_fragments)
            expert_frags = [len(e.get('fragments', [])) for e in model._lite.experts]
            avg_load = sum(expert_frags) / max(1, len(expert_frags))
            active = sum(1 for f in expert_frags if f > 0)

            print(f"  Epoch {epoch+1:5d} | 已学: {learned:6d} | "
                  f"活跃专家: {active:2d} | 均载: {avg_load:.0f} | "
                  f"用时: {elapsed:.1f}s")

            growth_log.append({
                "epoch": epoch + 1,
                "learned": learned,
                "active_experts": active,
                "avg_load": round(avg_load, 1),
            })

    total_time = time.time() - start_time
    print(f"\n  ✅ 训练完成！总用时: {total_time:.2f}s")

    # ---------------------------------------------------------------------
    # 4. 训练后评估
    # ---------------------------------------------------------------------
    print("\n[4/6] 训练后模型评估...")

    learned = len(model._lite._learned_fragments)
    expert_frags = [(e.get('name', '?'), len(e.get('fragments', []))) for e in model._lite.experts]
    active = sum(1 for _, f in expert_frags if f > 0)

    print(f"\n  📊 模型规模:")
    print(f"    已学语料: {learned} 条")
    print(f"    活跃专家: {active} / {len(model._lite.experts)}")
    print(f"    各专家负载:")
    for name, frags in expert_frags:
        bar = "█" * min(50, frags // 100)
        print(f"      {name:12s} [{bar}] {frags:5d}")

    # ---------------------------------------------------------------------
    # 5. 对比生成质量
    # ---------------------------------------------------------------------
    print(f"\n[5/6] 训练前后生成对比...")

    print("\n  --- 训练后生成结果 ---")
    improved = 0
    total = 0
    for p in baseline_prompts:
        after = model._lite.generate(p, max_new_tokens=30)
        before = baseline_results[p]
        total += 1
        if len(after) > len(before):
            improved += 1
        print(f"\n  [{p}]")
        print(f"    训练前: {before[:60]}")
        print(f"    训练后: {after[:60]}")

    # ---------------------------------------------------------------------
    # 6. 保存
    # ---------------------------------------------------------------------
    print(f"\n[6/6] 保存训练报告...")

    report = {
        "version": "v2",
        "training_time_seconds": round(total_time, 2),
        "total_corpus": len(corpus),
        "real_corpus": len(real_corpus),
        "synth_corpus": len(synth_corpus),
        "epochs": num_epochs,
        "fragments_learned": learned,
        "active_experts": active,
        "expert_load": {name: frags for name, frags in expert_frags},
        "growth_log": growth_log,
        "comparison": {
            p: {
                "before": baseline_results[p],
                "after": model._lite.generate(p, max_new_tokens=30),
            }
            for p in baseline_prompts
        },
        "improved_count": improved,
        "total_compared": total,
    }

    report_path = os.path.join(os.path.dirname(__file__), "trainer_v2_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  报告: {report_path}")

    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v2")
    model._lite.save(ckpt_dir)
    print(f"  检查点: {ckpt_dir}")

    # 总结
    print("\n" + "=" * 60)
    print("  📈 v2 训练总结")
    print("=" * 60)
    print(f"""
  📚 语料规模: {len(real_corpus)} 真实 + {len(synth_corpus)} 合成 = {len(corpus)} 总计
  🔄 训练轮次: {num_epochs}
  🧠 吸收片段: {learned} 条
  👥 活跃专家: {active} / {len(model._lite.experts)}
  ⏱️ 训练用时: {total_time:.2f}s
  📈 生成提升: {improved}/{total} 个 prompt 训练后更长

  积少成多进度:
    v1: 1000 轮 / 5000 片段
    v2: 5000 轮 / {learned} 片段 ✅

  下一步:
    ✅ 10000 轮训练
    ✅ 接入更多真实项目代码
    ✅ 双态切换到真实 API
""")


def _load_real_corpus() -> list:
    """加载从 xuni 项目提取的真实代码"""
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _generate_extended_corpus() -> list:
    """扩展合成语料：覆盖更多 Python 领域"""

    corpus = []

    # --- 设计模式 ---
    patterns = [
        "class Observer:\n    def __init__(self):\n        self._observers = []\n    def attach(self, observer):\n        self._observers.append(observer)\n    def detach(self, observer):\n        self._observers.remove(observer)\n    def notify(self, *args, **kwargs):\n        for obs in self._observers:\n            obs.update(*args, **kwargs)",
        "class Subject:\n    def __init__(self):\n        self._state = None\n        self._observers = []\n    def get_state(self):\n        return self._state\n    def set_state(self, state):\n        self._state = state\n        self._notify()",
        "class Strategy:\n    def __init__(self, strategy_func):\n        self._strategy = strategy_func\n    def execute(self, data):\n        return self._strategy(data)",
        "class Command:\n    def __init__(self, receiver, action):\n        self.receiver = receiver\n        self.action = action\n    def execute(self):\n        getattr(self.receiver, self.action)()",
        "class Factory:\n    @staticmethod\n    def create_product(product_type):\n        if product_type == 'A':\n            return ProductA()\n        elif product_type == 'B':\n            return ProductB()\n        raise ValueError(f'Unknown type: {product_type}')",
        "class Builder:\n    def __init__(self):\n        self.product = Product()\n    def build_part_a(self):\n        self.product.add('PartA')\n        return self\n    def build_part_b(self):\n        self.product.add('PartB')\n        return self\n    def get_result(self):\n        return self.product",
        "class Adapter:\n    def __init__(self, adaptee):\n        self.adaptee = adaptee\n    def request(self):\n        return self._translate(self.adaptee.specific_request())",
        "class Facade:\n    def __init__(self):\n        self.subsystem_a = SubsystemA()\n        self.subsystem_b = SubsystemB()\n    def operation(self):\n        result_a = self.subsystem_a.operation_a()\n        result_b = self.subsystem_b.operation_b()\n        return f'{result_a} + {result_b}'",
        "class Proxy:\n    def __init__(self, real_subject):\n        self._real_subject = real_subject\n        self._cached = None\n    def request(self):\n        if self._cached is None:\n            self._cached = self._real_subject.request()\n        return self._cached",
        "class Iterator:\n    def __init__(self, collection):\n        self._collection = collection\n        self._index = 0\n    def __next__(self):\n        if self._index >= len(self._collection):\n            raise StopIteration\n        item = self._collection[self._index]\n        self._index += 1\n        return item",
    ]
    corpus.extend(patterns)

    # --- 数据科学 ---
    data_science = [
        "import pandas as pd\nimport numpy as np\n\ndef load_and_clean(filepath):\n    df = pd.read_csv(filepath)\n    df = df.dropna()\n    df = df.drop_duplicates()\n    return df",
        "def correlation_matrix(df):\n    return df.corr()\n\ndef plot_heatmap(matrix):\n    import seaborn as sns\n    import matplotlib.pyplot as plt\n    sns.heatmap(matrix, annot=True, cmap='coolwarm')\n    plt.show()",
        "def standardize(df, columns):\n    for col in columns:\n        mean = df[col].mean()\n        std = df[col].std()\n        df[col] = (df[col] - mean) / std\n    return df",
        "def one_hot_encode(df, column):\n    dummies = pd.get_dummies(df[column], prefix=column)\n    df = pd.concat([df.drop(column, axis=1), dummies], axis=1)\n    return df",
        "def kmeans_cluster(X, k=3, max_iter=100):\n    centroids = X[np.random.choice(X.shape[0], k, replace=False)]\n    for _ in range(max_iter):\n        distances = np.linalg.norm(X[:, None] - centroids, axis=2)\n        labels = np.argmin(distances, axis=1)\n        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])\n        if np.all(centroids == new_centroids):\n            break\n        centroids = new_centroids\n    return labels, centroids",
        "def pca(X, n_components=2):\n    X_centered = X - X.mean(axis=0)\n    cov_matrix = np.cov(X_centered.T)\n    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)\n    idx = np.argsort(eigenvalues)[::-1][:n_components]\n    return X_centered @ eigenvectors[:, idx]",
        "def gradient_descent(X, y, lr=0.01, epochs=1000):\n    n, d = X.shape\n    w = np.zeros(d)\n    b = 0.0\n    for _ in range(epochs):\n        pred = X @ w + b\n        error = pred - y\n        w -= lr * (X.T @ error) / n\n        b -= lr * np.mean(error)\n    return w, b",
        "def random_forest_features(X, y, n_trees=10):\n    from sklearn.ensemble import RandomForestClassifier\n    rf = RandomForestClassifier(n_estimators=n_trees)\n    rf.fit(X, y)\n    return dict(zip(range(X.shape[1]), rf.feature_importances_))",
    ]
    corpus.extend(data_science)

    # --- 异步编程 ---
    async_code = [
        "import asyncio\n\nasync def fetch_data(url):\n    await asyncio.sleep(1)\n    return {'data': 'result'}",
        "async def gather_all(urls):\n    tasks = [fetch_data(url) for url in urls]\n    results = await asyncio.gather(*tasks)\n    return results",
        "async def producer(queue, items):\n    for item in items:\n        await queue.put(item)\n    await queue.put(None)",
        "async def consumer(queue):\n    while True:\n        item = await queue.get()\n        if item is None:\n            break\n        process(item)",
        "async def with_timeout(coro, timeout=5.0):\n    try:\n        return await asyncio.wait_for(coro, timeout=timeout)\n    except asyncio.TimeoutError:\n        return None",
        "class AsyncContextManager:\n    async def __aenter__(self):\n        await self.connect()\n        return self\n    async def __aexit__(self, *args):\n        await self.disconnect()",
        "async def retry_async(func, *args, max_retries=3, delay=1):\n    for attempt in range(max_retries):\n        try:\n            return await func(*args)\n        except Exception:\n            if attempt == max_retries - 1:\n                raise\n            await asyncio.sleep(delay)",
    ]
    corpus.extend(async_code)

    # --- 文件/IO ---
    io_code = [
        "import json\n\ndef read_json(filepath):\n    with open(filepath, 'r', encoding='utf-8') as f:\n        return json.load(f)\n\ndef write_json(filepath, data):\n    with open(filepath, 'w', encoding='utf-8') as f:\n        json.dump(data, f, ensure_ascii=False, indent=2)",
        "import csv\n\ndef read_csv(filepath):\n    with open(filepath, 'r', newline='', encoding='utf-8') as f:\n        reader = csv.DictReader(f)\n        return list(reader)",
        "import pickle\n\ndef save_pickle(filepath, obj):\n    with open(filepath, 'wb') as f:\n        pickle.dump(obj, f)\n\ndef load_pickle(filepath):\n    with open(filepath, 'rb') as f:\n        return pickle.load(f)",
        "import os\n\ndef walk_directory(root):\n    for dirpath, dirnames, filenames in os.walk(root):\n        for filename in filenames:\n            filepath = os.path.join(dirpath, filename)\n            yield filepath",
        "def read_large_file(filepath, chunk_size=8192):\n    with open(filepath, 'r', encoding='utf-8') as f:\n        while True:\n            chunk = f.read(chunk_size)\n            if not chunk:\n                break\n            yield chunk",
        "import gzip\n\ndef read_gzip(filepath):\n    with gzip.open(filepath, 'rt', encoding='utf-8') as f:\n        return f.read()",
    ]
    corpus.extend(io_code)

    # --- 测试 ---
    test_code = [
        "import unittest\n\nclass TestStringMethods(unittest.TestCase):\n    def test_upper(self):\n        self.assertEqual('hello'.upper(), 'HELLO')\n    def test_split(self):\n        s = 'hello world'\n        self.assertEqual(s.split(), ['hello', 'world'])",
        "import pytest\n\ndef test_addition():\n    assert 1 + 1 == 2\n\ndef test_string():\n    assert 'hello'.upper() == 'HELLO'",
        "from unittest.mock import Mock, patch\n\ndef test_api_call():\n    with patch('requests.get') as mock_get:\n        mock_get.return_value.json.return_value = {'status': 'ok'}\n        result = fetch_api_data('http://test.com')\n        assert result['status'] == 'ok'",
        "import pytest\n\n@pytest.fixture\ndef sample_data():\n    return [1, 2, 3, 4, 5]\n\ndef test_sum(sample_data):\n    assert sum(sample_data) == 15",
        "def parametrize_test():\n    import pytest\n    @pytest.mark.parametrize('input,expected', [\n        (1, 1), (2, 4), (3, 9), (4, 16)\n    ])\n    def test_square(input, expected):\n        assert input ** 2 == expected",
    ]
    corpus.extend(test_code)

    # --- 正则/文本处理 ---
    text_code = [
        "import re\n\ndef find_all(pattern, text):\n    return re.findall(pattern, text)\n\ndef replace(pattern, repl, text):\n    return re.sub(pattern, repl, text)",
        "def tokenize(text):\n    return re.findall(r'\\b\\w+\\b', text.lower())\n\ndef word_count(text):\n    tokens = tokenize(text)\n    from collections import Counter\n    return Counter(tokens)",
        "def extract_urls(text):\n    pattern = r'https?://[\\w./-]+'\n    return re.findall(pattern, text)",
        "def clean_text(text):\n    text = re.sub(r'<[^>]+>', '', text)\n    text = re.sub(r'\\s+', ' ', text)\n    return text.strip()",
        "from collections import defaultdict\n\ndef group_by(items, key_func):\n    groups = defaultdict(list)\n    for item in items:\n        groups[key_func(item)].append(item)\n    return dict(groups)",
    ]
    corpus.extend(text_code)

    # --- 类型注解/Pydantic ---
    type_code = [
        "from typing import List, Dict, Optional, Union\n\ndef process_items(items: List[str]) -> Dict[str, int]:\n    return {item: len(item) for item in items}",
        "from dataclasses import dataclass\nfrom typing import List\n\n@dataclass\nclass User:\n    name: str\n    age: int\n    email: str = ''\n    tags: List[str] = None",
        "from pydantic import BaseModel\n\nclass UserSchema(BaseModel):\n    name: str\n    age: int\n    email: str = ''\n    class Config:\n        from_attributes = True",
        "from typing import Protocol\n\nclass Drawable(Protocol):\n    def draw(self) -> None: ...",
        "from typing import TypeVar, Generic, List\n\nT = TypeVar('T')\n\nclass Stack(Generic[T]):\n    def __init__(self):\n        self._items: List[T] = []\n    def push(self, item: T) -> None:\n        self._items.append(item)\n    def pop(self) -> T:\n        return self._items.pop()",
    ]
    corpus.extend(type_code)

    # 扩展：切分长代码
    expanded = []
    for snippet in corpus:
        expanded.append(snippet)
        lines = snippet.split('\n')
        if len(lines) > 4:
            expanded.append('\n'.join(lines[:3]))
        if len(lines) > 6:
            expanded.append('\n'.join(lines[3:6]))

    return expanded


if __name__ == "__main__":
    main()
