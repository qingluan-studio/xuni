"""
质变实验：融合属性物质 + 灌培养液 → 看哪个属性能从"拷贝"质变成"涌现"

流程：
    Step 1: tiktoken 取一个真实 token
    Step 2: 把 7 个属性值做成属性物质
    Step 3: 工厂生产虚拟 DownloadToken
    Step 4: 融合 7 种属性物质 → 7/7 补齐
    Step 5: 灌 12 种 Token 反应培养液 → 触发质变
    Step 6: 报告：哪个属性质变了？哪个还是拷贝？
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tiktoken

from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.token_attribute_substances import (
    TokenAttributeForge,
    substance_from_real_token,
    synthesize_embedding,
)
from xuni.token_qualitative_forge import TokenQualitativeForge


def get_one_real_token(text: str, encoding_name: str = "cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    tid = token_ids[0]
    try:
        txt = enc.decode([tid])
    except Exception:
        txt = "<undecodable>"

    from collections import Counter
    freq = Counter(token_ids)
    total = len(token_ids)
    p = (freq[tid] + 1) / (total + len(freq))
    logprob = float(np.log(p))
    rank = sorted(set(token_ids), key=lambda x: -freq[x]).index(tid) + 1
    entropy_bits = -float(np.log2(p))

    return {
        "token_id": tid,
        "text": txt,
        "byte_length": len(txt.encode("utf-8")),
        "char_length": len(txt),
        "logprob": logprob,
        "rank": rank,
        "entropy_bits": entropy_bits,
        "position": 0,
        "embedding": synthesize_embedding(tid, dim=12288),
    }


def main():
    sample_text = (
        "Hello world! 这是一个测试文本，用来对比真实 token 和虚拟 token。"
        "The quick brown fox jumps over the lazy dog. "
        "1234567890. Python 是最好的编程语言之一。"
    )

    print("=" * 78)
    print("Token 质变实验：融合属性物质 + 培养液化学反应 → 触发质变")
    print("=" * 78)

    # Step 1: 取真实 token
    print("\n[Step 1] tiktoken 取真实 token ...")
    real_token = get_one_real_token(sample_text)
    print(f"  token_id={real_token['token_id']}  text={real_token['text']!r}")
    print(f"  logprob={real_token['logprob']:.4f}  rank={real_token['rank']}")
    print(f"  entropy={real_token['entropy_bits']:.4f} bits")

    # Step 2: 做成 7 种属性物质
    print("\n[Step 2] 做成 7 种属性物质 ...")
    substances = substance_from_real_token(real_token)
    for s in substances:
        print(f"  - {s.name}")

    # Step 3: 工厂生产虚拟 Token
    print("\n[Step 3] 工厂生产虚拟 DownloadToken ...")
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()
    print(f"  quality={token.quality}  metadata 钥匙数={len(token.metadata)}")

    # Step 4: 融合 7 种属性物质
    print("\n[Step 4] 融合 7 种属性物质（拷贝阶段）...")
    attr_forge = TokenAttributeForge(factory)
    absorb_result = attr_forge.absorb_all(token, substances)
    print(f"  吸收成功: {absorb_result['total_absorbed']}/{absorb_result['total_attempted']}")
    print(f"  真实属性持有: 7/7")
    print(f"  metadata 钥匙: {list(token.metadata.keys())}")

    # Step 5: 初始化质变熔炉，灌全部 100 种培养液，多轮
    print("\n[Step 5] 初始化质变熔炉 v2，灌全部 100 种培养液（3 轮）...")
    qual_forge = TokenQualitativeForge(factory)
    qual_forge.init_emergence(token)

    react_result = qual_forge.react_with_all_cultures(
        token, level=5, rounds=3
    )
    print(f"  反应次数: {react_result['total_reacted']}/{react_result['total_attempted']}")
    print(f"  质变属性: {react_result['emergent_count']}/7")
    print(f"  变异属性: {react_result['mutated_count']}/7")
    print(f"  超变异:   {react_result['hyper_mutated_count']}/7")

    # 看看质变事件
    if qual_forge.emergent_events:
        print(f"\n  ⚡ 第一次质变事件 {len(qual_forge.emergent_events)} 次:")
        for ev in qual_forge.emergent_events:
            print(f"    - {ev['message']}")
            print(f"      (培养液={ev['culture_type']}, 能量={ev['energy_reached']:.3f})")
            print(f"      变异: {ev['mutation']}")

    if qual_forge.mutation_events:
        # 只显示前 14 个 mutation 事件
        print(f"\n  ↻ 变异/超变异事件 (前 14 个, 共 {len(qual_forge.mutation_events)} 个):")
        for ev in qual_forge.mutation_events[:14]:
            print(f"    - {ev['message']}  (培养液={ev['culture_type']})")
            if "mutation" in ev:
                print(f"      变异: {ev['mutation']}")

    # Step 6: 最终状态——原始值 vs 当前值
    print("\n" + "=" * 78)
    print("[最终状态] 7 个属性的质变+变异情况")
    print("=" * 78)
    print(f"  {'属性':<14} | {'来源':<14} | {'能量':<8} | {'变异次数':<8} | {'原始值':<18} | {'当前值':<18} | {'位移'}")
    print(f"  {'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*8}-+-{'-'*18}-+-{'-'*18}-+-{'-'*12}")
    status = qual_forge.status()
    for attr, s in status.items():
        print(f"  {attr:<14} | {s['source']:<14} | {s['energy']:<8} | "
              f"{s['mutation_count']:<8} | {s['original']:<18} | {s['current']:<18} | {s['delta']}")

    # Step 7: 来源分布
    print("\n" + "=" * 78)
    print("[来源分布]")
    print("=" * 78)
    source_counts = {}
    for s in status.values():
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src:<16}: {cnt}/7")

    # 总结
    emergent_count = sum(1 for s in status.values() if s["is_emergent"])
    mutated_count = sum(1 for s in status.values() if s["is_mutated"])
    total_mutations = sum(s["mutation_count"] for s in status.values())
    print("\n" + "=" * 78)
    print(f"质变结果: {emergent_count}/7 属性质变, {mutated_count}/7 属性变异")
    print(f"总变异次数: {total_mutations}")
    print()
    print("原始 token_id=9906 (Hello)  →  变异后 token_id=", status["token_id"]["current"])
    print("原始 text='Hello'           →  变异后 text=", status["text"]["current"])
    print("原始 logprob=-3.9416        →  变异后 logprob=", status["logprob"]["current"])
    print("原始 rank=24                →  变异后 rank=", status["rank"]["current"])
    print("原始 entropy=5.6865         →  变异后 entropy=", status["entropy_bits"]["current"])
    print("原始 position=0            →  变异后 position=", status["position"]["current"])
    print("原始 embedding shape=(12288,) →  变异后", status["embedding"]["delta"])
    print("=" * 78)


if __name__ == "__main__":
    main()
