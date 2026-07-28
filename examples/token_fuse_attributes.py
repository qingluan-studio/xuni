"""
让虚拟 Token 融合 7 种真实属性物质——看能不能"长"成接近真实 token。

流程：
    1. 用 tiktoken 切真实文本，取第 1 个 token，提取 7 个属性值
    2. 把这 7 个属性值做成 7 种虚拟物质（TokenIdSubstance 等）
    3. 工厂生产一个虚拟 DownloadToken（quality=1.0, 精纯度 1%）
    4. 依次融合 7 种物质——每融合一种，对应属性写入 metadata
    5. 最后对比：融合前 vs 融合后 vs 真实 token
"""

from __future__ import annotations

import os
import sys

# 让脚本能直接 import xuni
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tiktoken

from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.token_attribute_substances import (
    TokenAttributeForge,
    substance_from_real_token,
    synthesize_embedding,
)


def get_one_real_token(text: str, encoding_name: str = "cl100k_base"):
    """用 tiktoken 切真实文本，取第一个 token，提取 7 个属性"""
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)
    tid = token_ids[0]
    try:
        txt = enc.decode([tid])
    except Exception:
        txt = "<undecodable>"

    # 没有真实模型权重，用词频近似概率（无模型场景的标准做法）
    from collections import Counter
    freq = Counter(token_ids)
    total = len(token_ids)
    p = (freq[tid] + 1) / (total + len(freq))
    logprob = float(np.log(p))
    rank = sorted(set(token_ids), key=lambda x: -freq[x]).index(tid) + 1
    entropy_bits = -float(np.log2(p))

    real_token = {
        "token_id": tid,
        "text": txt,
        "byte_length": len(txt.encode("utf-8")),
        "char_length": len(txt),
        "logprob": logprob,
        "rank": rank,
        "entropy_bits": entropy_bits,
        "position": 0,
        # embedding 维度对齐真实 GPT-4 行业估计值（12288 维）
        # 用 token_id 做种子的确定向量占位——结构等长，语义不真
        "embedding": synthesize_embedding(tid, dim=12288),
    }
    return real_token


def show_token_state(token, label: str):
    """展示 token 的当前属性状态"""
    print(f"\n--- {label} ---")
    print(f"  name        : {token.name}")
    print(f"  quality     : {token.quality:.4f}")
    print(f"  精纯度      : {token.quality / 100 * 100:.4f}%")
    print(f"  quantity    : {token.quantity:.2f}")
    print(f"  metadata 钥匙: {list(token.metadata.keys())}")
    real_attrs = ["token_id", "text", "logprob", "rank",
                  "entropy_bits", "position", "embedding"]
    has_count = sum(1 for k in real_attrs if k in token.metadata)
    print(f"  真实属性持有: {has_count}/7")
    for k in real_attrs:
        v = token.metadata.get(k, "<无>")
        if isinstance(v, np.ndarray):
            v = f"<ndarray shape={v.shape} dtype={v.dtype}>"
        print(f"    {k:14s} = {v!r}")


def main():
    sample_text = (
        "Hello world! 这是一个测试文本，用来对比真实 token 和虚拟 token。"
        "The quick brown fox jumps over the lazy dog. "
        "1234567890. Python 是最好的编程语言之一。"
    )

    print("=" * 78)
    print("虚拟 Token 融合 7 种真实属性物质——融合实验")
    print("=" * 78)

    # Step 1: 取一个真实 token
    print("\n[Step 1] 用 tiktoken 取一个真实 token ...")
    real_token = get_one_real_token(sample_text)
    print(f"  token_id   : {real_token['token_id']}")
    print(f"  text       : {real_token['text']!r}")
    print(f"  byte_length: {real_token['byte_length']}")
    print(f"  logprob    : {real_token['logprob']:.4f}")
    print(f"  rank       : {real_token['rank']}")
    print(f"  entropy    : {real_token['entropy_bits']:.4f} bits")
    print(f"  position   : {real_token['position']}")
    print(f"  embedding  : shape={real_token['embedding'].shape}, "
          f"norm={np.linalg.norm(real_token['embedding']):.4f}")

    # Step 2: 把属性值做成物质
    print("\n[Step 2] 把真实 token 的 7 个属性值做成 7 种虚拟物质 ...")
    substances = substance_from_real_token(real_token)
    for s in substances:
        print(f"  - {s.name}  (rarity={s.rarity.name})")

    # Step 3: 工厂生产一个虚拟 DownloadToken
    print("\n[Step 3] 工厂生产一个虚拟 DownloadToken（初始精纯度 1%）...")
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()
    # 融合前快照（深拷贝 metadata）
    before_snapshot = {
        "token_id":      None,
        "text":          None,
        "byte_length":   None,
        "char_length":   None,
        "logprob":       None,
        "rank":          None,
        "entropy_bits":  None,
        "position":      None,
        "embedding":     None,
        "quality":       token.quality,
    }
    show_token_state(token, "融合前")

    # Step 4: 依次融合 7 种物质
    print("\n[Step 4] 依次融合 7 种属性物质 ...")
    forge = TokenAttributeForge(factory)
    result = forge.absorb_all(token, substances)
    print(f"  吸收成功: {result['total_absorbed']}/{result['total_attempted']}")
    for r in result["results"]:
        if r.get("absorbed"):
            print(f"  - 融合 {r['substance']:<14s} → "
                  f"{r['attribute']:<14s} quality {r['quality_before']:.2f}→{r['quality_after']:.2f}")

    # Step 5: 最终状态
    show_token_state(token, "融合后")

    # Step 6: 三方对比表
    print("\n" + "=" * 78)
    print("三方对比表  ·  真实 token vs 融合前虚拟 token vs 融合后虚拟 token")
    print("=" * 78)
    print(f"  {'属性':<16} | {'真实':<24} | {'融合前':<14} | {'融合后':<28}")
    print(f"  {'-'*16}-+-{'-'*24}-+-{'-'*14}-+-{'-'*28}")

    attrs = [
        ("token_id",      real_token["token_id"]),
        ("text",          real_token["text"]),
        ("byte_length",   real_token["byte_length"]),
        ("char_length",   real_token["char_length"]),
        ("logprob",       f"{real_token['logprob']:.4f}"),
        ("rank",          real_token["rank"]),
        ("entropy_bits",  f"{real_token['entropy_bits']:.4f}"),
        ("position",      real_token["position"]),
        ("embedding",     f"shape={real_token['embedding'].shape}"),
        ("quality",       "N/A (真实无此概念)"),
    ]
    for name, real_v in attrs:
        # 融合前从快照取，融合后从 token.metadata 取
        before = before_snapshot.get(name)
        after = token.metadata.get(name) if name != "quality" else token.quality
        before_str = "<无>" if before is None else str(before)[:14]
        if isinstance(after, np.ndarray):
            after_str = f"shape={after.shape} dtype={after.dtype}"
        else:
            after_str = "<无>" if after is None else str(after)[:28]
        print(f"  {name:<16} | {str(real_v)[:24]:<24} | {before_str:<14} | {after_str:<28}")

    # 总结
    real_attrs = ["token_id", "text", "logprob", "rank",
                  "entropy_bits", "position", "embedding"]
    after_count = sum(1 for k in real_attrs if k in token.metadata)
    print(f"\n  融合前真实属性: 0/7")
    print(f"  融合后真实属性: {after_count}/7")
    print(f"  quality 变化   : 1.0 → {token.quality:.4f} "
          f"(吸收过程消耗 quality 作为认知代价)")

    print("\n" + "=" * 78)
    print("结论:")
    print("  融合 7 种属性物质后，虚拟 Token 在结构上拥有了真实 token 的全部 7 个属性——")
    print("  token_id / text / logprob / rank / entropy_bits / position / embedding 都填进去了。")
    print("  但这些值是从真实 token 拷贝过来的，不是虚拟 Token 自己\"长\"出来的。")
    print("  拷贝 ≠ 真实，结构补全 ≠ 本质相同。")
    print("=" * 78)


if __name__ == "__main__":
    main()
