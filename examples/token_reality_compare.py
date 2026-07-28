"""
真实 Token vs 虚拟 Token 数值对比分析

目标：
    1. 用 tiktoken (cl100k_base, GPT-4 用的词表) 把真实文本 tokenize
    2. 提取每个真实 token 的各项数值属性
    3. 用 xuni 的负负得正链路造一个虚拟 Token (quality 已被推到 100%)
    4. 提取虚拟 Token 的各项数值属性
    5. 做成对比表，看本质差什么

只做分析，不下结论。让数据自己说话。
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List

import numpy as np
import tiktoken

from xuni.multiverse_resources import MultiverseResourceFactory, DownloadToken
from xuni.purity_forge import PurityForge


# ============================================================
# 1. 真实 Token 分析
# ============================================================

def analyze_real_tokens(text: str, encoding_name: str = "cl100k_base") -> Dict[str, Any]:
    """
    用 tiktoken 把文本切成真实 token，提取数值属性。

    真实 token 有的属性：
        - token_id: 整数 ID
        - text: 子词文本
        - byte_length: 字节长度
        - char_length: 字符长度
        - is_ascii: 是否纯 ASCII
        - is_punct: 是否标点
        - is_digit: 是否数字
        - is_space: 是否空格
        - is_word: 是否普通词
        - position: 在序列中的位置
        - logprob_proxy: 用词频近似代替对数概率（无模型时的标准做法）
        - rank_proxy: 用词频排名近似
        - entropy_proxy: 单 token 信息量 -log2(p)（无模型时用均匀分布近似下界）
        - embedding_dim: 真实模型的 embedding 维度（GPT-4 未公开，但行业是 12288/4096）
    """
    enc = tiktoken.get_encoding(encoding_name)
    token_ids = enc.encode(text)

    # 词频统计（用本语料的频次当近似——真实场景要用大规模语料统计）
    from collections import Counter
    freq = Counter(token_ids)
    total = len(token_ids)

    # 按频次排序得到 rank
    sorted_ids = [tid for tid, _ in freq.most_common()]
    rank_map = {tid: r + 1 for r, tid in enumerate(sorted_ids)}

    samples: List[Dict[str, Any]] = []
    for pos, tid in enumerate(token_ids):
        # 解码回文本
        try:
            txt = enc.decode([tid])
        except Exception:
            txt = "<undecodable>"

        byte_len = len(txt.encode("utf-8"))
        char_len = len(txt)
        is_ascii = txt.isascii() and txt.isprintable()
        is_space = txt.isspace()
        is_digit = txt.isdigit()
        is_punct = all(not c.isalnum() and not c.isspace() for c in txt) and len(txt) > 0
        is_word = txt.isalpha()

        # 概率近似：频次/总长（拉普拉斯平滑避免 0）
        p = (freq[tid] + 1) / (total + len(freq))
        logprob = float(np.log(p))
        rank = rank_map[tid]
        entropy_bits = -float(np.log2(p))  # 自信息量

        samples.append({
            "token_id": tid,
            "text": txt,
            "byte_length": byte_len,
            "char_length": char_len,
            "is_ascii": is_ascii,
            "is_space": is_space,
            "is_digit": is_digit,
            "is_punct": is_punct,
            "is_word": is_word,
            "position": pos,
            "logprob": logprob,
            "rank": rank,
            "entropy_bits": entropy_bits,
            # embedding 维度：真实模型未公开，标行业值
            "embedding_dim_real": 12288,  # GPT-4 估计值
        })

    # 聚合统计
    logprobs = [s["logprob"] for s in samples]
    entropies = [s["entropy_bits"] for s in samples]
    byte_lens = [s["byte_length"] for s in samples]
    char_lens = [s["char_length"] for s in samples]
    ranks = [s["rank"] for s in samples]

    return {
        "source": "real (tiktoken cl100k_base)",
        "encoding": encoding_name,
        "vocab_size": enc.n_vocab,
        "text_length_chars": len(text),
        "num_tokens": len(samples),
        "tokens_per_char": len(samples) / max(1, len(text)),
        "samples": samples,
        "stats": {
            "logprob_mean": statistics.mean(logprobs),
            "logprob_min": min(logprobs),
            "logprob_max": max(logprobs),
            "entropy_mean_bits": statistics.mean(entropies),
            "entropy_min_bits": min(entropies),
            "entropy_max_bits": max(entropies),
            "byte_length_mean": statistics.mean(byte_lens),
            "char_length_mean": statistics.mean(char_lens),
            "rank_mean": statistics.mean(ranks),
            "unique_tokens": len(set(s["token_id"] for s in samples)),
            "type_token_ratio": len(set(s["token_id"] for s in samples)) / max(1, len(samples)),
            "ascii_ratio": sum(1 for s in samples if s["is_ascii"]) / max(1, len(samples)),
            "space_ratio": sum(1 for s in samples if s["is_space"]) / max(1, len(samples)),
            "digit_ratio": sum(1 for s in samples if s["is_digit"]) / max(1, len(samples)),
            "punct_ratio": sum(1 for s in samples if s["is_punct"]) / max(1, len(samples)),
            "word_ratio": sum(1 for s in samples if s["is_word"]) / max(1, len(samples)),
            "embedding_dim": 12288,
        },
    }


# ============================================================
# 2. 虚拟 Token 分析（负负得正链路造出来的）
# ============================================================

def analyze_virtual_token(cycles: int = 100, virtual_power: float = 1_000_000) -> Dict[str, Any]:
    """
    用 PurityForge 走负负得正链路造一个虚拟 Token，提取属性。

    链路：虚拟电 → 反相 → 真实电力 → 注入 DownloadToken 推 quality
    """
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()
    forge = PurityForge(factory)

    initial_quality = token.quality
    initial_purity_pct = initial_quality / 100 * 100

    # 跑负负得正提纯
    result = forge.purify_batch(
        token,
        cycles=cycles,
        virtual_power_per_cycle=virtual_power,
        verbose=False,
    )

    final_quality = token.quality
    final_purity_pct = final_quality / 100 * 100

    # 提取虚拟 token 的所有属性
    sample = {
        "resource_id": token.resource_id,
        "name": token.name,
        "dimension": token.dimension.name if token.dimension else None,
        "rarity": token.rarity.name,
        "quantity": token.quantity,
        "quality": token.quality,
        "purity_pct": final_purity_pct,
        "level": token.level,
        "concurrent_limit": token.concurrent_limit,
        "speed_multiplier": token.speed_multiplier,
        "unlimited": token.unlimited,
        "created_at": token.created_at,
        "metadata_keys": list(token.metadata.keys()),
        # 对应真实 token 的属性（虚拟里有没有）
        "token_id": None,             # 无
        "text": None,                 # 无
        "byte_length": None,          # 无
        "char_length": None,          # 无
        "is_ascii": None,             # 无
        "is_space": None,             # 无
        "is_digit": None,             # 无
        "is_punct": None,             # 无
        "is_word": None,              # 无
        "position": None,             # 无
        "logprob": None,              # 无
        "rank": None,                 # 无
        "entropy_bits": None,         # 无
        "embedding_dim_real": 0,      # 无（虚拟里没接 embedding）
        "embedding_dim_virtual": 0,   # 见 ultra_context.MemoryPoint，是 64
    }

    # 超长上下文记忆的 embedding 维度参考
    try:
        from xuni.ultra_context import MemoryPoint
        sample["embedding_dim_virtual_ref"] = 64  # MemoryPoint._embed 默认 64
    except Exception:
        sample["embedding_dim_virtual_ref"] = None

    return {
        "source": "virtual (xuni PurityForge, 负负得正链路)",
        "chain": "虚拟电 → 反相 → 真实电力 → 注入 DownloadToken",
        "cycles": cycles,
        "virtual_power_per_cycle": virtual_power,
        "initial_quality": initial_quality,
        "initial_purity_pct": initial_purity_pct,
        "final_quality": final_quality,
        "final_purity_pct": final_purity_pct,
        "real_power_total": result["real_power_total"],
        "total_loss": result["total_loss"],
        "target_reached": result["target_reached"],
        "sample": sample,
        "stats": {
            "quality": final_quality,
            "purity_pct": final_purity_pct,
            "quantity": token.quantity,
            "concurrent_limit": token.concurrent_limit,
            "speed_multiplier": token.speed_multiplier,
            "level": token.level,
            "metadata_count": len(token.metadata),
            "embedding_dim": 0,  # 虚拟 token 没有 embedding
        },
    }


# ============================================================
# 3. 对比表生成
# ============================================================

def print_comparison(real: Dict[str, Any], virtual: Dict[str, Any]) -> None:
    """打印对比表"""

    print("=" * 78)
    print("真实 Token vs 虚拟 Token  ·  数值对比表")
    print("=" * 78)
    print()

    # ---- 顶层信息 ----
    print("【1. 来源信息】")
    print(f"  {'项目':<24} | {'真实 Token':<28} | {'虚拟 Token':<24}")
    print(f"  {'-'*24}-+-{'-'*28}-+-{'-'*24}")
    print(f"  {'来源':<24} | {real['source']:<28} | {virtual['source']:<24}")
    vocab_str = f"cl100k_base (vocab={real['vocab_size']})"
    print(f"  {'词表/构造方式':<24} | {vocab_str:<28} | {'DownloadToken + PurityForge':<24}")
    print(f"  {'生产方式':<24} | {'tiktoken.encode(text)':<28} | {'工厂.produce_download_token()':<24}")
    print()

    # ---- 数值属性对比 ----
    print("【2. 核心数值属性对比】")
    print(f"  {'属性':<24} | {'真实 Token':<28} | {'虚拟 Token':<24} | {'差距':<14}")
    print(f"  {'-'*24}-+-{'-'*28}-+-{'-'*24}-+-{'-'*14}")

    rs = real["stats"]
    vs = virtual["stats"]
    rsamp = real["samples"][0] if real["samples"] else {}
    vsamp = virtual["sample"]

    rows = [
        # (属性名, 真实值, 虚拟值, 差距说明)
        ("token_id",
         f"{rsamp.get('token_id', 'N/A')} (整数)",
         f"{vsamp.get('token_id')}",
         "虚拟无此字段"),
        ("text (子词)",
         f"'{rsamp.get('text', 'N/A')}'",
         f"{vsamp.get('text')}",
         "虚拟无语义内容"),
        ("byte_length",
         f"{rsamp.get('byte_length', 'N/A')} bytes",
         f"{vsamp.get('byte_length')}",
         "虚拟无字节"),
        ("char_length",
         f"{rsamp.get('char_length', 'N/A')}",
         f"{vsamp.get('char_length')}",
         "虚拟无字符"),
        ("position (序列位置)",
         f"{rsamp.get('position', 'N/A')}",
         f"{vsamp.get('position')}",
         "虚拟无位置概念"),
        ("logprob (对数概率)",
         f"{rs['logprob_mean']:.4f} (均值)",
         f"{vsamp.get('logprob')}",
         "虚拟无概率分布"),
        ("rank (排名)",
         f"{rs['rank_mean']:.1f} (均值)",
         f"{vsamp.get('rank')}",
         "虚拟无候选分布"),
        ("entropy_bits (信息量)",
         f"{rs['entropy_mean_bits']:.4f} bits",
         f"{vsamp.get('entropy_bits')}",
         "虚拟无信息熵"),
        ("embedding_dim",
         f"{rs['embedding_dim']} 维",
         f"{vs['embedding_dim']} 维 (vs ref={vsamp.get('embedding_dim_virtual_ref')})",
         f"差 {rs['embedding_dim'] - vs['embedding_dim']} 维"),
        ("quality (质量)",
         "N/A (真实无此概念)",
         f"{vs['quality']}",
         "虚拟独有"),
        ("purity_pct (精纯度)",
         "100% (本质真实)",
         f"{vs['purity_pct']:.4f}%",
         "虚拟100%≠真实100%"),
        ("quantity (数量)",
         f"{real['num_tokens']} 个",
         f"{vs['quantity']}",
         "虚拟是传输容量"),
        ("level",
         "N/A",
         f"{vs['level']}",
         "虚拟独有"),
        ("concurrent_limit",
         "N/A",
         f"{vs['concurrent_limit']}",
         "虚拟独有(并发数)"),
        ("speed_multiplier",
         "N/A",
         f"{vs['speed_multiplier']}",
         "虚拟独有(速度倍率)"),
    ]

    for name, real_v, virt_v, gap in rows:
        print(f"  {name:<24} | {str(real_v):<28} | {str(virt_v):<24} | {gap:<14}")

    print()

    # ---- 类型分布 ----
    print("【3. 类型分布（真实 token 的语义结构）】")
    print(f"  {'类型':<20} | {'真实占比':<12} | {'虚拟':<12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'ASCII':<20} | {rs['ascii_ratio']*100:>6.2f}%   | {'N/A':<12}")
    print(f"  {'纯空格':<20} | {rs['space_ratio']*100:>6.2f}%   | {'N/A':<12}")
    print(f"  {'数字':<20} | {rs['digit_ratio']*100:>6.2f}%   | {'N/A':<12}")
    print(f"  {'标点':<20} | {rs['punct_ratio']*100:>6.2f}%   | {'N/A':<12}")
    print(f"  {'普通词':<20} | {rs['word_ratio']*100:>6.2f}%   | {'N/A':<12}")
    print()

    # ---- 链路信息 ----
    print("【4. 虚拟 Token 的负负得正链路信息】")
    print(f"  链路: {virtual['chain']}")
    print(f"  提纯轮数: {virtual['cycles']}")
    print(f"  每轮虚拟电: {virtual['virtual_power_per_cycle']:.0e}")
    print(f"  quality 变化: {virtual['initial_quality']} → {virtual['final_quality']}")
    print(f"  精纯度变化: {virtual['initial_purity_pct']:.4f}% → {virtual['final_purity_pct']:.4f}%")
    print(f"  真实电力累计: {virtual['real_power_total']:.2e}")
    print(f"  数量损耗: {virtual['total_loss']:.2f}")
    print()

    # ---- 本质差距 ----
    print("【5. 本质差距总结】")
    real_attrs = {"token_id","text","byte_length","char_length","is_ascii","is_space",
                  "is_digit","is_punct","is_word","position","logprob","rank",
                  "entropy_bits","embedding_dim_real"}
    virt_attrs = set(vsamp.keys()) - {"embedding_dim_virtual_ref"}
    missing = real_attrs - virt_attrs
    print(f"  真实 token 独有属性（虚拟完全没有）: {len(missing)} 个")
    for m in sorted(missing):
        print(f"    - {m}")
    print()
    print(f"  虚拟 token 独有属性（真实没有）: {len(virt_attrs - real_attrs)} 个")
    for e in sorted(virt_attrs - real_attrs):
        print(f"    - {e}")
    print()
    print("  结论: 虽然虚拟 token 的 quality 爬到了 100%，")
    print("        但它没有 token_id、text、logprob、rank、embedding 这些")
    print("        真实 token 的核心属性。100% 是 quality 字段的 100%，")
    print("        不是\"成为真实 token\"的 100%。")
    print("=" * 78)


# ============================================================
# 主入口
# ============================================================

def main():
    # 用一段真实的中英文混合文本做样本
    sample_text = (
        "Hello world! 这是一个测试文本，用来对比真实 token 和虚拟 token。"
        "The quick brown fox jumps over the lazy dog. "
        "1234567890. Python 是最好的编程语言之一。"
        " Artificial intelligence is changing the world. "
        "让我们看看 token 是怎么切的：tokenization, embeddings, transformers. "
        "GPT-4 uses cl100k_base vocabulary with 100277 tokens."
    )

    print("正在用 tiktoken 分析真实 token ...")
    real = analyze_real_tokens(sample_text)
    print(f"  文本: {real['text_length_chars']} 字符 → {real['num_tokens']} 个真实 token")
    print(f"  tokens/char = {real['tokens_per_char']:.4f}")
    print()

    print("正在用 PurityForge 造虚拟 token（100轮，百万虚拟电/轮）...")
    virtual = analyze_virtual_token(cycles=100, virtual_power=1_000_000)
    print(f"  quality: {virtual['initial_quality']} → {virtual['final_quality']}")
    print(f"  精纯度: {virtual['initial_purity_pct']:.2f}% → {virtual['final_purity_pct']:.2f}%")
    print()

    print_comparison(real, virtual)


if __name__ == "__main__":
    main()
