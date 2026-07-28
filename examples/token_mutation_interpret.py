"""
变异后数值解读——把每个属性从原始→当前的变化逐一解释。

读 token.metadata 里所有变异后的字段，跟原始值对比，解释：
    - 这个数值代表什么
    - 从多少变成多少
    - 变化幅度多大
    - 这个变化意味着什么（语义层面）
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
    print("变异后数值解读")
    print("=" * 78)

    # 重建场景：取真实 token + 融合属性 + 灌 100 种培养液 3 轮
    real_token = get_one_real_token(sample_text)
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()

    substances = substance_from_real_token(real_token)
    attr_forge = TokenAttributeForge(factory)
    attr_forge.absorb_all(token, substances)

    qual_forge = TokenQualitativeForge(factory)
    qual_forge.init_emergence(token)
    qual_forge.react_with_all_cultures(token, level=5, rounds=3)

    # ============================================================
    # 逐属性解读
    # ============================================================
    print("\n" + "─" * 78)
    print("【1. token_id —— 整数 ID】")
    print("─" * 78)
    orig_id = real_token["token_id"]
    curr_id = token.metadata.get("token_id")
    source = token.metadata.get("token_id_source", "copied")
    energy = token.metadata.get("token_id_energy", 0.0)
    print(f"  代表什么 : 在 GPT-4 cl100k_base 词表（共 100277 个 token）里的整数编号")
    print(f"  原始值   : {orig_id}  (解码='Hello')")
    print(f"  变异后   : {curr_id}")
    print(f"  变化幅度 : Δ={curr_id - orig_id:+d}  ({(curr_id - orig_id) / orig_id * 100:+.1f}%)")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    # 试着解码变异后的 ID
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        decoded = enc.decode([curr_id])
        print(f"  变异后 ID 解码 : {decoded!r}")
    except Exception as e:
        print(f"  变异后 ID 无法解码 : {e}")

    print("\n" + "─" * 78)
    print("【2. text —— 子词文本】")
    print("─" * 78)
    orig_text = real_token["text"]
    curr_text = token.metadata.get("text")
    source = token.metadata.get("text_source", "copied")
    energy = token.metadata.get("text_energy", 0.0)
    print(f"  代表什么 : 这个 token 在词表里对应的子词文本")
    print(f"  原始值   : {orig_text!r}  (长度={len(orig_text)})")
    print(f"  变异后   : {curr_text!r}  (长度={len(curr_text)})")
    print(f"  变化幅度 : 长度 +{len(curr_text) - len(orig_text)}")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    # 解析前缀——每个 [xxxx] 是一次培养液注入的标记
    import re
    prefixes = re.findall(r'\[([a-z]{4})\]', curr_text)
    print(f"  注入的培养液前缀 ({len(prefixes)} 个): {prefixes}")
    print(f"  语义影响 : 文本被各种培养液标记污染，每个前缀代表一次 token_fusion / token_composer / creative 类反应")

    print("\n" + "─" * 78)
    print("【3. logprob —— 对数概率】")
    print("─" * 78)
    orig_lp = real_token["logprob"]
    curr_lp = token.metadata.get("logprob")
    source = token.metadata.get("logprob_source", "copied")
    energy = token.metadata.get("logprob_energy", 0.0)
    print(f"  代表什么 : 这个 token 在词频分布中的对数概率（ln p），越大代表越常见")
    print(f"  原始值   : {orig_lp:.6f}  → p = e^(-3.94) = {np.exp(orig_lp):.6f}")
    print(f"  变异后   : {curr_lp:.6f}  → p = e^({curr_lp:.4f}) = {np.exp(curr_lp):.6f}")
    print(f"  变化幅度 : Δ={curr_lp - orig_lp:+.4f}")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    delta_p = np.exp(curr_lp) - np.exp(orig_lp)
    print(f"  语义影响 : 概率从 {np.exp(orig_lp)*100:.3f}% 变成 {np.exp(curr_lp)*100:.3f}%  "
          f"(Δ={delta_p*100:+.3f}%)")
    if curr_lp > orig_lp:
        print(f"            → token 变得更\"常见\"了（词频意义上的）")
    else:
        print(f"            → token 变得更\"罕见\"了")

    print("\n" + "─" * 78)
    print("【4. rank —— 候选排名】")
    print("─" * 78)
    orig_rank = real_token["rank"]
    curr_rank = token.metadata.get("rank")
    source = token.metadata.get("rank_source", "copied")
    energy = token.metadata.get("rank_energy", 0.0)
    print(f"  代表什么 : 在候选分布中按概率从高到低排的第几名（1 = 最可能）")
    print(f"  原始值   : {orig_rank}  (第 {orig_rank} 名)")
    print(f"  变异后   : {curr_rank}  (第 {curr_rank} 名)")
    print(f"  变化幅度 : Δ={curr_rank - orig_rank:+d}")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    if curr_rank < orig_rank:
        print(f"  语义影响 : 排名上升 {orig_rank - curr_rank} 位  → 更被\"选中\"")
    elif curr_rank > orig_rank:
        print(f"  语义影响 : 排名下降 {curr_rank - orig_rank} 位  → 更被\"淘汰\"")
    else:
        print(f"  语义影响 : 排名不变")

    print("\n" + "─" * 78)
    print("【5. entropy_bits —— 自信息量】")
    print("─" * 78)
    orig_e = real_token["entropy_bits"]
    curr_e = token.metadata.get("entropy_bits")
    source = token.metadata.get("entropy_bits_source", "copied")
    energy = token.metadata.get("entropy_bits_energy", 0.0)
    print(f"  代表什么 : 单个 token 携带的信息量（-log2 p, 单位 bit）")
    print(f"             信息量越大 = 这个 token 越\"意外\"")
    print(f"  原始值   : {orig_e:.6f} bits")
    print(f"  变异后   : {curr_e:.6f} bits")
    print(f"  变化幅度 : Δ={curr_e - orig_e:+.4f} bits")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    if curr_e > orig_e:
        print(f"  语义影响 : 信息量增加 {curr_e - orig_e:.4f} bits  → token 变得更\"意外\"")
    else:
        print(f"  语义影响 : 信息量减少 {orig_e - curr_e:.4f} bits  → token 变得更\"可预测\"")

    print("\n" + "─" * 78)
    print("【6. position —— 序列位置】")
    print("─" * 78)
    orig_pos = real_token["position"]
    curr_pos = token.metadata.get("position")
    source = token.metadata.get("position_source", "copied")
    energy = token.metadata.get("position_energy", 0.0)
    print(f"  代表什么 : token 在输入序列中的位置（0 = 第一个）")
    print(f"  原始值   : {orig_pos}")
    print(f"  变异后   : {curr_pos}")
    print(f"  变化幅度 : Δ={curr_pos - orig_pos:+d}")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    print(f"  语义影响 : 位置从第 {orig_pos} 个漂移到第 {curr_pos} 个")

    print("\n" + "─" * 78)
    print("【7. embedding —— 高维向量】")
    print("─" * 78)
    orig_emb = real_token["embedding"]
    curr_emb = token.metadata.get("embedding")
    source = token.metadata.get("embedding_source", "copied")
    energy = token.metadata.get("embedding_energy", 0.0)
    print(f"  代表什么 : token 的向量表示（12288 维），用于语义相似度计算")
    print(f"  原始值   : shape={orig_emb.shape}, norm={np.linalg.norm(orig_emb):.6f}")
    print(f"  变异后   : shape={curr_emb.shape}, norm={np.linalg.norm(curr_emb):.6f}")
    # 计算位移和相似度
    shift = float(np.linalg.norm(curr_emb - orig_emb))
    cos_sim = float(np.dot(orig_emb, curr_emb) / (
        np.linalg.norm(orig_emb) * np.linalg.norm(curr_emb)
    ))
    print(f"  变化幅度 : L2 位移 = {shift:.6f}")
    print(f"  余弦相似度 : {cos_sim:.6f}  (1.0 = 完全相同, 0 = 正交, -1 = 完全相反)")
    print(f"  来源     : {source}  (能量={energy:.3f})")
    # 向量每个维度的统计
    delta_vec = curr_emb - orig_emb
    print(f"  扰动统计 :")
    print(f"    均值  : {np.mean(delta_vec):+.6f}")
    print(f"    标准差: {np.std(delta_vec):.6f}")
    print(f"    最大值: {np.max(delta_vec):+.6f}")
    print(f"    最小值: {np.min(delta_vec):+.6f}")
    if cos_sim > 0.9:
        print(f"  语义影响 : 向量基本没变（相似度 > 0.9）")
    elif cos_sim > 0.5:
        print(f"  语义影响 : 向量部分变化（相似度 0.5~0.9）")
    elif cos_sim > 0:
        print(f"  语义影响 : 向量大幅变化（相似度 0~0.5），语义已偏离")
    else:
        print(f"  语义影响 : 向量正交或反向，语义完全不同")

    # ============================================================
    # metadata 多出来的字段
    # ============================================================
    print("\n" + "─" * 78)
    print("【8. metadata 多出来的所有字段】")
    print("─" * 78)
    all_keys = sorted(token.metadata.keys())
    print(f"  总钥匙数: {len(all_keys)}")
    # 7 个核心属性
    core_attrs = ["token_id", "text", "logprob", "rank",
                  "entropy_bits", "position", "embedding"]
    # 多出来的是 _source 和 _energy 后缀
    extra_source_keys = [k for k in all_keys if k.endswith("_source")]
    extra_energy_keys = [k for k in all_keys if k.endswith("_energy")]
    print(f"\n  ▸ 7 个核心属性: {core_attrs}")
    print(f"\n  ▸ 7 个 _source 字段（来源标记）:")
    for k in extra_source_keys:
        attr = k.replace("_source", "")
        print(f"      {k:<22} = {token.metadata[k]}  ← {attr} 的来源")
    print(f"\n  ▸ 7 个 _energy 字段（质变能量值）:")
    for k in extra_energy_keys:
        attr = k.replace("_energy", "")
        print(f"      {k:<22} = {token.metadata[k]:.3f}  ← {attr} 累积的质变能量")

    # ============================================================
    # collision_history
    # ============================================================
    print("\n" + "─" * 78)
    print("【9. collision_history —— 反应历史】")
    print("─" * 78)
    print(f"  总记录数: {len(token.collision_history)}")
    print(f"  示例（前 5 条）:")
    for i, h in enumerate(token.collision_history[:5]):
        print(f"    [{i+1}] {h}")
    print(f"  示例（后 5 条）:")
    for i, h in enumerate(token.collision_history[-5:]):
        idx = len(token.collision_history) - 5 + i + 1
        print(f"    [{idx}] {h}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 78)
    print("【变异解读总结】")
    print("=" * 78)
    print(f"  原始 token : id={orig_id} text={orig_text!r}  (一个完整的真实 token)")
    print(f"  变异后     : id={curr_id} text={curr_text[:40]!r}...")
    print()
    print(f"  7 个属性的来源全部从 'copied' 变成 '{token.metadata.get('token_id_source')}'")
    print(f"  数值全部漂移:")
    print(f"    - 整数 ID 漂移到词表边界")
    print(f"    - 文本被培养液前缀污染")
    print(f"    - 概率从 {np.exp(orig_lp)*100:.3f}% 变成 {np.exp(curr_lp)*100:.3f}%")
    print(f"    - 排名 {orig_rank} → {curr_rank}")
    print(f"    - 信息量 {orig_e:.2f} → {curr_e:.2f} bits")
    print(f"    - 位置 {orig_pos} → {curr_pos}")
    print(f"    - embedding 余弦相似度 {cos_sim:.4f}  (1.0=完全相同)")
    print()
    print(f"  metadata 从 7 个钥匙扩展到 {len(all_keys)} 个钥匙")
    print(f"  多出来的是 7 个 _source + 7 个 _energy = 14 个质变跟踪字段")
    print()
    print(f"  collision_history 记录了 {len(token.collision_history)} 次化学反应")
    print("=" * 78)


if __name__ == "__main__":
    main()
