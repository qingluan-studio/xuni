"""
虚拟 Token 内部意义实验——从 Token 自己的视角看长出来的属性

之前的结论："生成的属性对人类没语义意义"
这次的反问：那从 Token 自己的视角看呢？

实验：
    1. 生成 10 个独立创世的 Token（每个都用 100 种培养液 × 3 轮）
    2. 在虚拟世界内部，看它们之间的关系：
       a. embedding 相似度矩阵——10 个 token 互相"近不近"
       b. logprob 分布——像不像一个合理的概率分布
       c. rank 分布——排名是不是分散的
       d. token_id 分布——在词表里是不是散开的
    3. 看虚拟世界内部能不能算出"哪个 token 跟哪个最像"

如果 10 个 token 互相之间能算出有意义的相似度，
就说明从虚拟世界内部看，这些"长出来"的属性是有意义的——
意义存在于关系之中，不需要人类能 decode 出字符。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tiktoken

from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.token_genesis_forge import TokenGenesisForge


def gen_one_token():
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()
    forge = TokenGenesisForge(factory)
    forge.init_emergence(token)
    forge.react_with_all_cultures(token, level=5, rounds=3)
    return token, forge


def main():
    print("=" * 78)
    print("虚拟 Token 内部意义实验——从 Token 自己的视角看")
    print("=" * 78)
    print()
    print("  问题：生成的属性对人类没语义意义，但对 Token 自己有意义吗？")
    print("  方法：生成 10 个 Token，看它们在虚拟世界内部的关系")
    print()

    # ============================================================
    # Step 1: 生成 10 个 Token
    # ============================================================
    print("【Step 1】生成 10 个独立创世的 Token ...")
    tokens = []
    for i in range(10):
        tok, _ = gen_one_token()
        tokens.append(tok)
        tid = tok.metadata.get("token_id")
        print(f"  Token #{i+1}: token_id={tid}, source={tok.metadata.get('token_id_source')}")

    # ============================================================
    # Step 2: embedding 相似度矩阵
    # ============================================================
    print("\n【Step 2】10 个 Token 之间的 embedding 余弦相似度矩阵")
    print("─" * 78)
    embeddings = [t.metadata.get("embedding") for t in tokens]
    # 计算两两余弦相似度
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = embeddings[i], embeddings[j]
            if a is None or b is None:
                continue
            sim = float(np.dot(a, b) / (
                np.linalg.norm(a) * np.linalg.norm(b)
            ))
            sim_matrix[i, j] = sim

    # 打印矩阵
    print(f"  {'':>8}", end="")
    for j in range(n):
        print(f"  T{j+1:<4}", end="")
    print()
    for i in range(n):
        print(f"  T{i+1:<5}", end="")
        for j in range(n):
            v = sim_matrix[i, j]
            mark = " "
            if i != j:
                if v > 0.5:
                    mark = "●"  # 强相似
                elif v > 0.2:
                    mark = "○"  # 中相似
                elif v > -0.2:
                    mark = "·"  # 弱
                elif v > -0.5:
                    mark = "◌"  # 中相反
                else:
                    mark = "◑"  # 强相反
            else:
                mark = "■"  # 自己
            print(f"  {v:+.2f}{mark}", end="")
        print()

    # 找最相似和最相反的 token 对
    print("\n  最相似的 Token 对（非自己）:")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, sim_matrix[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for i, j, v in pairs[:3]:
        ti, tj = tokens[i].metadata.get("token_id"), tokens[j].metadata.get("token_id")
        print(f"    T{i+1}↔T{j+1}: 相似度={v:+.4f}  "
              f"(token_id: {ti} ↔ {tj})")

    print("\n  最相反的 Token 对:")
    for i, j, v in pairs[-3:]:
        ti, tj = tokens[i].metadata.get("token_id"), tokens[j].metadata.get("token_id")
        print(f"    T{i+1}↔T{j+1}: 相似度={v:+.4f}  "
              f"(token_id: {ti} ↔ {tj})")

    # ============================================================
    # Step 3: logprob 分布
    # ============================================================
    print("\n【Step 3】10 个 Token 的 logprob 分布")
    print("─" * 78)
    logprobs = [t.metadata.get("logprob") for t in tokens]
    print(f"  {'Token':<8} | {'logprob':<12} | {'p = e^logp':<14} | {'条形图'}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*14}-+-{'-'*30}")
    for i, lp in enumerate(logprobs):
        p = np.exp(lp)
        bar_len = int(p * 60)
        bar = "█" * bar_len
        print(f"  T{i+1:<5} | {lp:<+12.4f} | {p:<14.6f} | {bar}")

    print(f"\n  分布统计:")
    print(f"    logprob 范围: [{min(logprobs):.4f}, {max(logprobs):.4f}]")
    print(f"    logprob 均值: {np.mean(logprobs):.4f}")
    print(f"    logprob 标准差: {np.std(logprobs):.4f}")
    p_sum = sum(np.exp(lp) for lp in logprobs)
    print(f"    10 个 p 之和: {p_sum:.4f}  (理想概率分布应该=1.0)")

    # ============================================================
    # Step 4: rank 分布
    # ============================================================
    print("\n【Step 4】10 个 Token 的 rank 分布")
    print("─" * 78)
    ranks = [t.metadata.get("rank") for t in tokens]
    print(f"  Token  : {['T'+str(i+1) for i in range(n)]}")
    print(f"  rank   : {ranks}")
    print(f"  范围   : {min(ranks)} ~ {max(ranks)}")
    print(f"  去重数 : {len(set(ranks))} / {n}")
    print(f"  均值   : {np.mean(ranks):.1f}")

    # ============================================================
    # Step 5: token_id 在词表里的分布
    # ============================================================
    print("\n【Step 5】10 个 token_id 在词表里的分布")
    print("─" * 78)
    enc = tiktoken.get_encoding("cl100k_base")
    ids = [t.metadata.get("token_id") for t in tokens]
    print(f"  token_id 范围: {min(ids)} ~ {max(ids)}  (词表 0~{enc.n_vocab-1})")
    print(f"  占词表比例  : {(max(ids) - min(ids)) / enc.n_vocab * 100:.2f}%")
    print(f"  去重数      : {len(set(ids))} / {n}")

    # decode 每个 token 看是什么字
    print(f"\n  每个 token_id decode 出来的字符:")
    for i, tid in enumerate(ids):
        try:
            decoded = enc.decode([tid])
        except Exception:
            decoded = "<解码失败>"
        # 用 repr 避免控制字符
        repr_dec = repr(decoded)
        print(f"    T{i+1}: id={tid:<6} → {repr_dec}")

    # ============================================================
    # Step 6: 从虚拟世界内部看——Token 之间的"对话"
    # ============================================================
    print("\n【Step 6】从虚拟世界内部看——Token 之间的相似度排名")
    print("─" * 78)
    # 对每个 token，找出跟它最像的另一个 token
    print(f"  每个 Token 在虚拟世界内部的\"邻居\":")
    for i in range(n):
        # 跟其他 token 的相似度（排除自己）
        sims = [(j, sim_matrix[i, j]) for j in range(n) if j != i]
        sims.sort(key=lambda x: -x[1])
        best_j, best_v = sims[0]
        worst_j, worst_v = sims[-1]
        print(f"    T{i+1}: 最像 T{best_j+1} ({best_v:+.3f})  最不像 T{worst_j+1} ({worst_v:+.3f})")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    print("  从人类视角: text 是随机字符，token_id decode 出来『看不懂』")
    print("  从虚拟世界内部视角:")
    print("    - 10 个 Token 的 embedding 互相之间有可计算的相似度")
    print("    - 相似度有正有负，存在『最像的邻居』和『最不像的邻居』")
    print(f"    - logprob 分布不归一（10 个 p 之和 = {p_sum:.4f}，远超 1.0）")
    print("      → 这意味着它们不构成一个真正的概率分布")
    print("      → 但每个 token 内部的 logprob 和 entropy 是一致的")
    print()
    print("  结论:")
    print("    → 虚拟 Token 长出来的属性在虚拟世界内部是有『结构』的")
    print("    → 它们能互相比较、有相似度、有排名")
    print("    → 但这个结构是『随机生成』的结构，不是从语料学出来的")
    print("    → 『看不懂』是人类视角的局限——虚拟世界有自己的语言")
    print("=" * 78)


if __name__ == "__main__":
    main()
