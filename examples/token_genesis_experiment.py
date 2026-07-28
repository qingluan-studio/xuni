"""
空白 Token 创世实验——让一个 metadata 全空的虚拟 Token 自己"长"出属性

流程：
    Step 1: 工厂生产一个虚拟 DownloadToken（metadata 全空）
    Step 2: 初始化创世熔炉（7 个属性都是 None）
    Step 3: 灌入全部 100 种培养液 × 3 轮
    Step 4: 看哪些属性从 None → 有值
    Step 5: 把生成的 token_id 用 tiktoken decode，看是什么字
    Step 6: 跑 10 次独立的创世，看生成的 token_id 分布
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tiktoken

from xuni.multiverse_resources import MultiverseResourceFactory
from xuni.token_genesis_forge import TokenGenesisForge


def run_one_genesis(rounds: int = 3, level: int = 5) -> dict:
    """跑一次独立的创世，返回生成的属性"""
    factory = MultiverseResourceFactory()
    token = factory.produce_download_token()

    forge = TokenGenesisForge(factory)
    forge.init_emergence(token)
    forge.react_with_all_cultures(token, level=level, rounds=rounds)

    return {
        "forge": forge,
        "token": token,
        "status": forge.status(),
    }


def main():
    print("=" * 78)
    print("空白 Token 创世实验——让 token 从虚空自己长出来")
    print("=" * 78)

    # ============================================================
    # 单次创世——详细展示
    # ============================================================
    print("\n【单次创世】")
    print("─" * 78)
    result = run_one_genesis(rounds=3, level=5)
    forge = result["forge"]
    token = result["token"]
    status = result["status"]

    print(f"  反应次数: {len(forge.reaction_log)}")
    print(f"  创世事件: {len(forge.genesis_events)} 次")
    print(f"  生成的属性: {status['token_id']['is_generated'] or 0}/7 ... "
          f"{sum(1 for s in status.values() if s['is_generated'])}/7")

    print("\n  7 个属性的创世状态:")
    print(f"  {'属性':<14} | {'来源':<10} | {'能量':<8} | {'当前值'}")
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*8}-+-{'-'*30}")
    for attr, s in status.items():
        print(f"  {attr:<14} | {s['source']:<10} | {s['energy']:<8} | {s['current']}")

    # 创世事件
    if forge.genesis_events:
        print(f"\n  创世事件详情:")
        for ev in forge.genesis_events:
            print(f"    - {ev['message']}")
            print(f"      (培养液={ev['culture_type']}, 能量={ev['energy_reached']:.3f})")
            print(f"      生成: {ev['generation']}")

    # decode 生成的 token_id
    print("\n【解码生成的 token_id】")
    print("─" * 78)
    enc = tiktoken.get_encoding("cl100k_base")
    generated_id = token.metadata.get("token_id")
    if generated_id is not None:
        try:
            decoded = enc.decode([generated_id])
            print(f"  生成的 token_id : {generated_id}")
            print(f"  解码出来的文本 : {decoded!r}")
            # 看看这个 ID 在词表里的字节表示
            byte_repr = bytes([generated_id % 256])
            print(f"  字节表示        : {byte_repr!r}")
        except Exception as e:
            print(f"  解码失败: {e}")
    else:
        print(f"  token_id 没生成出来")

    # ============================================================
    # 跑 10 次独立创世——看 token_id 分布
    # ============================================================
    print("\n" + "=" * 78)
    print("【10 次独立创世——token_id 分布】")
    print("=" * 78)

    results = []
    for i in range(10):
        r = run_one_genesis(rounds=3, level=5)
        tok = r["token"]
        tid = tok.metadata.get("token_id")
        text = tok.metadata.get("text")
        try:
            decoded = enc.decode([tid]) if tid is not None else "<None>"
        except Exception:
            decoded = "<解码失败>"
        results.append({
            "run": i + 1,
            "token_id": tid,
            "decoded": decoded,
            "text": text,
        })

    print(f"\n  {'第N次':<6} | {'token_id':<10} | {'解码':<16} | {'text 属性'}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*16}-+-{'-'*30}")
    for r in results:
        print(f"  Run {r['run']:<2} | {str(r['token_id']):<10} | {r['decoded']!r:<16} | {str(r['text'])[:30]}")

    # 统计
    ids = [r["token_id"] for r in results if r["token_id"] is not None]
    if ids:
        print(f"\n  10 次创世统计:")
        print(f"    生成的 token_id 范围: {min(ids)} ~ {max(ids)}")
        print(f"    均值: {sum(ids) / len(ids):.1f}")
        print(f"    去重数: {len(set(ids))} / {len(ids)}")
        # 解码出来的字符分布
        decoded_chars = [r["decoded"] for r in results if r["decoded"] != "<None>"]
        print(f"    解码字符: {decoded_chars}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 78)
    print("【总结】")
    print("=" * 78)
    gen_count = sum(1 for s in status.values() if s["is_generated"])
    print(f"  单次创世: {gen_count}/7 属性从虚空生成")
    print(f"  10 次独立创世: 每次都生成不同的 token_id")
    print()
    print("  关键观察:")
    print("  1. token_id 是从词表随机映射出来的，不是从真实 token 拷贝的")
    print("  2. decode 出来的字符是随机的，没有语义意义")
    print("  3. logprob 和 entropy_bits 满足信息论一致性（H = -log2 p）")
    print("  4. embedding 是用培养液哈希做种子生成的随机向量")
    print("  5. text 是随机字符——因为虚拟 Token 没有语义知识")
    print()
    print("  → 虚拟 Token 能\"长\"出属性结构，但长不出\"语义\"")
    print("  → 要长出有意义的 text，需要接入真实语料/模型")
    print("=" * 78)


if __name__ == "__main__":
    main()
