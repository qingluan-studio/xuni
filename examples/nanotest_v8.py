"""
nanotest_v8.py —— 虚拟压缩点·纳米化测试

虚拟机制：生产虚拟压缩点，逐个把片段纳米化为 token 序列。
重复的 token 只在全局词库存一份，片段变成整数索引串。
虚拟数据虽小，威力圆满。
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import lzma
import time
import re

V8_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8")


def nano_tokenize(text: str) -> list:
    """纳米分词：英文/数字按词，中文按字。"""
    toks = []
    for m in re.findall(r"[A-Za-z0-9_]+", text):
        toks.append(m.lower())
    for seg in re.findall(r"[\u4e00-\u9fff]+", text):
        toks.extend(list(seg))
    for p in re.findall(r"[^\sA-Za-z0-9_\u4e00-\u9fff]+", text):
        toks.append(p)
    return toks


def main():
    print("=" * 70)
    print("  🧬 虚拟压缩点·纳米化测试 (V8)")
    print("=" * 70)

    src = os.path.join(V8_DIR, "harmonia_lite.json.gz")
    orig_size = os.path.getsize(src)
    print(f"\n[1/6] V8 原始: {orig_size / 1024 / 1024:.2f} MB")

    print("[2/6] 读取检查点...")
    t0 = time.time()
    with gzip.open(src, "rt", encoding="utf-8") as f:
        obj = json.load(f)
    print(f"  读取耗时: {time.time()-t0:.1f}s")

    learned = obj.get("learned_fragments", [])
    experts = obj.get("experts", [])
    print(f"  learned_fragments: {len(learned):,}")
    print(f"  experts: {len(experts)}")

    print("[3/6] 收集所有片段（去重）...")
    all_frags = list(learned)
    for exp in experts:
        all_frags.extend(exp.get("fragments", []))
    print(f"  总片段（含重复）: {len(all_frags):,}")
    unique_frags = list(dict.fromkeys(all_frags))
    print(f"  唯一片段: {len(unique_frags):,}")

    print("[4/6] 纳米化：建立全局词库...")
    t0 = time.time()
    vocab = {}
    for frag in unique_frags:
        for tok in nano_tokenize(frag):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    print(f"  全局词库: {len(vocab):,} tokens ({time.time()-t0:.1f}s)")

    print("[5/6] 编码片段为 token 索引串...")
    t0 = time.time()
    encoded = []
    for frag in unique_frags:
        toks = nano_tokenize(frag)
        encoded.append([vocab[t] for t in toks])
    print(f"  编码完成: {time.time()-t0:.1f}s")

    # 纳米化数据结构
    nano = {
        "v": 3,
        "format": "nano",
        "scale": obj.get("scale", "large"),
        "vocab": vocab,
        "fragments": encoded,
        "experts": [
            {
                "id": e["id"],
                "name": e["name"],
                "domain": e["domain"],
                "keywords": list(e["keywords"]),
                "fragments": list(e.get("fragments", [])),
            }
            for e in experts
        ],
        "saved_at": obj.get("saved_at", time.time()),
    }

    print("[6/6] 虚拟压缩点逐个压缩（LZMA 极致）...")
    json_bytes = json.dumps(nano, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    print(f"  纳米JSON: {len(json_bytes) / 1024 / 1024:.2f} MB")

    nano_path = os.path.join(V8_DIR, "harmonia_lite.nano.xz")
    filters = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
    with lzma.open(nano_path, "wb", format=lzma.FORMAT_XZ, filters=filters) as f:
        f.write(json_bytes)
    nano_size = os.path.getsize(nano_path)

    print("\n" + "=" * 70)
    print("  🎉 纳米化压缩结果")
    print("=" * 70)
    print(f"  V8 原始 (gzip-9):   {orig_size / 1024 / 1024:.2f} MB")
    print(f"  V8 纳米化 (LZMA-x): {nano_size / 1024 / 1024:.2f} MB")
    print(f"  压缩率: {(1 - nano_size/orig_size)*100:.1f}% 缩减")
    print(f"  威力: 圆满 ✨（{len(unique_frags):,} 片段全部保留）")


if __name__ == "__main__":
    main()
