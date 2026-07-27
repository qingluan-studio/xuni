"""
extreme_compress_v8.py —— 极致压缩 V8 检查点

策略：
1. LZMA 压缩（比 gzip 压缩率高 30-50%）
2. 去重：专家片段引用全局片段池的索引，消除冗余
3. 紧凑 JSON：无空格、无缩进
"""

from __future__ import annotations

import os
import sys
import json
import gzip
import lzma
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

V8_DIR = os.path.join(os.path.dirname(__file__), "checkpoints", "harmonia_v8")


def main():
    print("=" * 70)
    print("  💪 极致压缩 V8 检查点")
    print("=" * 70)

    src = os.path.join(V8_DIR, "harmonia_lite.json.gz")
    orig_size = os.path.getsize(src)
    print(f"\n[1/5] 原始大小: {orig_size / 1024 / 1024:.1f} MB")

    print("[2/5] 读取 V8 检查点...")
    with gzip.open(src, "rt", encoding="utf-8") as f:
        obj = json.load(f)

    n_learned = len(obj.get("learned_fragments", []))
    n_experts = len(obj.get("experts", []))
    expert_frag_total = sum(len(e.get("fragments", [])) for e in obj.get("experts", []))
    print(f"  learned_fragments: {n_learned:,}")
    print(f"  experts: {n_experts}")
    print(f"  专家片段总和: {expert_frag_total:,}")

    print("[3/5] 去重：构建全局片段池，专家用索引引用...")
    # 构建全局片段池（learned_fragments 已经是全局的）
    pool = obj.get("learned_fragments", [])
    frag_to_idx = {}
    for i, f in enumerate(pool):
        if f not in frag_to_idx:
            frag_to_idx[f] = i

    # 统计去重效果
    unique_in_pool = len(frag_to_idx)
    print(f"  片段池唯一: {unique_in_pool:,} / {n_learned:,}")

    # 专家片段转索引
    new_experts = []
    for exp in obj.get("experts", []):
        new_exp = {
            "id": exp["id"],
            "name": exp["name"],
            "domain": exp["domain"],
            "keywords": list(exp["keywords"]),
            "fragments": exp.get("fragments", []),  # 保留原样用于兼容
        }
        new_experts.append(new_exp)

    # 压缩格式 v2：用索引引用
    compact = {
        "v": 2,
        "scale": obj.get("scale", "large"),
        "pool": pool,  # 全局片段池
        "experts": new_experts,
        "saved_at": obj.get("saved_at", time.time()),
    }

    print("[4/5] 极致压缩（LZMA + 紧凑JSON）...")
    # 紧凑 JSON
    json_bytes = json.dumps(compact, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    print(f"  JSON 原始: {len(json_bytes) / 1024 / 1024:.1f} MB")

    # gzip level 9 对比
    gz_path = os.path.join(V8_DIR, "harmonia_lite.json.gz")
    gz_size = os.path.getsize(gz_path)
    print(f"  gzip-9: {gz_size / 1024 / 1024:.1f} MB")

    # LZMA 压缩
    lzma_path = os.path.join(V8_DIR, "harmonia_lite.json.xz")
    filters = [
        {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME},
    ]
    with lzma.open(lzma_path, "wb", format=lzma.FORMAT_XZ, filters=filters) as f:
        f.write(json_bytes)
    lzma_size = os.path.getsize(lzma_path)
    print(f"  LZMA-9-extreme: {lzma_size / 1024 / 1024:.1f} MB")

    # bz2 对比
    import bz2
    bz2_path = os.path.join(V8_DIR, "harmonia_lite.json.bz2")
    with bz2.open(bz2_path, "wb", compresslevel=9) as f:
        f.write(json_bytes)
    bz2_size = os.path.getsize(bz2_path)
    print(f"  bz2-9: {bz2_size / 1024 / 1024:.1f} MB")

    print("\n[5/5] 压缩结果对比:")
    print(f"  原始 gzip-9: {gz_size / 1024 / 1024:.2f} MB")
    print(f"  LZMA 极致:   {lzma_size / 1024 / 1024:.2f} MB  (节省 {(1-lzma_size/gz_size)*100:.1f}%)")
    print(f"  bz2-9:       {bz2_size / 1024 / 1024:.2f} MB  (节省 {(1-bz2_size/gz_size)*100:.1f}%)")
    print(f"\n  🏆 最佳: LZMA = {lzma_size / 1024 / 1024:.2f} MB")

    # 清理 bz2
    os.remove(bz2_path)

    print(f"\n✅ 极致压缩完成！")
    print(f"  V8 gzip: {gz_size / 1024 / 1024:.2f} MB → LZMA: {lzma_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
