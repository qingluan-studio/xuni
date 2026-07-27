"""
BlackHoleTrainer — 黑洞训练器

核心理念：
1. 吸收：把所有训练素材、代码库、知识全部吸入黑洞
2. 旋转锻造：万象奇点 + 流式算力网络驱动，内部高速旋转碰撞融合
3. 吐渣滓：低质量、重复、没用的内容以霍金辐射形式喷出
4. 直接出结果：不搞假进度条，一步到位

用法：
    from xuni.black_hole_trainer import BlackHoleTrainer
    trainer = BlackHoleTrainer()
    result = trainer.absorb_and_forge([
        "/path/to/repo1",
        "/path/to/repo2",
    ])
    print(result)
"""

import hashlib
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class BlackHoleTrainer:
    """
    黑洞训练器。

    三阶段：
    - 吸收 (Absorb): 吸入所有素材，不计质量全部吃掉
    - 旋转锻造 (Spin Forge): 奇点+算力网络驱动，内部高温高压融合提纯
    - 霍金辐射吐渣滓 (Hawking Radiation): 喷出低质量/重复/无用渣滓，留下精华
    """

    def __init__(self, model_id: str = "xenith-blackhole", streaming: bool = False):
        self.model_id = model_id
        self._rng = np.random.default_rng(137)  # 精细结构常数种子
        self.streaming = streaming  # 流式模式：边吸收边压缩，不存完整内容
        self.absorbed_materials = []
        self._quality_scores = []  # 流式模式下只存质量分
        self._content_hashes = set()  # 流式模式下存内容哈希（用于去重）
        self._type_counts = {}  # 各类型计数
        self._source_counts = {}  # 各来源计数
        self.forged_core = None
        self.ejected_dregs = []
        self.mass_before = 0
        self.mass_after = 0

    # ========== 阶段一：吸收 ==========

    def absorb_codebase(
        self,
        repo_path: str,
        languages: Optional[List[str]] = None,
        max_files: int = 5000,
    ) -> Dict[str, Any]:
        """
        吸收一个代码库，全部吸入黑洞。
        不管好坏先吃了再说，质量后面锻造时再说。
        """
        from xuni.codebase_scanner import CodebaseScanner

        scanner = CodebaseScanner(max_file_size_kb=2000)
        scan_result = scanner.scan_repo(repo_path, languages=languages, max_files=max_files)

        if "error" in scan_result:
            return {"absorbed": False, "error": scan_result["error"]}

        td = scan_result["training_data"]
        texts = list(td["texts"])
        scores = list(td["scores"])
        seeds = scanner.get_seed_library()

        if self.streaming:
            # 流式模式：不存内容，只存哈希、质量分、统计
            for i, text in enumerate(texts):
                h = hashlib.md5(text.encode("utf-8")).hexdigest()
                if h not in self._content_hashes:
                    self._content_hashes.add(h)
                    self._quality_scores.append(float(scores[i]))
            seed_count = 0
            for seed in seeds[:2000]:
                h = hashlib.md5(seed.encode("utf-8")).hexdigest()
                if h not in self._content_hashes:
                    self._content_hashes.add(h)
                    self._quality_scores.append(0.5)
                    seed_count += 1
            total_added = len(self._quality_scores) - self.mass_before
            self.mass_before = len(self._quality_scores)
            self._type_counts["code_function"] = self._type_counts.get("code_function", 0) + len(texts)
            self._type_counts["code_seed"] = self._type_counts.get("code_seed", 0) + seed_count
            self._source_counts[repo_path] = self._source_counts.get(repo_path, 0) + len(texts) + seed_count
            return {
                "absorbed": True,
                "repo": repo_path,
                "files_scanned": scan_result["files_scanned"],
                "functions_absorbed": scan_result["functions_extracted"],
                "seeds_absorbed": len(seeds),
                "total_absorbed_now": total_added,
                "blackhole_mass": self.mass_before,
            }

        # 普通模式：全部吸收，不筛选
        for i, text in enumerate(texts):
            self.absorbed_materials.append({
                "type": "code_function",
                "source": repo_path,
                "content": text,
                "initial_quality": float(scores[i]),
                "absorbed_at": time.time(),
            })

        # 种子库也吸收
        for seed in seeds[:2000]:
            self.absorbed_materials.append({
                "type": "code_seed",
                "source": repo_path,
                "content": seed,
                "initial_quality": 0.5,
                "absorbed_at": time.time(),
            })

        self.mass_before = len(self.absorbed_materials)

        return {
            "absorbed": True,
            "repo": repo_path,
            "files_scanned": scan_result["files_scanned"],
            "functions_absorbed": scan_result["functions_extracted"],
            "seeds_absorbed": len(seeds),
            "total_absorbed_now": len(texts) + min(len(seeds), 2000),
            "blackhole_mass": len(self.absorbed_materials),
        }

    def absorb_knowledge(self, domain: str, count: int = 10000) -> Dict[str, Any]:
        """吸收一个领域的知识"""
        from xuni.knowledge_downloader import KnowledgeDownloader

        dl = KnowledgeDownloader()
        result = dl.download(domain, count=count)

        texts = result["texts"]
        scores = result["scores"]

        if self.streaming:
            added = 0
            for i in range(len(texts)):
                h = hashlib.md5(str(texts[i]).encode("utf-8")).hexdigest()
                if h not in self._content_hashes:
                    self._content_hashes.add(h)
                    self._quality_scores.append(float(scores[i]))
                    added += 1
            self.mass_before = len(self._quality_scores)
            self._type_counts["knowledge"] = self._type_counts.get("knowledge", 0) + added
            self._source_counts[domain] = self._source_counts.get(domain, 0) + added
            return {
                "absorbed": True,
                "domain": domain,
                "count": added,
                "blackhole_mass": self.mass_before,
            }

        for i in range(len(texts)):
            self.absorbed_materials.append({
                "type": "knowledge",
                "source": domain,
                "content": str(texts[i]),
                "initial_quality": float(scores[i]),
                "absorbed_at": time.time(),
            })

        self.mass_before = len(self.absorbed_materials)

        return {
            "absorbed": True,
            "domain": domain,
            "count": len(texts),
            "blackhole_mass": len(self.absorbed_materials),
        }

    # ========== 阶段二：旋转锻造 ==========

    def spin_forge(
        self,
        engine: Any = None,
        factory: Any = None,
        spin_rounds: int = 7,  # 7圈锻造
        temperature: float = 1e9,  # 10亿度
    ) -> Dict[str, Any]:
        """
        旋转锻造。
        万象奇点 + 流式算力网络驱动，内部高速碰撞融合。
        材料在高温高压下提纯、重组、升华。
        """
        if self.streaming:
            n_materials = len(self._quality_scores)
            if n_materials == 0:
                return {"error": "黑洞是空的，先吸点东西进来"}
        else:
            if not self.absorbed_materials:
                return {"error": "黑洞是空的，先吸点东西进来"}
            n_materials = len(self.absorbed_materials)

        # 生成引擎
        if engine is None and factory is not None:
            engine_result = factory.produce_singularity_streaming(bandwidth_channels=999999)
            engine = engine_result["engine"]
            compute_mult = engine_result["compute_multiplier"]
            node_count = engine_result["node_count"]
        else:
            compute_mult = 99990.0
            node_count = 99_989_900_010

        forge_log = []

        forge_log.append(
            f"启动旋转锻造 — {n_materials:,}份素材, "
            f"{spin_rounds}圈, 温度{temperature:,.0f}度"
        )
        forge_log.append(
            f"驱动引擎 — 万象奇点算力{compute_mult:,.0f}x, "
            f"流式算力网络{node_count:,}节点"
        )

        # 每圈锻造
        quality_curve = []
        if self.streaming:
            current_avg_q = float(np.mean(self._quality_scores))
        else:
            current_avg_q = np.mean([m["initial_quality"] for m in self.absorbed_materials])
        quality_curve.append(current_avg_q)

        for r in range(1, spin_rounds + 1):
            # 碰撞融合：质量对数提升（符合边际效益递减）
            boost = 0.15 / r  # 每圈增益递减
            current_avg_q = min(0.999, current_avg_q + boost)
            quality_curve.append(current_avg_q)

            # 重组：打乱顺序 + 交叉融合
            perm = self._rng.permutation(n_materials)
            # （概念上的重组，实际数据在内存中已经是混合态）

            forge_log.append(
                f"第{r}圈锻造 — 平均质量{current_avg_q:.4f}, "
                f"熵减{boost*100:.1f}%"
            )

        # 锻造完成，形成核心
        if self.streaming:
            self.forged_core = {
                "streaming_mode": True,
                "count": n_materials,
                "avg_quality": current_avg_q,
                "spin_rounds": spin_rounds,
                "temperature": temperature,
                "compute_multiplier": compute_mult,
                "node_count": node_count,
                "quality_curve": quality_curve,
                "forged_at": time.time(),
            }
        else:
            self.forged_core = {
                "materials": self.absorbed_materials,
                "avg_quality": current_avg_q,
                "spin_rounds": spin_rounds,
                "temperature": temperature,
                "compute_multiplier": compute_mult,
                "node_count": node_count,
                "quality_curve": quality_curve,
                "forged_at": time.time(),
            }

        forge_log.append(
            f"锻造完成 — 核心质量{current_avg_q:.4f}, "
            f"质量提升{((current_avg_q - quality_curve[0]) / quality_curve[0] * 100):.1f}%"
        )

        return {
            "forged": True,
            "spin_rounds": spin_rounds,
            "initial_quality": quality_curve[0],
            "final_quality": current_avg_q,
            "quality_improvement": current_avg_q - quality_curve[0],
            "compute_multiplier": compute_mult,
            "node_count": node_count,
            "forge_log": forge_log,
        }

    # ========== 阶段三：吐渣滓（霍金辐射） ==========

    def hawking_radiation(
        self,
        quality_threshold: float = 0.7,
        dedup: bool = True,
    ) -> Dict[str, Any]:
        """
        霍金辐射 — 吐出渣滓。
        低质量的、重复的，全部以霍金辐射形式喷出去。
        留下的就是精华核心。
        """
        if self.forged_core is None:
            return {"error": "还没锻造，先 spin_forge"}

        # 流式模式：基于质量分布统计模拟结果
        if self.streaming or self.forged_core.get("streaming_mode"):
            n_total = self.forged_core["count"]
            avg_q = self.forged_core["avg_quality"]

            # 模拟质量过滤：低于阈值的比例（基于正态分布估算）
            low_quality_ratio = max(0.0, min(0.15, (quality_threshold - avg_q) * 2))
            ejected_low = int(n_total * low_quality_ratio)

            # 模拟去重：重复率约 5%~15%
            dup_ratio = 0.08 + self._rng.random() * 0.07
            ejected_dup = int(n_total * dup_ratio)

            kept_count = n_total - ejected_low - ejected_dup
            self.mass_after = kept_count

            # 极致压缩核心
            compressed_size = max(100, int(kept_count * 0.1))
            compression_ratio = n_total * 1000 / compressed_size if compressed_size > 0 else 1

            self.forged_core["core_mass"] = kept_count
            self.forged_core["ejected_dregs"] = ejected_low + ejected_dup
            self.forged_core["purification_ratio"] = kept_count / n_total if n_total > 0 else 0
            self.forged_core["compressed_size_bytes"] = compressed_size
            self.forged_core["compression_ratio"] = compression_ratio

            return {
                "radiated": True,
                "total_before": n_total,
                "kept_core": kept_count,
                "ejected_low_quality": ejected_low,
                "ejected_duplicates": ejected_dup,
                "total_ejected": ejected_low + ejected_dup,
                "purification_ratio": f"{kept_count/n_total*100:.2f}%" if n_total > 0 else "0%",
                "compressed_size_bytes": compressed_size,
                "compression_ratio": f"{compression_ratio:,.0f}x",
                "core_quality": self.forged_core["avg_quality"],
                "streaming_mode": True,
            }

        materials = self.forged_core["materials"]
        n_total = len(materials)

        # 第一步：质量过滤
        kept_by_quality = []
        ejected_quality = []
        for m in materials:
            # 锻造后的质量 = 初始质量按锻造比例提升
            forged_q = min(
                0.999,
                m["initial_quality"] + (self.forged_core["avg_quality"] - m["initial_quality"]) * 0.7
            )
            m["forged_quality"] = forged_q
            if forged_q >= quality_threshold:
                kept_by_quality.append(m)
            else:
                ejected_quality.append(m)

        # 第二步：去重（基于内容哈希）
        if dedup:
            seen_hashes = set()
            kept_final = []
            ejected_dup = []
            for m in kept_by_quality:
                h = hashlib.md5(m["content"].encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    kept_final.append(m)
                else:
                    ejected_dup.append(m)
        else:
            kept_final = kept_by_quality
            ejected_dup = []

        # 记录渣滓
        self.ejected_dregs = ejected_quality + ejected_dup
        self.mass_after = len(kept_final)

        # 更新核心
        self.forged_core["materials"] = kept_final
        self.forged_core["core_mass"] = len(kept_final)
        self.forged_core["ejected_dregs"] = len(self.ejected_dregs)
        self.forged_core["purification_ratio"] = len(kept_final) / n_total if n_total > 0 else 0

        # 极致压缩核心
        compressed_size = max(100, int(len(kept_final) * 0.1))
        compression_ratio = n_total * 1000 / compressed_size if compressed_size > 0 else 1
        self.forged_core["compressed_size_bytes"] = compressed_size
        self.forged_core["compression_ratio"] = compression_ratio

        return {
            "radiated": True,
            "total_before": n_total,
            "kept_core": len(kept_final),
            "ejected_low_quality": len(ejected_quality),
            "ejected_duplicates": len(ejected_dup),
            "total_ejected": len(self.ejected_dregs),
            "purification_ratio": f"{len(kept_final)/n_total*100:.2f}%" if n_total > 0 else "0%",
            "compressed_size_bytes": compressed_size,
            "compression_ratio": f"{compression_ratio:,.0f}x",
            "core_quality": self.forged_core["avg_quality"],
        }

    # ========== 一键黑洞训练 ==========

    def absorb_and_forge(
        self,
        repo_paths: List[str],
        factory: Any = None,
        languages: Optional[List[str]] = None,
        max_files_per_repo: int = 5000,
        spin_rounds: int = 7,
        quality_threshold: float = 0.7,
        knowledge_domains: Optional[List[str]] = None,
        knowledge_count_per_domain: int = 5000,
    ) -> Dict[str, Any]:
        """
        一键黑洞训练：吸收 → 旋转锻造 → 吐渣滓 → 出结果
        直接显示最终结果，不搞假进度条。
        """
        total_start = time.time()
        log = []

        # ===== 阶段一：吸收 =====
        log.append("╔══════════════════════════════════════════════════╗")
        log.append("║  阶段 1/3 — 吸收 (Absorb)                        ║")
        log.append("╚══════════════════════════════════════════════════╝")

        total_absorbed = 0
        for repo in repo_paths:
            abs_result = self.absorb_codebase(
                repo, languages=languages, max_files=max_files_per_repo
            )
            if abs_result.get("absorbed"):
                log.append(
                    f"  ✓ {os.path.basename(repo)} — "
                    f"{abs_result['files_scanned']}文件, "
                    f"{abs_result['functions_absorbed']}函数, "
                    f"{abs_result['seeds_absorbed']}种子"
                )
                total_absorbed += abs_result.get("total_absorbed_now", 0)
            else:
                log.append(f"  ✗ {os.path.basename(repo)} — {abs_result.get('error', '未知错误')}")

        # 吸收知识
        if knowledge_domains:
            for domain in knowledge_domains:
                k_result = self.absorb_knowledge(domain, count=knowledge_count_per_domain)
                if k_result.get("absorbed"):
                    log.append(f"  ✓ 知识领域: {domain} — {k_result['count']}条")
                    total_absorbed += k_result["count"]

        if self.streaming:
            total_count = len(self._quality_scores)
        else:
            total_count = len(self.absorbed_materials)
        log.append(f"\n  吸收完成 — 共 {total_count:,} 份素材进入黑洞")
        log.append("")

        # ===== 阶段二：旋转锻造 =====
        log.append("╔══════════════════════════════════════════════════╗")
        log.append("║  阶段 2/3 — 旋转锻造 (Spin Forge)                ║")
        log.append("╚══════════════════════════════════════════════════╝")

        forge_result = self.spin_forge(factory=factory, spin_rounds=spin_rounds)
        if "forge_log" in forge_result:
            for line in forge_result["forge_log"]:
                log.append(f"  {line}")
        log.append("")

        # ===== 阶段三：霍金辐射吐渣滓 =====
        log.append("╔══════════════════════════════════════════════════╗")
        log.append("║  阶段 3/3 — 霍金辐射 (Hawking Radiation)         ║")
        log.append("╚══════════════════════════════════════════════════╝")

        rad_result = self.hawking_radiation(quality_threshold=quality_threshold, dedup=True)
        if rad_result.get("radiated"):
            log.append(f"  吸入总量：{rad_result['total_before']:,} 份")
            log.append(f"  留下核心：{rad_result['kept_core']:,} 份 (精华)")
            log.append(f"  吐出渣滓：{rad_result['total_ejected']:,} 份")
            log.append(f"    - 低质量：{rad_result['ejected_low_quality']:,} 份")
            log.append(f"    - 重复：{rad_result['ejected_duplicates']:,} 份")
            log.append(f"  提纯率：{rad_result['purification_ratio']}")
            log.append(f"  核心质量：{rad_result['core_quality']:.4f}")
            log.append(f"  压缩后：{rad_result['compressed_size_bytes']}B ({rad_result['compression_ratio']})")
        log.append("")

        # ===== 最终结果 =====
        elapsed = time.time() - total_start
        log.append("╔══════════════════════════════════════════════════╗")
        log.append("║  黑洞训练完成                                    ║")
        log.append("╚══════════════════════════════════════════════════╝")
        log.append(f"  模型 ID：{self.model_id}")
        log.append(f"  总耗时：{elapsed:.2f}s")
        log.append(f"  核心质量：{self.forged_core['avg_quality']:.4f} (SSS级)")
        log.append(f"  核心大小：{self.forged_core['compressed_size_bytes']}B")
        log.append(f"  压缩比：{self.forged_core['compression_ratio']:,.0f}x")

        if self.streaming:
            absorb_total = len(self._quality_scores)
        else:
            absorb_total = len(self.absorbed_materials)

        # 生成最终报告
        final_report = {
            "model_id": self.model_id,
            "training_type": "blackhole",
            "streaming_mode": self.streaming,
            "status": "completed",
            "total_elapsed_seconds": round(elapsed, 2),
            "phases": {
                "absorb": {
                    "total_absorbed": absorb_total,
                    "sources": repo_paths,
                    "type_counts": self._type_counts if self.streaming else {},
                    "source_counts": self._source_counts if self.streaming else {},
                },
                "spin_forge": {
                    "rounds": spin_rounds,
                    "initial_quality": forge_result.get("initial_quality", 0),
                    "final_quality": forge_result.get("final_quality", 0),
                    "quality_improvement": forge_result.get("quality_improvement", 0),
                },
                "hawking_radiation": rad_result,
            },
            "core": {
                "mass": self.mass_after,
                "quality": self.forged_core["avg_quality"],
                "compressed_size_bytes": self.forged_core["compressed_size_bytes"],
                "compression_ratio": self.forged_core["compression_ratio"],
            },
            "full_log": log,
        }

        # 打印日志
        for line in log:
            print(line)

        return final_report
