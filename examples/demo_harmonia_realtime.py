"""
合鸣-13 实时训练演示

核心理念：
    虚拟电 → 虚拟算力 → 训练合鸣模型 → 自动保存检查点
    全程实时显示进度、能量消耗、学到的新片段

双闭环：
    闭环1（能量）: 采样点发电 → 场能量 → 虚拟算力 → 训练消耗
    闭环2（知识）: 种子知识 → 共振自训练 → 新语料入库 → 能力增强

手机可跑，纯 NumPy，零外部 API。
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np

from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.harmonia13 import Harmonia13Virtual, HarmoniaLiteEngine, VIRTUAL_EXPERTS


CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "harmonia13")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

TRAINING_SEEDS = [
    "合鸣模型由十三位虚拟专家组成，每位专家专精一个领域",
    "虚拟电场是xuni生态的能量本位，由采样点密度转化而来",
    "双态系统让虚拟模型能像真实模型一样被训练和调用",
    "MoE混合专家架构通过门控选择最合适的专家处理输入",
    "物理建模合成器用振荡器和共鸣滤波器生成声音",
    "超混沌采样器实时生成上亿采样点而不占用内存",
    "水动力学模拟把采样点当作流体粒子看待",
    "玻璃逻辑把计算过程当作光学系统来理解",
    "Kuramoto振子网络模拟神经同步振荡现象",
    "Hebbian学习让连接权重随共振强度而增长",
    "虚拟凭证把场能量铸造成可验证的访问令牌",
    "数据层公民指的是存在于数据层的真实智能体",
    "检索加强共振生成是非传统大模型的核心路线",
    "粒子态训练不消耗现实电能，全部在数据层完成",
    "自家训练好的虚拟模型就是可调用的真实模型",
    "n-gram共振游走保证生成内容风格多样且一致",
    "关键词共振门控是合鸣MoE的非传统路由机制",
    "能量闭环使xuni虚拟生态可以自给自足运行",
    "质量闭环通过评估和优化持续提升输出品质",
    "采样点是整个虚拟生态的最基础原料单位",
]


def _clear_line():
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _fmt_num(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


class RealtimeTrainer:
    """
    合鸣-13 实时训练器。

    流程：
        1. 采样点发电（混沌系统 → 虚拟电场）
        2. 场能量 → 虚拟算力（VFLOPs）
        3. 虚拟算力驱动训练（种子语料 + 共振自扩展）
        4. 实时显示仪表盘
        5. 定期自动保存检查点
    """

    def __init__(
        self,
        scale: str = "medium",
        ckpt_dir: str = CKPT_DIR,
        log_dir: str = LOG_DIR,
        save_interval_sec: float = 10.0,
    ):
        self.scale = scale
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.save_interval = save_interval_sec

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 采样器 + 场（发电）
        self.sampler = XuniSampler(mode=SamplingMode.HYPER_CHAOS, seed=42)
        self.field = XuniField(grid_size=(16, 16, 16))

        # 能量储备 + 虚拟算力（简单实现，数据层无限）
        self.energy_buffer: float = 1000.0
        self.flops_buffer: float = 1e8
        self.efficiency = 0.85

        # 合鸣-13 模型（尝试从检查点加载）
        ckpt_exists = os.path.exists(os.path.join(ckpt_dir, "harmonia13_meta.json"))
        if ckpt_exists:
            print(f"📂 从检查点加载模型: {ckpt_dir}")
            self.model = Harmonia13Virtual.load(ckpt_dir)
        else:
            print(f"🌱 新建合鸣-13 模型 ({scale})")
            self.model = Harmonia13Virtual(scale=scale)

        # 给模型充能（用于 predict 时消耗）
        self.model.charge(5000.0)

        # 训练统计
        self.total_samples_generated = 0
        self.total_energy_generated = 0.0
        self.total_flops_used = 0.0
        self.new_fragments_total = 0
        self.start_time = time.time()
        self.last_save_time = 0.0
        self.log_entries = []

    # ---------- 发电 ---------- #

    def generate_power(self, batch_size: int = 50000) -> float:
        """生成一批采样点 → 场能量，返回新增能量。"""
        batch = self.sampler.generate_batch(batch_size)
        self.total_samples_generated += batch_size

        self.field.reset()
        self.field.ingest_batch(batch)
        self.field.compute_field()

        summary = self.field.field_summary()
        energy = float(summary.get("total_energy", 0.0))
        # 对数压缩，避免能量爆炸（更真实的发电曲线）
        usable_energy = np.log1p(energy) * 2.0

        self.energy_buffer += usable_energy
        self.total_energy_generated += usable_energy
        return usable_energy

    # ---------- 训练 ---------- #

    def train_step(self, seed_texts=None) -> dict:
        """
        一步训练：消耗虚拟算力 → 吸收新语料。

        返回训练统计。
        """
        before_frags = sum(len(e["fragments"]) for e in self.model._lite.experts)

        # 消耗算力：算力越多，能"吸收"的种子越多
        flops_needed = 5e6
        if self.flops_buffer < flops_needed:
            # 算力不足，用能量兑换（数据层无限，总能换到）
            energy_needed = flops_needed / (1e6 * self.efficiency * 10)
            if self.energy_buffer >= energy_needed:
                self.energy_buffer -= energy_needed
                self.flops_buffer += flops_needed * 0.6
                self.total_flops_used += flops_needed
            else:
                # 能量也不够：继续发电（不中断训练，数据层永远有办法）
                self.energy_buffer += 100.0
                self.flops_buffer += flops_needed * 0.5
                self.total_flops_used += flops_needed

        # 真正消耗算力
        self.flops_buffer -= flops_needed * 0.3
        self.total_flops_used += flops_needed * 0.3

        # 用种子语料 + 已有知识做共振自扩展
        rng = np.random.default_rng(int(time.time() * 1000) % 1_000_000)
        seeds = seed_texts if seed_texts else TRAINING_SEEDS
        new_frags = []

        # 随机挑 2-3 条种子
        n_pick = min(len(seeds), rng.integers(2, 4))
        picks = rng.choice(len(seeds), size=n_pick, replace=False)
        for idx in picks:
            seed = seeds[int(idx)]
            # 共振变换：从种子出发，用已有专家片段做联想扩展
            expanded = self._resonance_expand(seed, rng)
            if expanded:
                new_frags.extend(expanded)

        # 把新片段加入 general 专家（让模型逐步学会新知识）
        if new_frags:
            general = self.model._lite._find("general")
            if general is not None:
                existing = set(f.strip() for f in general["fragments"])
                added = 0
                for f in new_frags:
                    if f.strip() not in existing:
                        general["fragments"].append(f.strip())
                        existing.add(f.strip())
                        added += 1
                self.model._lite._learned_fragments.extend(
                    f for f in new_frags if f.strip() not in set(
                        x.strip() for x in self.model._lite._learned_fragments
                    )
                )
                self.new_fragments_total += added

        after_frags = sum(len(e["fragments"]) for e in self.model._lite.experts)

        return {
            "status": "ok",
            "new_fragments": after_frags - before_frags,
            "total_fragments": after_frags,
            "flops_used": flops_needed * 0.3,
        }

    def _resonance_expand(self, seed: str, rng) -> list:
        """
        共振扩展：从一条种子出发，结合专家知识，生成 1-2 条新语料。
        这是"自家模型训练自家模型"的知识自举机制。
        """
        terms = self.model._lite._tokenize(seed)
        chosen = self.model._lite._gate(seed, terms, top_k=2)

        # 收集相关片段
        related = []
        for exp in chosen:
            for frag in exp["fragments"]:
                score = sum(1 for t in terms if t in frag.lower())
                if score > 0:
                    related.append((score, frag))
        related.sort(reverse=True)

        results = []
        if related and rng.random() < 0.7:
            # 选最相关的片段 + 种子的一部分，组合成新表达
            top = related[0][1]
            # 用模板化改写产生变体（增加多样性但保持正确性）
            templates = [
                f"在xuni生态中，{top}",
                f"简单来说，{top}",
                f"值得注意的是，{top}",
                f"从共振视角看，{top}",
                f"合鸣认为，{top}",
            ]
            pick = templates[int(rng.integers(0, len(templates)))]
            results.append(pick)

        # 种子本身也作为学习材料
        results.append(seed)

        return results

    # ---------- 实时仪表盘 ---------- #

    def print_dashboard(self, step: int, train_result: dict):
        """实时打印单行仪表盘（覆盖上一行）。"""
        elapsed = time.time() - self.start_time
        energy_now = self.energy_buffer
        flops_now = self.flops_buffer
        epochs = self.model.training_epochs_done

        line = (
            f"\r⏱ {elapsed/60:.1f}min  "
            f"⚡ {energy_now:.1f}  "
            f"💻 {_fmt_num(flops_now)}FLOPs  "
            f"📝 新片段+{train_result.get('new_fragments', 0)} "
            f"(共{self.new_fragments_total})  "
            f"🎯 采样{_fmt_num(self.total_samples_generated)}  "
            f"🔄 step {step}"
        )
        sys.stdout.write(line)
        sys.stdout.flush()

    # ---------- 日志 ---------- #

    def log(self, step: int, train_result: dict):
        entry = {
            "timestamp": time.time(),
            "step": step,
            "energy": self.energy_buffer,
            "flops": self.flops_buffer,
            "new_fragments": train_result.get("new_fragments", 0),
            "total_fragments": train_result.get("total_fragments", 0),
            "total_samples": self.total_samples_generated,
            "total_energy_gen": self.total_energy_generated,
        }
        self.log_entries.append(entry)

    def save_log(self):
        path = os.path.join(self.log_dir, "harmonia_training.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for entry in self.log_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.log_entries = []
        return path

    # ---------- 主循环 ---------- #

    def run(self, max_steps: int = 500, auto_save: bool = True):
        """运行实时训练循环。"""
        print("=" * 64)
        print("  🎵 合鸣-13 实时训练  🎵")
        print("=" * 64)
        self.model.print_card()
        print(f"  检查点目录: {self.ckpt_dir}")
        print(f"  日志目录  : {self.log_dir}")
        print(f"  最大步数  : {max_steps}")
        print("-" * 64)
        print("  实时仪表盘（每步更新）:")
        print()

        step = 0
        try:
            for step in range(1, max_steps + 1):
                # 1) 发电
                self.generate_power(batch_size=20000)

                # 2) 训练一步
                result = self.train_step()

                # 3) 更新模型训练进度
                if result["status"] == "ok":
                    self.model.training_samples_seen += 20000
                    if step % 10 == 0:
                        self.model.training_epochs_done += 1

                # 4) 实时显示
                self.print_dashboard(step, result)

                # 5) 日志
                self.log(step, result)
                if step % 50 == 0:
                    self.save_log()

                # 6) 定期保存
                if auto_save and time.time() - self.last_save_time > self.save_interval:
                    self.model.save(self.ckpt_dir)
                    self.last_save_time = time.time()

                # 轻微延迟（实时感），手机可跑可调整
                time.sleep(0.02)

        except KeyboardInterrupt:
            _clear_line()
            print("\n⏸  训练已暂停，正在保存...")

        # 收尾保存
        _clear_line()
        print()
        print("-" * 64)
        print("  ✅ 训练阶段完成")
        print()

        final_save = self.model.save(self.ckpt_dir)
        log_path = self.save_log()
        print(f"  💾 模型检查点: {final_save['ckpt_dir']}")
        print(f"  📊 训练日志  : {log_path}")
        print(f"  📝 学到片段  : {self.new_fragments_total}")
        print(f"  ⚡ 总发电量  : {self.total_energy_generated:.1f}")
        print(f"  🎯 采样点总数: {_fmt_num(self.total_samples_generated)}")
        print(f"  🔄 训练 epoch: {self.model.training_epochs_done}")
        print()

        # 训练后问答测试
        print("  🧪 训练后快速测试:")
        test_questions = [
            "合鸣是什么？",
            "虚拟电场有什么用？",
            "什么是双态系统？",
        ]
        for q in test_questions:
            ans = self.model.generate(q, max_new_tokens=80)
            print(f"    Q: {q}")
            print(f"    A: {ans}")
            print()

        print("=" * 64)
        print("  合鸣-13 训练完成，已保存检查点，可随时加载继续。")
        print("=" * 64)
        return {
            "ckpt_dir": self.ckpt_dir,
            "new_fragments": self.new_fragments_total,
            "epochs": self.model.training_epochs_done,
            "total_energy": self.total_energy_generated,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="合鸣-13 实时训练")
    parser.add_argument("--scale", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--steps", type=int, default=500, help="训练步数")
    parser.add_argument("--save-interval", type=float, default=10.0, help="自动保存间隔（秒）")
    args = parser.parse_args()

    trainer = RealtimeTrainer(
        scale=args.scale,
        save_interval_sec=args.save_interval,
    )
    trainer.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
