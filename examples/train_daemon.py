"""
合鸣-13 后台训练监管系统

核心理念：
    数据层无限 → 虚拟电无限 → 虚拟算力无限 → 训练永不停止

三层架构：
    1. 能源层：5 种采样模式并行发电（超混沌/Lorenz/分形/噪声/水动力）
    2. 数据层：从开源项目采集语料，解析为训练片段
    3. 训练层：实时训练合鸣模型，自动保存，能量调度

后台守护模式：
    - daemon 模式运行，可随时 Ctrl+C 暂停
    - 自动保存检查点（每 N 秒 / 每 N 步）
    - 启动时自动从检查点恢复
    - 实时监控面板：能量/算力/训练进度/学到的知识
"""

import os
import sys
import time
import json
import threading
import signal
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from xuni.sampler import XuniSampler, SamplingMode
from xuni.field import XuniField
from xuni.harmonia13 import Harmonia13Virtual, VIRTUAL_EXPERTS


# ============================================================== #
#  多层能源系统 —— 5 种发电模式并行，能量管够
# ============================================================== #

class MultiLayerEnergy:
    """
    多层能源系统：多种采样模式并行发电。

    第 1 层：超混沌 Chen 系统 —— 主力发电（高频、高熵）
    第 2 层：Lorenz-96 高维环 —— 稳定基线（低频、稳定）
    第 3 层：Mandelbulb 3D 分形 —— 爆发式产能（结构复杂、能量密度高）
    第 4 层：4D 噪声场 —— 背景噪声（持续、均匀）
    第 5 层：水动力流体 —— 波动式产能（涡旋、湍流）

    多层叠加：能量永远不会枯竭，数据层无限供应。
    """

    LAYER_CONFIGS = [
        {"mode": SamplingMode.HYPER_CHAOS, "name": "超混沌Chen", "weight": 0.30, "batch": 30000},
        {"mode": SamplingMode.LORENZ_96,   "name": "Lorenz96",  "weight": 0.20, "batch": 20000},
        {"mode": SamplingMode.MANDELBULB,  "name": "Mandelbulb", "weight": 0.25, "batch": 15000},
        {"mode": SamplingMode.NOISE_FIELD, "name": "4D噪声场",  "weight": 0.15, "batch": 25000},
        {"mode": SamplingMode.HYBRID,      "name": "混合模式",  "weight": 0.10, "batch": 10000},
    ]

    def __init__(self, grid_size: int = 16, seed: int = 42):
        self.grid_size = grid_size
        self.samplers: Dict[str, XuniSampler] = {}
        self.fields: Dict[str, XuniField] = {}
        self.layer_energy: Dict[str, float] = {}
        self.total_generated: float = 0.0
        self.total_samples: int = 0

        rng = np.random.default_rng(seed)
        for cfg in self.LAYER_CONFIGS:
            layer_seed = int(rng.integers(0, 1_000_000))
            self.samplers[cfg["name"]] = XuniSampler(mode=cfg["mode"], seed=layer_seed)
            self.fields[cfg["name"]] = XuniField(grid_size=(grid_size, grid_size, grid_size))
            self.layer_energy[cfg["name"]] = 0.0

    def generate(self, target_energy: float = 50.0) -> Dict[str, Any]:
        """
        多层发电，直到总能量达到 target。
        返回各层产出和总能量。
        """
        total_energy = 0.0
        total_samples = 0
        layer_output = {}

        for cfg in self.LAYER_CONFIGS:
            name = cfg["name"]
            weight = cfg["weight"]
            batch_size = cfg["batch"]
            target_layer = target_energy * weight

            sampler = self.samplers[name]
            field = self.fields[name]

            layer_energy = 0.0
            layer_samples = 0
            max_iter = 5  # 每层最多迭代几次

            for i in range(max_iter):
                batch = sampler.generate_batch(batch_size)
                layer_samples += batch_size
                field.reset()
                field.ingest_batch(batch)
                field.compute_field()
                summary = field.field_summary()
                e = float(summary.get("total_energy", 0.0))
                usable = np.log1p(e) * 1.5
                layer_energy += usable
                if layer_energy >= target_layer:
                    break

            layer_output[name] = {
                "energy": round(layer_energy, 2),
                "samples": layer_samples,
            }
            total_energy += layer_energy
            total_samples += layer_samples
            self.layer_energy[name] += layer_energy

        self.total_generated += total_energy
        self.total_samples += total_samples

        return {
            "total_energy": total_energy,
            "total_samples": total_samples,
            "layers": layer_output,
        }

    def get_layer_stats(self) -> Dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "total_samples": self.total_samples,
            "layers": dict(self.layer_energy),
        }


# ============================================================== #
#  开源数据采集管道 —— 从代码仓库提取训练语料
# ============================================================== #

class CodeCorpusExtractor:
    """
    从代码仓库提取高质量训练语料。

    提取策略：
    1. 文档文件（.md）→ 直接按段落/标题切分
    2. 代码文件 → 提取函数/类注释、docstring、关键逻辑片段
    3. 配置文件 → 提取配置说明和默认值
    """

    SUPPORTED_EXTENSIONS = {
        ".go", ".ts", ".tsx", ".js", ".jsx", ".py",
        ".md", ".json", ".yaml", ".yml", ".toml",
    }

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.fragments: List[str] = []
        self.file_count = 0
        self.line_count = 0

    def extract(self, max_fragments: int = 10000) -> List[str]:
        """遍历仓库，提取训练片段。"""
        for root, dirs, files in os.walk(self.repo_path):
            # 跳过无用目录
            skip_dirs = {"node_modules", ".git", "__pycache__", "vendor",
                         "dist", "build", ".next", ".cache"}
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue
                fpath = os.path.join(root, fname)
                try:
                    self._extract_file(fpath, ext)
                except Exception:
                    continue

                if len(self.fragments) >= max_fragments:
                    return self.fragments

        return self.fragments

    def _extract_file(self, fpath: str, ext: str):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return

        if not content.strip():
            return

        self.file_count += 1
        lines = content.split("\n")
        self.line_count += len(lines)

        rel_path = os.path.relpath(fpath, self.repo_path)

        if ext == ".md":
            self._extract_markdown(content, rel_path)
        elif ext in {".py", ".go", ".ts", ".tsx", ".js", ".jsx"}:
            self._extract_code(content, rel_path, ext)
        elif ext in {".json", ".yaml", ".yml", ".toml"}:
            self._extract_config(content, rel_path, ext)

    def _extract_markdown(self, content: str, rel_path: str):
        """从 Markdown 提取干净的知识段落。"""
        lines = content.split("\n")
        current_para: List[str] = []

        in_code_block = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            # 标题也作为简短知识
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title and len(title) > 3:
                    self.fragments.append(title[:200])
                if current_para:
                    para_text = " ".join(current_para).strip()
                    if len(para_text) > 20 and not para_text.startswith("<"):
                        self.fragments.append(para_text[:250])
                    current_para = []
            elif stripped == "":
                if current_para:
                    para_text = " ".join(current_para).strip()
                    if len(para_text) > 20 and not para_text.startswith("<"):
                        self.fragments.append(para_text[:250])
                    current_para = []
            else:
                # 过滤纯链接、纯标签
                if stripped.startswith(("http", "![")):
                    continue
                current_para.append(stripped)

        if current_para:
            para_text = " ".join(current_para).strip()
            if len(para_text) > 20 and not para_text.startswith("<"):
                self.fragments.append(para_text[:250])

    def _extract_code(self, content: str, rel_path: str, ext: str):
        """从代码文件提取有信息量的注释和说明。"""
        lines = content.split("\n")
        comment_lines: List[str] = []
        in_docstring = False
        docstring_lines: List[str] = []

        comment_prefix = {
            ".py": "#",
            ".go": "//",
            ".ts": "//",
            ".tsx": "//",
            ".js": "//",
            ".jsx": "//",
        }.get(ext, "//")

        def _flush_comments():
            nonlocal comment_lines
            if not comment_lines:
                return
            text = " ".join(comment_lines).strip()
            # 只保留有信息量的（不是 TODO/FIXME/版权声明等）
            skip_starts = ("TODO", "FIXME", "NOTE", "Copyright", "license",
                           "Licensed", "Package ", "package ", "import ")
            if (len(text) > 25 and not any(text.startswith(s) for s in skip_starts)
                    and not text.startswith(("-", "=", "*"))):
                self.fragments.append(text[:200])
            comment_lines = []

        for line in lines:
            stripped = line.strip()

            # Python docstring
            if ext == ".py":
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    if in_docstring:
                        if docstring_lines:
                            doc = " ".join(docstring_lines).strip()
                            if len(doc) > 25:
                                self.fragments.append(doc[:200])
                        docstring_lines = []
                        in_docstring = False
                    else:
                        in_docstring = True
                    continue
                if in_docstring:
                    docstring_lines.append(stripped.strip('"\''))
                    continue

            # 单行注释
            if stripped.startswith(comment_prefix):
                text = stripped[len(comment_prefix):].strip()
                # 过滤噪声
                if (len(text) > 8
                        and not text.startswith(("---", "===", "***", "///", "---"))
                        and not text.startswith(("TODO", "FIXME", "Copyright"))
                        and not set(text) <= set("-=* ")):
                    comment_lines.append(text)
                else:
                    _flush_comments()
            else:
                _flush_comments()

        _flush_comments()

    def _extract_config(self, content: str, rel_path: str, ext: str):
        """从配置文件提取有价值的注释说明。"""
        lines = content.split("\n")
        comment_lines: List[str] = []
        comment_prefix = "#" if ext in {".yaml", ".yml", ".toml"} else "//"

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(comment_prefix):
                text = stripped[len(comment_prefix):].strip()
                if len(text) > 15 and not text.startswith(("---", "===")):
                    comment_lines.append(text)
            else:
                if comment_lines:
                    text = " ".join(comment_lines).strip()
                    if len(text) > 20:
                        self.fragments.append(text[:200])
                    comment_lines = []

        if comment_lines:
            text = " ".join(comment_lines).strip()
            if len(text) > 20:
                self.fragments.append(text[:200])

    def stats(self) -> Dict[str, Any]:
        return {
            "files": self.file_count,
            "lines": self.line_count,
            "fragments": len(self.fragments),
        }


# ============================================================== #
#  后台训练监管系统
# ============================================================== #

class TrainingDaemon:
    """
    合鸣-13 后台训练守护进程。

    功能：
    - 后台持续训练，能量自动维持
    - 实时监控面板
    - 定期自动保存检查点
    - 启动自动恢复
    - 信号处理（Ctrl+C 优雅退出）
    """

    def __init__(
        self,
        repo_path: str,
        ckpt_dir: str = "checkpoints/harmonia13",
        log_dir: str = "logs",
        scale: str = "large",
        save_interval_sec: float = 30.0,
        max_fragments: int = 8000,
    ):
        self.repo_path = repo_path
        self.ckpt_dir = ckpt_dir
        self.log_dir = log_dir
        self.scale = scale
        self.save_interval = save_interval_sec
        self.max_fragments = max_fragments

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 运行状态
        self.running = False
        self.paused = False
        self.start_time = 0.0
        self.last_save_time = 0.0
        self.step_count = 0

        # 统计
        self.total_new_fragments = 0
        self.total_energy_used = 0.0

        # 初始化能源系统
        self.energy = MultiLayerEnergy(grid_size=16, seed=42)
        self.energy_buffer = 5000.0

        # 初始化模型
        ckpt_meta = os.path.join(ckpt_dir, "harmonia13_meta.json")
        if os.path.exists(ckpt_meta):
            print(f"📂 从检查点加载: {ckpt_dir}")
            self.model = Harmonia13Virtual.load(ckpt_dir)
        else:
            print(f"🌱 新建合鸣-13 模型 ({scale})")
            self.model = Harmonia13Virtual(scale=scale)

        self.model.charge(10000.0)  # 模型能量缓冲

        # 数据提取
        self.extractor = CodeCorpusExtractor(repo_path)
        self.corpus: List[str] = []
        self.corpus_index = 0

        # 注册信号
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n\n⏸  收到信号 {signum}，正在优雅退出...")
        self.running = False

    # ---------- 数据准备 ---------- #

    def prepare_corpus(self, max_fragments: int = 8000):
        """准备训练语料。"""
        print(f"📦 正在从 {self.repo_path} 提取训练语料...")
        self.corpus = self.extractor.extract(max_fragments=max_fragments)
        stats = self.extractor.stats()
        print(f"   文件数: {stats['files']}")
        print(f"   代码行数: {stats['lines']:,}")
        print(f"   训练片段: {stats['fragments']:,}")
        return stats

    # ---------- 单步训练 ---------- #

    def train_step(self) -> Dict[str, Any]:
        """一步训练：发电 → 算力兑换 → 吸收语料。"""
        # 1) 发电（仅在能量不足时，避免每步都做昂贵的采样计算）
        if self.energy_buffer < 10.0:
            power = self.energy.generate(target_energy=30.0)
            self.energy_buffer += power["total_energy"]
        else:
            power = {"total_energy": 0.0, "layers": {}}

        # 2) 算力兑换
        flops_needed = 8e6
        energy_cost = flops_needed / 1e7
        if self.energy_buffer >= energy_cost:
            self.energy_buffer -= energy_cost
            self.total_energy_used += energy_cost
        else:
            extra = self.energy.generate(target_energy=100.0)
            self.energy_buffer += extra["total_energy"]

        # 3) 从语料库取一批训练
        batch_size = 5
        new_frags: List[str] = []
        for _ in range(batch_size):
            if self.corpus_index < len(self.corpus):
                frag = self.corpus[self.corpus_index].strip()
                self.corpus_index += 1
                if frag and len(frag) > 15:
                    new_frags.append(frag)
                    # 对较短的知识片段做共振复述（增加记忆强度）
                    if len(frag) < 120:
                        rephrased = f"值得了解的是，{frag}"
                        new_frags.append(rephrased)
            else:
                self.corpus_index = 0
                break

        # 4) 加入模型（去重）
        added = 0
        if new_frags:
            general = self.model._lite._find("general")
            if general is not None:
                existing = set(f.strip() for f in general["fragments"])
                for f in new_frags:
                    f = f.strip()
                    if f and f not in existing and len(f) > 15:
                        general["fragments"].append(f)
                        existing.add(f)
                        added += 1
                # 同步到 learned_fragments
                learned_set = set(x.strip() for x in self.model._lite._learned_fragments)
                for f in new_frags:
                    f = f.strip()
                    if f and f not in learned_set and len(f) > 15:
                        self.model._lite._learned_fragments.append(f)
                        learned_set.add(f)

        self.total_new_fragments += added
        self.model.training_samples_seen += batch_size
        if self.step_count % 20 == 0:
            self.model.training_epochs_done += 1

        return {
            "new_fragments": added,
            "energy_generated": power["total_energy"],
            "energy_buffer": self.energy_buffer,
            "corpus_progress": f"{self.corpus_index}/{len(self.corpus)}",
            "layers": power["layers"],
        }

    def _resonance_expand(self, fragment: str) -> List[str]:
        """共振扩展：从一个片段生成多个变体，增加知识密度。"""
        results = [fragment]

        # 模板化改写
        templates = [
            f"在MonkeyCode项目中，{fragment}",
            f"根据代码文档，{fragment}",
            f"技术要点：{fragment}",
        ]
        # 只用较短的片段做扩展，避免超长
        if len(fragment) < 100:
            results.append(templates[0])
        if len(fragment) < 80:
            results.append(templates[2])

        return results

    # ---------- 实时监控面板 ---------- #

    def print_dashboard(self, step_result: Dict[str, Any]):
        """实时打印单行监控面板。"""
        elapsed = time.time() - self.start_time
        total_frags = sum(len(e["fragments"]) for e in self.model._lite.experts)

        layers = step_result.get("layers", {})
        layer_str = " ".join(
            f"{name[:3]}{v['energy']:.0f}" for name, v in list(layers.items())[:3]
        )

        line = (
            f"\r⏱ {elapsed/60:.1f}m  "
            f"⚡{self.energy_buffer:.0f} "
            f"[三层 {layer_str}]  "
            f"📝 +{step_result['new_fragments']} "
            f"(总{self.total_new_fragments:,}/{total_frags:,})  "
            f"📚 {step_result['corpus_progress']}  "
            f"🔄 step {self.step_count}"
        )
        sys.stdout.write(line[:120])
        sys.stdout.flush()

    # ---------- 日志 ---------- #

    def log_step(self, result: Dict[str, Any]):
        entry = {
            "timestamp": time.time(),
            "step": self.step_count,
            "energy_buffer": self.energy_buffer,
            "new_fragments": result["new_fragments"],
            "total_new": self.total_new_fragments,
            "corpus_index": self.corpus_index,
            "energy_generated": result["energy_generated"],
        }
        log_path = os.path.join(self.log_dir, "harmonia_daemon.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------- 主循环 ---------- #

    def start(self, max_steps: Optional[int] = None):
        """启动后台训练。"""
        print("=" * 64)
        print("  🎵 合鸣-13 后台训练监管系统  🎵")
        print("=" * 64)
        self.model.print_card()
        print(f"  语料来源 : {self.repo_path}")
        print(f"  检查点   : {self.ckpt_dir}")
        print(f"  日志目录 : {self.log_dir}")
        print(f"  能源层级 : {len(MultiLayerEnergy.LAYER_CONFIGS)} 层")
        if max_steps:
            print(f"  最大步数 : {max_steps}")
        print("-" * 64)

        # 准备语料
        self.prepare_corpus(max_fragments=self.max_fragments)

        print("-" * 64)
        print("  启动实时训练... (Ctrl+C 暂停并保存)")
        print()

        self.running = True
        self.start_time = time.time()
        self.last_save_time = time.time()

        try:
            while self.running:
                self.step_count += 1

                if self.paused:
                    time.sleep(0.1)
                    continue

                # 训练一步
                result = self.train_step()

                # 显示
                self.print_dashboard(result)

                # 日志
                if self.step_count % 10 == 0:
                    self.log_step(result)

                # 自动保存
                if time.time() - self.last_save_time > self.save_interval:
                    self.model.save(self.ckpt_dir)
                    self.last_save_time = time.time()

                # 步数限制
                if max_steps and self.step_count >= max_steps:
                    break

                # 轻微延迟（手机友好）
                time.sleep(0.01)

        except KeyboardInterrupt:
            pass

        # 收尾
        print("\n")
        print("-" * 64)
        print("  🏁 训练结束，正在保存...")

        save_result = self.model.save(self.ckpt_dir)
        elapsed = time.time() - self.start_time

        print()
        print(f"  ⏱  运行时长   : {elapsed/60:.1f} 分钟")
        print(f"  🔄 总步数     : {self.step_count:,}")
        print(f"  📝 新学片段   : {self.total_new_fragments:,}")
        print(f"  ⚡ 总耗能     : {self.total_energy_used:.1f}")
        print(f"  🎯 总采样点   : {self.energy.total_samples:,}")
        print(f"  🔋 总发电量   : {self.energy.total_generated:.1f}")
        print(f"  💾 检查点     : {save_result['ckpt_dir']}")
        print(f"  📊 训练日志   : {os.path.join(self.log_dir, 'harmonia_daemon.jsonl')}")

        # 训练后测试
        print()
        print("  🧪 训练后问答测试:")
        test_questions = [
            "MonkeyCode是什么？",
            "什么是代码安全扫描？",
            "如何部署MonkeyCode？",
            "合鸣模型是什么？",
            "虚拟电场有什么用？",
        ]
        for q in test_questions:
            ans = self.model.generate(q, max_new_tokens=100)
            print(f"    Q: {q}")
            print(f"    A: {ans}")
            print()

        print("=" * 64)
        print("  合鸣-13 后台训练完成，已保存检查点。")
        print("  下次启动自动从检查点继续。")
        print("=" * 64)

        return {
            "steps": self.step_count,
            "new_fragments": self.total_new_fragments,
            "elapsed": elapsed,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="合鸣-13 后台训练监管系统")
    parser.add_argument("--repo", required=True, help="开源代码仓库路径")
    parser.add_argument("--scale", default="large",
                        choices=["small", "medium", "large"])
    parser.add_argument("--steps", type=int, default=None,
                        help="最大训练步数（默认无限）")
    parser.add_argument("--ckpt", default="checkpoints/harmonia13",
                        help="检查点目录")
    parser.add_argument("--save-interval", type=float, default=30.0,
                        help="自动保存间隔（秒）")
    parser.add_argument("--max-fragments", type=int, default=8000,
                        help="最大提取片段数（默认8000，大语料建议20000+）")
    args = parser.parse_args()

    if not os.path.isdir(args.repo):
        print(f"❌ 仓库路径不存在: {args.repo}")
        sys.exit(1)

    daemon = TrainingDaemon(
        repo_path=args.repo,
        ckpt_dir=args.ckpt,
        scale=args.scale,
        save_interval_sec=args.save_interval,
        max_fragments=args.max_fragments,
    )
    daemon.start(max_steps=args.steps)


if __name__ == "__main__":
    main()
