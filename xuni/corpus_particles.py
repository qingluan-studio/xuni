"""
CorpusParticles —— 训练语料粒子化容器

核心理念：
    原始语料仓库（几十 MB~几 GB）→ 粒子化（指纹+摘要）→ 几 KB 入库
    跨窗口/跨会话直接"复制"粒子容器，不用重新下载原始仓库。

粒子化后的内容：
    - 仓库指纹（SHA256）
    - 文件清单 + 每个文件的指纹
    - 提取出的训练片段（完整内容，这部分是真正要用的）
    - 统计元数据

不存的内容：
    - 原始仓库的 .git 历史
    - 完整源码文件（只存指纹 + 提取的片段）
    - 二进制资源

这样：
    70MB 语料 → 几 MB 粒子容器 → 入库成"待推送"
    换窗口后 git pull 即可拿到全部训练素材，直接继续训练
"""

import hashlib
import json
import os
import time
import zlib
import base64
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple


SUPPORTED_EXTS = {".py", ".go", ".ts", ".tsx", ".js", ".jsx",
                  ".md", ".json", ".yaml", ".yml", ".toml"}
SKIP_DIRS = {"node_modules", ".git", "__pycache__", "vendor",
             "dist", "build", ".next", ".cache", ".venv", "venv"}


@dataclass
class FileParticle:
    """单文件粒子——只存指纹和元数据，不存完整内容"""
    rel_path: str
    ext: str
    size_bytes: int
    line_count: int
    fingerprint: str          # SHA256
    summary_b64: str          # 前 256 字节压缩摘要（base64）
    extracted_fragments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.rel_path,
            "ext": self.ext,
            "size": self.size_bytes,
            "lines": self.line_count,
            "fingerprint": self.fingerprint[:16] + "...",
            "fragments": len(self.extracted_fragments),
        }


@dataclass
class CorpusParticleContainer:
    """
    语料粒子容器——多个仓库的粒子化集合

    这是跨窗口"复制"的载体：入库后任意窗口 git pull 即可拿到。
    """
    container_id: str
    created_at: float
    repos: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_particles: List[FileParticle] = field(default_factory=list)
    all_fragments: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_repo(self, repo_path: str, name: Optional[str] = None,
                 max_fragments_per_repo: int = 8000) -> Dict[str, Any]:
        """把一个仓库粒子化加入容器"""
        name = name or os.path.basename(repo_path.rstrip("/"))
        repo_fingerprint = hashlib.sha256(name.encode()).hexdigest()[:16]
        repo_files = 0
        repo_lines = 0
        repo_frags = 0

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    continue
                fpath = os.path.join(root, fname)
                fp = self._particle_file(fpath, repo_path, ext)
                if fp is None:
                    continue
                self.file_particles.append(fp)
                self.all_fragments.extend(fp.extracted_fragments)
                repo_files += 1
                repo_lines += fp.line_count
                repo_frags += len(fp.extracted_fragments)
                if repo_frags >= max_fragments_per_repo:
                    break
            if repo_frags >= max_fragments_per_repo:
                break

        self.repos[name] = {
            "fingerprint": repo_fingerprint,
            "files": repo_files,
            "lines": repo_lines,
            "fragments": repo_frags,
            "path_origin": repo_path,
        }
        return self.repos[name]

    def _particle_file(self, fpath: str, repo_root: str, ext: str) -> Optional[FileParticle]:
        """单个文件粒子化"""
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return None
        if not content.strip():
            return None

        raw = content.encode("utf-8")
        fingerprint = hashlib.sha256(raw).hexdigest()
        summary = zlib.compress(raw, level=9)[:256]
        summary_b64 = base64.b64encode(summary).decode("ascii")

        rel_path = os.path.relpath(fpath, repo_root)
        lines = content.count("\n") + 1

        fragments = self._extract_fragments(content, ext)

        return FileParticle(
            rel_path=rel_path,
            ext=ext,
            size_bytes=len(raw),
            line_count=lines,
            fingerprint=fingerprint,
            summary_b64=summary_b64,
            extracted_fragments=fragments,
        )

    @staticmethod
    def _extract_fragments(content: str, ext: str) -> List[str]:
        """从文件提取训练片段（与 CodeCorpusExtractor 策略一致，但精简）"""
        frags: List[str] = []
        lines = content.split("\n")

        if ext == ".md":
            para: List[str] = []
            in_code = False
            for line in lines:
                s = line.strip()
                if s.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    continue
                if s.startswith("#"):
                    t = s.lstrip("#").strip()
                    if len(t) > 5:
                        frags.append(t[:200])
                    if para:
                        t2 = " ".join(para).strip()
                        if len(t2) > 20 and not t2.startswith(("<", "http")):
                            frags.append(t2[:250])
                        para = []
                elif s == "":
                    if para:
                        t2 = " ".join(para).strip()
                        if len(t2) > 20 and not t2.startswith(("<", "http")):
                            frags.append(t2[:250])
                        para = []
                else:
                    if s.startswith(("http", "![")):
                        continue
                    para.append(s)
            if para:
                t2 = " ".join(para).strip()
                if len(t2) > 20 and not t2.startswith(("<", "http")):
                    frags.append(t2[:250])

        elif ext == ".py":
            in_doc = False
            doc_buf: List[str] = []
            for line in lines:
                s = line.strip()
                if s.startswith('"""') or s.startswith("'''"):
                    if in_doc:
                        doc = " ".join(doc_buf).strip()
                        if len(doc) > 25:
                            frags.append(doc[:200])
                        doc_buf = []
                        in_doc = False
                    else:
                        in_doc = True
                    continue
                if in_doc:
                    doc_buf.append(s.strip("\"'"))
                    continue
                if s.startswith("#"):
                    t = s[1:].strip()
                    if (len(t) > 12
                            and not t.startswith(("TODO", "FIXME", "Copyright",
                                                  "license", "Licensed", "---", "==="))
                            and not set(t) <= set("-=* ")):
                        frags.append(t[:200])
        elif ext in {".go", ".ts", ".tsx", ".js", ".jsx"}:
            for line in lines:
                s = line.strip()
                if s.startswith("//"):
                    t = s[2:].strip()
                    if (len(t) > 12
                            and not t.startswith(("TODO", "FIXME", "Copyright",
                                                  "package ", "import "))):
                        frags.append(t[:200])

        return [f for f in frags if f.strip()]

    def compute_stats(self) -> Dict[str, Any]:
        total_size = sum(fp.size_bytes for fp in self.file_particles)
        total_lines = sum(fp.line_count for fp in self.file_particles)
        total_frags = len(self.all_fragments)
        ext_dist: Dict[str, int] = {}
        for fp in self.file_particles:
            ext_dist[fp.ext] = ext_dist.get(fp.ext, 0) + 1
        self.stats = {
            "repos": len(self.repos),
            "files": len(self.file_particles),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_lines": total_lines,
            "total_fragments": total_frags,
            "ext_distribution": ext_dist,
            "container_size_estimate_kb": round(
                len(json.dumps([asdict(fp) for fp in self.file_particles],
                               ensure_ascii=False)) / 1024, 1
            ),
        }
        return self.stats

    def save(self, path: str) -> Dict[str, Any]:
        """保存粒子容器到 JSON 文件（入库用）"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.compute_stats()
        data = {
            "container_id": self.container_id,
            "created_at": self.created_at,
            "saved_at": time.time(),
            "repos": self.repos,
            "stats": self.stats,
            "file_particles": [asdict(fp) for fp in self.file_particles],
            "all_fragments": self.all_fragments,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        size_kb = os.path.getsize(path) / 1024
        return {
            "path": path,
            "size_kb": round(size_kb, 1),
            "fragments": len(self.all_fragments),
            "files": len(self.file_particles),
        }

    @classmethod
    def load(cls, path: str) -> "CorpusParticleContainer":
        """从 JSON 加载粒子容器"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        c = cls(
            container_id=data["container_id"],
            created_at=data["created_at"],
            repos=data.get("repos", {}),
        )
        for fp_data in data.get("file_particles", []):
            c.file_particles.append(FileParticle(**fp_data))
        c.all_fragments = data.get("all_fragments", [])
        c.stats = data.get("stats", {})
        return c

    def to_training_corpus(self) -> List[str]:
        """导出为训练用的片段列表（直接喂 train_daemon）"""
        return list(self.all_fragments)


def particleize_corpora(repo_paths: Dict[str, str],
                        output_path: str,
                        max_per_repo: int = 8000) -> Dict[str, Any]:
    """
    便捷函数：批量粒子化多个仓库

    Args:
        repo_paths: {仓库名: 仓库本地路径}
        output_path: 粒子容器输出路径
        max_per_repo: 每个仓库最多提取片段数

    Returns:
        保存结果
    """
    container = CorpusParticleContainer(
        container_id=hashlib.sha256(
            "|".join(repo_paths.keys()).encode()
        ).hexdigest()[:16],
        created_at=time.time(),
    )
    for name, path in repo_paths.items():
        if not os.path.isdir(path):
            print(f"  ⚠️ 跳过不存在的仓库: {name} ({path})")
            continue
        info = container.add_repo(path, name=name, max_fragments_per_repo=max_per_repo)
        print(f"  ✅ {name}: {info['files']} 文件, {info['lines']} 行, {info['fragments']} 片段")

    result = container.save(output_path)
    print(f"\n  📦 粒子容器已保存: {result['path']}")
    print(f"     大小: {result['size_kb']} KB")
    print(f"     片段: {result['fragments']}")
    print(f"     文件: {result['files']}")
    return result


if __name__ == "__main__":
    import sys
    # 用法: python corpus_particles.py /workspace/corpus output.json
    if len(sys.argv) < 3:
        print("用法: python corpus_particles.py <corpus_root> <output.json>")
        print("示例: python corpus_particles.py /workspace/corpus corpus_particles.json")
        sys.exit(1)
    corpus_root = sys.argv[1]
    output = sys.argv[2]
    repos = {}
    for d in os.listdir(corpus_root):
        full = os.path.join(corpus_root, d)
        if os.path.isdir(full):
            repos[d] = full
    particleize_corpora(repos, output, max_per_repo=10000)
