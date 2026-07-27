"""
CorpusDownloader —— 语料下载器

核心理念：
    工厂自己具备"获取食物"的能力——不需要外部工具喂料。
    从 GitHub raw / PyPI / 本地仓库下载真实代码 → 粒子化 → 训练。

闭环：
    CorpusDownloader.fetch() → CorpusParticleContainer.add_repo() → train()

支持来源：
    1. GitHub raw 文件（指定 repo + 文件列表）
    2. GitHub 仓库压缩包（zipball）
    3. 本地目录扫描（已由 CorpusParticleContainer 支持）
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import urllib.request
import urllib.error
import tempfile
import zipfile
import threading
import concurrent.futures
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# 经典 Python 标准库文件（从 CPython 仓库下载）
CPYTHON_RAW_BASE = "https://raw.githubusercontent.com/python/cpython/main/Lib"

DEFAULT_PYTHON_FILES = [
    "collections/__init__.py",
    "functools.py",
    "itertools.py",
    "heapq.py",
    "bisect.py",
    "json/decoder.py",
    "json/encoder.py",
    "json/scanner.py",
    "re/__init__.py",
    "os.py",
    "posixpath.py",
    "genericpath.py",
    "string.py",
    "textwrap.py",
    "pprint.py",
    "enum.py",
    "abc.py",
    "copy.py",
    "copyreg.py",
    "types.py",
    "typing.py",
    "dataclasses.py",
    "contextlib.py",
    "pathlib.py",
    "shutil.py",
    "tempfile.py",
    "io.py",
    "csv.py",
    "configparser.py",
    "hashlib.py",
    "hmac.py",
    "secrets.py",
    "logging/__init__.py",
    "unittest/__init__.py",
    "unittest/case.py",
    "unittest/mock.py",
    "argparse.py",
    "uuid.py",
    "weakref.py",
    "decimal.py",
    "fractions.py",
    "statistics.py",
    "random.py",
    "datetime.py",
    "calendar.py",
    "calendar.py",
    "xml/etree/ElementTree.py",
    "html/__init__.py",
    "html/parser.py",
    "urllib/parse.py",
    "urllib/request.py",
    "urllib/error.py",
    "http/client.py",
    "http/server.py",
    "email/__init__.py",
    "email/parser.py",
    "email/mime/text.py",
    "sqlite3/__init__.py",
    "ssl.py",
    "socket.py",
    "select.py",
    "selectors.py",
    "subprocess.py",
    "signal.py",
    "threading.py",
    "queue.py",
    "concurrent/futures/__init__.py",
    "concurrent/futures/thread.py",
    "asyncio/__init__.py",
    "asyncio/base_events.py",
    "asyncio/streams.py",
    "inspect.py",
    "traceback.py",
    "warnings.py",
    "contextvars.py",
    "ast.py",
    "tokenize.py",
    "keyword.py",
    "token.py",
    "dis.py",
    "compileall.py",
    "py_compile.py",
    "importlib/__init__.py",
    "importlib/util.py",
    "importlib/metadata.py",
    "pickle.py",
    "shelve.py",
    "json/__init__.py",
    "xmlrpc/client.py",
    "multiprocessing/__init__.py",
    "multiprocessing/pool.py",
    "multiprocessing/queue.py",
]


@dataclass
class DownloadResult:
    """单个文件下载结果"""
    url: str
    filepath: str
    success: bool
    size_bytes: int = 0
    error: str = ""
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "filepath": self.filepath,
            "success": self.success,
            "size_bytes": self.size_bytes,
            "error": self.error,
        }


class CorpusDownloader:
    """
    语料下载器——工厂的"觅食"能力。

    用法：
        dl = CorpusDownloader()
        results = dl.fetch_cpython_stdlib(max_files=30)
        texts = dl.get_texts()
        # 然后喂给 HarmoniaLiteEngine.train()
    """

    def __init__(self, cache_dir: Optional[str] = None, timeout: int = 30):
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="xuni_corpus_")
        self.timeout = timeout
        self._results: List[DownloadResult] = []
        os.makedirs(self.cache_dir, exist_ok=True)

    def fetch_url(self, url: str, filepath: Optional[str] = None) -> DownloadResult:
        """下载单个 URL"""
        if filepath is None:
            # 从 URL 生成文件名
            fname = url.rsplit("/", 1)[-1] or "unnamed.py"
            subpath = url.split("/main/Lib/")[-1] if "/main/Lib/" in url else fname
            filepath = os.path.join(self.cache_dir, subpath.replace("/", "_"))

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "xuni-factory/1.0 (corpus-downloader)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
                text = data.decode("utf-8", errors="ignore")

            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(text)

            result = DownloadResult(
                url=url, filepath=filepath, success=True,
                size_bytes=len(data), content=text,
            )
        except Exception as e:
            result = DownloadResult(
                url=url, filepath=filepath, success=False, error=str(e)
            )

        self._results.append(result)
        return result

    def fetch_cpython_stdlib(
        self, max_files: int = 50, file_list: Optional[List[str]] = None,
        parallel: int = 8,
    ) -> List[DownloadResult]:
        """
        从 CPython 仓库并发下载标准库源码（虚拟流量加速）。

        Args:
            max_files: 最多下载多少个文件
            file_list: 自定义文件列表（相对路径），默认用 DEFAULT_PYTHON_FILES
            parallel: 并发线程数（虚拟流量），默认 8
        """
        files = (file_list or DEFAULT_PYTHON_FILES)[:max_files]
        results: List[DownloadResult] = []
        results_lock = threading.Lock()

        def _download_one(fpath: str) -> DownloadResult:
            url = f"{CPYTHON_RAW_BASE}/{fpath}"
            result = self.fetch_url(url)
            with results_lock:
                results.append(result)
            return result

        print(f"  🌐 工厂虚拟流量加速: {parallel} 并发 x {len(files)} 文件")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_download_one, f): f for f in files}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                done += 1
                result = fut.result()
                fpath = futures[fut]
                status = "✅" if result.success else "❌"
                size = result.size_bytes if result.success else 0
                print(f"  [{done:3d}/{len(files)}] {status} {fpath} ({size} bytes)")

        results.sort(key=lambda r: files.index(
            r.url.replace(f"{CPYTHON_RAW_BASE}/", "")
        ) if r.url.replace(f"{CPYTHON_RAW_BASE}/", "") in files else 999)

        success_count = sum(1 for r in results if r.success)
        total_bytes = sum(r.size_bytes for r in results if r.success)
        print(f"  下载完成: {success_count}/{len(results)} 成功, "
              f"共 {total_bytes} bytes ({total_bytes/1024:.1f} KB)")

        return results

    def fetch_github_raw(
        self, repo: str, branch: str, paths: List[str]
    ) -> List[DownloadResult]:
        """
        从指定 GitHub 仓库下载 raw 文件。

        Args:
            repo: "owner/repo" 格式
            branch: 分支名
            paths: 文件路径列表
        """
        base = f"https://raw.githubusercontent.com/{repo}/{branch}"
        results = []
        for path in paths:
            url = f"{base}/{path}"
            result = self.fetch_url(url)
            results.append(result)
        return results

    def fetch_github_zip(self, repo: str, branch: str = "main") -> Optional[str]:
        """
        下载 GitHub 仓库的 zip 压缩包并解压。

        Returns:
            解压后的目录路径，失败返回 None
        """
        url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
        zip_path = os.path.join(self.cache_dir, f"{repo.replace('/', '_')}_{branch}.zip")

        try:
            print(f"  下载 {repo} ({branch}) zip...")
            req = urllib.request.Request(url, headers={
                "User-Agent": "xuni-factory/1.0"
            })
            with urllib.request.urlopen(req, timeout=self.timeout * 3) as resp:
                with open(zip_path, "wb") as f:
                    f.write(resp.read())

            extract_dir = os.path.join(self.cache_dir, f"{repo.replace('/', '_')}_{branch}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

            # 找到解压后的根目录（通常有一个前缀目录）
            entries = os.listdir(extract_dir)
            if len(entries) == 1:
                root = os.path.join(extract_dir, entries[0])
            else:
                root = extract_dir

            print(f"  ✅ 解压完成: {root}")
            return root

        except Exception as e:
            print(f"  ❌ 下载失败: {e}")
            return None

    def scan_directory(self, dir_path: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        扫描目录，返回所有匹配扩展名的文件内容。

        Args:
            dir_path: 目录路径
            extensions: 文件扩展名列表，默认 ['.py']
        """
        if extensions is None:
            extensions = [".py"]

        texts = []
        for root, dirs, files in os.walk(dir_path):
            # 跳过 __pycache__、.git、test 等无关目录
            dirs[:] = [d for d in dirs if d not in (
                "__pycache__", ".git", "test", "tests", "idlelib",
                "tkinter", "turtledemo", "site-packages"
            )]

            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                        if len(text.strip()) > 50:
                            texts.append(text)
                    except Exception:
                        pass

        return texts

    def fetch_python_zip(self, max_size_mb: int = 50) -> Optional[str]:
        """
        一键获取 CPython 标准库：下载 zip → 解压 → 扫描 .py 文件。

        这是最快的方式——一次请求搞定整个标准库。

        Returns:
            解压后的 Lib 目录路径，失败返回 None
        """
        print("  🚀 工厂一键获取 CPython 标准库 zip...")
        root = self.fetch_github_zip("python/cpython", branch="main")
        if root is None:
            return None

        # 找 Lib 目录
        lib_dir = os.path.join(root, "Lib")
        if not os.path.isdir(lib_dir):
            # 尝试直接用根目录
            for d in os.listdir(root):
                candidate = os.path.join(root, d)
                if os.path.isdir(candidate) and d == "Lib":
                    lib_dir = candidate
                    break

        print(f"  📂 扫描 Lib 目录: {lib_dir}")
        texts = self.scan_directory(lib_dir if os.path.isdir(lib_dir) else root)
        print(f"  📚 发现 {len(texts)} 个 Python 源文件")

        # 注册为下载结果
        total_size = 0
        for text in texts:
            size = len(text.encode("utf-8"))
            total_size += size
            self._results.append(DownloadResult(
                url="cpython-zip",
                filepath="",
                success=True,
                size_bytes=size,
                content=text,
            ))

        print(f"  ✅ 载入完成: {total_size} bytes ({total_size/1024:.1f} KB)")
        return lib_dir if os.path.isdir(lib_dir) else root

    def get_texts(self) -> List[str]:
        """获取所有已下载文件的内容（用于训练）"""
        return [r.content for r in self._results if r.success and r.content]

    def get_fragments(self, max_lines_per_fragment: int = 15) -> List[str]:
        """
        把下载的代码切成训练片段。

        每个片段是一个完整的函数/类定义，控制在 max_lines_per_fragment 行以内。
        """
        fragments = []
        for text in self.get_texts():
            frags = _extract_code_fragments(text, max_lines_per_fragment)
            fragments.extend(frags)
        return fragments

    def get_stats(self) -> Dict[str, Any]:
        """下载统计"""
        success = [r for r in self._results if r.success]
        failed = [r for r in self._results if not r.success]
        total_bytes = sum(r.size_bytes for r in success)
        return {
            "total_urls": len(self._results),
            "success": len(success),
            "failed": len(failed),
            "total_bytes": total_bytes,
            "total_kb": round(total_bytes / 1024, 1),
            "cache_dir": self.cache_dir,
            "failed_urls": [r.url for r in failed],
        }

    def save_report(self, path: str):
        """保存下载报告"""
        report = {
            "stats": self.get_stats(),
            "results": [r.to_dict() for r in self._results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


def _extract_code_fragments(text: str, max_lines: int = 15) -> List[str]:
    """
    从代码文本中提取函数/类定义片段。
    每个片段是一个完整的 def/class 块。
    """
    lines = text.split("\n")
    fragments = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # 检测 def / class 开头
        if stripped.startswith(("def ", "class ", "async def ")):
            # 收集这个定义的所有行（直到下一个同缩进或更少缩进的定义）
            indent = len(line) - len(stripped)
            frag_lines = [line]
            j = i + 1

            while j < len(lines) and j - i < max_lines:
                next_line = lines[j]
                next_stripped = next_line.lstrip()

                # 如果遇到同缩进的新定义，停止
                if next_stripped and not next_stripped.startswith("#"):
                    next_indent = len(next_line) - len(next_stripped)
                    if next_indent <= indent and next_stripped.startswith(
                        ("def ", "class ", "async def ", "@", "import ", "from ")
                    ):
                        break

                frag_lines.append(next_line)
                j += 1

            # 去掉尾部空行
            while frag_lines and not frag_lines[-1].strip():
                frag_lines.pop()

            if len(frag_lines) >= 3:  # 至少 3 行才值得训练
                fragments.append("\n".join(frag_lines))

            i = j
        else:
            i += 1

    return fragments
