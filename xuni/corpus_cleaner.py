"""
训练语料质量筛选器
从粒子化语料中筛出真正高质量的训练片段
"""
import re
import json
from typing import List, Dict, Tuple


# 严格噪声规则
NOISE_REGEX = [
    r'^https?://',                    # 纯 URL
    r'^\[!\[',                        # 徽章 [![...](...
    r'^!\[',                          # 图片 ![...](...
    r'^\s*[\{\}\[\]\(\)<>]+\s*$',     # 纯符号
    r'^\s*(import|from|package)\s+',  # import 语句
    r'^\s*(def|class|func|fn)\s+\w+', # 定义语句
    r'^\s*[@\$]',                     # 装饰器/变量
    r'^\s*\d+(\.\d+)?\s*$',           # 纯数字
    r'^\s*[a-z_]+\s*=\s*$',           # 未完成赋值
    r'^\s*#',                         # 纯注释符号
    r'^\s*//',                        # 纯注释符号
    r'^\s*\*?\s*$',                   # 纯星号
    r'^\s*[\-\+\*]\s+\S+\s*$',        # 单词列表项
    r'^\s*(http|ftp|www)\.',          # 链接前缀
    r'^\s*[a-zA-Z]\s*$',              # 单字符
    r'^\s*(true|false|null|none|nil)\s*$',  # 单关键字
    r'^\s*[A-Z][a-z]+\s*\?+\s*$',     # 纯英文问句 "What?" "Why?"
    r'^.*\?\s*$',                     # 以问号结尾的纯问句
    r'^\s*NOT\w+',                    # NOT_EXPECTED 类常量
    r'^\s*\w+\s*\?\?\?',              # 多问号 "xxx???"
    r'^[A-Z_][A-Z_0-9]+\s*[:=]',      # 全大写常量定义
    r'^\s*(NOT|EXPECTED|RE)_',        # 测试常量前缀
    r'^\s*```\s*$',                   # 代码块标记
    r'^\s*\|.*\|\s*$',                # markdown 表格行
    r'^\s*[*\-+]\s+\[[ xX]\]\s*$',    # todo 列表空项
]

# 必须过滤的 URL/链接模式
URL_PATTERNS = [
    r'https?://\S+',
    r'www\.\S+',
    r'\[!\[.*?\]\(.*?\)\]\(.*?\)',    # 嵌套徽章
    r'!\[.*?\]\(.*?\)',               # 图片
    r'\[[^\]]+\]\([^)]+\)',           # 任何 markdown 链接 [text](url)
    r'shields\.io',
    r'pypi\.org',
    r'github\.com/.+/(blob|tree)/',
    r'youtube\.com',
    r'youtu\.be',
    r'<br\s*/?>',                     # HTML 换行
    r'\.md\)',                        # markdown 文件引用
    r'\.\./',                         # 相对路径引用
]

# 必须过滤的关键词（噪声内容）
NOISE_KEYWORDS = [
    'shields.io', 'img.shields', 'pypi.org/project',
    'codecov.io', 'coveralls.io', 'circleci.com',
    'travis-ci', 'github.com/.*/actions',
    'pydantic.main', 'pydantic.fields',
    'NOTEXPECTED', 'EXPECTED_RE_TYPE',
]


def is_noise(text: str) -> bool:
    """判断是否是噪声"""
    t = text.strip()
    if len(t) < 25 or len(t) > 220:
        return True
    # 正则规则
    for pat in NOISE_REGEX:
        if re.match(pat, t):
            return True
    # 含 URL 模式直接过滤
    for pat in URL_PATTERNS:
        if re.search(pat, t):
            return True
    # 含噪声关键词
    for kw in NOISE_KEYWORDS:
        if kw in t.lower():
            return True
    # 全是符号
    if set(t) <= set('._-*=#/[\\](){}<>@&|'):
        return True
    # 字母数字占比太低（<50%）且无中文
    chinese = sum(1 for c in t if '\u4e00' <= c <= '\u9fff')
    alpha = sum(1 for c in t if c.isalpha())
    if chinese == 0 and alpha < len(t) * 0.4:
        return True
    # 重复字符超过 50%
    if len(set(t)) < len(t) * 0.3:
        return True
    return False


def clean_fragments(fragments: List[str]) -> Tuple[List[str], Dict]:
    """筛选片段，返回 (干净片段, 统计)"""
    clean = []
    removed_reasons = {'too_short': 0, 'too_long': 0, 'noise': 0, 'duplicate': 0}

    seen = set()
    for frag in fragments:
        t = frag.strip()
        if len(t) < 25:
            removed_reasons['too_short'] += 1
            continue
        if len(t) > 220:
            removed_reasons['too_long'] += 1
            continue
        if is_noise(t):
            removed_reasons['noise'] += 1
            continue
        if t in seen:
            removed_reasons['duplicate'] += 1
            continue
        seen.add(t)
        clean.append(t)

    stats = {
        'original': len(fragments),
        'clean': len(clean),
        'removed_total': len(fragments) - len(clean),
        'removed_reasons': removed_reasons,
        'retention_rate': round(len(clean) / max(len(fragments), 1) * 100, 1),
    }
    return clean, stats


def clean_particle_corpus(input_path: str, output_path: str) -> Dict:
    """筛选粒子容器中的语料"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    fragments = data.get('all_fragments', [])
    clean, stats = clean_fragments(fragments)

    data['all_fragments'] = clean
    data['clean_stats'] = stats

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    print(f"筛选: {stats['original']} → {stats['clean']} (保留 {stats['retention_rate']}%)")
    print(f"  过短: {stats['removed_reasons']['too_short']}")
    URL_PATTERNS_list = stats['removed_reasons']
    print(f"  过长: {stats['removed_reasons']['too_long']}")
    print(f"  噪声: {stats['removed_reasons']['noise']}")
    print(f"  重复: {stats['removed_reasons']['duplicate']}")
    return stats


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python corpus_cleaner.py <input.json> <output.json>")
        sys.exit(1)
    clean_particle_corpus(sys.argv[1], sys.argv[2])
