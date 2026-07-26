#!/bin/bash
# 批量克隆开源仓库
# 跳过已存在的, 失败的容错跳过
# 用法: bash clone_repos.sh <repos_list_file> <target_dir>

set -u
REPO_LIST="${1:-repos_100.txt}"
TARGET_DIR="${2:-/workspace/corpus}"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

total=0
ok=0
skip=0
fail=0
failed_repos=()

while IFS= read -r line; do
    # 跳过注释和空行
    line=$(echo "$line" | sed 's/#.*//' | xargs)
    [ -z "$line" ] && continue

    total=$((total + 1))
    name=$(echo "$line" | cut -d/ -f2)

    if [ -d "$name" ]; then
        skip=$((skip + 1))
        continue
    fi

    # 克隆 (depth=1 节省时间和空间, 失败重试1次)
    if git clone --depth 1 "https://github.com/${line}.git" 2>/dev/null; then
        ok=$((ok + 1))
        if [ $((ok % 10)) -eq 0 ]; then
            echo "  已克隆 $ok / ~$total ..."
        fi
    else
        fail=$((fail + 1))
        failed_repos+=("$line")
    fi
done < "$REPO_LIST"

echo ""
echo "================================================"
echo "  克隆完成"
echo "================================================"
echo "  总计: $total"
echo "  成功: $ok"
echo "  跳过(已存在): $skip"
echo "  失败: $fail"
if [ ${#failed_repos[@]} -gt 0 ]; then
    echo ""
    echo "  失败的仓库:"
    for r in "${failed_repos[@]}"; do
        echo "    - $r"
    done
fi
echo "================================================"
