#!/bin/bash
# 合鸣-13 一键推送到 GitHub
# 用法: 在任意窗口 cd /workspace/xuni && bash push_to_github.sh
#
# 这个脚本会把所有"待推送"的 commit 推到 GitHub
# 包括: 训练产物 + 粒子化语料 + 代码改动
#
# 如果没有 push 权限, 会提示你配置 token

set -e
cd "$(dirname "$0")"

echo "============================================================"
echo "  合鸣-13 推送到 GitHub"
echo "============================================================"
echo ""

# 检查待推送的 commit
AHEAD=$(git rev-list --count origin/master..HEAD 2>/dev/null || echo "0")
if [ "$AHEAD" = "0" ]; then
    echo "✅ 没有待推送的 commit，已是最新"
    exit 0
fi

echo "📦 待推送 commit: $AHEAD 个"
git log origin/master..HEAD --oneline
echo ""

# 显示待推送内容大小
echo "📊 待推送内容:"
git diff --stat origin/master..HEAD | tail -5
echo ""

# 尝试推送
echo "🚀 推送中..."
if git push origin master 2>&1; then
    echo ""
    echo "✅ 推送成功!"
    echo ""
    echo "训练产物已保存在 GitHub:"
    echo "  - checkpoints/harmonia13/ (训练好的模型)"
    echo "  - corpus_particles.json (粒子化语料)"
    echo "  - logs/harmonia_daemon.jsonl (训练日志)"
    echo ""
    echo "任何窗口 git clone/pull 即可拿到全部数据"
else
    echo ""
    echo "❌ 推送失败 — 需要 GitHub 认证"
    echo ""
    echo "解决方法 (任选一种):"
    echo ""
    echo "方法1 - 用 token (推荐, 临时):"
    echo "  1. 去 https://github.com/settings/tokens 生成 token (仅 repo 权限)"
    echo "  2. 运行:"
    echo "     git push https://<你的token>@github.com/qingluan-studio/xuni.git master"
    echo "  3. 推送成功后立刻去 GitHub 撤销 token"
    echo ""
    echo "方法2 - 配置 git credential:"
    echo "  git config credential.helper store"
    echo "  git push  # 会提示输入用户名和密码(token)"
    echo ""
    echo "方法3 - SSH (如果配过 SSH key):"
    echo "  git remote set-url origin git@github.com:qingluan-studio/xuni.git"
    echo "  git push"
fi
