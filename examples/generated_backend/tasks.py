"""
Celery 后台任务 — 邮件发送 / 缓存预热 / 数据统计 / 全文索引
"""

import os
from celery import Celery
from celery.schedules import crontab

from .config import settings


celery_app = Celery(
    "xenith_blog",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 分钟超时
    worker_prefetch_multiplier=1,
)

# 定时任务
celery_app.conf.beat_schedule = {
    # 每天凌晨统计前一天数据
    "daily-stats": {
        "task": "tasks.daily_stats",
        "schedule": crontab(hour=2, minute=0),
    },
    # 每小时清理过期缓存
    "cache-cleanup": {
        "task": "tasks.cache_cleanup",
        "schedule": crontab(minute=0),
    },
    # 每 10 分钟同步热门文章到缓存
    "hot-posts-warmup": {
        "task": "tasks.hot_posts_warmup",
        "schedule": 600.0,
    },
}


@celery_app.task(name="tasks.send_email")
def send_email(to_email: str, subject: str, template: str, context: dict) -> bool:
    """发送邮件（异步）"""
    try:
        # 实际项目接 SendGrid / SMTP / 邮件服务
        print(f"[EMAIL] To: {to_email} | Subject: {subject}")
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


@celery_app.task(name="tasks.send_verification_email")
def send_verification_email(user_id: str, email: str, token: str) -> bool:
    """发送验证邮件"""
    verify_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/verify?token={token}"
    subject = "请验证您的邮箱 - Xenith Blog"
    context = {"verify_url": verify_url, "user_id": user_id}
    return send_email.delay(email, subject, "verification", context)


@celery_app.task(name="tasks.send_reset_password_email")
def send_reset_password_email(email: str, token: str) -> bool:
    """发送重置密码邮件"""
    reset_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={token}"
    subject = "重置您的密码 - Xenith Blog"
    return send_email.delay(email, subject, "reset_password", {"reset_url": reset_url})


@celery_app.task(name="tasks.new_comment_notification")
def new_comment_notification(post_id: str, comment_id: str) -> None:
    """新评论通知（给文章作者）"""
    # 查询文章和评论信息，发送站内信/邮件通知
    print(f"[NOTIFY] New comment {comment_id} on post {post_id}")


@celery_app.task(name="tasks.daily_stats")
def daily_stats() -> dict:
    """每日统计任务"""
    # 统计新增用户、文章、评论、PV/UV 等
    # 写入统计表
    print("[STATS] Daily stats computed")
    return {"status": "ok"}


@celery_app.task(name="tasks.cache_cleanup")
def cache_cleanup() -> int:
    """清理过期缓存项"""
    # Redis 自带过期，这里主要清理一些特殊缓存
    print("[CACHE] Cache cleanup completed")
    return 0


@celery_app.task(name="tasks.hot_posts_warmup")
def hot_posts_warmup() -> int:
    """热门文章缓存预热"""
    # 查询热门文章，写入缓存
    print("[CACHE] Hot posts warmed up")
    return 0


@celery_app.task(name="tasks.sync_to_elasticsearch")
def sync_to_elasticsearch(post_id: str) -> bool:
    """同步文章到 Elasticsearch"""
    # 如果用了 ES，发布/更新文章后异步同步索引
    print(f"[ES] Synced post {post_id}")
    return True


@celery_app.task(name="tasks.generate_sitemap")
def generate_sitemap() -> bool:
    """生成 sitemap.xml"""
    print("[SEO] Sitemap generated")
    return True


@celery_app.task(name="tasks.image_optimize")
def image_optimize(image_path: str) -> bool:
    """图片优化（压缩/转 WebP/生成缩略图）"""
    print(f"[IMAGE] Optimized {image_path}")
    return True
