"""
search router — 全文搜索
"""

from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Post, PostStatus, User
from ..config import settings

router = APIRouter()


@router.get("")
async def search(
    q: str = Query(..., min_length=1, max_length=100),
    type: str = Query("all", regex="^(all|posts|users|tags)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """全局搜索（简单版，基于 ILIKE；生产建议用 ES）"""
    offset = (page - 1) * page_size
    results = {"posts": [], "users": [], "total": 0}

    pattern = f"%{q}%"

    if type in ("all", "posts"):
        query = (
            select(Post)
            .where(Post.status == PostStatus.PUBLISHED)
            .where(
                or_(
                    Post.title.ilike(pattern),
                    Post.summary.ilike(pattern),
                    Post.content.ilike(pattern),
                )
            )
            .order_by(Post.view_count.desc())
            .offset(offset)
            .limit(page_size)
        )
        post_result = await db.execute(query)
        posts = post_result.scalars().all()

        results["posts"] = [
            {
                "id": str(p.id),
                "title": p.title,
                "summary": p.summary,
                "slug": p.slug,
                "cover_image": p.cover_image,
                "view_count": p.view_count,
                "like_count": p.like_count,
                "author": {
                    "id": str(p.author_id),
                    "username": "author",
                },
                "published_at": p.published_at.isoformat() if p.published_at else None,
                "highlight": _highlight(p.title or p.summary or "", q),
            }
            for p in posts
        ]

    if type in ("all", "users"):
        query = (
            select(User)
            .where(User.is_active == True)
            .where(
                or_(
                    User.username.ilike(pattern),
                    User.nickname.ilike(pattern),
                )
            )
            .order_by(User.post_count.desc())
            .offset(offset)
            .limit(page_size)
        )
        user_result = await db.execute(query)
        users = user_result.scalars().all()

        results["users"] = [
            {
                "id": str(u.id),
                "username": u.username,
                "nickname": u.nickname,
                "avatar_url": u.avatar_url,
                "post_count": u.post_count,
                "highlight": _highlight(u.nickname or u.username, q),
            }
            for u in users
        ]

    results["total"] = len(results["posts"]) + len(results["users"])
    return results


def _highlight(text: str, keyword: str) -> str:
    """简单关键词高亮"""
    import re
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group()}</mark>", text, count=3)


@router.get("/suggest")
async def search_suggest(
    q: str = Query(..., min_length=1, max_length=50),
    limit: int = Query(10, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """搜索建议（自动补全）"""
    pattern = f"{q}%"

    # 文章标题建议
    post_query = (
        select(Post.title)
        .where(Post.status == PostStatus.PUBLISHED)
        .where(Post.title.ilike(pattern))
        .limit(limit)
    )
    post_result = await db.execute(post_query)
    titles = [r[0] for r in post_result.fetchall()]

    return {"suggestions": titles[:limit]}
