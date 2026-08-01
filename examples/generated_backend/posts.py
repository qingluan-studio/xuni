"""
posts router — 文章 CRUD + 列表 + 搜索 + 点赞 + 收藏
"""

from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, desc, asc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Post, User, PostStatus, Tag
from ..auth import get_current_user, require_role
from ..schemas.post import (
    PostCreate, PostUpdate, PostResponse,
    PostListResponse, PostListItem,
)
from ..redis_client import cache_get, cache_set, cache_delete, cache_invalidate_pattern
from ..config import settings
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("", response_model=PostListResponse)
async def list_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    status: Optional[PostStatus] = PostStatus.PUBLISHED,
    tag: Optional[str] = None,
    author_id: Optional[UUID] = None,
    keyword: Optional[str] = None,
    sort_by: str = Query("published_at", regex="^(published_at|created_at|view_count|like_count)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    db: AsyncSession = Depends(get_db),
):
    """文章列表"""
    cache_key = f"posts:list:{page}:{page_size}:{status}:{tag}:{author_id}:{keyword}:{sort_by}:{order}"
    cached = await cache_get(cache_key)
    if cached:
        return PostListResponse(**cached)

    offset = (page - 1) * page_size

    # 构建查询
    query = select(Post)
    count_query = select(func.count(Post.id))

    if status:
        query = query.where(Post.status == status)
        count_query = count_query.where(Post.status == status)

    if author_id:
        query = query.where(Post.author_id == author_id)
        count_query = count_query.where(Post.author_id == author_id)

    if keyword:
        like_pattern = f"%{keyword}%"
        query = query.where(or_(Post.title.ilike(like_pattern), Post.summary.ilike(like_pattern)))
        count_query = count_query.where(or_(Post.title.ilike(like_pattern), Post.summary.ilike(like_pattern)))

    if tag:
        query = query.where(Post.tag_list.any(tag))
        count_query = count_query.where(Post.tag_list.any(tag))

    # 排序
    sort_col = getattr(Post, sort_by)
    query = query.order_by(desc(sort_col) if order == "desc" else asc(sort_col))

    # 分页
    query = query.offset(offset).limit(page_size)

    result = await db.execute(query)
    posts = result.scalars().all()

    # 总数
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    total_pages = (total + page_size - 1) // page_size

    items = [PostListItem.model_validate(p) for p in posts]
    response = PostListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )

    await cache_set(cache_key, response.model_dump(), ttl=60)
    return response


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: UUID, db: AsyncSession = Depends(get_db)):
    """获取文章详情"""
    cache_key = f"posts:detail:{post_id}"
    cached = await cache_get(cache_key)
    if cached:
        # 增加浏览量（异步更好，这里简化）
        result = await db.execute(select(Post).where(Post.id == post_id))
        post = result.scalar_one_or_none()
        if post:
            post.view_count += 1
            await db.commit()
        return PostResponse(**cached)

    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="文章不存在")

    if post.status != PostStatus.PUBLISHED:
        raise HTTPException(status_code=404, detail="文章不存在")

    # 浏览量+1
    post.view_count += 1
    await db.commit()
    await db.refresh(post)

    response = PostResponse.model_validate(post)
    await cache_set(cache_key, response.model_dump(), ttl=300)
    return response


@router.post("", response_model=PostResponse, status_code=201)
async def create_post(
    post_in: PostCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建文章"""
    # 生成 slug
    slug = post_in.slug or _slugify(post_in.title)

    # 检查 slug 重复
    result = await db.execute(select(Post).where(Post.slug == slug))
    if result.scalar_one_or_none():
        slug = f"{slug}-{UUID(int=hash(slug) & 0xffffffff).hex[:8]}"

    post = Post(
        title=post_in.title,
        slug=slug,
        summary=post_in.summary,
        content=post_in.content,
        cover_image=post_in.cover_image,
        status=post_in.status or PostStatus.DRAFT,
        author_id=current_user.id,
        tag_list=post_in.tags or [],
        word_count=len(post_in.content),
        reading_time_minutes=max(1, len(post_in.content) // 500),
        published_at=datetime.utcnow() if post_in.status == PostStatus.PUBLISHED else None,
    )
    db.add(post)
    await db.flush()

    # 处理标签
    if post_in.tags:
        for tag_name in post_in.tags:
            result = await db.execute(select(Tag).where(Tag.name == tag_name))
            tag = result.scalar_one_or_none()
            if not tag:
                tag = Tag(name=tag_name, slug=_slugify(tag_name))
                db.add(tag)
                await db.flush()
            tag.post_count += 1
            post.tags.append(tag)

    current_user.post_count += 1
    await db.commit()
    await db.refresh(post)

    # 清缓存
    await cache_invalidate_pattern("posts:list:*")

    logger.info(f"用户 {current_user.username} 创建文章: {post.title}")
    return PostResponse.model_validate(post)


@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: UUID,
    post_in: PostUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """更新文章"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="文章不存在")

    if post.author_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权修改此文章")

    update_data = post_in.model_dump(exclude_unset=True)

    if "title" in update_data:
        post.title = update_data["title"]
        if not post.slug:
            post.slug = _slugify(update_data["title"])

    for field in ["summary", "content", "cover_image", "status"]:
        if field in update_data:
            setattr(post, field, update_data[field])

    if "content" in update_data:
        post.word_count = len(update_data["content"])
        post.reading_time_minutes = max(1, len(update_data["content"]) // 500)

    if update_data.get("status") == PostStatus.PUBLISHED and not post.published_at:
        post.published_at = datetime.utcnow()

    if "tags" in update_data:
        post.tag_list = update_data["tags"] or []

    post.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(post)

    # 清缓存
    await cache_delete(f"posts:detail:{post_id}")
    await cache_invalidate_pattern("posts:list:*")

    return PostResponse.model_validate(post)


@router.delete("/{post_id}", status_code=204)
async def delete_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除文章"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="文章不存在")

    if post.author_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权删除此文章")

    await db.delete(post)
    current_user.post_count = max(0, current_user.post_count - 1)
    await db.commit()

    await cache_delete(f"posts:detail:{post_id}")
    await cache_invalidate_pattern("posts:list:*")

    logger.info(f"用户 {current_user.username} 删除文章: {post_id}")


@router.post("/{post_id}/like")
async def like_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """点赞文章"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post or post.status != PostStatus.PUBLISHED:
        raise HTTPException(status_code=404, detail="文章不存在")

    # 简单实现：直接 +1（实际应该用 like 表防重复）
    post.like_count += 1
    await db.commit()

    await cache_delete(f"posts:detail:{post_id}")
    return {"liked": True, "like_count": post.like_count}


@router.post("/{post_id}/bookmark")
async def bookmark_post(
    post_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """收藏文章"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post or post.status != PostStatus.PUBLISHED:
        raise HTTPException(status_code=404, detail="文章不存在")

    post.bookmark_count += 1
    await db.commit()

    return {"bookmarked": True, "bookmark_count": post.bookmark_count}


def _slugify(text: str) -> str:
    """简单 slug 生成（中文直接用拼音或原文，这里简化）"""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:100] or "post"
