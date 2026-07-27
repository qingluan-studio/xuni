"""
comments router — 评论 CRUD + 嵌套 + 点赞
"""

from typing import Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Comment, Post, User, PostStatus
from ..auth import get_current_user
from ..schemas.comment import (
    CommentCreate, CommentUpdate, CommentResponse,
    CommentListResponse,
)
from ..config import settings
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("", response_model=CommentListResponse)
async def list_comments(
    post_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """获取文章的评论列表（按创建时间倒序，顶层评论）"""
    offset = (page - 1) * page_size

    # 只查顶层评论（parent_id is NULL）
    query = (
        select(Comment)
        .where(Comment.post_id == post_id)
        .where(Comment.parent_id.is_(None))
        .where(Comment.is_approved == True)
        .order_by(desc(Comment.created_at))
        .offset(offset)
        .limit(page_size)
    )

    count_query = (
        select(func.count(Comment.id))
        .where(Comment.post_id == post_id)
        .where(Comment.parent_id.is_(None))
        .where(Comment.is_approved == True)
    )

    result = await db.execute(query)
    comments = result.scalars().all()

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    total_pages = (total + page_size - 1) // page_size

    items = [CommentResponse.model_validate(c) for c in comments]
    return CommentListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{comment_id}/replies")
async def get_replies(
    comment_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """获取评论的回复（子评论）"""
    offset = (page - 1) * page_size

    query = (
        select(Comment)
        .where(Comment.parent_id == comment_id)
        .where(Comment.is_approved == True)
        .order_by(Comment.created_at)
        .offset(offset)
        .limit(page_size)
    )

    result = await db.execute(query)
    replies = result.scalars().all()

    return [CommentResponse.model_validate(r) for r in replies]


@router.post("", response_model=CommentResponse, status_code=201)
async def create_comment(
    comment_in: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """创建评论"""
    # 检查文章是否存在且已发布
    post_result = await db.execute(
        select(Post).where(Post.id == comment_in.post_id)
    )
    post = post_result.scalar_one_or_none()
    if not post or post.status != PostStatus.PUBLISHED:
        raise HTTPException(status_code=404, detail="文章不存在")

    # 如果是回复，检查父评论
    parent = None
    depth = 0
    path = ""
    if comment_in.parent_id:
        parent_result = await db.execute(
            select(Comment).where(Comment.id == comment_in.parent_id)
        )
        parent = parent_result.scalar_one_or_none()
        if not parent or parent.post_id != comment_in.post_id:
            raise HTTPException(status_code=400, detail="父评论不存在")
        depth = parent.depth + 1
        if depth > 5:
            raise HTTPException(status_code=400, detail="评论嵌套深度不能超过5层")
        path = f"{parent.path}.{parent.id}" if parent.path else str(parent.id)

    comment = Comment(
        content=comment_in.content,
        post_id=comment_in.post_id,
        user_id=current_user.id,
        parent_id=comment_in.parent_id,
        depth=depth,
        path=path,
        is_approved=True,
    )
    db.add(comment)
    await db.flush()

    # 更新文章评论数
    post.comment_count += 1

    # 更新父评论回复数
    if parent:
        parent.reply_count += 1

    await db.commit()
    await db.refresh(comment)

    logger.info(f"用户 {current_user.username} 评论文章 {post.id}")
    return CommentResponse.model_validate(comment)


@router.put("/{comment_id}", response_model=CommentResponse)
async def update_comment(
    comment_id: UUID,
    comment_in: CommentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """更新评论"""
    result = await db.execute(select(Comment).where(Comment.id == comment_id))
    comment = result.scalar_one_or_none()

    if not comment:
        raise HTTPException(status_code=404, detail="评论不存在")

    if comment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权修改此评论")

    comment.content = comment_in.content
    comment.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(comment)

    return CommentResponse.model_validate(comment)


@router.delete("/{comment_id}", status_code=204)
async def delete_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """删除评论（软删除）"""
    result = await db.execute(select(Comment).where(Comment.id == comment_id))
    comment = result.scalar_one_or_none()

    if not comment:
        raise HTTPException(status_code=404, detail="评论不存在")

    if comment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="无权删除此评论")

    comment.deleted_at = datetime.utcnow()
    comment.is_approved = False
    comment.content = "[已删除]"

    # 更新文章评论数
    post_result = await db.execute(select(Post).where(Post.id == comment.post_id))
    post = post_result.scalar_one_or_none()
    if post:
        post.comment_count = max(0, post.comment_count - 1)

    await db.commit()
    logger.info(f"用户 {current_user.username} 删除评论 {comment_id}")


@router.post("/{comment_id}/like")
async def like_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """点赞评论"""
    result = await db.execute(select(Comment).where(Comment.id == comment_id))
    comment = result.scalar_one_or_none()

    if not comment or not comment.is_approved:
        raise HTTPException(status_code=404, detail="评论不存在")

    comment.like_count += 1
    await db.commit()

    return {"liked": True, "like_count": comment.like_count}
