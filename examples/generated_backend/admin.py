"""
admin router — 管理员后台接口
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import User, Post, Comment, Tag, UserRole, PostStatus
from ..auth import get_current_user, require_role, hash_password
from ..config import settings
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("/stats")
async def admin_stats(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """仪表盘统计数据"""
    # 用户统计
    total_users = (await db.execute(select(func.count(User.id)))).scalar() or 0
    active_users = (await db.execute(
        select(func.count(User.id)).where(User.is_active == True)
    )).scalar() or 0
    new_today = 0  # 简化

    # 文章统计
    total_posts = (await db.execute(select(func.count(Post.id)))).scalar() or 0
    published_posts = (await db.execute(
        select(func.count(Post.id)).where(Post.status == PostStatus.PUBLISHED)
    )).scalar() or 0
    draft_posts = total_posts - published_posts

    # 评论统计
    total_comments = (await db.execute(select(func.count(Comment.id)))).scalar() or 0
    pending_comments = (await db.execute(
        select(func.count(Comment.id)).where(Comment.is_approved == False)
    )).scalar() or 0

    # 标签统计
    total_tags = (await db.execute(select(func.count(Tag.id)))).scalar() or 0

    # 角色分布
    role_stats = {}
    for role in UserRole:
        count = (await db.execute(
            select(func.count(User.id)).where(User.role == role)
        )).scalar() or 0
        role_stats[role.value] = count

    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "new_today": new_today,
            "by_role": role_stats,
        },
        "posts": {
            "total": total_posts,
            "published": published_posts,
            "draft": draft_posts,
        },
        "comments": {
            "total": total_comments,
            "pending": pending_comments,
        },
        "tags": {
            "total": total_tags,
        },
    }


@router.get("/users")
async def admin_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    keyword: Optional[str] = None,
    role: Optional[UserRole] = None,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """管理用户列表"""
    offset = (page - 1) * page_size

    query = select(User)
    count_query = select(func.count(User.id))

    if keyword:
        pattern = f"%{keyword}%"
        query = query.where(
            (User.username.ilike(pattern)) |
            (User.email.ilike(pattern)) |
            (User.nickname.ilike(pattern))
        )
        count_query = count_query.where(
            (User.username.ilike(pattern)) |
            (User.email.ilike(pattern)) |
            (User.nickname.ilike(pattern))
        )

    if role:
        query = query.where(User.role == role)
        count_query = count_query.where(User.role == role)

    query = query.order_by(desc(User.created_at)).offset(offset).limit(page_size)

    result = await db.execute(query)
    users = result.scalars().all()

    total = (await db.execute(count_query)).scalar() or 0

    return {
        "items": [
            {
                "id": str(u.id),
                "username": u.username,
                "email": u.email,
                "nickname": u.nickname,
                "role": u.role.value,
                "is_active": u.is_active,
                "is_verified": u.is_verified,
                "post_count": u.post_count,
                "created_at": u.created_at.isoformat(),
                "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            }
            for u in users
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.patch("/users/{user_id}/role")
async def set_user_role(
    user_id: UUID,
    role: UserRole,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """设置用户角色"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    old_role = user.role
    user.role = role
    await db.commit()

    logger.info(f"管理员 {current_user.username} 将用户 {user.username} 角色从 {old_role} 改为 {role}")
    return {"message": "角色更新成功", "role": role.value}


@router.patch("/users/{user_id}/ban")
async def ban_user(
    user_id: UUID,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """封禁用户"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    if user.id == current_user.id:
        raise HTTPException(status_code=400, detail="不能封禁自己")

    user.is_active = False
    await db.commit()

    logger.warning(f"管理员 {current_user.username} 封禁用户 {user.username}")
    return {"message": "用户已封禁"}


@router.patch("/users/{user_id}/unban")
async def unban_user(
    user_id: UUID,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """解封用户"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    user.is_active = True
    await db.commit()

    logger.info(f"管理员 {current_user.username} 解封用户 {user.username}")
    return {"message": "用户已解封"}


@router.delete("/posts/{post_id}", status_code=204)
async def admin_delete_post(
    post_id: UUID,
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.EDITOR)),
    db: AsyncSession = Depends(get_db),
):
    """管理员删除文章"""
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="文章不存在")

    # 减少作者计数
    author_result = await db.execute(select(User).where(User.id == post.author_id))
    author = author_result.scalar_one_or_none()
    if author:
        author.post_count = max(0, author.post_count - 1)

    await db.delete(post)
    await db.commit()

    logger.warning(f"管理员 {current_user.username} 删除文章 {post_id}")
