"""
tags router — 标签管理
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import Tag, UserRole
from ..auth import get_current_user, require_role
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("")
async def list_tags(
    sort_by: str = Query("post_count", regex="^(name|post_count|created_at)$"),
    order: str = Query("desc", regex="^(asc|desc)$"),
    limit: int = Query(50, ge=1, le=200),
    is_hot: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
):
    """标签列表"""
    query = select(Tag)

    if is_hot is not None:
        query = query.where(Tag.is_hot == is_hot)

    sort_col = getattr(Tag, sort_by)
    query = query.order_by(desc(sort_col) if order == "desc" else sort_col).limit(limit)

    result = await db.execute(query)
    tags = result.scalars().all()

    return [
        {
            "id": str(t.id),
            "name": t.name,
            "slug": t.slug,
            "description": t.description,
            "color": t.color,
            "icon": t.icon,
            "post_count": t.post_count,
            "is_hot": t.is_hot,
        }
        for t in tags
    ]


@router.get("/hot")
async def hot_tags(
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    """热门标签"""
    query = (
        select(Tag)
        .order_by(desc(Tag.post_count))
        .limit(limit)
    )
    result = await db.execute(query)
    tags = result.scalars().all()

    return [
        {
            "id": str(t.id),
            "name": t.name,
            "post_count": t.post_count,
            "color": t.color,
        }
        for t in tags
    ]


@router.get("/{tag_id}")
async def get_tag(tag_id: UUID, db: AsyncSession = Depends(get_db)):
    """获取单个标签详情"""
    result = await db.execute(select(Tag).where(Tag.id == tag_id))
    tag = result.scalar_one_or_none()

    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")

    return {
        "id": str(tag.id),
        "name": tag.name,
        "slug": tag.slug,
        "description": tag.description,
        "color": tag.color,
        "icon": tag.icon,
        "post_count": tag.post_count,
        "is_hot": tag.is_hot,
        "created_at": tag.created_at,
    }


@router.post("", status_code=201)
async def create_tag(
    name: str,
    description: Optional[str] = None,
    color: str = "#3b82f6",
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.EDITOR)),
    db: AsyncSession = Depends(get_db),
):
    """创建标签（管理员/编辑）"""
    result = await db.execute(select(Tag).where(Tag.name == name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="标签已存在")

    import re
    slug = re.sub(r"[-\s]+", "-", name.lower().strip())[:50]

    tag = Tag(
        name=name,
        slug=slug,
        description=description,
        color=color,
    )
    db.add(tag)
    await db.commit()
    await db.refresh(tag)

    logger.info(f"管理员 {current_user.username} 创建标签: {name}")
    return {"id": str(tag.id), "name": tag.name, "slug": tag.slug}


@router.delete("/{tag_id}", status_code=204)
async def delete_tag(
    tag_id: UUID,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: AsyncSession = Depends(get_db),
):
    """删除标签（管理员）"""
    result = await db.execute(select(Tag).where(Tag.id == tag_id))
    tag = result.scalar_one_or_none()

    if not tag:
        raise HTTPException(status_code=404, detail="标签不存在")

    if tag.post_count > 0:
        raise HTTPException(status_code=400, detail="该标签下还有文章，无法删除")

    await db.delete(tag)
    await db.commit()

    logger.info(f"管理员 {current_user.username} 删除标签: {tag.name}")
