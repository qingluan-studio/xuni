"""
users router — 用户信息 / 个人资料 / 关注
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..models import User, UserRole
from ..auth import get_current_user, hash_password
from ..schemas.user import (
    UserResponse, UserUpdate, UserListResponse,
    ChangePasswordRequest,
)
from ..config import settings
from ..utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserResponse.model_validate(current_user)


@router.put("/me", response_model=UserResponse)
async def update_me(
    user_in: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """更新当前用户信息"""
    update_data = user_in.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if hasattr(current_user, field) and value is not None:
            setattr(current_user, field, value)

    await db.commit()
    await db.refresh(current_user)

    logger.info(f"用户 {current_user.username} 更新了个人资料")
    return UserResponse.model_validate(current_user)


@router.post("/me/change-password")
async def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """修改密码"""
    from ..auth import verify_password

    if not verify_password(req.old_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="原密码错误")

    current_user.password_hash = hash_password(req.new_password)
    await db.commit()

    logger.info(f"用户 {current_user.username} 修改了密码")
    return {"message": "密码修改成功"}


@router.get("", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    keyword: Optional[str] = None,
    role: Optional[UserRole] = None,
    db: AsyncSession = Depends(get_db),
):
    """用户列表"""
    from ..schemas.user import UserListItem

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

    query = query.order_by(User.created_at.desc()).offset(offset).limit(page_size)

    result = await db.execute(query)
    users = result.scalars().all()

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    total_pages = (total + page_size - 1) // page_size

    items = [UserListItem.model_validate(u) for u in users]
    return UserListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: UUID, db: AsyncSession = Depends(get_db)):
    """获取指定用户信息"""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(status_code=404, detail="用户不存在")

    return UserResponse.model_validate(user)


@router.post("/{user_id}/follow")
async def follow_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """关注用户（简化版，实际需要 follow 表）"""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能关注自己")

    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    if not target or not target.is_active:
        raise HTTPException(status_code=404, detail="用户不存在")

    target.follower_count += 1
    current_user.following_count += 1
    await db.commit()

    return {"followed": True, "follower_count": target.follower_count}


@router.post("/{user_id}/unfollow")
async def unfollow_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """取消关注"""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="不能取关自己")

    result = await db.execute(select(User).where(User.id == user_id))
    target = result.scalar_one_or_none()
    if not target:
        raise HTTPException(status_code=404, detail="用户不存在")

    target.follower_count = max(0, target.follower_count - 1)
    current_user.following_count = max(0, current_user.following_count - 1)
    await db.commit()

    return {"unfollowed": True, "follower_count": target.follower_count}
