"""
用户相关 Pydantic Schema
"""

import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, field_validator

from ..models import UserRole


class UserBase(BaseModel):
    username: str
    email: EmailStr
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or len(v) < 3 or len(v) > 50:
            raise ValueError("用户名只能包含字母数字下划线，长度3-50")
        return v


class UserCreate(UserBase):
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("密码至少8位")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码至少包含一个数字")
        return v


class UserUpdate(BaseModel):
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    preferences: Optional[dict] = None


class UserResponse(UserBase):
    id: uuid.UUID
    role: UserRole
    is_active: bool
    is_verified: bool
    post_count: int = 0
    follower_count: int = 0
    following_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class UserListItem(BaseModel):
    id: uuid.UUID
    username: str
    nickname: Optional[str]
    avatar_url: Optional[str]
    role: UserRole
    post_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    items: List[UserListItem]
    total: int
    page: int
    page_size: int
    total_pages: int


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("新密码至少8位")
        return v
