"""
评论相关 Pydantic Schema
"""

import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


class CommentBase(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("评论内容不能为空")
        if len(v) > 2000:
            raise ValueError("评论不能超过2000字")
        return v


class CommentCreate(CommentBase):
    post_id: uuid.UUID
    parent_id: Optional[uuid.UUID] = None


class CommentUpdate(BaseModel):
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("评论内容不能为空")
        return v


class CommentUser(BaseModel):
    id: uuid.UUID
    username: str
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True


class CommentResponse(BaseModel):
    id: uuid.UUID
    content: str
    post_id: uuid.UUID
    user: CommentUser
    parent_id: Optional[uuid.UUID]
    depth: int
    like_count: int
    reply_count: int
    is_approved: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CommentListResponse(BaseModel):
    items: List[CommentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
