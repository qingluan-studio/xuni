"""
文章相关 Pydantic Schema
"""

import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator

from ..models import PostStatus


class PostBase(BaseModel):
    title: str
    summary: Optional[str] = None
    content: str
    cover_image: Optional[str] = None
    status: Optional[PostStatus] = PostStatus.DRAFT
    tags: Optional[List[str]] = None
    slug: Optional[str] = None

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("标题不能为空")
        if len(v) > 300:
            raise ValueError("标题不能超过300字")
        return v

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("内容不能为空")
        return v


class PostCreate(PostBase):
    pass


class PostUpdate(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    cover_image: Optional[str] = None
    status: Optional[PostStatus] = None
    tags: Optional[List[str]] = None


class PostListItem(BaseModel):
    id: uuid.UUID
    title: str
    slug: str
    summary: Optional[str]
    cover_image: Optional[str]
    status: PostStatus
    view_count: int
    like_count: int
    comment_count: int
    tag_list: List[str] = []
    word_count: int = 0
    reading_time_minutes: int = 5
    author_id: uuid.UUID
    created_at: datetime
    published_at: Optional[datetime]

    class Config:
        from_attributes = True


class PostResponse(PostListItem):
    content: str
    content_html: Optional[str] = None
    bookmark_count: int = 0
    is_featured: bool = False
    is_sticky: bool = False

    class Config:
        from_attributes = True


class PostListResponse(BaseModel):
    items: List[PostListItem]
    total: int
    page: int
    page_size: int
    total_pages: int
