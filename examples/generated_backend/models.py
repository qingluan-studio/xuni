"""
数据模型 — 用户 / 文章 / 评论 / 标签
"""

import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime,
    ForeignKey, Table, Index, Enum, UUID,
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from .database import Base


class UserRole(str, PyEnum):
    """用户角色"""
    ADMIN = "admin"
    EDITOR = "editor"
    USER = "user"


class PostStatus(str, PyEnum):
    """文章状态"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


# 中间表：文章-标签 多对多
post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", UUID(as_uuid=True), ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    Index("idx_post_tags_post", "post_id"),
    Index("idx_post_tags_tag", "tag_id"),
)


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    nickname = Column(String(100))
    avatar_url = Column(String(500))
    bio = Column(Text)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    is_verified = Column(Boolean, default=False)

    # 统计字段（反范式优化）
    post_count = Column(Integer, default=0)
    follower_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)

    # 元数据
    settings = Column(JSONB, default=dict)
    preferences = Column(JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)

    # 关系
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_users_active_role", "is_active", "role"),
        Index("idx_users_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


class Post(Base):
    """文章模型"""
    __tablename__ = "posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(300), nullable=False, index=True)
    slug = Column(String(300), unique=True, nullable=False, index=True)
    summary = Column(String(500))
    content = Column(Text, nullable=False)
    content_html = Column(Text)
    cover_image = Column(String(500))

    status = Column(Enum(PostStatus), default=PostStatus.DRAFT, index=True)
    is_featured = Column(Boolean, default=False, index=True)
    is_sticky = Column(Boolean, default=False)

    # 计数（反范式）
    view_count = Column(Integer, default=0, index=True)
    like_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    bookmark_count = Column(Integer, default=0)

    # 关联
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    author = relationship("User", back_populates="posts")

    # 标签（多对多）
    tags = relationship("Tag", secondary=post_tags, back_populates="posts")
    tag_list = Column(ARRAY(String(50)), default=list)

    # 扩展
    meta = Column(JSONB, default=dict)
    toc = Column(JSONB, default=list)
    word_count = Column(Integer, default=0)
    reading_time_minutes = Column(Integer, default=5)

    # 时间
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, index=True)

    # 关系
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_posts_status_published", "status", "published_at"),
        Index("idx_posts_author_status", "author_id", "status"),
        Index("idx_posts_views", "view_count"),
        Index("idx_posts_featured", "is_featured", "status"),
    )

    def __repr__(self) -> str:
        return f"<Post {self.title[:50]}>"


class Comment(Base):
    """评论模型"""
    __tablename__ = "comments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)

    # 嵌套评论
    parent_id = Column(UUID(as_uuid=True), ForeignKey("comments.id", ondelete="CASCADE"))
    depth = Column(Integer, default=0)
    path = Column(String(500), default="", index=True)

    # 关联
    post_id = Column(UUID(as_uuid=True), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False, index=True)
    post = relationship("Post", back_populates="comments")
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    user = relationship("User", back_populates="comments")

    # 状态
    is_approved = Column(Boolean, default=True, index=True)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)

    __table_args__ = (
        Index("idx_comments_post_created", "post_id", "created_at"),
        Index("idx_comments_user", "user_id"),
    )


class Tag(Base):
    """标签模型"""
    __tablename__ = "tags"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(50), unique=True, nullable=False, index=True)
    slug = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    color = Column(String(7), default="#3b82f6")
    icon = Column(String(50))

    post_count = Column(Integer, default=0, index=True)
    is_hot = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    posts = relationship("Post", secondary=post_tags, back_populates="tags")

    def __repr__(self) -> str:
        return f"<Tag {self.name}>"
