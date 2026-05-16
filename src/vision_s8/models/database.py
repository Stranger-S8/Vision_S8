"""
Database models and session management for Vision_S8.

Uses SQLAlchemy with async support via aiosqlite.
"""

from datetime import datetime
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class Audit(Base):
    """Audit history table."""

    __tablename__ = "audits"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    image_path = Column(String(500), nullable=False)
    image_hash = Column(String(64), nullable=True)  # SHA256 hash for deduplication
    audit_type = Column(String(50), nullable=False)  # single, compare, enhance, ab_test, seo, compliance
    platform = Column(String(50), nullable=True)  # amazon, shopify, instagram, etc.
    score = Column(Float, nullable=True)
    result_json = Column(Text, nullable=True)  # Full JSON response stored as text
    processing_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to batch items
    batch_item = relationship("BatchItem", back_populates="audit", uselist=False)


class BatchJob(Base):
    """Batch processing jobs table."""

    __tablename__ = "batch_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(200), nullable=True)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed, cancelled
    total_images = Column(Integer, nullable=False)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    audit_type = Column(String(50), nullable=False)  # Type of audit to perform
    platform = Column(String(50), nullable=True)
    report_path = Column(String(500), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationship to items
    items = relationship("BatchItem", back_populates="batch_job", cascade="all, delete-orphan")

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.processed_images / self.total_images) * 100


class BatchItem(Base):
    """Individual items within a batch job."""

    __tablename__ = "batch_items"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    batch_id = Column(String(36), ForeignKey("batch_jobs.id", ondelete="CASCADE"), nullable=False)
    image_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=True)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    score = Column(Float, nullable=True)
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    audit_id = Column(String(36), ForeignKey("audits.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    # Relationships
    batch_job = relationship("BatchJob", back_populates="items")
    audit = relationship("Audit", back_populates="batch_item")


class EnhancedImage(Base):
    """Stored enhanced images."""

    __tablename__ = "enhanced_images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    original_path = Column(String(500), nullable=False)
    enhanced_path = Column(String(500), nullable=False)
    audit_id = Column(String(36), ForeignKey("audits.id"), nullable=True)
    enhancements_applied = Column(Text, nullable=True)  # JSON list of enhancements
    original_score = Column(Float, nullable=True)
    enhanced_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database engine and session factory
_engine = None
_session_factory = None


async def init_database(database_url: str) -> None:
    """
    Initialize the database connection and create tables.

    Args:
        database_url: SQLAlchemy database URL
    """
    global _engine, _session_factory

    _engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    """Close database connection."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get an async database session.

    Yields:
        AsyncSession for database operations
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get the session factory."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _session_factory
