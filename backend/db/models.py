"""
SQLAlchemy ORM 模型

目前使用 PostgreSQL 持久化分析记录和历史排行。
"""

from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    JSON,
    Index,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.utils.config import settings

Base = declarative_base()


class AnalysisRecord(Base):
    """单次分析记录"""

    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, nullable=False, index=True)
    book_id = Column(String(32), nullable=False, index=True)

    # 三路评分
    bert_score = Column(Float, default=0.0)
    dialogue_score = Column(Float, default=0.0)
    verb_score = Column(Float, default=0.0)
    final_score = Column(Float, default=0.0)

    # 权重快照
    weight_bert = Column(Float, default=0.4)
    weight_dialogue = Column(Float, default=0.35)
    weight_verb = Column(Float, default=0.25)

    # LLM 综合分析 (JSON)
    analysis = Column(JSON, nullable=True)

    # 元数据
    text_hash = Column(String(32), nullable=True, comment="输入文本MD5")
    status = Column(String(16), default="completed")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        Index("ix_book_final_score", "book_id", "final_score"),
    )


class BookRanking(Base):
    """排行榜聚合视图"""

    __tablename__ = "book_rankings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    book_id = Column(String(32), unique=True, nullable=False, index=True)

    final_score = Column(Float, default=0.0)
    bert_score = Column(Float, default=0.0)
    dialogue_score = Column(Float, default=0.0)
    verb_score = Column(Float, default=0.0)

    level = Column(String(32), default="微百合/友情向")
    analysis_count = Column(Integer, default=1)

    # LLM 综合分析快照
    plot_summary = Column(Text, nullable=True)
    analysis_json = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ── 数据库引擎工厂 ────────────────────────

def get_engine():
    return create_engine(
        settings.postgres.url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db():
    """创建所有表（幂等操作）"""
    engine = get_engine()
    Base.metadata.create_all(engine)
