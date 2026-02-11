"""
数据库会话管理
"""

from contextlib import contextmanager
from sqlalchemy.orm import Session

from backend.db.models import get_session_factory

_SessionFactory = None


def _get_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = get_session_factory()
    return _SessionFactory


@contextmanager
def get_db():
    """
    FastAPI 依赖注入用的数据库会话

    Usage:
        with get_db() as db:
            db.query(...)
    """
    session: Session = _get_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_dependency():
    """FastAPI Depends() 用生成器"""
    session: Session = _get_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
