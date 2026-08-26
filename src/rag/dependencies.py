"""RAG module dependencies.

This module intentionally uses AI_DB_* instead of the main Travel API DB_*
settings. RAG data lives in the AI/RAG database.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

load_dotenv()

_ai_engine = None
_AI_SESSION = None


def build_ai_db_url() -> str:
    host = os.getenv("AI_DB_HOST", "localhost")
    port = os.getenv("AI_DB_PORT", "5432")
    name = os.getenv("AI_DB_NAME")
    user = os.getenv("AI_DB_USER")
    password = os.getenv("AI_DB_PASSWORD")
    if not all([name, user, password]):
        raise RuntimeError("AI_DB_NAME, AI_DB_USER and AI_DB_PASSWORD are required")
    return f"postgresql://{user}:{quote_plus(password)}@{host}:{port}/{name}"


def get_ai_engine():
    global _ai_engine, _AI_SESSION
    if _ai_engine is None:
        _ai_engine = create_engine(
            build_ai_db_url(),
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        _AI_SESSION = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=_ai_engine)
        )
    return _ai_engine


def get_ai_session() -> Generator[Session, None, None]:
    global _AI_SESSION
    get_ai_engine()
    db = _AI_SESSION()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def ai_session_scope() -> Generator[Session, None, None]:
    global _AI_SESSION
    get_ai_engine()
    db = _AI_SESSION()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
