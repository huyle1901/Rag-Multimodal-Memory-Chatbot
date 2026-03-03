from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json


@dataclass
class ChatSession:
    id: str
    title: str
    summary: str
    summary_message_count: int
    created_at: datetime
    updated_at: datetime


@dataclass
class ChatMessage:
    id: int
    session_id: str
    role: str
    content: str
    sources: list[str]
    created_at: datetime


class ChatStore:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id UUID PRIMARY KEY,
                        title TEXT NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        summary_message_count INTEGER NOT NULL DEFAULT 0,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS chat_messages_session_id_id_idx
                    ON chat_messages (session_id, id)
                    """
                )
            conn.commit()

    def list_sessions(self, limit: int = 50) -> list[ChatSession]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, summary, summary_message_count, created_at, updated_at
                    FROM chat_sessions
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [self._session_from_row(row) for row in rows]

    def create_session(self, title: str = "New chat") -> ChatSession:
        session_id = str(uuid4())
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (id, title)
                    VALUES (%s, %s)
                    RETURNING id, title, summary, summary_message_count, created_at, updated_at
                    """,
                    (session_id, title),
                )
                row = cur.fetchone()
            conn.commit()
        return self._session_from_row(row)

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, summary, summary_message_count, created_at, updated_at
                    FROM chat_sessions
                    WHERE id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
        return self._session_from_row(row) if row else None

    def rename_session(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET title = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (title, session_id),
                )
            conn.commit()

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[str] | None = None,
    ) -> ChatMessage:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, sources)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, session_id, role, content, sources, created_at
                    """,
                    (session_id, role, content, Json(sources or [])),
                )
                row = cur.fetchone()
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = NOW()
                    WHERE id = %s
                    """,
                    (session_id,),
                )
            conn.commit()
        return self._message_from_row(row)

    def get_messages(self, session_id: str) -> list[ChatMessage]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, session_id, role, content, sources, created_at
                    FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY id ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        return [self._message_from_row(row) for row in rows]

    def update_summary(self, session_id: str, summary: str, summary_message_count: int) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET summary = %s,
                        summary_message_count = %s,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (summary, summary_message_count, session_id),
                )
            conn.commit()

    @staticmethod
    def _session_from_row(row: dict) -> ChatSession:
        return ChatSession(
            id=str(row["id"]),
            title=row["title"],
            summary=row["summary"],
            summary_message_count=row["summary_message_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _message_from_row(row: dict) -> ChatMessage:
        return ChatMessage(
            id=row["id"],
            session_id=str(row["session_id"]),
            role=row["role"],
            content=row["content"],
            sources=row["sources"] or [],
            created_at=row["created_at"],
        )
