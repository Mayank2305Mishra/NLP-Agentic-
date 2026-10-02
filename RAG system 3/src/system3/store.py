from __future__ import annotations

import logging
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config
from .schema import Conversation, Turn

log = logging.getLogger("system3.store")

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS conv_fts USING fts5(
    transcript_id UNINDEXED,
    conversation_text,
    content='conversations',
    content_rowid='conv_pk',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    speaker,
    text_normalized,
    content='turns',
    content_rowid='turn_pk',
    tokenize='unicode61 remove_diacritics 2'
);
"""

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conv_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id TEXT NOT NULL UNIQUE,
    time_of_interaction TEXT,
    domain TEXT,
    intent TEXT,
    reason_for_call TEXT,
    conversation_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_intent ON conversations(intent);
CREATE INDEX IF NOT EXISTS idx_conversations_domain ON conversations(domain);

CREATE TABLE IF NOT EXISTS turns (
    turn_pk INTEGER PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    speaker TEXT,
    speaker_original TEXT,
    text_original TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
    is_post_call_artifact INTEGER NOT NULL DEFAULT 0,
    speaker_inferred INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(transcript_id) REFERENCES conversations(transcript_id)
);
CREATE INDEX IF NOT EXISTS idx_turns_transcript ON turns(transcript_id, turn_id);

""" + _FTS_DDL

_SPECIAL = re.compile(r'["]')


def quote_term(term: str) -> str:
    cleaned = _SPECIAL.sub("", term).strip()
    if not cleaned:
        return ""
    return '"' + cleaned.replace('"', "") + '"'


def build_match(terms: Sequence[str], operator: str = "AND", limit: int = 12) -> str:
    quoted = [quote_term(t) for t in terms if t]
    quoted = [q for q in quoted if q][:limit]
    if not quoted:
        return ""
    return f" {operator} ".join(quoted)


class LexicalStore:
    """SQLite FTS5/BM25 store.

    Connections are thread-local so a single shared store (Streamlit cached
    pipeline, parallel benchmark workers) can serve concurrent requests:
    WAL gives us concurrent readers and each thread gets its own connection.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Read-path tuning: keep the index hot in RAM, mmap instead of read(2),
        # and spill FTS temp structures to memory rather than disk.
        conn.execute("PRAGMA cache_size=-8000")  # 8 MiB page cache
        conn.execute("PRAGMA mmap_size=268435456")  # 256 MiB
        conn.execute("PRAGMA temp_store=MEMORY")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def close(self) -> None:
        with self._conns_lock:
            conns, self._conns = self._conns, []
        # Drop the thread-local registry so any surviving thread opens fresh.
        self._local = threading.local()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def create_schema(self) -> None:
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def build_fts(self) -> None:
        self.conn.executescript("DROP TABLE IF EXISTS turns_fts; DROP TABLE IF EXISTS conv_fts;")
        self.conn.executescript(_FTS_DDL)
        self.conn.execute(
            "INSERT INTO conv_fts(rowid, transcript_id, conversation_text) "
            "SELECT conv_pk, transcript_id, conversation_text FROM conversations"
        )
        self.conn.execute(
            "INSERT INTO turns_fts(rowid, speaker, text_normalized) "
            "SELECT turn_pk, speaker, text_normalized FROM turns"
        )
        self.conn.commit()
        self.conn.execute("INSERT INTO conv_fts(conv_fts) VALUES('optimize')")
        self.conn.execute("INSERT INTO turns_fts(turns_fts) VALUES('optimize')")
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            DROP TABLE IF EXISTS turns_fts;
            DROP TABLE IF EXISTS conv_fts;
            DROP TABLE IF EXISTS turns;
            DROP TABLE IF EXISTS conversations;
            """
        )
        self.conn.commit()
        self.create_schema()

    def insert_conversations(self, conversations: Iterable[Conversation], mode: str) -> int:
        rows = []
        for conv in conversations:
            text = self._conversation_text(conv, mode)
            rows.append(
                (
                    conv.transcript_id,
                    conv.time_of_interaction,
                    conv.domain,
                    conv.intent,
                    conv.reason_for_call,
                    text,
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO conversations
            (transcript_id, time_of_interaction, domain, intent, reason_for_call, conversation_text)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    @staticmethod
    def _conversation_text(conv: Conversation, mode: str) -> str:
        parts: list[str] = []
        if mode in ("metadata", "full"):
            parts.extend(
                str(x) for x in (conv.domain, conv.intent, conv.reason_for_call) if x
            )
        if mode == "full":
            parts.extend(t.text_normalized for t in conv.turns if t.text_normalized)
        return " \n ".join(parts)

    def insert_turns(self, conversations: Iterable[Conversation]) -> int:
        next_pk = self.conn.execute("SELECT COALESCE(MAX(turn_pk), 0) FROM turns").fetchone()[0]
        count = 0
        batch: list[tuple[Any, ...]] = []
        for conv in conversations:
            for turn in conv.turns:
                next_pk += 1
                batch.append(
                    (
                        next_pk,
                        turn.transcript_id,
                        turn.turn_id,
                        turn.speaker,
                        turn.speaker_original,
                        turn.text_original,
                        turn.text_normalized,
                        int(turn.is_post_call_artifact),
                        int(turn.speaker_inferred),
                    )
                )
                if len(batch) >= 5000:
                    self._flush_turns(batch)
                    count += len(batch)
                    batch = []
        if batch:
            self._flush_turns(batch)
            count += len(batch)
        return count

    def _flush_turns(self, batch: list[tuple[Any, ...]]) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO turns
            (turn_pk, transcript_id, turn_id, speaker, speaker_original,
             text_original, text_normalized, is_post_call_artifact, speaker_inferred)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        self.conn.commit()

    def optimize(self) -> None:
        self.conn.execute("ANALYZE")
        self.conn.commit()

    def counts(self) -> dict[str, int]:
        return {
            "conversations": self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
            "turns": self.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0],
            "conv_fts": self.conn.execute("SELECT COUNT(*) FROM (SELECT rowid FROM conv_fts)").fetchone()[0],
            "turns_fts": self.conn.execute("SELECT COUNT(*) FROM (SELECT rowid FROM turns_fts)").fetchone()[0],
        }

    def intents(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT intent FROM conversations WHERE intent IS NOT NULL ORDER BY intent"
        ).fetchall()
        return [r[0] for r in rows]

    def search_conversations(
        self, match: str, limit: int, intent: str | None = None
    ) -> list[sqlite3.Row]:
        if not match:
            return []
        sql = (
            "SELECT c.transcript_id, bm25(conv_fts, 1.0, 10.0) AS bm, c.intent, c.domain, "
            "c.reason_for_call, c.time_of_interaction "
            "FROM conv_fts JOIN conversations c ON c.conv_pk = conv_fts.rowid "
            "WHERE conv_fts MATCH ?"
        )
        params: list[Any] = [match]
        if intent:
            sql += " AND c.intent = ?"
            params.append(intent)
        sql += " ORDER BY bm LIMIT ?"
        params.append(limit)
        return list(self.conn.execute(sql, params))

    def search_turns(
        self,
        match: str,
        transcript_ids: Sequence[str],
        limit: int,
        include_post_call: bool = False,
    ) -> list[sqlite3.Row]:
        if not match or not transcript_ids:
            return []
        placeholders = ",".join("?" for _ in transcript_ids)
        sql = (
            "SELECT t.transcript_id, t.turn_id, bm25(turns_fts, 1.0, 10.0) AS bm "
            "FROM turns_fts JOIN turns t ON t.turn_pk = turns_fts.rowid "
            f"WHERE turns_fts MATCH ? AND t.transcript_id IN ({placeholders})"
        )
        params: list[Any] = [match, *transcript_ids]
        if not include_post_call:
            sql += " AND t.is_post_call_artifact = 0"
        sql += " ORDER BY bm LIMIT ?"
        params.append(limit)
        return list(self.conn.execute(sql, params))

    def search_turns_global(self, match: str, limit: int) -> list[sqlite3.Row]:
        if not match:
            return []
        sql = (
            "SELECT t.transcript_id, t.turn_id, bm25(turns_fts, 1.0, 10.0) AS bm "
            "FROM turns_fts JOIN turns t ON t.turn_pk = turns_fts.rowid "
            "WHERE turns_fts MATCH ?"
        )
        return list(self.conn.execute(sql, (match, limit)))

    def fetch_meta(self, transcript_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not transcript_ids:
            return {}
        placeholders = ",".join("?" for _ in transcript_ids)
        rows = self.conn.execute(
            "SELECT transcript_id, time_of_interaction, domain, intent, reason_for_call "
            f"FROM conversations WHERE transcript_id IN ({placeholders})",
            list(transcript_ids),
        ).fetchall()
        return {r["transcript_id"]: dict(r) for r in rows}

    def fetch_turns(
        self, transcript_ids: Sequence[str], include_post_call: bool = False
    ) -> dict[str, list[Turn]]:
        if not transcript_ids:
            return {}
        placeholders = ",".join("?" for _ in transcript_ids)
        sql = (
            "SELECT transcript_id, turn_id, speaker, speaker_original, text_original, "
            "text_normalized, is_post_call_artifact, speaker_inferred "
            f"FROM turns WHERE transcript_id IN ({placeholders})"
        )
        params: list[Any] = list(transcript_ids)
        if not include_post_call:
            sql += " AND is_post_call_artifact = 0"
        sql += " ORDER BY transcript_id, turn_id"
        rows = self.conn.execute(sql, params).fetchall()
        out: dict[str, list[Turn]] = {}
        for r in rows:
            out.setdefault(r["transcript_id"], []).append(
                Turn(
                    transcript_id=r["transcript_id"],
                    turn_id=r["turn_id"],
                    speaker_original=r["speaker_original"],
                    speaker=r["speaker"],
                    text_original=r["text_original"],
                    text_normalized=r["text_normalized"],
                    is_post_call_artifact=bool(r["is_post_call_artifact"]),
                    speaker_inferred=bool(r["speaker_inferred"]),
                )
            )
        return out

    def conversation_text(self, transcript_id: str) -> str:
        row = self.conn.execute(
            "SELECT conversation_text FROM conversations WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()
        return row[0] if row else ""

    def fetch_conversation_texts(
        self, transcript_ids: Sequence[str]
    ) -> dict[str, str]:
        """Batch alternative to ``conversation_text`` — one query for the whole
        candidate pool instead of one query per candidate (N+1 hot path)."""
        if not transcript_ids:
            return {}
        unique = list(dict.fromkeys(transcript_ids))
        placeholders = ",".join("?" for _ in unique)
        rows = self.conn.execute(
            "SELECT transcript_id, conversation_text FROM conversations "
            f"WHERE transcript_id IN ({placeholders})",
            unique,
        ).fetchall()
        return {r["transcript_id"]: r["conversation_text"] or "" for r in rows}
