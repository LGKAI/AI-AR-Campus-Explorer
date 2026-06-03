"""
Memory — SQLite + FAISS dual storage.

Optimization: lazy FAISS rebuild.
Instead of rebuilding the entire FAISS index on every add() call (O(n) cost),
we batch updates and only rebuild when the pending count exceeds the threshold
or when a search is requested and the index is stale.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from config import settings
from embedding import Embedder
from vector_store import FaissStore

class Memory:
    def __init__(self) -> None:
        self.db_path:      Path      = settings.memory_db_path
        self.embedder                 = Embedder()
        self.vector      = FaissStore(settings.memory_index_path, settings.memory_meta_path)
        self.reward_vector = FaissStore(settings.reward_index_path, settings.reward_meta_path)

        # Lazy rebuild counters
        self._pending_interactions: int = 0
        self._pending_feedback:     int = 0

        self._init_db()

    # Schema

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       TEXT    NOT NULL,
                role     TEXT    NOT NULL,
                content  TEXT    NOT NULL,
                turn_id  TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ts               TEXT    NOT NULL,
                turn_id          TEXT    NOT NULL,
                user_query       TEXT    NOT NULL,
                assistant_answer TEXT    NOT NULL,
                reward           INTEGER NOT NULL,
                note             TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    # Write

    def add(self, role: str, content: str, turn_id: str | None = None) -> None:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO interactions(ts, role, content, turn_id) VALUES(?,?,?,?)",
            (datetime.utcnow().isoformat(), role, content, turn_id),
        )
        conn.commit()
        conn.close()

        self._pending_interactions += 1
        if self._pending_interactions >= settings.memory_rebuild_threshold:
            self._rebuild_vector_index()
            self._pending_interactions = 0

    def add_feedback(
        self,
        turn_id: str,
        user_query: str,
        assistant_answer: str,
        reward: int,
        note: str = "",
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            """
            INSERT INTO feedback(ts, turn_id, user_query, assistant_answer, reward, note)
            VALUES(?,?,?,?,?,?)
            """,
            (datetime.utcnow().isoformat(), turn_id, user_query, assistant_answer,
             int(reward), note),
        )
        conn.commit()
        conn.close()

        # Always rebuild reward index immediately — feedback is infrequent
        self._rebuild_reward_index()

    # Read

    def clear(self) -> None:
        """Clear all conversation history and the in-memory FAISS index."""
        import shutil
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM interactions")
        conn.commit()
        conn.close()
        # Reset in-memory FAISS index
        self.index = None
        self._pending_interactions = 0
        # Remove persisted index files
        for p in (self.index_path, self.meta_path):
            if p.exists():
                p.unlink()

    def recent(self, n: int = 6) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            "SELECT ts, role, content FROM interactions ORDER BY id DESC LIMIT ?", (n,)
        )
        rows = cur.fetchall()
        conn.close()
        return [{"ts": r[0], "role": r[1], "content": r[2]} for r in reversed(rows)]

    def recall(self, query: str, k: int = 4) -> list[dict]:
        # Force rebuild if there are pending unsaved interactions
        if self._pending_interactions > 0:
            self._rebuild_vector_index()
            self._pending_interactions = 0

        if not self.vector.exists():
            return []
        self.vector.load()
        qv = self.embedder.encode([query])
        # Guard: stale FAISS index built with a different embedding model
        if self.vector.index is not None and self.vector.index.d != qv.shape[-1]:
            return []
        return self.vector.search(qv, k)

    def reinforced_recall(self, query: str, k: int = 3) -> list[dict]:
        if not self.reward_vector.exists():
            return []
        self.reward_vector.load()
        qv = self.embedder.encode([query])
        # Guard: stale FAISS index built with a different embedding model
        if self.reward_vector.index is not None and self.reward_vector.index.d != qv.shape[-1]:
            return []
        return self.reward_vector.search(qv, k)

    # Index rebuild helpers

    def _rebuild_vector_index(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            "SELECT id, role, content, ts, turn_id FROM interactions ORDER BY id ASC"
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return
        texts = [f"{r[1]}: {r[2]}" for r in rows]
        vecs  = self.embedder.encode(texts)
        meta  = [
            {"id": r[0], "role": r[1], "content": r[2], "ts": r[3], "turn_id": r[4]}
            for r in rows
        ]
        self.vector.build(vecs, meta)
        self.vector.save()

    def _rebuild_reward_index(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur  = conn.cursor()
        cur.execute(
            """
            SELECT id, turn_id, user_query, assistant_answer, reward, note, ts
            FROM feedback
            WHERE reward >= 4
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return
        texts = [f"Q: {r[2]}\nA: {r[3]}" for r in rows]
        vecs  = self.embedder.encode(texts)
        meta  = [
            {
                "id":               r[0],
                "turn_id":          r[1],
                "user_query":       r[2],
                "assistant_answer": r[3],
                "reward":           r[4],
                "note":             r[5],
                "ts":               r[6],
            }
            for r in rows
        ]
        self.reward_vector.build(vecs, meta)
        self.reward_vector.save()
