"""Vector index backed by sqlite-vec.

Two tables in one .db file:
- vec_chunks: sqlite-vec virtual table, stores `dim`-dim vectors
- chunk_meta: chunk metadata (file/lines/code/header/...) keyed by rowid

The two tables share rowid as the join key; sqlite-vec returns top-k
rowids by vector distance, we look up the metadata to rebuild Chunks.

One VectorIndex instance == one .db file == one indexed repo.
Concurrent write from multiple processes is not supported.
"""

import sqlite3
import struct
from typing import List

import numpy as np
import sqlite_vec

from utils.chunk import Chunk


class VectorIndex:
    def __init__(self, db_path: str, dim: int):
        self.dim = dim
        # check_same_thread=False: build runs in a worker thread (asyncio.to_thread),
        # search runs in the main thread. We never write concurrently, so this is safe.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding FLOAT[{self.dim}]
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_meta (
                rowid INTEGER PRIMARY KEY,
                file TEXT,
                start_line INTEGER,
                end_line INTEGER,
                language TEXT,
                code TEXT,
                header TEXT,
                token_count INTEGER
            )
            """
        )

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Insert chunks + their vectors. embeddings.shape == (len(chunks), dim)."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        vec_rows = [
            (struct.pack(f"{self.dim}f", *emb.tolist()),) for emb in embeddings
        ]
        with self.conn:
            cursor = self.conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM vec_chunks")
            next_rowid = cursor.fetchone()[0] + 1

            self.conn.executemany("INSERT INTO vec_chunks(embedding) VALUES (?)", vec_rows)

            meta_rows = [
                (
                    next_rowid + i,
                    chunk.file, chunk.start_line, chunk.end_line,
                    chunk.language, chunk.code, chunk.header, chunk.token_count,
                )
                for i, chunk in enumerate(chunks)
            ]
            self.conn.executemany(
                """
                INSERT INTO chunk_meta
                    (rowid, file, start_line, end_line, language, code, header, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                meta_rows,
            )

    def search(self, query_emb: np.ndarray, top_k: int = 10) -> List[Chunk]:
        """Return top_k Chunks ranked by vector distance (smaller = closer).

        The returned Chunks have `score` filled with the L2 distance.
        """
        query_bytes = struct.pack(f"{self.dim}f", *query_emb.tolist())
        rows = self.conn.execute(
            """
            SELECT m.code, m.header, m.file, m.start_line, m.end_line,
                   m.language, m.token_count, v.distance
            FROM vec_chunks v
            JOIN chunk_meta m ON v.rowid = m.rowid
            WHERE v.embedding MATCH ? AND k = ?
            """,
            (query_bytes, top_k),
        ).fetchall()

        return [Chunk(*row) for row in rows]

    def clear(self) -> None:
        """Wipe the index. Used when re-indexing a repo from scratch."""
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self.conn.execute("DROP TABLE IF EXISTS chunk_meta")
        self._create_tables()

    def count(self) -> int:
        """Number of indexed chunks."""
        row = self.conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()
        return row[0]

    def close(self) -> None:
        self.conn.close()
