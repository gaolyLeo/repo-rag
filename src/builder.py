"""Glue: walk repo, chunk every file, embed in batches, write to index.

Stateless. Builds a fresh VectorIndex from scratch. Incremental update
is a Phase 2 concern.
"""

import asyncio
from typing import List

from chunking.chunker import Chunker
from indexing.embedder import Embedder
from indexing.index import VectorIndex
from utils.chunk import Chunk
from utils.repo import iter_files


async def build_index(
    repo_path: str,
    db_path: str,
    chunker: Chunker,
    embedder: Embedder,
) -> VectorIndex:
    """Index every supported file under repo_path into a sqlite-vec db.

    Async wrapper around the sync build pipeline. Runs heavy operations
    (chunking, encoding, inserting) in a thread pool to avoid blocking
    the event loop.

    Returns a VectorIndex with all chunks added. If db_path already exists,
    appends to it (does not wipe). Call index.clear() first if you want a
    fresh build.
    """
    # Run the entire sync pipeline in a thread pool
    return await asyncio.to_thread(_build_index_sync, repo_path, db_path, chunker, embedder)


def _build_index_sync(
    repo_path: str,
    db_path: str,
    chunker: Chunker,
    embedder: Embedder,
) -> VectorIndex:
    """Synchronous implementation of build_index."""
    index = VectorIndex(db_path, dim=embedder.dim)

    # Scan and chunk
    files = list(iter_files(repo_path))
    print(f"Scanning {len(files)} files in {repo_path}...")

    all_chunks: List[Chunk] = []
    for file_path in files:
        all_chunks.extend(chunker.chunk(file_path))

    if not all_chunks:
        print("No indexable chunks found")
        return index

    print(f"Chunked into {len(all_chunks)} chunks")

    # Encode (code only, no header injection)
    texts = [chunk.code for chunk in all_chunks]
    embeddings = embedder.encode(texts)

    # Insert
    print(f"Inserting {len(all_chunks)} chunks into index...")
    index.add(all_chunks, embeddings)
    print(f"Done. Total indexed: {index.count()}")

    return index
