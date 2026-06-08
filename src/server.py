"""MCP server: expose repo-rag search as a tool for Claude Code.

On startup, indexes the current working directory (cwd) — Claude Code
launches MCP servers with cwd set to the user's project root. The index
build runs in the background; the MCP loop starts immediately so the
server stays responsive.

When search_code is called before the index is ready, it awaits the
build instead of returning early — the caller gets real results, just
later.

The index is persisted under <repo>/.repo-rag/index.db; subsequent starts
reuse it (no build needed). If a previous build left an empty db (count=0,
e.g. crashed mid-build), it is detected and rebuilt.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from builder import build_index
from chunking.chunker import Chunker
from indexing.embedder import Embedder
from indexing.index import VectorIndex
from indexing.search import Searcher


# Global state, populated asynchronously by _build_index_task()
chunker: Chunker | None = None
embedder: Embedder | None = None
index: VectorIndex | None = None
searcher: Searcher | None = None

# Set when the background build finishes (success or failure)
_index_ready: asyncio.Event = asyncio.Event()

mcp = FastMCP("repo-rag")
log = logging.getLogger("repo-rag")

QUERY_LOG_MAX = 200
SLOW_THRESHOLD = 1.0


def _setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_h = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(file_h)
    log.addHandler(stream_h)
    log.propagate = False


@mcp.tool()
async def search_code(query: str, top_k: int = 10) -> str:
    """
    Search for code chunks similar to a code snippet.

    Pass a code snippet (NOT natural language). Typically Claude generates
    a "what the implementation might look like" snippet and uses it as the
    query. Returns top-k chunks with file paths, line numbers, and the
    surrounding header context.

    If the index is still building, this call awaits build completion
    before searching — the caller gets real results, possibly after a
    delay.

    Args:
        query: A code snippet to search for.
        top_k: Number of results to return (default 10).
    """
    if not _index_ready.is_set():
        log.info("search_code: waiting for background build to finish")
        await _index_ready.wait()

    if searcher is None:
        log.error("search_code: index build failed, no searcher available")
        return "Error: index build failed. Check repo-rag.log for details."

    short = query[:QUERY_LOG_MAX].replace('\n', ' ')
    suffix = "..." if len(query) > QUERY_LOG_MAX else ""
    log.info(f'search_code: query="{short}{suffix}" top_k={top_k}')

    t0 = time.time()
    try:
        results = searcher.search(query, top_k=top_k)
    except Exception:
        log.error(f"search_code: failed\n{traceback.format_exc()}")
        raise
    elapsed = time.time() - t0

    top3_lines = "\n".join(
        f"  [{i}] {c.file}:{c.start_line}-{c.end_line} score={c.score:.3f}"
        for i, c in enumerate(results[:3], 1)
    )
    log.info(f"search_code: returned {len(results)} results in {elapsed:.2f}s\n{top3_lines}")
    if elapsed > SLOW_THRESHOLD:
        log.warning(f"search_code: slow call ({elapsed:.2f}s)")

    lines = [f"Found {len(results)} results:\n"]
    for i, chunk in enumerate(results, 1):
        lines.append(f"{i}. {chunk.file}:{chunk.start_line}-{chunk.end_line} (score: {chunk.score:.4f})")
        lines.append(f"   Language: {chunk.language}")
        if chunk.header:
            lines.append(f"   Header: {chunk.header}")
        lines.append(f"   Code:\n{chunk.code}\n")
    return "\n".join(lines)


async def _build_index_task(repo_path: str, db_path: str):
    """Background task: load models, build/load index, populate searcher.

    On any failure, the event is still set so search_code stops waiting
    and reports the error.
    """
    global chunker, embedder, index, searcher
    try:
        chunker = Chunker(max_size=512)
        embedder = Embedder()

        db_file = Path(db_path)
        if db_file.exists():
            existing = VectorIndex(db_path, dim=embedder.dim)
            if existing.count() > 0:
                log.info(f"background: reusing existing index ({existing.count()} chunks)")
                index = existing
            else:
                log.warning("background: existing index is empty (failed previous build), rebuilding")
                existing.close()
                db_file.unlink()
                index = await build_index(repo_path, db_path, chunker, embedder)
        else:
            log.info("background: building fresh index (this can take a while)")
            index = await build_index(repo_path, db_path, chunker, embedder)

        searcher = Searcher(index, embedder)
        log.info(f"background: ready — {index.count()} chunks indexed")
    except Exception:
        log.error(f"background: build failed\n{traceback.format_exc()}")
    finally:
        _index_ready.set()


async def main():
    repo_path = os.getcwd()
    repo_rag_dir = Path(repo_path) / ".repo-rag"
    db_path = repo_rag_dir / "index.db"
    log_path = repo_rag_dir / "repo-rag.log"

    _setup_logging(log_path)
    log.info(f"startup: project={repo_path}")
    log.info(f"startup: db={db_path}")

    # Kick off index build in background — don't await
    asyncio.create_task(_build_index_task(repo_path, str(db_path)))

    # Enter MCP stdio loop immediately
    log.info("startup: MCP server entering stdio loop")
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
