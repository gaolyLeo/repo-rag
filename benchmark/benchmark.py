"""End-to-end benchmark for repo-rag.

Measures the parts that are reproducible on any machine:
  1. Chunking throughput   (files/s, chunks produced)
  2. Embedding throughput  (chunks/s) — the headline metric in the README
  3. Index build + search  (end-to-end smoke test, search latency)

Usage:
    python benchmark.py [target_repo]

Defaults to benchmarking repo-rag's own src/ directory. Pass a path to
benchmark against a larger codebase.

Run from the repo-rag/ root with the venv active. The script inserts
src/ onto sys.path so the package imports resolve the same way the MCP
server sees them.
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Make `chunking`, `indexing`, `utils` importable (same as server.py's cwd=src).
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from chunking.chunker import Chunker          # noqa: E402
from indexing.embedder import Embedder        # noqa: E402
from indexing.index import VectorIndex        # noqa: E402
from indexing.search import Searcher          # noqa: E402
from utils.repo import iter_files             # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 56}\n{title}\n{'=' * 56}")


def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else str(SRC)
    target = os.path.abspath(target)
    print(f"Benchmark target: {target}")

    # ---- 1. Chunking -----------------------------------------------------
    section("1. Chunking")
    chunker = Chunker(max_size=512)
    files = list(iter_files(target))
    t0 = time.perf_counter()
    chunks = []
    for f in files:
        chunks.extend(chunker.chunk(f))
    t_chunk = time.perf_counter() - t0
    print(f"  files          : {len(files)}")
    print(f"  chunks         : {len(chunks)}")
    print(f"  time           : {t_chunk:.3f}s")
    if t_chunk > 0:
        print(f"  files/s        : {len(files) / t_chunk:.1f}")
    if not chunks:
        print("\nNo chunks produced — nothing to embed. Stopping.")
        return

    # ---- 2. Embedding ----------------------------------------------------
    section("2. Embedding")
    t0 = time.perf_counter()
    embedder = Embedder()
    t_load = time.perf_counter() - t0
    print(f"  model load     : {t_load:.1f}s (device={embedder.device})")

    texts = [c.code for c in chunks]
    # Warm-up: first batch pays one-time MPS/graph init cost; exclude it
    # from the throughput number so chunks/s reflects steady state.
    embedder.encode(texts[: min(32, len(texts))])

    t0 = time.perf_counter()
    embeddings = embedder.encode(texts)
    t_embed = time.perf_counter() - t0
    print(f"  embeddings     : {embeddings.shape}")
    print(f"  time           : {t_embed:.3f}s")
    if t_embed > 0:
        print(f"  chunks/s       : {len(chunks) / t_embed:.1f}")

    # ---- 3. Index + search ----------------------------------------------
    section("3. Index build + search")
    tmp = tempfile.mkdtemp(prefix="repo-rag-bench-")
    db_path = os.path.join(tmp, "bench.db")
    index = VectorIndex(db_path, dim=embedder.dim)

    t0 = time.perf_counter()
    index.add(chunks, embeddings)
    t_insert = time.perf_counter() - t0
    print(f"  inserted       : {index.count()} chunks in {t_insert:.3f}s")

    searcher = Searcher(index, embedder)
    query = texts[0]  # query with an indexed chunk → should match itself
    t0 = time.perf_counter()
    results = searcher.search(query, top_k=5)
    t_search = time.perf_counter() - t0
    print(f"  search latency : {t_search * 1000:.1f}ms (top_k=5)")
    print(f"  results        : {len(results)}")

    # Self-retrieval sanity: the exact query text is in the index, so the
    # nearest neighbour should be essentially distance 0.
    if results:
        top = results[0]
        print(f"  top match      : {Path(top.file).name}:{top.start_line}-{top.end_line} "
              f"score={top.score:.4f}")
        ok = top.score < 0.01
        print(f"  self-retrieval : {'PASS' if ok else 'WARN'} "
              f"(expected score≈0, got {top.score:.4f})")

    index.close()

    section("Summary")
    print(f"  {len(files)} files -> {len(chunks)} chunks")
    print(f"  chunking : {len(files) / t_chunk:.1f} files/s" if t_chunk else "")
    print(f"  embedding: {len(chunks) / t_embed:.1f} chunks/s (device={embedder.device})"
          if t_embed else "")
    print(f"  search   : {t_search * 1000:.1f}ms")


if __name__ == "__main__":
    main()
