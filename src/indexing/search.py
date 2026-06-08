"""Top-level retrieval entry point.

Searcher = (loaded VectorIndex) + (loaded Embedder). Given a code snippet,
encode it and ask the index for top-k matches. That's it.

Note: Claude Code's MCP only ever gives us *code* as the query (it generates
a "what the code I'm looking for might look like" snippet). We do NOT support
NL→code; that's a different model and a different problem.
"""

from typing import List

from utils.chunk import Chunk
from indexing.embedder import Embedder
from indexing.index import VectorIndex


class Searcher:
    def __init__(self, index: VectorIndex, embedder: Embedder):
        self.index = index
        self.embedder = embedder

    def search(self, query_code: str, top_k: int = 10) -> List[Chunk]:
        """Encode query_code, return top_k matching Chunks (score filled)."""
        query_emb = self.embedder.encode([query_code])[0]
        results = self.index.search(query_emb, top_k=top_k)
        return results
