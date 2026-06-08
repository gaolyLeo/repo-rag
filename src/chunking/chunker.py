from typing import List
from pathlib import Path

from tree_sitter import Node

from chunking.language import get_adapter
from utils.chunk import Chunk
from utils.tokenizer import token_count, truncate


# Chunks smaller than this are dropped — too short to carry useful semantic
# signal. Typical victims: bare decorator lines, empty function bodies,
# section-divider comments. Setting this too high would discard genuine
# small functions; 20 tokens ≈ a single non-trivial line of code.
MIN_CHUNK_TOKENS = 20


class Chunker:
    def __init__(self, max_size=256):
        self.max_size = max_size

    def chunk(self, file_path: Path) -> List[Chunk]:
        self.file = file_path
        adapter = get_adapter(file_path)
        if not adapter:
            return []
        self.adapter = adapter
        self.chunks: List[Chunk] = []
        self._is_definition: List[bool] = []
        self.headers: List[str] = []
        self._visit(adapter.parse(file_path))
        merged = self._merge_chunks()
        return [c for c in merged if c.token_count >= MIN_CHUNK_TOKENS]

    def _visit(self, node: Node):
        self_header = self.adapter.header_for(node)
        code = self.adapter.get_text(node)

        # Too big: recurse into children, or truncate if leaf
        if token_count(code) > self.max_size:
            if node.named_child_count:
                self.headers.append(self_header)
                for child in node.named_children:
                    self._visit(child)
                self.headers.pop()
                return
            code = truncate(code, self.max_size)

        header = "\n".join(filter(None, self.headers))
        self._emit(node, code, header, token_count(code), bool(self_header))

    def _emit(self, node: Node, code: str, header: str, tokens: int, is_definition: bool):
        self.chunks.append(Chunk(
            code=code,
            header=header,
            file=str(self.file),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            language=self.adapter.name,
            token_count=tokens,
        ))
        self._is_definition.append(is_definition)

    def _merge_chunks(self) -> List[Chunk]:
        merged: List[Chunk] = []
        merged_is_def: List[bool] = []
        for chunk, is_def in zip(self.chunks, self._is_definition):
            if merged:
                last = merged[-1]
                both_definitions = merged_is_def[-1] and is_def
                same_header = last.header == chunk.header
                fits = last.token_count + chunk.token_count <= self.max_size
                if not both_definitions and same_header and fits:
                    last.code = f"{last.code}\n{chunk.code}"
                    last.end_line = chunk.end_line
                    last.token_count = last.token_count + chunk.token_count
                    continue
            merged.append(chunk)
            merged_is_def.append(is_def)
        return merged
