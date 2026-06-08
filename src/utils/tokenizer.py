"""Shared jina-code tokenizer.

Loaded lazily on first use. Used by both chunker (for size decisions
during AST splitting) and embedder (for encoding). Sharing one
tokenizer instance means chunker's size estimates exactly match what
the model sees at encode time.
"""

from functools import lru_cache
from typing import cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def token_count(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def safe_index_for_tokens(text: str, max_tokens: int) -> int:
    """Return the char index such that text[:index] is at most max_tokens tokens."""
    offsets = cast(
        list[tuple[int, int]],
        get_tokenizer()(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"],
    )
    if len(offsets) <= max_tokens:
        return len(text)
    return offsets[max_tokens - 1][1]


def truncate(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens (token-aligned)."""
    return text[:safe_index_for_tokens(text, max_tokens)]
