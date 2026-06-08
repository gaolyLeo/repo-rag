from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from tree_sitter import Node


class LanguageAdapter(ABC):
    name: str = ""
    HEADER_NODE_TYPES: set[str] = set()
    HEADER_PREFIXES: dict[str, str] = {}  # node type → prefix word ("class", "def", "struct", ...)

    @abstractmethod
    def parse(self, file_path: Path) -> Node:
        """Parse the file and return the root node of the syntax tree."""
        ...

    def header_for(self, node: Node) -> str:
        """Return a short header for definition nodes, or "" otherwise.

        Format: "{prefix} {name}" — e.g. "def foo", "class Bar", "struct Point".
        Drops parameters/return type/decorators/modifiers — those are noise
        for embedding-based context.
        """
        if node.type not in self.HEADER_NODE_TYPES:
            return ""
        name = self._node_name(node)
        if not name:
            return ""
        prefix = self.HEADER_PREFIXES.get(node.type, "")
        return f"{prefix} {name}" if prefix else name

    def _node_name(self, node: Node) -> str:
        """Extract the identifier name of a definition node.

        Default: read the `name` field (works for Python class/function,
        C++ class_specifier/struct_specifier/namespace_definition).
        Override for C/C++ function_definition where name is nested.
        """
        n = node.child_by_field_name("name")
        if n and n.text:
            return n.text.decode("utf-8", errors="replace")
        return ""

    def get_text(self, node: Node | None) -> str:
        """Helper to get the text content of a node."""
        return node.text.decode('utf-8') if node and node.text else ""

    def signature_text(self, node: Node) -> str:
        """Return the source text from node start to its `body` field start.

        Kept for callers that still want the full signature (not used by
        header_for anymore).
        """
        body = node.child_by_field_name("body")
        if body is None or node.text is None:
            return ""
        offset = body.start_byte - node.start_byte
        return node.text[:offset].decode("utf-8", errors="replace").rstrip()


_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
}

_LANG_TO_CLASS: dict[str, str] = {
    "python": "PythonAdapter",
    "c": "CAdapter",
    "cpp": "CppAdapter",
}


@lru_cache(maxsize=8)
def _load_adapter(lang: str) -> LanguageAdapter:
    """Load and instantiate an adapter for the given language name.

    Cached so each language adapter is only instantiated once.
    """
    from importlib import import_module
    cls_name = _LANG_TO_CLASS[lang]
    module = import_module(f"chunking.adapters.{lang}")
    return getattr(module, cls_name)()


def get_adapter(file_path: Path) -> LanguageAdapter | None:
    """Return a cached LanguageAdapter for the file's extension, or None."""
    lang = _EXT_TO_LANG.get(file_path.suffix.lower())
    return _load_adapter(lang) if lang else None