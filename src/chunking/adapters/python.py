from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_python
from chunking.language import LanguageAdapter


class PythonAdapter(LanguageAdapter):
    name = "python"
    HEADER_NODE_TYPES = {
        "class_definition",
        "function_definition",
    }
    HEADER_PREFIXES = {
        "class_definition": "class",
        "function_definition": "def",
    }

    def __init__(self):
        self.language = Language(tree_sitter_python.language())
        self.parser = Parser(self.language)

    def parse(self, file_path: Path) -> Node:
        content = file_path.read_bytes()
        tree = self.parser.parse(content)
        return tree.root_node