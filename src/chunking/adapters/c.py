from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_c

from chunking.language import LanguageAdapter


class CAdapter(LanguageAdapter):
    name = "c"
    HEADER_NODE_TYPES = {
        "function_definition",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
    }
    HEADER_PREFIXES = {
        "struct_specifier": "struct",
        "union_specifier": "union",
        "enum_specifier": "enum",
        # function_definition: no prefix, just the name
    }

    def __init__(self):
        self.language = Language(tree_sitter_c.language())
        self.parser = Parser(self.language)

    def parse(self, file_path: Path) -> Node:
        content = file_path.read_bytes()
        return self.parser.parse(content).root_node

    def _node_name(self, node: Node) -> str:
        if node.type == "function_definition":
            # name is in nested declarator: function_definition → function_declarator → identifier
            decl = node.child_by_field_name("declarator")
            while decl is not None:
                inner = decl.child_by_field_name("declarator")
                if inner is None:
                    return decl.text.decode("utf-8", errors="replace") if decl.text else ""
                decl = inner
            return ""
        return super()._node_name(node)
