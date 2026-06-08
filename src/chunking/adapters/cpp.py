from pathlib import Path

from tree_sitter import Language, Node, Parser
import tree_sitter_cpp

from chunking.language import LanguageAdapter


class CppAdapter(LanguageAdapter):
    name = "cpp"
    HEADER_NODE_TYPES = {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "union_specifier",
        "enum_specifier",
        "namespace_definition",
    }
    HEADER_PREFIXES = {
        "class_specifier": "class",
        "struct_specifier": "struct",
        "union_specifier": "union",
        "enum_specifier": "enum",
        "namespace_definition": "namespace",
        # function_definition: no prefix
    }

    def __init__(self):
        self.language = Language(tree_sitter_cpp.language())
        self.parser = Parser(self.language)

    def parse(self, file_path: Path) -> Node:
        content = file_path.read_bytes()
        return self.parser.parse(content).root_node

    def header_for(self, node: Node) -> str:
        h = super().header_for(node)
        if h:
            return h
        # Template wrapper: use the inner declaration's name
        if node.type == "template_declaration":
            inner = node.child_by_field_name("declaration") or self._inner_decl(node)
            return self.header_for(inner) if inner else ""
        return ""

    def _node_name(self, node: Node) -> str:
        if node.type == "function_definition":
            # name is in nested declarator (may include qualifiers like Class::method)
            decl = node.child_by_field_name("declarator")
            while decl is not None:
                inner = decl.child_by_field_name("declarator")
                if inner is None:
                    return decl.text.decode("utf-8", errors="replace") if decl.text else ""
                decl = inner
            return ""
        return super()._node_name(node)

    def _inner_decl(self, node: Node) -> Node | None:
        for child in node.children:
            if child.type in self.HEADER_NODE_TYPES:
                return child
        return None
