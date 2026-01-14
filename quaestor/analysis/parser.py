"""
Python Parser for Agent Code Analysis.

Uses tree-sitter for fast, accurate Python AST parsing.
Extracts functions, classes, imports, and decorators for workflow analysis.

Part of Phase 1: Core Analysis Engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tree_sitter_python as ts_python
from tree_sitter import Language, Parser, Node


# Initialize tree-sitter Python language
PY_LANGUAGE = Language(ts_python.language())


@dataclass
class SourceLocation:
    """Location in source code."""
    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int


@dataclass
class Parameter:
    """Function parameter definition."""
    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_required: bool = True


@dataclass
class Decorator:
    """Decorator applied to a function or class."""
    name: str
    arguments: list[str] = field(default_factory=list)
    location: SourceLocation | None = None


@dataclass
class FunctionDef:
    """Extracted function definition."""
    name: str
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    decorators: list[Decorator] = field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    location: SourceLocation | None = None
    body_text: str | None = None


@dataclass
class ClassDef:
    """Extracted class definition."""
    name: str
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionDef] = field(default_factory=list)
    class_variables: list[str] = field(default_factory=list)
    docstring: str | None = None
    decorators: list[Decorator] = field(default_factory=list)
    location: SourceLocation | None = None
    body_text: str | None = None


@dataclass
class Import:
    """Extracted import statement."""
    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_from_import: bool = False
    location: SourceLocation | None = None
    
    @property
    def is_from(self) -> bool:
        """Alias for is_from_import for compatibility."""
        return self.is_from_import


@dataclass
class ParsedCode:
    """
    Complete parsing result for a Python file.
    
    Contains all extracted structural information needed
    for workflow analysis and linting.
    """
    source_file: str
    source_code: str
    functions: list[FunctionDef] = field(default_factory=list)
    classes: list[ClassDef] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    module_docstring: str | None = None
    errors: list[str] = field(default_factory=list)
    syntax_errors: list[str] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        """Check if parsing encountered any errors."""
        return len(self.errors) > 0 or len(self.syntax_errors) > 0
    
    @property
    def has_syntax_errors(self) -> bool:
        """Check if parsing encountered syntax errors."""
        return len(self.syntax_errors) > 0
    
    def get_function_by_name(self, name: str) -> FunctionDef | None:
        """Get a function by name."""
        for func in self.functions:
            if func.name == name:
                return func
        return None
    
    def get_class_by_name(self, name: str) -> ClassDef | None:
        """Get a class by name."""
        for cls in self.classes:
            if cls.name == name:
                return cls
        return None
    
    def get_decorated_functions(self, decorator_name: str) -> list[FunctionDef]:
        """Get all functions with a specific decorator."""
        result = []
        for func in self.functions:
            for dec in func.decorators:
                if dec.name == decorator_name:
                    result.append(func)
                    break
        return result
    
    def get_decorated_classes(self, decorator_name: str) -> list[ClassDef]:
        """Get all classes with a specific decorator."""
        result = []
        for cls in self.classes:
            for dec in cls.decorators:
                if dec.name == decorator_name:
                    result.append(cls)
                    break
        return result


class PythonParser:
    """
    Tree-sitter based Python parser for agent code analysis.
    
    Extracts:
    - Function definitions with parameters, return types, decorators
    - Class definitions with methods and inheritance
    - Import statements
    - Docstrings
    
    Usage:
        parser = PythonParser()
        result = parser.parse_file("agent.py")
        # or
        result = parser.parse_string(source_code, "agent.py")
    """
    
    def __init__(self):
        """Initialize the parser with tree-sitter Python language."""
        self._parser = Parser(PY_LANGUAGE)
    
    def parse_file(self, file_path: str | Path) -> ParsedCode:
        """
        Parse a Python file and extract structural information.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            ParsedCode with extracted functions, classes, imports
            
        Raises:
            FileNotFoundError: If file does not exist
            PermissionError: If file cannot be read
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        source_code = path.read_text(encoding="utf-8")
        return self.parse_string(source_code, str(path))
    
    def parse_string(self, source_code: str, file_name: str = "<string>") -> ParsedCode:
        """
        Parse Python source code string.
        
        Args:
            source_code: Python source code
            file_name: Name to use for source location reporting
            
        Returns:
            ParsedCode with extracted structures
        """
        result = ParsedCode(
            source_file=file_name,
            source_code=source_code,
        )
        
        # Parse with tree-sitter
        tree = self._parser.parse(bytes(source_code, "utf-8"))
        root = tree.root_node
        
        # Check for syntax errors
        if root.has_error:
            result.errors.append("Syntax error detected in source code")
            self._collect_errors(root, result.syntax_errors)
        
        # Extract module docstring
        result.module_docstring = self._extract_module_docstring(root, source_code)
        
        # Extract all structures
        self._extract_imports(root, source_code, file_name, result)
        self._extract_functions(root, source_code, file_name, result)
        self._extract_classes(root, source_code, file_name, result)
        
        return result
    
    def _collect_errors(self, node: Node, errors: list[str]) -> None:
        """Recursively collect error nodes."""
        if node.type == "ERROR":
            start = node.start_point
            errors.append(f"Syntax error at line {start[0] + 1}, column {start[1]}")
        for child in node.children:
            self._collect_errors(child, errors)
    
    def _extract_module_docstring(self, root: Node, source: str) -> str | None:
        """Extract module-level docstring if present."""
        for child in root.children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type == "string":
                        return self._clean_docstring(self._get_node_text(expr, source))
            elif child.type not in ("comment", "newline"):
                break
        return None
    
    def _extract_imports(
        self, 
        root: Node, 
        source: str, 
        file_name: str, 
        result: ParsedCode
    ) -> None:
        """Extract import statements."""
        for child in root.children:
            if child.type == "import_statement":
                imp = self._parse_import(child, source, file_name)
                if imp:
                    result.imports.append(imp)
            elif child.type == "import_from_statement":
                imp = self._parse_from_import(child, source, file_name)
                if imp:
                    result.imports.append(imp)
    
    def _parse_import(self, node: Node, source: str, file_name: str) -> Import | None:
        """Parse 'import x' statement."""
        names = []
        for child in node.children:
            if child.type == "dotted_name":
                names.append(self._get_node_text(child, source))
            elif child.type == "aliased_import":
                for sub in child.children:
                    if sub.type == "dotted_name":
                        names.append(self._get_node_text(sub, source))
                        break
        
        if names:
            return Import(
                module=names[0],
                names=names,
                is_from_import=False,
                location=self._get_location(node, file_name),
            )
        return None
    
    def _parse_from_import(self, node: Node, source: str, file_name: str) -> Import | None:
        """Parse 'from x import y' statement."""
        module = None
        names = []
        found_module = False
        
        for child in node.children:
            if child.type == "dotted_name":
                if not found_module:
                    # First dotted_name is the module
                    module = self._get_node_text(child, source)
                    found_module = True
                else:
                    # Subsequent dotted_names are imported names
                    names.append(self._get_node_text(child, source))
            elif child.type == "import_prefix":
                module = self._get_node_text(child, source)
                found_module = True
            elif child.type == "identifier":
                names.append(self._get_node_text(child, source))
            elif child.type == "aliased_import":
                for sub in child.children:
                    if sub.type == "identifier" or sub.type == "dotted_name":
                        names.append(self._get_node_text(sub, source))
                        break
            elif child.type == "wildcard_import":
                names.append("*")
        
        if module:
            return Import(
                module=module,
                names=names,
                is_from_import=True,
                location=self._get_location(node, file_name),
            )
        return None
    
    def _extract_functions(
        self,
        root: Node,
        source: str,
        file_name: str,
        result: ParsedCode,
    ) -> None:
        """Extract top-level function definitions."""
        for child in root.children:
            if child.type == "function_definition":
                func = self._parse_function(child, source, file_name, is_method=False)
                if func:
                    result.functions.append(func)
            elif child.type == "decorated_definition":
                func = self._parse_decorated_function(child, source, file_name, is_method=False)
                if func:
                    result.functions.append(func)
    
    def _extract_classes(
        self,
        root: Node,
        source: str,
        file_name: str,
        result: ParsedCode,
    ) -> None:
        """Extract class definitions."""
        for child in root.children:
            if child.type == "class_definition":
                cls = self._parse_class(child, source, file_name, decorators=[])
                if cls:
                    result.classes.append(cls)
            elif child.type == "decorated_definition":
                cls = self._parse_decorated_class(child, source, file_name)
                if cls:
                    result.classes.append(cls)
    
    def _parse_function(
        self,
        node: Node,
        source: str,
        file_name: str,
        is_method: bool = False,
        decorators: list[Decorator] | None = None,
    ) -> FunctionDef | None:
        """Parse a function definition node."""
        name = None
        parameters: list[Parameter] = []
        return_type = None
        docstring = None
        is_async = False
        body_text = None
        
        # Check for async
        for child in node.children:
            if child.type == "async":
                is_async = True
                break
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source)
            elif child.type == "parameters":
                parameters = self._parse_parameters(child, source)
            elif child.type == "type":
                return_type = self._get_node_text(child, source)
            elif child.type == "block":
                docstring = self._extract_docstring(child, source)
                body_text = self._get_node_text(child, source)
        
        if name:
            return FunctionDef(
                name=name,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                decorators=decorators or [],
                is_async=is_async,
                is_method=is_method,
                location=self._get_location(node, file_name),
                body_text=body_text,
            )
        return None
    
    def _parse_decorated_function(
        self,
        node: Node,
        source: str,
        file_name: str,
        is_method: bool = False,
    ) -> FunctionDef | None:
        """Parse a decorated function definition."""
        decorators: list[Decorator] = []
        
        for child in node.children:
            if child.type == "decorator":
                dec = self._parse_decorator(child, source, file_name)
                if dec:
                    decorators.append(dec)
            elif child.type == "function_definition":
                return self._parse_function(
                    child, source, file_name, is_method, decorators
                )
        return None
    
    def _parse_class(
        self,
        node: Node,
        source: str,
        file_name: str,
        decorators: list[Decorator],
    ) -> ClassDef | None:
        """Parse a class definition node."""
        name = None
        bases: list[str] = []
        methods: list[FunctionDef] = []
        class_variables: list[str] = []
        docstring = None
        body_text = None
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source)
            elif child.type == "argument_list":
                # Parse base classes
                for arg in child.children:
                    if arg.type == "identifier":
                        bases.append(self._get_node_text(arg, source))
                    elif arg.type == "attribute":
                        bases.append(self._get_node_text(arg, source))
            elif child.type == "block":
                body_text = self._get_node_text(child, source)
                docstring = self._extract_docstring(child, source)
                # Extract methods and class variables
                for block_child in child.children:
                    if block_child.type == "function_definition":
                        method = self._parse_function(
                            block_child, source, file_name, is_method=True
                        )
                        if method:
                            methods.append(method)
                    elif block_child.type == "decorated_definition":
                        method = self._parse_decorated_function(
                            block_child, source, file_name, is_method=True
                        )
                        if method:
                            methods.append(method)
                    elif block_child.type == "expression_statement":
                        # Could be class variable assignment
                        for expr in block_child.children:
                            if expr.type == "assignment":
                                for sub in expr.children:
                                    if sub.type == "identifier":
                                        class_variables.append(
                                            self._get_node_text(sub, source)
                                        )
                                        break
        
        if name:
            return ClassDef(
                name=name,
                bases=bases,
                methods=methods,
                class_variables=class_variables,
                docstring=docstring,
                decorators=decorators,
                location=self._get_location(node, file_name),
                body_text=body_text,
            )
        return None
    
    def _parse_decorated_class(
        self,
        node: Node,
        source: str,
        file_name: str,
    ) -> ClassDef | None:
        """Parse a decorated class definition."""
        decorators: list[Decorator] = []
        
        for child in node.children:
            if child.type == "decorator":
                dec = self._parse_decorator(child, source, file_name)
                if dec:
                    decorators.append(dec)
            elif child.type == "class_definition":
                return self._parse_class(child, source, file_name, decorators)
        return None
    
    def _parse_decorator(
        self,
        node: Node,
        source: str,
        file_name: str,
    ) -> Decorator | None:
        """Parse a decorator."""
        name = None
        arguments: list[str] = []
        
        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_text(child, source)
            elif child.type == "attribute":
                name = self._get_node_text(child, source)
            elif child.type == "call":
                for sub in child.children:
                    if sub.type in ("identifier", "attribute"):
                        name = self._get_node_text(sub, source)
                    elif sub.type == "argument_list":
                        for arg in sub.children:
                            if arg.type not in ("(", ")", ","):
                                arguments.append(self._get_node_text(arg, source))
        
        if name:
            return Decorator(
                name=name,
                arguments=arguments,
                location=self._get_location(node, file_name),
            )
        return None
    
    def _parse_parameters(self, node: Node, source: str) -> list[Parameter]:
        """Parse function parameters."""
        params: list[Parameter] = []
        
        for child in node.children:
            if child.type == "identifier":
                params.append(Parameter(name=self._get_node_text(child, source)))
            elif child.type == "typed_parameter":
                name = None
                type_ann = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = self._get_node_text(sub, source)
                    elif sub.type == "type":
                        type_ann = self._get_node_text(sub, source)
                if name:
                    params.append(Parameter(name=name, type_annotation=type_ann))
            elif child.type == "default_parameter":
                name = None
                default = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = self._get_node_text(sub, source)
                    elif sub.type not in ("=",):
                        default = self._get_node_text(sub, source)
                if name:
                    params.append(Parameter(
                        name=name,
                        default_value=default,
                        is_required=False,
                    ))
            elif child.type == "typed_default_parameter":
                name = None
                type_ann = None
                default = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = self._get_node_text(sub, source)
                    elif sub.type == "type":
                        type_ann = self._get_node_text(sub, source)
                    elif sub.type not in ("=", ":"):
                        default = self._get_node_text(sub, source)
                if name:
                    params.append(Parameter(
                        name=name,
                        type_annotation=type_ann,
                        default_value=default,
                        is_required=False,
                    ))
        
        return params
    
    def _extract_docstring(self, block_node: Node, source: str) -> str | None:
        """Extract docstring from a block node."""
        for child in block_node.children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type == "string":
                        return self._clean_docstring(self._get_node_text(expr, source))
            elif child.type not in ("newline", "indent", "dedent", "comment"):
                break
        return None
    
    def _clean_docstring(self, docstring: str) -> str:
        """Clean docstring by removing quotes and extra whitespace."""
        # Remove triple quotes
        if docstring.startswith('"""') and docstring.endswith('"""'):
            docstring = docstring[3:-3]
        elif docstring.startswith("'''") and docstring.endswith("'''"):
            docstring = docstring[3:-3]
        elif docstring.startswith('"') and docstring.endswith('"'):
            docstring = docstring[1:-1]
        elif docstring.startswith("'") and docstring.endswith("'"):
            docstring = docstring[1:-1]
        return docstring.strip()
    
    def _get_node_text(self, node: Node, source: str) -> str:
        """Get the text content of a node."""
        return source[node.start_byte:node.end_byte]
    
    def _get_location(self, node: Node, file_name: str) -> SourceLocation:
        """Get source location for a node."""
        start = node.start_point
        end = node.end_point
        return SourceLocation(
            file_path=file_name,
            start_line=start[0] + 1,  # 1-indexed
            start_column=start[1],
            end_line=end[0] + 1,
            end_column=end[1],
        )


# Convenience function
def parse_python_file(file_path: str | Path) -> ParsedCode:
    """
    Parse a Python file and extract structural information.
    
    Convenience function that creates a parser and parses the file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        ParsedCode with extracted structures
    """
    parser = PythonParser()
    return parser.parse_file(file_path)


def parse_python_string(source_code: str, file_name: str = "<string>") -> ParsedCode:
    """
    Parse Python source code string.
    
    Convenience function that creates a parser and parses the string.
    
    Args:
        source_code: Python source code
        file_name: Name for source location reporting
        
    Returns:
        ParsedCode with extracted structures
    """
    parser = PythonParser()
    return parser.parse_string(source_code, file_name)
