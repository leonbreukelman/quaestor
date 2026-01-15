"""
Tests for the Python Parser module.

Tests tree-sitter based AST extraction functionality.
"""

import pytest

from quaestor.analysis.parser import (
    ParsedCode,
    PythonParser,
    parse_python_string,
)


class TestPythonParser:
    """Tests for PythonParser class."""

    @pytest.fixture
    def parser(self) -> PythonParser:
        """Create a parser instance."""
        return PythonParser()

    def test_parse_simple_function(self, parser: PythonParser) -> None:
        """Test parsing a simple function."""
        source = '''
def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''
        result = parser.parse_string(source)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "hello"
        assert func.docstring == "Greet someone."
        assert not func.is_async
        assert len(func.parameters) == 1
        assert func.parameters[0].name == "name"
        assert func.parameters[0].type_annotation == "str"
        assert func.return_type == "str"

    def test_parse_async_function(self, parser: PythonParser) -> None:
        """Test parsing an async function."""
        source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        result = parser.parse_string(source)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "fetch_data"
        assert func.is_async is True
        assert func.docstring == "Fetch data from URL."

    def test_parse_function_with_decorators(self, parser: PythonParser) -> None:
        """Test parsing functions with decorators."""
        source = '''
@tool
@retry(max_attempts=3)
def process_data(data: list) -> dict:
    """Process the data."""
    return {"result": data}
'''
        result = parser.parse_string(source)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "process_data"
        assert len(func.decorators) == 2
        assert func.decorators[0].name == "tool"
        assert func.decorators[1].name == "retry"

    def test_parse_class_with_methods(self, parser: PythonParser) -> None:
        """Test parsing a class with methods."""
        source = '''
class MyAgent:
    """An agent that does things."""

    def __init__(self, name: str):
        self.name = name

    async def run(self, task: str) -> str:
        """Run a task."""
        return f"{self.name} completed {task}"

    @tool
    def search(self, query: str) -> list:
        """Search for something."""
        return []
'''
        result = parser.parse_string(source)

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "MyAgent"
        assert cls.docstring == "An agent that does things."
        assert len(cls.methods) == 3

        method_names = [m.name for m in cls.methods]
        assert "__init__" in method_names
        assert "run" in method_names
        assert "search" in method_names

    def test_parse_class_with_inheritance(self, parser: PythonParser) -> None:
        """Test parsing class with base classes."""
        source = '''
class SpecialAgent(BaseAgent, Mixin):
    """A special agent."""
    pass
'''
        result = parser.parse_string(source)

        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "SpecialAgent"
        assert "BaseAgent" in cls.bases
        assert "Mixin" in cls.bases

    def test_parse_imports(self, parser: PythonParser) -> None:
        """Test parsing import statements."""
        source = """
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from quaestor.analysis import parser as p
"""
        result = parser.parse_string(source)

        # Should have multiple imports
        assert len(result.imports) >= 4

        # Check standard imports
        import_modules = [i.module for i in result.imports]
        assert "os" in import_modules
        assert "sys" in import_modules

        # Check from imports
        from_imports = [i for i in result.imports if i.is_from]
        pathlib_import = next((i for i in from_imports if i.module == "pathlib"), None)
        assert pathlib_import is not None
        assert "Path" in pathlib_import.names

    def test_parse_function_default_values(self, parser: PythonParser) -> None:
        """Test parsing function with default parameter values."""
        source = '''
def configure(name: str, enabled: bool = True, count: int = 10) -> None:
    """Configure something."""
    pass
'''
        result = parser.parse_string(source)

        func = result.functions[0]
        assert len(func.parameters) == 3

        # Check defaults
        name_param = func.parameters[0]
        assert name_param.name == "name"
        assert name_param.default_value is None

        enabled_param = func.parameters[1]
        assert enabled_param.name == "enabled"
        assert enabled_param.default_value == "True"

        count_param = func.parameters[2]
        assert count_param.name == "count"
        assert count_param.default_value == "10"

    def test_parse_empty_file(self, parser: PythonParser) -> None:
        """Test parsing an empty file."""
        result = parser.parse_string("")

        assert result.source_code == ""
        assert len(result.functions) == 0
        assert len(result.classes) == 0
        assert len(result.imports) == 0

    def test_parse_syntax_error(self, parser: PythonParser) -> None:
        """Test handling of syntax errors."""
        source = """
def broken(
    # Missing closing paren and colon
"""
        result = parser.parse_string(source)

        # Should still return a result but with syntax errors
        assert result.has_syntax_errors is True
        assert len(result.syntax_errors) > 0

    def test_parse_complex_agent(self, parser: PythonParser) -> None:
        """Test parsing a complex agent file."""
        source = '''
"""A research agent implementation."""

from typing import List, Optional
from langchain.agents import AgentExecutor
from langchain.tools import tool

class ResearchAgent:
    """Agent that researches topics."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.tools = [self.search, self.summarize]

    @tool
    def search(self, query: str) -> List[str]:
        """Search for information on a topic."""
        return ["result1", "result2"]

    @tool
    async def summarize(self, text: str) -> str:
        """Summarize the given text."""
        return text[:100]

    async def run(self, task: str) -> str:
        """Execute the research task."""
        try:
            results = await self._execute(task)
            return results
        except Exception as e:
            return f"Error: {e}"

    async def _execute(self, task: str) -> str:
        """Internal execution method."""
        return f"Completed: {task}"
'''
        result = parser.parse_string(source)

        # Check module docstring
        assert result.module_docstring == "A research agent implementation."

        # Check imports
        assert len(result.imports) >= 3

        # Check class
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "ResearchAgent"
        assert len(cls.methods) == 5

        # Find tool-decorated methods
        tool_methods = [m for m in cls.methods if any(d.name == "tool" for d in m.decorators)]
        assert len(tool_methods) == 2


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parse_python_string(self) -> None:
        """Test parse_python_string convenience function."""
        source = "def foo(): pass"
        result = parse_python_string(source)

        assert isinstance(result, ParsedCode)
        assert len(result.functions) == 1
        assert result.functions[0].name == "foo"


class TestSourceLocation:
    """Tests for source location tracking."""

    def test_function_location(self) -> None:
        """Test that function locations are tracked."""
        source = """
# Comment

def first():
    pass

def second():
    pass
"""
        result = parse_python_string(source)

        assert len(result.functions) == 2

        first = result.functions[0]
        second = result.functions[1]

        assert first.location is not None
        assert second.location is not None

        # Second function should be on a later line
        assert second.location.start_line > first.location.start_line

    def test_class_location(self) -> None:
        """Test that class locations are tracked."""
        source = """
class First:
    pass

class Second:
    pass
"""
        result = parse_python_string(source)

        assert len(result.classes) == 2

        first = result.classes[0]
        second = result.classes[1]

        assert first.location is not None
        assert second.location is not None
        assert second.location.start_line > first.location.start_line


class TestParsedCodeHelpers:
    """Tests for ParsedCode helper methods."""

    def test_lookup_helpers(self) -> None:
        source = """
@tool
def foo():
    return 1

@other
def bar():
    return 2

@decorator
class MyClass:
    pass
"""
        result = parse_python_string(source)

        assert result.get_function_by_name("foo") is not None
        assert result.get_function_by_name("missing") is None

        assert result.get_class_by_name("MyClass") is not None
        assert result.get_class_by_name("Missing") is None

        decorated_funcs = result.get_decorated_functions("tool")
        assert [f.name for f in decorated_funcs] == ["foo"]

        decorated_classes = result.get_decorated_classes("decorator")
        assert [c.name for c in decorated_classes] == ["MyClass"]

    def test_has_errors_aliases(self) -> None:
        source = """
def broken(
    # Missing closing paren and colon
"""
        result = parse_python_string(source)

        assert result.has_syntax_errors is True
        assert result.has_errors is True

    def test_parse_file_roundtrip(self, tmp_path) -> None:
        parser = PythonParser()

        file_path = tmp_path / "example.py"
        file_path.write_text(
            '"""Module doc."""\n\n@tool\ndef run():\n    return 1\n', encoding="utf-8"
        )

        result = parser.parse_file(file_path)
        assert result.source_file.endswith("example.py")
        assert result.module_docstring == "Module doc."
        assert result.get_function_by_name("run") is not None

    def test_parse_file_missing_raises(self, tmp_path) -> None:
        parser = PythonParser()

        missing = tmp_path / "missing.py"
        with pytest.raises(FileNotFoundError):
            parser.parse_file(missing)
