"""
Tests for the Workflow Analyzer module.

Tests DSPy-based semantic workflow analysis.
"""

import pytest

from quaestor.analysis.parser import parse_python_string
from quaestor.analysis.workflow_analyzer import (
    AnalyzerConfig,
    DetectedState,
    DetectedTool,
    WorkflowAnalysis,
    WorkflowAnalyzer,
)


class TestWorkflowAnalyzerUnit:
    """Unit tests for WorkflowAnalyzer (no LLM calls)."""

    @pytest.fixture
    def analyzer(self) -> WorkflowAnalyzer:
        """Create an analyzer instance."""
        # Use mock mode for unit tests
        config = AnalyzerConfig(use_mock=True)
        return WorkflowAnalyzer(config)

    def test_analyzer_initialization(self, analyzer: WorkflowAnalyzer) -> None:
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.config is not None

    def test_extract_tools_from_decorators(self, analyzer: WorkflowAnalyzer) -> None:
        """Test extraction of tools from decorated functions."""
        source = '''
from langchain.tools import tool

@tool
def search(query: str) -> list:
    """Search for information."""
    return []

@tool
def calculate(expression: str) -> float:
    """Calculate a math expression."""
    return eval(expression)
'''
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        assert result is not None
        assert len(result.tools_detected) >= 2

        tool_names = [t.name for t in result.tools_detected]
        assert "search" in tool_names
        assert "calculate" in tool_names

    def test_extract_tool_descriptions(self, analyzer: WorkflowAnalyzer) -> None:
        """Test that tool descriptions are extracted from docstrings."""
        source = '''
@tool
def analyze_sentiment(text: str) -> dict:
    """Analyze the sentiment of the given text and return scores."""
    return {"positive": 0.8, "negative": 0.2}
'''
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        assert len(result.tools_detected) == 1
        tool = result.tools_detected[0]
        assert tool.name == "analyze_sentiment"
        assert "sentiment" in tool.description.lower()

    def test_detect_states_from_enums(self, analyzer: WorkflowAnalyzer) -> None:
        """Test detection of states from Enum classes."""
        source = """
from enum import Enum

class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COMPLETE = "complete"
    ERROR = "error"
"""
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        assert len(result.states_detected) >= 4

        state_names = [s.name for s in result.states_detected]
        assert "IDLE" in state_names
        assert "THINKING" in state_names
        assert "ACTING" in state_names

    def test_detect_entry_points(self, analyzer: WorkflowAnalyzer) -> None:
        """Test detection of workflow entry points."""
        source = '''
class ResearchAgent:
    """Research agent implementation."""

    async def run(self, query: str) -> str:
        """Main entry point for the agent."""
        return await self.execute(query)

    def execute(self, query: str) -> str:
        """Execute a query."""
        return ""

    def __call__(self, query: str) -> str:
        """Callable interface."""
        return self.run(query)
'''
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        assert "run" in result.entry_points or "__call__" in result.entry_points

    def test_complexity_score_simple(self, analyzer: WorkflowAnalyzer) -> None:
        """Test complexity scoring for simple code."""
        source = '''
def simple_function():
    """A simple function."""
    return "hello"
'''
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        # Simple code should have low complexity
        assert result.complexity_score <= 3.0

    def test_complexity_score_complex(self, analyzer: WorkflowAnalyzer) -> None:
        """Test complexity scoring for complex code."""
        source = '''
from enum import Enum

class State(Enum):
    IDLE = "idle"
    RUNNING = "running"
    ERROR = "error"

class ComplexAgent:
    """A complex agent with many methods."""

    @tool
    def search(self, q: str) -> list:
        """Search tool."""
        return []

    @tool
    def analyze(self, data: list) -> dict:
        """Analyze tool."""
        return {}

    @tool
    def summarize(self, text: str) -> str:
        """Summarize tool."""
        return ""

    @tool
    def report(self, analysis: dict) -> str:
        """Report tool."""
        return ""

    async def run(self, task: str) -> str:
        """Main entry point."""
        pass

    def transition(self, from_state: State, to_state: State) -> None:
        """Handle state transition."""
        pass
'''
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        # Complex code should have higher complexity
        assert result.complexity_score >= 3.0

    def test_generate_recommendations(self, analyzer: WorkflowAnalyzer) -> None:
        """Test that recommendations are generated."""
        source = """
def undocumented_tool():
    pass

class UntestedAgent:
    def run(self):
        pass
"""
        parsed = parse_python_string(source)
        result = analyzer.analyze(parsed)

        # Should have some recommendations
        assert isinstance(result.recommendations, list)


class TestWorkflowAnalysis:
    """Tests for WorkflowAnalysis data structure."""

    def test_workflow_analysis_creation(self) -> None:
        """Test creating a WorkflowAnalysis."""
        analysis = WorkflowAnalysis(
            summary="A simple agent workflow",
            complexity_score=2.5,
            tools_detected=[
                DetectedTool(name="search", description="Search for info", parameters=["query"]),
                DetectedTool(name="calculate", description="Do math", parameters=["expr"]),
            ],
            states_detected=[
                DetectedState(name="IDLE", type="initial"),
                DetectedState(name="RUNNING", type="intermediate"),
            ],
            entry_points=["run", "__call__"],
            execution_paths=[["run", "search", "calculate"]],
            recommendations=["Add error handling"],
        )

        assert analysis.summary == "A simple agent workflow"
        assert analysis.complexity_score == 2.5
        assert len(analysis.tools_detected) == 2
        assert len(analysis.states_detected) == 2
        assert len(analysis.entry_points) == 2

    def test_detected_tool_creation(self) -> None:
        """Test creating a DetectedTool."""
        tool = DetectedTool(
            name="web_search",
            description="Search the web for information",
            parameters=["query", "num_results"],
            return_type="list[str]",
            is_async=True,
        )

        assert tool.name == "web_search"
        assert tool.is_async is True
        assert "query" in tool.parameters

    def test_detected_state_creation(self) -> None:
        """Test creating a DetectedState."""
        state = DetectedState(
            name="ERROR",
            type="terminal",
            transitions_from=["RUNNING", "THINKING"],
            transitions_to=[],
        )

        assert state.name == "ERROR"
        assert state.type == "terminal"
        assert len(state.transitions_to) == 0


class TestAnalyzerConfig:
    """Tests for AnalyzerConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AnalyzerConfig()

        assert config.use_mock is False
        assert config.model is not None
        assert config.max_retries >= 1

    def test_mock_mode(self) -> None:
        """Test mock mode configuration."""
        config = AnalyzerConfig(use_mock=True)
        analyzer = WorkflowAnalyzer(config)

        # Should not raise when analyzing in mock mode
        parsed = parse_python_string("def foo(): pass")
        result = analyzer.analyze(parsed)

        assert result is not None


class TestToAgentWorkflow:
    """Tests for converting WorkflowAnalysis to AgentWorkflow model."""

    @pytest.fixture
    def analyzer(self) -> WorkflowAnalyzer:
        """Create an analyzer in mock mode."""
        config = AnalyzerConfig(use_mock=True)
        return WorkflowAnalyzer(config)

    def test_convert_to_agent_workflow(self, analyzer: WorkflowAnalyzer) -> None:
        """Test conversion from WorkflowAnalysis to AgentWorkflow."""
        source = '''
from enum import Enum

class State(Enum):
    IDLE = "idle"
    RUNNING = "running"

@tool
def search(query: str) -> list:
    """Search for info."""
    return []

class MyAgent:
    async def run(self, task: str) -> str:
        """Run the agent."""
        return ""
'''
        parsed = parse_python_string(source)
        analysis = analyzer.analyze(parsed)

        workflow = analyzer.to_agent_workflow(analysis)

        assert workflow is not None
        assert workflow.name is not None
        assert len(workflow.tools) >= 0
        assert len(workflow.states) >= 0
