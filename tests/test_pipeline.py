"""
Tests for the Analysis Pipeline module.

Tests the unified API orchestrating parser, analyzer, and linter.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from quaestor.analysis.linter import LinterConfig
from quaestor.analysis.pipeline import (
    AnalysisLevel,
    AnalysisPipeline,
    AnalysisReport,
    OutputFormat,
    PipelineConfig,
    analyze_string,
    lint_only,
)


class TestAnalysisPipeline:
    """Tests for AnalysisPipeline class."""

    @pytest.fixture
    def pipeline(self) -> AnalysisPipeline:
        """Create a pipeline instance with mock analyzer."""
        from quaestor.analysis.workflow_analyzer import AnalyzerConfig

        config = PipelineConfig(
            analyzer_config=AnalyzerConfig(use_mock=True),
        )
        return AnalysisPipeline(config)

    def test_pipeline_initialization(self, pipeline: AnalysisPipeline) -> None:
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.config is not None

    def test_analyze_simple_code(self, pipeline: AnalysisPipeline) -> None:
        """Test analyzing simple Python code."""
        source = '''
def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"
'''
        report = pipeline.analyze_string(source, "test.py")

        assert report is not None
        assert report.file_path == "test.py"
        assert report.parsed is not None
        assert len(report.parsed.functions) == 1
        assert not report.has_errors

    def test_analyze_with_lint_issues(self, pipeline: AnalysisPipeline) -> None:
        """Test that lint issues are captured."""
        source = """
def undocumented():
    pass

def dangerous():
    eval("code")
"""
        report = pipeline.analyze_string(source)

        assert report.lint_result is not None
        assert report.has_lint_issues is True
        assert report.lint_result.error_count >= 1  # eval usage

    def test_analyze_with_workflow(self, pipeline: AnalysisPipeline) -> None:
        """Test that workflow analysis is performed."""
        source = '''
from enum import Enum

class State(Enum):
    IDLE = "idle"
    RUNNING = "running"

@tool
def search(query: str) -> list:
    """Search for information."""
    return []

class Agent:
    """An agent that searches."""

    async def run(self, task: str) -> str:
        """Run a task."""
        return ""
'''
        report = pipeline.analyze_string(source)

        assert report.workflow_analysis is not None
        # Mock analyzer should still detect tools/states
        assert report.workflow_analysis.tools_detected is not None


class TestAnalysisLevels:
    """Tests for different analysis levels."""

    def test_parse_only_level(self) -> None:
        """Test PARSE_ONLY level - no lint, no analyzer."""
        config = PipelineConfig(level=AnalysisLevel.PARSE_ONLY)
        pipeline = AnalysisPipeline(config)

        source = "def foo(): eval('bad')"
        report = pipeline.analyze_string(source)

        assert report.parsed is not None
        assert report.lint_result is None
        assert report.workflow_analysis is None

    def test_lint_level(self) -> None:
        """Test LINT level - parse + lint, no analyzer."""
        config = PipelineConfig(level=AnalysisLevel.LINT)
        pipeline = AnalysisPipeline(config)

        source = "def foo(): eval('bad')"
        report = pipeline.analyze_string(source)

        assert report.parsed is not None
        assert report.lint_result is not None
        assert report.workflow_analysis is None
        assert report.lint_result.error_count >= 1

    def test_analyze_level(self) -> None:
        """Test ANALYZE level - full analysis."""
        from quaestor.analysis.workflow_analyzer import AnalyzerConfig

        config = PipelineConfig(
            level=AnalysisLevel.ANALYZE,
            analyzer_config=AnalyzerConfig(use_mock=True),
        )
        pipeline = AnalysisPipeline(config)

        source = '''
@tool
def search(q: str) -> list:
    """Search."""
    return []
'''
        report = pipeline.analyze_string(source)

        assert report.parsed is not None
        assert report.lint_result is not None
        assert report.workflow_analysis is not None


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.level == AnalysisLevel.ANALYZE
        assert config.output_format == OutputFormat.JSON
        assert config.read_only is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        linter_config = LinterConfig(disabled_rules={"Q001"})
        config = PipelineConfig(
            level=AnalysisLevel.LINT,
            linter_config=linter_config,
            output_format=OutputFormat.SARIF,
        )

        assert config.level == AnalysisLevel.LINT
        assert config.output_format == OutputFormat.SARIF
        assert "Q001" in config.linter_config.disabled_rules


class TestAnalysisReport:
    """Tests for AnalysisReport."""

    @pytest.fixture
    def sample_report(self) -> AnalysisReport:
        """Create a sample report."""
        from quaestor.analysis.workflow_analyzer import AnalyzerConfig

        config = PipelineConfig(
            level=AnalysisLevel.ANALYZE,
            analyzer_config=AnalyzerConfig(use_mock=True),
        )
        pipeline = AnalysisPipeline(config)

        source = '''
@tool
def search(query: str) -> list:
    """Search for info."""
    return []

class Agent:
    """An agent."""

    def run(self, task: str) -> str:
        """Run a task."""
        return ""
'''
        return pipeline.analyze_string(source, "sample.py")

    def test_report_summary(self, sample_report: AnalysisReport) -> None:
        """Test report summary generation."""
        summary = sample_report.summary

        assert "file" in summary
        assert "source_lines" in summary
        assert "level" in summary
        assert "functions_found" in summary
        assert "classes_found" in summary

    def test_report_to_dict(self, sample_report: AnalysisReport) -> None:
        """Test report serialization to dict."""
        data = sample_report.to_dict()

        assert "file_path" in data
        assert "source_lines" in data
        assert "analysis_level" in data

        if sample_report.parsed:
            assert "parsed" in data
            assert "functions" in data["parsed"]
            assert "classes" in data["parsed"]

        if sample_report.lint_result:
            assert "lint" in data
            assert "issues" in data["lint"]

    def test_report_has_errors(self) -> None:
        """Test has_errors property."""
        # Report with errors
        report = AnalysisReport(
            file_path="test.py",
            errors=["Something went wrong"],
        )
        assert report.has_errors is True

        # Report without errors
        clean_report = AnalysisReport(file_path="test.py")
        assert clean_report.has_errors is False


class TestFileAnalysis:
    """Tests for file-based analysis."""

    def test_analyze_file(self) -> None:
        """Test analyzing a file from disk."""
        from quaestor.analysis.workflow_analyzer import AnalyzerConfig

        config = PipelineConfig(
            level=AnalysisLevel.LINT,
            analyzer_config=AnalyzerConfig(use_mock=True),
        )
        pipeline = AnalysisPipeline(config)

        # Create a temp file
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
def hello():
    """Greet."""
    return "hello"
''')
            temp_path = f.name

        try:
            report = pipeline.analyze_file(temp_path)

            assert report is not None
            assert report.file_path == temp_path
            assert report.parsed is not None
            assert len(report.parsed.functions) == 1
        finally:
            Path(temp_path).unlink()

    def test_analyze_nonexistent_file(self) -> None:
        """Test analyzing a file that doesn't exist."""
        pipeline = AnalysisPipeline()

        report = pipeline.analyze_file("/nonexistent/path/file.py")

        assert report.has_errors is True
        assert "not found" in report.errors[0].lower()

    def test_analyze_non_python_file(self) -> None:
        """Test analyzing a non-Python file."""
        pipeline = AnalysisPipeline()

        with NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is not Python code.")
            temp_path = f.name

        try:
            report = pipeline.analyze_file(temp_path)

            assert report.has_errors is True
            assert "not a python file" in report.errors[0].lower()
        finally:
            Path(temp_path).unlink()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_analyze_string_function(self) -> None:
        """Test analyze_string convenience function."""
        source = "def foo(): pass"

        # Use LINT level to avoid LLM call
        report = analyze_string(source, level=AnalysisLevel.LINT)

        assert isinstance(report, AnalysisReport)
        assert report.parsed is not None

    def test_lint_only_function(self) -> None:
        """Test lint_only convenience function."""
        with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): eval('bad')")
            temp_path = f.name

        try:
            from quaestor.analysis.linter import LintResult

            result = lint_only(temp_path)

            assert isinstance(result, LintResult)
            assert result.error_count >= 1
        finally:
            Path(temp_path).unlink()


class TestErrorHandling:
    """Tests for error handling in the pipeline."""

    def test_syntax_error_handling(self) -> None:
        """Test handling of syntax errors in code."""
        pipeline = AnalysisPipeline(PipelineConfig(level=AnalysisLevel.LINT))

        source = """
def broken(
    # Missing closing paren
"""
        report = pipeline.analyze_string(source)

        # Should still return a report
        assert report is not None
        assert report.parsed is not None
        assert report.parsed.has_syntax_errors

    def test_pipeline_resilience(self) -> None:
        """Test that pipeline continues even if one stage has issues."""
        pipeline = AnalysisPipeline(PipelineConfig(level=AnalysisLevel.LINT))

        # Empty but valid Python
        report = pipeline.analyze_string("")

        assert report is not None
        assert report.source_lines == 1
        assert not report.has_errors
