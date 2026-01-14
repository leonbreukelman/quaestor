"""
Analysis Pipeline - Unified API for Agent Code Analysis.

Orchestrates the full analysis flow:
1. Python Parser (tree-sitter AST extraction)
2. WorkflowAnalyzer (DSPy semantic analysis)
3. Static Linter (no-LLM fast checks)

Part of Phase 1: Core Analysis Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from quaestor.analysis.linter import LinterConfig, LintResult, StaticLinter
from quaestor.analysis.models import AgentWorkflow
from quaestor.analysis.parser import ParsedCode, PythonParser
from quaestor.analysis.workflow_analyzer import (
    AnalyzerConfig,
    WorkflowAnalysis,
    WorkflowAnalyzer,
)


class AnalysisLevel(str, Enum):
    """Depth of analysis to perform."""

    PARSE_ONLY = "parse_only"
    """Just parse to AST - fastest, no LLM calls."""

    LINT = "lint"
    """Parse + static linting - fast, no LLM calls."""

    ANALYZE = "analyze"
    """Parse + lint + DSPy workflow analysis - full depth."""


class OutputFormat(str, Enum):
    """Output format for analysis reports."""

    JSON = "json"
    SARIF = "sarif"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    # Analysis depth
    level: AnalysisLevel = AnalysisLevel.ANALYZE

    # Component configs
    linter_config: LinterConfig | None = None
    analyzer_config: AnalyzerConfig | None = None

    # Output settings
    output_format: OutputFormat = OutputFormat.JSON
    include_source_snippets: bool = True

    # Performance settings
    parallel: bool = False  # Future: parallel file analysis
    cache_enabled: bool = True

    # Safety
    read_only: bool = True  # Never modify source files


@dataclass
class AnalysisReport:
    """
    Complete analysis report for a file or codebase.

    Combines results from all analysis stages:
    - Parsed AST structure
    - Workflow analysis (if level >= ANALYZE)
    - Lint results (if level >= LINT)
    """

    # Source info
    file_path: str
    source_lines: int = 0

    # Parsed structure
    parsed: ParsedCode | None = None

    # Workflow analysis (requires LLM)
    workflow_analysis: WorkflowAnalysis | None = None

    # Lint results (no LLM)
    lint_result: LintResult | None = None

    # Derived workflow model
    agent_workflow: AgentWorkflow | None = None

    # Metadata
    analysis_level: AnalysisLevel = AnalysisLevel.ANALYZE
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if analysis encountered errors."""
        return len(self.errors) > 0

    @property
    def has_lint_issues(self) -> bool:
        """Check if lint found any issues."""
        return self.lint_result is not None and len(self.lint_result.issues) > 0

    @property
    def summary(self) -> dict[str, Any]:
        """Get a summary of the analysis."""
        return {
            "file": self.file_path,
            "source_lines": self.source_lines,
            "level": self.analysis_level.value,
            "has_errors": self.has_errors,
            "functions_found": len(self.parsed.functions) if self.parsed else 0,
            "classes_found": len(self.parsed.classes) if self.parsed else 0,
            "lint_issues": len(self.lint_result.issues) if self.lint_result else 0,
            "tools_found": (
                len(self.workflow_analysis.tools_detected) if self.workflow_analysis else 0
            ),
            "states_found": (
                len(self.workflow_analysis.states_detected) if self.workflow_analysis else 0
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "file_path": self.file_path,
            "source_lines": self.source_lines,
            "analysis_level": self.analysis_level.value,
            "errors": self.errors,
        }

        if self.parsed:
            result["parsed"] = {
                "functions": [
                    {
                        "name": f.name,
                        "is_async": f.is_async,
                        "parameters": [p.name for p in f.parameters],
                        "decorators": [d.name for d in f.decorators],
                        "has_docstring": f.docstring is not None,
                    }
                    for f in self.parsed.functions
                ],
                "classes": [
                    {
                        "name": c.name,
                        "bases": c.bases,
                        "methods": [m.name for m in c.methods],
                        "has_docstring": c.docstring is not None,
                    }
                    for c in self.parsed.classes
                ],
                "imports": [
                    {"module": i.module, "names": i.names, "is_from": i.is_from}
                    for i in self.parsed.imports
                ],
            }

        if self.lint_result:
            result["lint"] = {
                "issues": [
                    {
                        "rule_id": issue.rule_id,
                        "severity": issue.severity.value,
                        "category": issue.category.value,
                        "message": issue.message,
                        "line": issue.line,
                        "suggestion": issue.suggestion,
                    }
                    for issue in self.lint_result.issues
                ],
                "error_count": self.lint_result.error_count,
                "warning_count": self.lint_result.warning_count,
                "info_count": self.lint_result.info_count,
            }

        if self.workflow_analysis:
            result["workflow"] = {
                "summary": self.workflow_analysis.summary,
                "complexity_score": self.workflow_analysis.complexity_score,
                "tools": [
                    {"name": t.name, "description": t.description}
                    for t in self.workflow_analysis.tools_detected
                ],
                "states": [
                    {"name": s.name, "type": s.type} for s in self.workflow_analysis.states_detected
                ],
                "entry_points": self.workflow_analysis.entry_points,
                "recommendations": self.workflow_analysis.recommendations,
            }

        return result


class AnalysisPipeline:
    """
    Unified analysis pipeline for agent code.

    Orchestrates all analysis components:
    1. PythonParser - Extract AST using tree-sitter
    2. StaticLinter - Fast no-LLM checks
    3. WorkflowAnalyzer - DSPy semantic analysis

    SAFETY: Operates in read-only mode by default.
    Never modifies source files.

    Usage:
        pipeline = AnalysisPipeline()
        report = pipeline.analyze_file("agent.py")

        # Or with custom config
        config = PipelineConfig(level=AnalysisLevel.LINT)
        pipeline = AnalysisPipeline(config)
        report = pipeline.analyze_string(source_code)
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline with optional configuration."""
        self.config = config or PipelineConfig()

        # Initialize components
        self._parser = PythonParser()
        self._linter = StaticLinter(self.config.linter_config)

        # Analyzer is lazy-loaded (requires LLM)
        self._analyzer: WorkflowAnalyzer | None = None

    @property
    def analyzer(self) -> WorkflowAnalyzer:
        """Get or create the workflow analyzer (lazy initialization)."""
        if self._analyzer is None:
            self._analyzer = WorkflowAnalyzer(self.config.analyzer_config)
        return self._analyzer

    def analyze_file(self, file_path: str | Path) -> AnalysisReport:
        """
        Analyze a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            AnalysisReport with all analysis results
        """
        path = Path(file_path)

        if not path.exists():
            return AnalysisReport(
                file_path=str(path),
                errors=[f"File not found: {path}"],
            )

        if not path.suffix == ".py":
            return AnalysisReport(
                file_path=str(path),
                errors=[f"Not a Python file: {path}"],
            )

        source_code = path.read_text()
        return self.analyze_string(source_code, str(path))

    def analyze_string(
        self,
        source_code: str,
        file_name: str = "<string>",
    ) -> AnalysisReport:
        """
        Analyze Python source code.

        Args:
            source_code: Python source code
            file_name: Name for reporting

        Returns:
            AnalysisReport with all analysis results
        """
        report = AnalysisReport(
            file_path=file_name,
            source_lines=source_code.count("\n") + 1,
            analysis_level=self.config.level,
        )

        # Stage 1: Parse (always)
        try:
            report.parsed = self._parser.parse_string(source_code, file_name)
            if report.parsed.has_syntax_errors:
                report.errors.append(f"Syntax errors in {file_name}: {report.parsed.syntax_errors}")
        except Exception as e:
            report.errors.append(f"Parse error: {e}")
            return report

        # Stage 2: Lint (if level >= LINT)
        if self.config.level in (AnalysisLevel.LINT, AnalysisLevel.ANALYZE):
            try:
                report.lint_result = self._linter.lint_parsed(report.parsed)
            except Exception as e:
                report.errors.append(f"Lint error: {e}")

        # Stage 3: Analyze workflow (if level == ANALYZE)
        if self.config.level == AnalysisLevel.ANALYZE:
            try:
                report.workflow_analysis = self.analyzer.analyze(report.parsed)

                # Convert to domain model
                if report.workflow_analysis:
                    report.agent_workflow = self.analyzer.to_agent_workflow(
                        report.workflow_analysis
                    )
            except Exception as e:
                report.errors.append(f"Workflow analysis error: {e}")

        return report

    def analyze_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        pattern: str = "*.py",
    ) -> list[AnalysisReport]:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Directory path
            recursive: Whether to search recursively
            pattern: Glob pattern for files

        Returns:
            List of AnalysisReport for each file
        """
        path = Path(directory)

        if not path.is_dir():
            return [
                AnalysisReport(
                    file_path=str(path),
                    errors=[f"Not a directory: {path}"],
                )
            ]

        # Find all matching files
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))

        # Analyze each file
        # TODO: Parallel analysis when config.parallel is True
        reports = []
        for file_path in files:
            report = self.analyze_file(file_path)
            reports.append(report)

        return reports


# Convenience functions
def analyze_file(
    file_path: str | Path,
    level: AnalysisLevel = AnalysisLevel.ANALYZE,
) -> AnalysisReport:
    """
    Analyze a Python file.

    Args:
        file_path: Path to the Python file
        level: Depth of analysis

    Returns:
        AnalysisReport with all analysis results
    """
    config = PipelineConfig(level=level)
    pipeline = AnalysisPipeline(config)
    return pipeline.analyze_file(file_path)


def analyze_string(
    source_code: str,
    file_name: str = "<string>",
    level: AnalysisLevel = AnalysisLevel.ANALYZE,
) -> AnalysisReport:
    """
    Analyze Python source code.

    Args:
        source_code: Python source code
        file_name: Name for reporting
        level: Depth of analysis

    Returns:
        AnalysisReport with all analysis results
    """
    config = PipelineConfig(level=level)
    pipeline = AnalysisPipeline(config)
    return pipeline.analyze_string(source_code, file_name)


def lint_only(file_path: str | Path) -> LintResult:
    """
    Lint a file without LLM analysis.

    Args:
        file_path: Path to the Python file

    Returns:
        LintResult with found issues
    """
    config = PipelineConfig(level=AnalysisLevel.LINT)
    pipeline = AnalysisPipeline(config)
    report = pipeline.analyze_file(file_path)
    return report.lint_result or LintResult(file_path=str(file_path))
