"""
Quaestor Analysis Engine.

Code analysis and workflow extraction for AI agents.

Components:
- parser: Tree-sitter based Python AST extraction
- workflow_analyzer: DSPy semantic workflow analysis
- linter: Static analysis for agent code quality
- pipeline: Unified API orchestrating all components
"""

# Domain models
# Linter
from quaestor.analysis.linter import (
    Category,
    LinterConfig,
    LintIssue,
    LintResult,
    LintRule,
    Severity,
    StaticLinter,
    lint_file,
    lint_string,
)
from quaestor.analysis.models import (
    AgentWorkflow,
    EntryPoint,
    Invariant,
    StateDefinition,
    ToolDefinition,
    Transition,
)

# Parser
from quaestor.analysis.parser import (
    ClassDef,
    Decorator,
    FunctionDef,
    Import,
    Parameter,
    ParsedCode,
    PythonParser,
    SourceLocation,
    parse_python_file,
    parse_python_string,
)

# Pipeline
from quaestor.analysis.pipeline import (
    AnalysisLevel,
    AnalysisPipeline,
    AnalysisReport,
    OutputFormat,
    PipelineConfig,
    analyze_file,
    analyze_string,
    lint_only,
)

# Workflow Analyzer
from quaestor.analysis.workflow_analyzer import (
    AnalyzerConfig,
    DetectedState,
    DetectedTool,
    WorkflowAnalysis,
    WorkflowAnalyzer,
)

__all__ = [
    # Domain models
    "AgentWorkflow",
    "ToolDefinition",
    "StateDefinition",
    "Transition",
    "Invariant",
    "EntryPoint",
    # Parser
    "PythonParser",
    "ParsedCode",
    "FunctionDef",
    "ClassDef",
    "Import",
    "Decorator",
    "Parameter",
    "SourceLocation",
    "parse_python_file",
    "parse_python_string",
    # Linter
    "StaticLinter",
    "LinterConfig",
    "LintResult",
    "LintIssue",
    "LintRule",
    "Severity",
    "Category",
    "lint_file",
    "lint_string",
    # Workflow Analyzer
    "WorkflowAnalyzer",
    "WorkflowAnalysis",
    "DetectedTool",
    "DetectedState",
    "AnalyzerConfig",
    # Pipeline
    "AnalysisPipeline",
    "AnalysisReport",
    "PipelineConfig",
    "AnalysisLevel",
    "OutputFormat",
    "analyze_file",
    "analyze_string",
    "lint_only",
]
