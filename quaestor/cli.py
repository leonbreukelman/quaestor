"""
Quaestor CLI - Command-line interface for agent testing.

Provides commands for analyzing, linting, testing, and reporting on AI agents.
Built on Smactorio governance infrastructure.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from quaestor.analysis.linter import Severity
from quaestor.analysis.pipeline import (
    AnalysisLevel,
    AnalysisPipeline,
    AnalysisReport,
    PipelineConfig,
)
from quaestor.analysis.workflow_analyzer import AnalyzerConfig

app = typer.Typer(
    name="quaestor",
    help="Self-optimizing agentic testing framework - pytest for AI agents",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    """Display version information."""
    if value:
        from quaestor import __version__

        console.print(f"[bold blue]Quaestor[/bold blue] v{__version__}")
        console.print("[dim]Built on Smactorio governance infrastructure[/dim]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Quaestor - Self-optimizing agentic testing framework."""
    pass


@app.command()
def analyze(
    path: str = typer.Argument(..., help="Path to agent code or directory"),
    output: str = typer.Option(None, "--output", "-o", help="Output file for workflow JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose output"),
    mock: bool = typer.Option(False, "--mock", help="Use mock mode (no LLM calls)"),
    format_: str = typer.Option("console", "--format", "-f", help="Output format: console, json"),
) -> None:
    """
    Analyze agent code and extract workflow definition.

    Uses DSPy-powered WorkflowAnalyzer to understand agent structure,
    tools, states, and transitions.
    """
    target_path = Path(path)

    if not target_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    # Configure pipeline
    analyzer_config = AnalyzerConfig(use_mock=mock) if mock else None
    config = PipelineConfig(
        level=AnalysisLevel.ANALYZE,
        analyzer_config=analyzer_config,
    )
    pipeline = AnalysisPipeline(config)

    console.print(
        Panel(
            f"[bold]Analyzing:[/bold] {path}"
            + ("\n[dim]Mock mode: LLM calls disabled[/dim]" if mock else ""),
            title="🔍 Quaestor Analyzer",
            border_style="blue",
        )
    )

    # Analyze file(s)
    if target_path.is_file():
        reports = [pipeline.analyze_file(target_path)]
    else:
        reports = pipeline.analyze_directory(target_path)

    # Handle output
    if format_ == "json" or output:
        json_output = [report.to_dict() for report in reports]
        if output:
            Path(output).write_text(json.dumps(json_output, indent=2))
            console.print(f"[green]✓[/green] Output written to {output}")
        else:
            console.print(json.dumps(json_output, indent=2))
    else:
        _display_analysis_reports(reports, verbose)

    # Summary
    error_count = sum(1 for r in reports if r.has_errors)
    if error_count > 0:
        console.print(f"\n[red]⚠ {error_count} file(s) had analysis errors[/red]")
        raise typer.Exit(1)


@app.command()
def lint(
    path: str = typer.Argument(..., help="Path to agent code or directory"),
    format_: str = typer.Option(
        "console", "--format", "-f", help="Output format: console, json, sarif"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output file for results"),
) -> None:
    """
    Run static analysis on agent code (no LLM calls).

    Fast feedback on common anti-patterns, security issues,
    and best practice violations.
    """
    target_path = Path(path)

    if not target_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    # Configure pipeline for lint-only
    config = PipelineConfig(level=AnalysisLevel.LINT)
    pipeline = AnalysisPipeline(config)

    console.print(
        Panel(
            f"[bold]Linting:[/bold] {path}",
            title="📋 Quaestor Linter",
            border_style="yellow",
        )
    )

    # Analyze file(s)
    if target_path.is_file():
        reports = [pipeline.analyze_file(target_path)]
    else:
        reports = pipeline.analyze_directory(target_path)

    # Aggregate lint results
    total_issues = sum(len(r.lint_result.issues) if r.lint_result else 0 for r in reports)

    # Handle output
    if format_ == "json" or (output and output.endswith(".json")):
        json_output = _lint_reports_to_json(reports)
        if output:
            Path(output).write_text(json.dumps(json_output, indent=2))
            console.print(f"[green]✓[/green] Output written to {output}")
        else:
            console.print(json.dumps(json_output, indent=2))
    elif format_ == "sarif" or (output and output.endswith(".sarif")):
        console.print("[yellow]⚠️  SARIF output not yet implemented[/yellow]")
        console.print("[dim]See TODO.md Phase 5 for implementation roadmap[/dim]")
    else:
        _display_lint_reports(reports)

    # Summary
    if total_issues > 0:
        error_count = sum(
            sum(1 for issue in r.lint_result.issues if issue.severity == Severity.ERROR)
            for r in reports
            if r.lint_result
        )
        console.print(f"\n[bold]Found {total_issues} issue(s)[/bold]")
        if error_count > 0:
            raise typer.Exit(1)


@app.command()
def test(
    path: str = typer.Argument(..., help="Path to agent code or test file"),
    level: str = typer.Option(
        "integration",
        "--level",
        "-l",
        help="Test level: unit, integration, scenario, redteam",
    ),
    _verbose: bool = typer.Option(False, "--verbose", "-V", help="Verbose output"),
    _fail_fast: bool = typer.Option(False, "--fail-fast", "-x", help="Stop on first failure"),
) -> None:
    """
    Run tests against an agent.

    Levels:
    - unit: Test individual tools/functions
    - integration: Test tool combinations
    - scenario: Full conversation flows
    - redteam: Adversarial testing (requires deepteam)
    """
    console.print(
        Panel(
            f"[bold]Testing:[/bold] {path}\n[dim]Level: {level}[/dim]",
            title="🧪 Quaestor Tester",
            border_style="green",
        )
    )

    # TODO: Implement test runner
    console.print("[yellow]⚠️  Test runner not yet implemented[/yellow]")
    console.print("[dim]See TODO.md Phase 3 for implementation roadmap[/dim]")


@app.command()
def coverage(
    path: str = typer.Argument(..., help="Path to agent code"),
    _output: str = typer.Option(
        "./quaestor-reports", "--output", "-o", help="Report output directory"
    ),
    _format: str = typer.Option(
        "html", "--format", "-f", help="Report format: html, json, console"
    ),
) -> None:
    """
    Generate coverage report for agent testing.

    Tracks coverage across:
    - Tools (which tools were exercised)
    - States (which states were reached)
    - Transitions (which paths were taken)
    - Invariants (which rules were verified)
    """
    console.print(
        Panel(
            f"[bold]Coverage:[/bold] {path}",
            title="📊 Quaestor Coverage",
            border_style="magenta",
        )
    )

    # TODO: Implement coverage tracker
    console.print("[yellow]⚠️  Coverage tracker not yet implemented[/yellow]")
    console.print("[dim]See TODO.md Phase 5 for implementation roadmap[/dim]")


@app.command()
def learn(
    examples_path: str = typer.Argument(..., help="Path to example test cases"),
    _optimize: bool = typer.Option(True, "--optimize", help="Run DSPy optimization"),
) -> None:
    """
    Bootstrap or optimize Quaestor from examples.

    Uses DSPy MIPROv2 to learn from successful test patterns
    and improve test generation.
    """
    console.print(
        Panel(
            f"[bold]Learning from:[/bold] {examples_path}",
            title="🧠 Quaestor Learner",
            border_style="cyan",
        )
    )

    # TODO: Implement learning system
    console.print("[yellow]⚠️  Learning system not yet implemented[/yellow]")
    console.print("[dim]See TODO.md Phase 6 for implementation roadmap[/dim]")


@app.command()
def init(
    path: str = typer.Argument(".", help="Path to initialize Quaestor config"),
) -> None:
    """
    Initialize Quaestor configuration in a project.

    Creates quaestor.yaml with default settings and
    integrates with existing Smactorio configuration.
    """
    console.print(
        Panel(
            f"[bold]Initializing:[/bold] {path}",
            title="🚀 Quaestor Init",
            border_style="blue",
        )
    )

    # TODO: Implement config initialization
    console.print("[yellow]⚠️  Init not yet implemented[/yellow]")
    console.print("[dim]Creates quaestor.yaml with project defaults[/dim]")


# =============================================================================
# Helper Functions
# =============================================================================


def _display_analysis_reports(reports: list[AnalysisReport], verbose: bool = False) -> None:
    """Display analysis reports in a human-readable format."""
    for report in reports:
        if report.has_errors:
            console.print(f"\n[red]✗ {report.file_path}[/red]")
            for error in report.errors:
                console.print(f"  [red]{error}[/red]")
            continue

        console.print(f"\n[green]✓ {report.file_path}[/green]")
        console.print(f"  [dim]{report.source_lines} lines[/dim]")

        # Show parsed structure
        if report.parsed and verbose:
            tree = Tree("[bold]Parsed Structure[/bold]")
            if report.parsed.functions:
                funcs = tree.add("📦 Functions")
                for func in report.parsed.functions:
                    func_label = f"{'async ' if func.is_async else ''}{func.name}()"
                    funcs.add(func_label)
            if report.parsed.classes:
                classes = tree.add("🏛️ Classes")
                for cls in report.parsed.classes:
                    cls_node = classes.add(cls.name)
                    for method in cls.methods:
                        cls_node.add(f"{'async ' if method.is_async else ''}{method.name}()")
            console.print(tree)

        # Show workflow analysis
        if report.workflow_analysis:
            console.print("\n  [bold]Workflow Analysis:[/bold]")
            console.print(f"    [cyan]Summary:[/cyan] {report.workflow_analysis.summary}")
            console.print(
                f"    [cyan]Complexity:[/cyan] {report.workflow_analysis.complexity_score}/10"
            )

            if report.workflow_analysis.tools_detected:
                console.print(
                    f"    [cyan]Tools:[/cyan] {len(report.workflow_analysis.tools_detected)}"
                )
                for tool in report.workflow_analysis.tools_detected:
                    console.print(f"      • {tool.name}: {tool.description}")

            if report.workflow_analysis.states_detected:
                console.print(
                    f"    [cyan]States:[/cyan] {len(report.workflow_analysis.states_detected)}"
                )
                for state in report.workflow_analysis.states_detected:
                    console.print(f"      • {state.name} ({state.type})")

            if report.workflow_analysis.recommendations:
                console.print("    [cyan]Recommendations:[/cyan]")
                for rec in report.workflow_analysis.recommendations:
                    console.print(f"      💡 {rec}")

        # Show lint issues
        if report.has_lint_issues and report.lint_result:
            console.print(f"\n  [bold]Lint Issues:[/bold] {len(report.lint_result.issues)}")
            for issue in report.lint_result.issues:
                severity_color = {
                    Severity.ERROR: "red",
                    Severity.WARNING: "yellow",
                    Severity.INFO: "blue",
                }.get(issue.severity, "white")
                console.print(
                    f"    [{severity_color}]{issue.severity.value.upper()}[/{severity_color}] "
                    f"L{issue.line}: {issue.message} [{issue.rule_id}]"
                )


def _display_lint_reports(reports: list[AnalysisReport]) -> None:
    """Display lint results in a human-readable format."""
    for report in reports:
        if report.has_errors:
            console.print(f"\n[red]✗ {report.file_path}[/red]")
            for error in report.errors:
                console.print(f"  [red]{error}[/red]")
            continue

        if not report.lint_result or not report.lint_result.issues:
            console.print(f"[green]✓ {report.file_path}[/green] - No issues")
            continue

        console.print(f"\n[yellow]⚠ {report.file_path}[/yellow]")

        # Group issues by severity
        table = Table(show_header=True, header_style="bold")
        table.add_column("Line", style="dim", width=6)
        table.add_column("Severity", width=10)
        table.add_column("Rule", style="cyan", width=20)
        table.add_column("Message")

        for issue in report.lint_result.issues:
            severity_color = {
                Severity.ERROR: "red",
                Severity.WARNING: "yellow",
                Severity.INFO: "blue",
            }.get(issue.severity, "white")
            table.add_row(
                str(issue.line),
                f"[{severity_color}]{issue.severity.value}[/{severity_color}]",
                issue.rule_id,
                issue.message,
            )

        console.print(table)


def _lint_reports_to_json(reports: list[AnalysisReport]) -> list[dict[str, object]]:
    """Convert lint reports to JSON-serializable format."""
    result: list[dict[str, object]] = []
    for report in reports:
        issues: list[dict[str, object]] = []
        if report.lint_result:
            issues = [
                {
                    "line": issue.line,
                    "column": issue.column,
                    "severity": issue.severity.value,
                    "rule_id": issue.rule_id,
                    "category": issue.category.value,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in report.lint_result.issues
            ]
        report_dict: dict[str, object] = {
            "file": report.file_path,
            "errors": report.errors,
            "issues": issues,
        }
        result.append(report_dict)
    return result


if __name__ == "__main__":
    app()
