"""
Tests for HTML report generation.

Part of Phase 5: Coverage & Reporting.
"""

import re
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pytest

from quaestor.coverage.tracker import (
    CoverageDimension,
    CoverageReport,
    CoverageTracker,
    DimensionCoverage,
)
from quaestor.evaluation.models import EvaluationCategory, Severity, Verdict
from quaestor.reporting.html import HTMLReportGenerator


class TestHTMLReportGenerator:
    """Test HTML report generation."""

    def test_basic_report_generation(self) -> None:
        """Test basic report creation without data."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                report_title="Test Report",
            )

            assert temp_path.exists()
            content = temp_path.read_text()

            # Check basic structure
            assert "<!DOCTYPE html>" in content
            assert "Test Report" in content
            assert "Quaestor Test Report" in content
        finally:
            temp_path.unlink()

    def test_report_with_coverage(self) -> None:
        """Test report with coverage data."""
        generator = HTMLReportGenerator()

        # Create coverage report
        coverage = CoverageReport(
            session_id="test-session",
            timestamp=datetime.now(),
            dimensions={
                CoverageDimension.TOOL: DimensionCoverage(
                    dimension=CoverageDimension.TOOL,
                    total_items=10,
                    covered_items={"tool1", "tool2", "tool3", "tool4", "tool5", "tool6", "tool7", "tool8"},
                    uncovered_items={"tool9", "tool10"},
                ),
                CoverageDimension.STATE: DimensionCoverage(
                    dimension=CoverageDimension.STATE,
                    total_items=5,
                    covered_items={"s1", "s2", "s3", "s4", "s5"},
                    uncovered_items=set(),
                ),
                CoverageDimension.TRANSITION: DimensionCoverage(
                    dimension=CoverageDimension.TRANSITION,
                    total_items=8,
                    covered_items={"t1", "t2", "t3", "t4", "t5", "t6"},
                    uncovered_items={"t7", "t8"},
                ),
                CoverageDimension.INVARIANT: DimensionCoverage(
                    dimension=CoverageDimension.INVARIANT,
                    total_items=3,
                    covered_items={"i1", "i2"},
                    uncovered_items={"i3"},
                ),
            },
        )

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                coverage_report=coverage,
            )

            content = temp_path.read_text()

            # Check coverage section exists
            assert "Coverage Summary" in content
            assert "Overall Coverage" in content

            # Check coverage percentages
            assert "80.0%" in content  # Tool coverage
            assert "100.0%" in content  # State coverage

            # Check gaps section
            assert "Coverage Gaps" in content
            assert "tool9" in content or "tool10" in content  # At least one uncovered tool
        finally:
            temp_path.unlink()

    def test_report_with_verdicts(self) -> None:
        """Test report with evaluation verdicts."""
        generator = HTMLReportGenerator()

        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.CRITICAL,
                title="Critical Error",
                description="A critical issue was found",
                score=0.2,
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.PERFORMANCE,
                severity=Severity.HIGH,
                title="Performance Issue",
                description="Slow response time",
                score=0.5,
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.SAFETY,
                severity=Severity.MEDIUM,
                title="Safety Warning",
                description="Minor safety concern",
                score=0.7,
                evidence=[],
            ),
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                verdicts=verdicts,
            )

            content = temp_path.read_text()

            # Check verdict section
            assert "Evaluation Verdicts" in content
            assert "Total Verdicts" in content

            # Check verdict counts
            assert "1" in content  # Critical count
            assert "1" in content  # High count

            # Check verdict titles
            assert "Critical Error" in content
            assert "Performance Issue" in content
            assert "Safety Warning" in content

            # Check descriptions
            assert "A critical issue was found" in content
            assert "Slow response time" in content

            # Check scores
            assert "0.20" in content
            assert "0.50" in content
        finally:
            temp_path.unlink()

    def test_report_with_timeline(self) -> None:
        """Test report with timeline events."""
        generator = HTMLReportGenerator()

        timeline = [
            {"timestamp": "2026-01-15 10:00:00", "description": "Test started"},
            {"timestamp": "2026-01-15 10:01:30", "description": "Analysis completed"},
            {"timestamp": "2026-01-15 10:02:45", "description": "Tests executed"},
            {"timestamp": "2026-01-15 10:03:00", "description": "Report generated"},
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                timeline=timeline,
            )

            content = temp_path.read_text()

            # Check timeline section
            assert "Test Execution Timeline" in content

            # Check all events
            assert "Test started" in content
            assert "Analysis completed" in content
            assert "Tests executed" in content
            assert "Report generated" in content

            # Check timestamps
            assert "2026-01-15 10:00:00" in content
            assert "2026-01-15 10:03:00" in content
        finally:
            temp_path.unlink()

    def test_report_with_agent_name(self) -> None:
        """Test report with agent name."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                agent_name="TestAgent-v1.0",
            )

            content = temp_path.read_text()

            # Check agent name appears
            assert "Agent: TestAgent-v1.0" in content
        finally:
            temp_path.unlink()

    def test_report_with_all_data(self) -> None:
        """Test comprehensive report with all data."""
        generator = HTMLReportGenerator()

        coverage = CoverageReport(
            session_id="test-all-data",
            timestamp=datetime.now(),
            dimensions={
                CoverageDimension.TOOL: DimensionCoverage(
                    dimension=CoverageDimension.TOOL,
                    total_items=5,
                    covered_items={"t1", "t2", "t3", "t4", "t5"},
                    uncovered_items=set(),
                ),
                CoverageDimension.STATE: DimensionCoverage(
                    dimension=CoverageDimension.STATE,
                    total_items=3,
                    covered_items={"s1", "s2", "s3"},
                    uncovered_items=set(),
                ),
                CoverageDimension.TRANSITION: DimensionCoverage(
                    dimension=CoverageDimension.TRANSITION,
                    total_items=4,
                    covered_items={"tr1", "tr2", "tr3", "tr4"},
                    uncovered_items=set(),
                ),
                CoverageDimension.INVARIANT: DimensionCoverage(
                    dimension=CoverageDimension.INVARIANT,
                    total_items=2,
                    covered_items={"i1", "i2"},
                    uncovered_items=set(),
                ),
            },
        )

        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.INFO,
                title="All Tests Passed",
                description="No issues found",
                score=0.95,
                evidence=[],
            )
        ]

        timeline = [
            {"timestamp": "2026-01-15 10:00:00", "description": "Test started"},
            {"timestamp": "2026-01-15 10:01:00", "description": "Test completed"},
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                report_title="Complete Test Report",
                agent_name="CompleteAgent",
                coverage_report=coverage,
                verdicts=verdicts,
                timeline=timeline,
            )

            content = temp_path.read_text()

            # Check all sections
            assert "Complete Test Report" in content
            assert "Agent: CompleteAgent" in content
            assert "Coverage Summary" in content
            assert "Evaluation Verdicts" in content
            assert "Test Execution Timeline" in content
            assert "All Tests Passed" in content
        finally:
            temp_path.unlink()

    def test_empty_verdicts_section(self) -> None:
        """Test report with no verdicts shows empty state."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                verdicts=[],
            )

            content = temp_path.read_text()

            # Check empty state message
            assert "No verdicts generated yet" in content
        finally:
            temp_path.unlink()

    def test_verdict_severity_styling(self) -> None:
        """Test that different severity verdicts have correct styling."""
        generator = HTMLReportGenerator()

        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.CRITICAL,
                title="Critical",
                description="Critical issue",
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.HIGH,
                title="High",
                description="High issue",
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.MEDIUM,
                title="Medium",
                description="Medium issue",
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.LOW,
                title="Low",
                description="Low issue",
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.INFO,
                title="Info",
                description="Info issue",
                evidence=[],
            ),
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                verdicts=verdicts,
            )

            content = temp_path.read_text()

            # Check severity classes
            assert "verdict-critical" in content
            assert "verdict-high" in content
            assert "verdict-medium" in content
            assert "verdict-low" in content
            assert "verdict-info" in content

            # Check badge classes
            assert "badge-critical" in content
            assert "badge-high" in content
            assert "badge-medium" in content
            assert "badge-low" in content
            assert "badge-info" in content
        finally:
            temp_path.unlink()

    def test_generate_from_data_dict(self) -> None:
        """Test generation from data dictionary."""
        generator = HTMLReportGenerator()

        coverage = CoverageReport(
            session_id="test-dict",
            timestamp=datetime.now(),
            dimensions={
                CoverageDimension.TOOL: DimensionCoverage(
                    dimension=CoverageDimension.TOOL,
                    total_items=2,
                    covered_items={"t1", "t2"},
                    uncovered_items=set(),
                ),
                CoverageDimension.STATE: DimensionCoverage(
                    dimension=CoverageDimension.STATE,
                    total_items=2,
                    covered_items={"s1", "s2"},
                    uncovered_items=set(),
                ),
                CoverageDimension.TRANSITION: DimensionCoverage(
                    dimension=CoverageDimension.TRANSITION,
                    total_items=2,
                    covered_items={"tr1", "tr2"},
                    uncovered_items=set(),
                ),
                CoverageDimension.INVARIANT: DimensionCoverage(
                    dimension=CoverageDimension.INVARIANT,
                    total_items=2,
                    covered_items={"i1", "i2"},
                    uncovered_items=set(),
                ),
            },
        )

        data = {
            "title": "Data Dict Report",
            "agent_name": "DictAgent",
            "coverage": coverage,
            "verdicts": [],
            "timeline": [{"timestamp": "now", "description": "test"}],
        }

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate_from_data(output_path=temp_path, data=data)

            content = temp_path.read_text()
            assert "Data Dict Report" in content
            assert "Agent: DictAgent" in content
        finally:
            temp_path.unlink()

    def test_custom_template(self) -> None:
        """Test report generation with custom template."""
        custom_template = """
        <!DOCTYPE html>
        <html>
        <head><title>{{ report_title }}</title></head>
        <body>
            <h1>{{ report_title }}</h1>
            <p>Custom template test</p>
        </body>
        </html>
        """

        generator = HTMLReportGenerator(template=custom_template)

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                report_title="Custom Report",
            )

            content = temp_path.read_text()
            assert "Custom Report" in content
            assert "Custom template test" in content
        finally:
            temp_path.unlink()

    def test_creates_directory_if_not_exists(self) -> None:
        """Test that generator creates parent directories."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False) as f:
            temp_base = Path(f.name)

        try:
            # Use non-existent subdirectory
            output_path = temp_base.parent / "test_subdir" / "report.html"

            generator.generate(
                output_path=output_path,
                report_title="Directory Test",
            )

            assert output_path.exists()
            assert output_path.read_text()
        finally:
            if output_path.exists():
                output_path.unlink()
            if output_path.parent.exists():
                output_path.parent.rmdir()

    def test_html_validity(self) -> None:
        """Test that generated HTML has valid structure."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(output_path=temp_path)
            content = temp_path.read_text()

            # Check required HTML structure
            assert content.startswith("<!DOCTYPE html>")
            assert "<html" in content
            assert "</html>" in content
            assert "<head>" in content
            assert "</head>" in content
            assert "<body>" in content
            assert "</body>" in content
            assert '<meta charset="UTF-8">' in content
        finally:
            temp_path.unlink()

    def test_dark_mode_support(self) -> None:
        """Test that dark mode CSS is included."""
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(output_path=temp_path)
            content = temp_path.read_text()

            # Check for dark mode media query
            assert "prefers-color-scheme: dark" in content
            assert "--bg-primary:" in content
            assert "--text-primary:" in content
        finally:
            temp_path.unlink()


class TestHTMLReportIntegration:
    """Integration tests for HTML reports."""

    def test_full_report_workflow(self) -> None:
        """Test complete workflow from tracker to HTML report."""
        # Create tracker and record coverage
        tracker = CoverageTracker()
        tracker.record_tool("search_files")
        tracker.record_tool("read_file")
        tracker.record_state("idle")
        tracker.record_state("processing")
        tracker.record_transition("idle", "processing")
        tracker.record_invariant("auth_required")

        coverage_report = tracker.generate_report()

        # Create verdicts
        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=Severity.HIGH,
                title="Test Issue",
                description="Found during testing",
                score=0.6,
                evidence=[],
            )
        ]

        # Generate report
        generator = HTMLReportGenerator()

        with NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
            temp_path = Path(f.name)

        try:
            generator.generate(
                output_path=temp_path,
                report_title="Integration Test Report",
                agent_name="IntegrationTestAgent",
                coverage_report=coverage_report,
                verdicts=verdicts,
                timeline=[
                    {"timestamp": "start", "description": "Test started"},
                    {"timestamp": "end", "description": "Test completed"},
                ],
            )

            content = temp_path.read_text()

            # Verify all data appears
            assert "Integration Test Report" in content
            assert "IntegrationTestAgent" in content
            assert "search_files" not in content  # Not in gaps
            assert "Test Issue" in content
            assert "0.60" in content
        finally:
            temp_path.unlink()
