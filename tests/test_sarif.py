"""
Tests for SARIF report generation.

Part of Phase 5: Coverage & Reporting.
"""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pytest

from quaestor.analysis.linter import Category, LintIssue, Severity
from quaestor.evaluation.models import (
    EvaluationCategory,
    Evidence,
    Severity as VerdictSeverity,
    Verdict,
)
from quaestor.reporting.sarif import (
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    create_sarif_from_issues,
    create_sarif_from_verdicts,
)


class TestSARIFLocation:
    """Test SARIF location creation."""

    def test_basic_location(self) -> None:
        """Test basic location with file and line."""
        location = SARIFLocation(
            uri="src/main.py",
            start_line=10,
        )

        result = location.to_dict()
        assert result["physicalLocation"]["artifactLocation"]["uri"] == "src/main.py"
        assert result["physicalLocation"]["region"]["startLine"] == 10
        assert result["physicalLocation"]["region"]["startColumn"] == 1

    def test_location_with_range(self) -> None:
        """Test location with line and column range."""
        location = SARIFLocation(
            uri="src/main.py",
            start_line=10,
            start_column=5,
            end_line=12,
            end_column=20,
        )

        result = location.to_dict()
        assert result["physicalLocation"]["region"]["startLine"] == 10
        assert result["physicalLocation"]["region"]["startColumn"] == 5
        assert result["physicalLocation"]["region"]["endLine"] == 12
        assert result["physicalLocation"]["region"]["endColumn"] == 20

    def test_location_without_end_markers(self) -> None:
        """Test location without end line/column."""
        location = SARIFLocation(
            uri="src/main.py",
            start_line=10,
        )

        result = location.to_dict()
        assert "endLine" not in result["physicalLocation"]["region"]
        assert "endColumn" not in result["physicalLocation"]["region"]


class TestSARIFRule:
    """Test SARIF rule creation."""

    def test_basic_rule(self) -> None:
        """Test basic rule creation."""
        rule = SARIFRule(
            id="Q-001",
            name="Test Rule",
            short_description="A test rule",
        )

        result = rule.to_dict()
        assert result["id"] == "Q-001"
        assert result["name"] == "Test Rule"
        assert result["shortDescription"]["text"] == "A test rule"
        assert result["defaultConfiguration"]["level"] == "warning"

    def test_rule_with_all_fields(self) -> None:
        """Test rule with all optional fields."""
        rule = SARIFRule(
            id="Q-002",
            name="Complete Rule",
            short_description="Short desc",
            full_description="Full description here",
            help_text="Help text here",
            help_uri="https://example.com/help",
            default_level="error",
        )

        result = rule.to_dict()
        assert result["fullDescription"]["text"] == "Full description here"
        assert result["help"]["text"] == "Help text here"
        assert result["helpUri"] == "https://example.com/help"
        assert result["defaultConfiguration"]["level"] == "error"

    def test_rule_severity_levels(self) -> None:
        """Test different severity levels."""
        for level in ["error", "warning", "note"]:
            rule = SARIFRule(
                id=f"Q-{level}",
                name=f"{level} Rule",
                short_description=f"A {level} rule",
                default_level=level,
            )
            result = rule.to_dict()
            assert result["defaultConfiguration"]["level"] == level


class TestSARIFResult:
    """Test SARIF result creation."""

    def test_basic_result(self) -> None:
        """Test basic result creation."""
        result = SARIFResult(
            rule_id="Q-001",
            message="Test finding",
        )

        output = result.to_dict()
        assert output["ruleId"] == "Q-001"
        assert output["message"]["text"] == "Test finding"
        assert output["level"] == "warning"

    def test_result_with_location(self) -> None:
        """Test result with location."""
        location = SARIFLocation(uri="src/test.py", start_line=42)
        result = SARIFResult(
            rule_id="Q-002",
            message="Issue at line 42",
            locations=[location],
        )

        output = result.to_dict()
        assert len(output["locations"]) == 1
        assert output["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/test.py"

    def test_result_with_properties(self) -> None:
        """Test result with custom properties."""
        result = SARIFResult(
            rule_id="Q-003",
            message="Test",
            properties={"custom": "value", "score": 0.95},
        )

        output = result.to_dict()
        assert output["properties"]["custom"] == "value"
        assert output["properties"]["score"] == 0.95


class TestSARIFReport:
    """Test SARIF report generation."""

    def test_empty_report(self) -> None:
        """Test empty report structure."""
        report = SARIFReport()
        output = report.to_dict()

        assert output["version"] == "2.1.0"
        assert "$schema" in output
        assert len(output["runs"]) == 1
        assert output["runs"][0]["tool"]["driver"]["name"] == "Quaestor"
        assert output["runs"][0]["results"] == []

    def test_report_with_custom_tool_info(self) -> None:
        """Test report with custom tool information."""
        report = SARIFReport(
            tool_name="Custom Tool",
            tool_version="1.2.3",
            tool_uri="https://example.com",
        )
        output = report.to_dict()

        driver = output["runs"][0]["tool"]["driver"]
        assert driver["name"] == "Custom Tool"
        assert driver["version"] == "1.2.3"
        assert driver["informationUri"] == "https://example.com"

    def test_add_rule_and_result(self) -> None:
        """Test adding rules and results."""
        report = SARIFReport()

        rule = SARIFRule(
            id="Q-001",
            name="Test Rule",
            short_description="Test",
        )
        report.add_rule(rule)

        result = SARIFResult(
            rule_id="Q-001",
            message="Finding",
        )
        report.add_result(result)

        output = report.to_dict()
        assert len(output["runs"][0]["tool"]["driver"]["rules"]) == 1
        assert len(output["runs"][0]["results"]) == 1
        assert output["runs"][0]["results"][0]["ruleId"] == "Q-001"

    def test_add_duplicate_rule(self) -> None:
        """Test that duplicate rules are handled correctly."""
        report = SARIFReport()

        rule1 = SARIFRule(id="Q-001", name="Rule 1", short_description="First")
        rule2 = SARIFRule(id="Q-001", name="Rule 2", short_description="Second")

        report.add_rule(rule1)
        report.add_rule(rule2)  # Should overwrite

        output = report.to_dict()
        rules = output["runs"][0]["tool"]["driver"]["rules"]
        assert len(rules) == 1
        assert rules[0]["name"] == "Rule 2"  # Second one wins

    def test_to_json(self) -> None:
        """Test JSON serialization."""
        report = SARIFReport()
        json_str = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"

    def test_write_to_file(self) -> None:
        """Test writing report to file."""
        report = SARIFReport()
        report.add_result(SARIFResult(rule_id="Q-001", message="Test"))

        with NamedTemporaryFile(mode="w", delete=False, suffix=".sarif") as f:
            temp_path = Path(f.name)

        try:
            report.write(temp_path)
            assert temp_path.exists()

            # Verify content
            content = json.loads(temp_path.read_text())
            assert content["version"] == "2.1.0"
            assert len(content["runs"][0]["results"]) == 1
        finally:
            temp_path.unlink()


class TestSARIFFromIssues:
    """Test SARIF generation from linter issues."""

    def test_add_single_issue(self) -> None:
        """Test adding a single linter issue."""
        report = SARIFReport()

        issue = LintIssue(
            rule_id="SEC001",
            category=Category.SECURITY,
            severity=Severity.ERROR,
            message="Hardcoded credentials detected",
            file_path="src/auth.py",
            line=42,
            column=10,
        )

        report.add_issue(issue)
        output = report.to_dict()

        rules = output["runs"][0]["tool"]["driver"]["rules"]
        results = output["runs"][0]["results"]

        assert len(rules) == 1
        assert rules[0]["id"] == "Q-SEC001"
        assert len(results) == 1
        assert results[0]["level"] == "error"
        assert results[0]["message"]["text"] == "Hardcoded credentials detected"

    def test_issue_severity_mapping(self) -> None:
        """Test severity mapping for issues."""
        report = SARIFReport()

        test_cases = [
            (Severity.ERROR, "error"),
            (Severity.WARNING, "warning"),
            (Severity.INFO, "note"),
        ]

        for severity, expected_level in test_cases:
            issue = LintIssue(
                rule_id=f"RULE-{severity.value}",
                category=Category.SECURITY,
                severity=severity,
                message=f"Test {severity.value}",
                file_path="test.py",
                line=1,
                column=1,
            )
            report.add_issue(issue)

        output = report.to_dict()
        results = output["runs"][0]["results"]

        for i, (_, expected_level) in enumerate(test_cases):
            assert results[i]["level"] == expected_level

    def test_issue_with_location(self) -> None:
        """Test issue with file location."""
        report = SARIFReport()

        issue = LintIssue(
            rule_id="STYLE001",
            category=Category.STYLE,
            severity=Severity.WARNING,
            message="Line too long",
            file_path="src/main.py",
            line=100,
            column=81,
        )

        report.add_issue(issue)
        output = report.to_dict()

        result = output["runs"][0]["results"][0]
        location = result["locations"][0]

        assert location["physicalLocation"]["artifactLocation"]["uri"] == "src/main.py"
        assert location["physicalLocation"]["region"]["startLine"] == 100

    def test_issue_without_location(self) -> None:
        """Test issue with fallback file path."""
        report = SARIFReport()

        issue = LintIssue(
            rule_id="GEN001",
            category=Category.CORRECTNESS,
            severity=Severity.WARNING,
            message="General issue",
            file_path="",
            line=0,
            column=0,
        )

        report.add_issue(issue, file_path="fallback.py")
        output = report.to_dict()

        result = output["runs"][0]["results"][0]
        assert len(result["locations"]) == 1
        assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "fallback.py"

    def test_create_sarif_from_issues_helper(self) -> None:
        """Test helper function for creating SARIF from issues."""
        issues = [
            LintIssue(
                rule_id="R1",
                category=Category.SECURITY,
                severity=Severity.ERROR,
                message="Issue 1",
                file_path="test.py",
                line=1,
                column=1,
            ),
            LintIssue(
                rule_id="R2",
                category=Category.STYLE,
                severity=Severity.WARNING,
                message="Issue 2",
                file_path="test.py",
                line=2,
                column=1,
            ),
        ]

        report = create_sarif_from_issues(issues)
        output = report.to_dict()

        assert len(output["runs"][0]["results"]) == 2

    def test_create_sarif_from_issues_with_file(self) -> None:
        """Test helper function writes to file."""
        issues = [
            LintIssue(
                rule_id="R1",
                category=Category.SECURITY,
                severity=Severity.ERROR,
                message="Test issue",
                file_path="test.py",
                line=1,
                column=1,
            )
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".sarif") as f:
            temp_path = Path(f.name)

        try:
            report = create_sarif_from_issues(issues, temp_path)
            assert temp_path.exists()

            content = json.loads(temp_path.read_text())
            assert len(content["runs"][0]["results"]) == 1
        finally:
            temp_path.unlink()


class TestSARIFFromVerdicts:
    """Test SARIF generation from evaluation verdicts."""

    def test_add_single_verdict(self) -> None:
        """Test adding a single verdict."""
        report = SARIFReport()

        verdict = Verdict(
            id=str(uuid4()),
            category=EvaluationCategory.CORRECTNESS,
            severity=VerdictSeverity.MEDIUM,
            title="Logic Error",
            description="Function returns incorrect value",
            score=0.75,
            evidence=[],
        )

        report.add_verdict(verdict)
        output = report.to_dict()

        rules = output["runs"][0]["tool"]["driver"]["rules"]
        results = output["runs"][0]["results"]

        assert len(rules) == 1
        assert rules[0]["id"] == "V-correctness"
        assert len(results) == 1
        assert results[0]["level"] == "warning"
        assert "Logic Error" in results[0]["message"]["text"]

    def test_verdict_severity_mapping(self) -> None:
        """Test severity mapping for verdicts."""
        report = SARIFReport()

        test_cases = [
            (VerdictSeverity.CRITICAL, "error"),
            (VerdictSeverity.HIGH, "error"),
            (VerdictSeverity.MEDIUM, "warning"),
            (VerdictSeverity.LOW, "warning"),
            (VerdictSeverity.INFO, "note"),
        ]

        for severity, expected_level in test_cases:
            verdict = Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.PERFORMANCE,
                severity=severity,
                title=f"Test {severity.value}",
                description="Test description",
                evidence=[],
            )
            report.add_verdict(verdict)

        output = report.to_dict()
        results = output["runs"][0]["results"]

        for i, (_, expected_level) in enumerate(test_cases):
            assert results[i]["level"] == expected_level

    def test_verdict_with_file_path(self) -> None:
        """Test verdict with explicit file path."""
        report = SARIFReport()

        verdict = Verdict(
            id=str(uuid4()),
            category=EvaluationCategory.SAFETY,
            severity=VerdictSeverity.HIGH,
            title="Security Issue",
            description="Potential vulnerability",
            evidence=[],
        )

        report.add_verdict(verdict, file_path="src/vulnerable.py")
        output = report.to_dict()

        result = output["runs"][0]["results"][0]
        assert len(result["locations"]) == 1
        assert result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] == "src/vulnerable.py"

    def test_verdict_with_evidence_location(self) -> None:
        """Test verdict with location from evidence."""
        report = SARIFReport()

        evidence = Evidence(
            type="observation",
            source="test-run-1",
            content="Found issue",
            metadata={"file": "src/main.py", "line": 50},
        )

        verdict = Verdict(
            id=str(uuid4()),
            category=EvaluationCategory.CORRECTNESS,
            severity=VerdictSeverity.MEDIUM,
            title="Error",
            description="Description",
            evidence=[evidence],
        )

        report.add_verdict(verdict)
        output = report.to_dict()

        result = output["runs"][0]["results"][0]
        assert len(result["locations"]) == 1
        location = result["locations"][0]
        assert location["physicalLocation"]["artifactLocation"]["uri"] == "src/main.py"
        assert location["physicalLocation"]["region"]["startLine"] == 50

    def test_verdict_with_score(self) -> None:
        """Test verdict properties include score."""
        report = SARIFReport()

        verdict = Verdict(
            id="test-verdict-1",
            category=EvaluationCategory.RELIABILITY,
            severity=VerdictSeverity.LOW,
            title="Minor Issue",
            description="Not critical",
            score=0.92,
            evidence=[],
        )

        report.add_verdict(verdict)
        output = report.to_dict()

        result = output["runs"][0]["results"][0]
        assert result["properties"]["score"] == 0.92
        assert result["properties"]["verdict_id"] == "test-verdict-1"

    def test_create_sarif_from_verdicts_helper(self) -> None:
        """Test helper function for creating SARIF from verdicts."""
        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.CORRECTNESS,
                severity=VerdictSeverity.HIGH,
                title="Error 1",
                description="Description 1",
                evidence=[],
            ),
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.PERFORMANCE,
                severity=VerdictSeverity.LOW,
                title="Warning 1",
                description="Description 2",
                evidence=[],
            ),
        ]

        report = create_sarif_from_verdicts(verdicts)
        output = report.to_dict()

        assert len(output["runs"][0]["results"]) == 2

    def test_create_sarif_from_verdicts_with_file(self) -> None:
        """Test helper function writes to file."""
        verdicts = [
            Verdict(
                id=str(uuid4()),
                category=EvaluationCategory.SAFETY,
                severity=VerdictSeverity.CRITICAL,
                title="Critical Issue",
                description="Needs attention",
                evidence=[],
            )
        ]

        with NamedTemporaryFile(mode="w", delete=False, suffix=".sarif") as f:
            temp_path = Path(f.name)

        try:
            report = create_sarif_from_verdicts(verdicts, temp_path)
            assert temp_path.exists()

            content = json.loads(temp_path.read_text())
            assert len(content["runs"][0]["results"]) == 1
            assert content["runs"][0]["results"][0]["level"] == "error"
        finally:
            temp_path.unlink()


class TestSARIFIntegration:
    """Integration tests for SARIF generation."""

    def test_mixed_issues_and_verdicts(self) -> None:
        """Test report with both issues and verdicts."""
        report = SARIFReport()

        # Add issues
        issue = LintIssue(
            rule_id="SEC001",
            category=Category.SECURITY,
            severity=Severity.ERROR,
            message="Security issue",
            file_path="test.py",
            line=1,
            column=1,
        )
        report.add_issue(issue)

        # Add verdicts
        verdict = Verdict(
            id=str(uuid4()),
            category=EvaluationCategory.CORRECTNESS,
            severity=VerdictSeverity.MEDIUM,
            title="Logic error",
            description="Wrong calculation",
            evidence=[],
        )
        report.add_verdict(verdict)

        output = report.to_dict()
        assert len(output["runs"][0]["results"]) == 2
        assert len(output["runs"][0]["tool"]["driver"]["rules"]) == 2

    def test_github_code_scanning_format(self) -> None:
        """Test that output is compatible with GitHub Code Scanning."""
        report = SARIFReport()

        issue = LintIssue(
            rule_id="TEST001",
            category=Category.SECURITY,
            severity=Severity.ERROR,
            message="Test security issue",
            file_path="src/test.py",
            line=10,
            column=5,
        )
        report.add_issue(issue)

        output = report.to_dict()

        # Verify required SARIF 2.1.0 fields
        assert output["version"] == "2.1.0"
        assert "$schema" in output
        assert len(output["runs"]) > 0

        run = output["runs"][0]
        assert "tool" in run
        assert "driver" in run["tool"]
        assert "results" in run

        # Verify GitHub-compatible fields
        driver = run["tool"]["driver"]
        assert "name" in driver
        assert "version" in driver
        assert "rules" in driver

        result = run["results"][0]
        assert "ruleId" in result
        assert "message" in result
        assert "level" in result
        assert "locations" in result
