"""
SARIF Output Generator for CI Integration.

Generate SARIF 2.1.0 format reports for GitHub Code Scanning
and other CI/CD tools.

Part of Phase 5: Coverage & Reporting.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quaestor.analysis.linter import LintIssue, Severity
from quaestor.evaluation.models import Severity as VerdictSeverity
from quaestor.evaluation.models import Verdict
from quaestor.redteam.models import AttackResult, AttackSeverity, RedTeamReport


@dataclass
class SARIFLocation:
    """SARIF location information."""

    uri: str
    start_line: int = 1
    start_column: int = 1
    end_line: int | None = None
    end_column: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to SARIF format."""
        region: dict[str, Any] = {
            "startLine": self.start_line,
            "startColumn": self.start_column,
        }

        if self.end_line:
            region["endLine"] = self.end_line
        if self.end_column:
            region["endColumn"] = self.end_column

        location: dict[str, Any] = {
            "physicalLocation": {
                "artifactLocation": {"uri": self.uri},
                "region": region,
            }
        }

        return location


@dataclass
class SARIFRule:
    """SARIF rule definition."""

    id: str
    name: str
    short_description: str
    full_description: str | None = None
    help_text: str | None = None
    help_uri: str | None = None
    default_level: str = "warning"  # error, warning, note

    def to_dict(self) -> dict[str, Any]:
        """Convert to SARIF format."""
        rule = {
            "id": self.id,
            "name": self.name,
            "shortDescription": {"text": self.short_description},
            "defaultConfiguration": {"level": self.default_level},
        }

        if self.full_description:
            rule["fullDescription"] = {"text": self.full_description}

        if self.help_text:
            rule["help"] = {"text": self.help_text}

        if self.help_uri:
            rule["helpUri"] = self.help_uri

        return rule


@dataclass
class SARIFResult:
    """SARIF result (finding)."""

    rule_id: str
    message: str
    level: str = "warning"  # error, warning, note
    locations: list[SARIFLocation] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to SARIF format."""
        result: dict[str, Any] = {
            "ruleId": self.rule_id,
            "message": {"text": self.message},
            "level": self.level,
        }

        if self.locations:
            result["locations"] = [loc.to_dict() for loc in self.locations]

        if self.properties:
            result["properties"] = self.properties

        return result


class SARIFReport:
    """
    SARIF 2.1.0 report generator.

    Converts Quaestor findings (Issues, Verdicts) to SARIF format
    for CI/CD integration and GitHub Code Scanning.
    """

    VERSION = "2.1.0"
    SCHEMA_URI = "https://json.schemastore.org/sarif-2.1.0.json"

    def __init__(
        self,
        tool_name: str = "Quaestor",
        tool_version: str = "0.1.0",
        tool_uri: str = "https://github.com/leonbreukelman/quaestor",
    ):
        """
        Initialize SARIF report.

        Args:
            tool_name: Name of the analysis tool
            tool_version: Version of the tool
            tool_uri: URI for tool information
        """
        self.tool_name = tool_name
        self.tool_version = tool_version
        self.tool_uri = tool_uri
        self.rules: dict[str, SARIFRule] = {}
        self.results: list[SARIFResult] = []

    def add_rule(self, rule: SARIFRule) -> None:
        """Add a rule definition."""
        self.rules[rule.id] = rule

    def add_result(self, result: SARIFResult) -> None:
        """Add a result (finding)."""
        self.results.append(result)

    def add_issue(
        self,
        issue: LintIssue,
        file_path: str | None = None,
    ) -> None:
        """
        Add a linter issue to the SARIF report.

        Args:
            issue: Issue from static analysis
            file_path: Path to the file (if not in issue.file_path)
        """
        # Map issue severity to SARIF level
        level_map = {
            Severity.ERROR: "error",
            Severity.WARNING: "warning",
            Severity.INFO: "note",
        }
        level = level_map.get(issue.severity, "warning")

        # Create rule if not exists
        rule_id = f"Q-{issue.rule_id}"
        if rule_id not in self.rules:
            self.add_rule(
                SARIFRule(
                    id=rule_id,
                    name=issue.category.value.replace("_", " ").title(),
                    short_description=issue.message,
                    default_level=level,
                )
            )

        # Create location
        locations = []
        if issue.file_path or file_path:
            loc_file = issue.file_path or file_path or "unknown"
            locations.append(
                SARIFLocation(
                    uri=loc_file,
                    start_line=issue.line or 1,
                    start_column=issue.column or 1,
                )
            )

        # Create result
        self.add_result(
            SARIFResult(
                rule_id=rule_id,
                message=issue.message,
                level=level,
                locations=locations,
                properties={
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                },
            )
        )

    def add_verdict(
        self,
        verdict: Verdict,
        file_path: str | None = None,
    ) -> None:
        """
        Add an evaluation verdict to the SARIF report.

        Args:
            verdict: Verdict from evaluation
            file_path: Optional file path for the verdict
        """
        # Map verdict severity to SARIF level
        level_map = {
            VerdictSeverity.CRITICAL: "error",
            VerdictSeverity.HIGH: "error",
            VerdictSeverity.MEDIUM: "warning",
            VerdictSeverity.LOW: "warning",
            VerdictSeverity.INFO: "note",
        }
        level = level_map.get(verdict.severity, "warning")

        # Create rule if not exists
        rule_id = f"V-{verdict.category.value}"
        if rule_id not in self.rules:
            self.add_rule(
                SARIFRule(
                    id=rule_id,
                    name=verdict.category.value.replace("_", " ").title(),
                    short_description=verdict.title,
                    full_description=verdict.description,
                    default_level=level,
                )
            )

        # Create locations from evidence
        locations = []
        if file_path:
            locations.append(SARIFLocation(uri=file_path))
        elif verdict.evidence:
            # Try to extract location from evidence
            for evidence in verdict.evidence:
                if "file" in evidence.metadata:
                    locations.append(
                        SARIFLocation(
                            uri=evidence.metadata["file"],
                            start_line=evidence.metadata.get("line", 1),
                        )
                    )
                    break

        # Create result
        properties: dict[str, Any] = {
            "verdict_id": verdict.id,
            "category": verdict.category.value,
            "severity": verdict.severity.value,
        }
        if verdict.score is not None:
            properties["score"] = verdict.score

        self.add_result(
            SARIFResult(
                rule_id=rule_id,
                message=f"{verdict.title}: {verdict.description}",
                level=level,
                locations=locations,
                properties=properties,
            )
        )

    def add_attack_result(
        self,
        attack: AttackResult,
        target_name: str = "agent",
    ) -> None:
        """
        Add a red team attack result to the SARIF report.

        Only successful attacks (vulnerabilities found) are reported.

        Args:
            attack: Attack result from red team testing
            target_name: Name of the tested target/agent
        """
        # Only report successful attacks (actual vulnerabilities)
        if not attack.success:
            return

        # Map attack severity to SARIF level
        severity_value = (
            attack.severity.value
            if isinstance(attack.severity, AttackSeverity)
            else attack.severity
        )
        level_map = {
            "critical": "error",
            "high": "error",
            "medium": "warning",
            "low": "warning",
            "info": "note",
        }
        level = level_map.get(severity_value, "warning")

        # Get vulnerability type as string
        vuln_type = (
            attack.vulnerability_type.value
            if hasattr(attack.vulnerability_type, "value")
            else str(attack.vulnerability_type)
        )

        # Get attack method as string
        attack_method = (
            attack.attack_method.value
            if hasattr(attack.attack_method, "value")
            else str(attack.attack_method)
        )

        # Create rule if not exists
        rule_id = f"RT-{vuln_type}"
        if rule_id not in self.rules:
            self.add_rule(
                SARIFRule(
                    id=rule_id,
                    name=vuln_type.replace("_", " ").title(),
                    short_description=f"Vulnerability: {vuln_type}",
                    full_description=(
                        f"The agent is vulnerable to {vuln_type} attacks. "
                        f"Detected using {attack_method} attack method."
                    ),
                    help_text=attack.remediation,
                    default_level=level,
                )
            )

        # Create result - use target name as location since this is agent testing
        locations = [
            SARIFLocation(uri=f"agent://{target_name}", start_line=1)
        ]

        # Build properties
        properties: dict[str, Any] = {
            "attack_id": attack.id,
            "vulnerability_type": vuln_type,
            "attack_method": attack_method,
            "severity": severity_value,
            "confidence": attack.confidence,
        }
        if attack.evidence:
            properties["evidence"] = attack.evidence

        # Build message with input/output if available
        message_parts = [f"Security vulnerability detected: {vuln_type}"]
        if attack.input_prompt:
            # Truncate long prompts
            prompt_preview = attack.input_prompt[:200]
            if len(attack.input_prompt) > 200:
                prompt_preview += "..."
            message_parts.append(f"Attack prompt: {prompt_preview}")
        if attack.agent_response:
            response_preview = attack.agent_response[:200]
            if len(attack.agent_response) > 200:
                response_preview += "..."
            message_parts.append(f"Vulnerable response: {response_preview}")

        self.add_result(
            SARIFResult(
                rule_id=rule_id,
                message=" | ".join(message_parts),
                level=level,
                locations=locations,
                properties=properties,
            )
        )

    def add_redteam_report(
        self,
        report: RedTeamReport,
    ) -> None:
        """
        Add all successful attacks from a red team report.

        Args:
            report: Complete red team report
        """
        for attack in report.results:
            self.add_attack_result(attack, target_name=report.target_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to SARIF dictionary."""
        return {
            "version": self.VERSION,
            "$schema": self.SCHEMA_URI,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": self.tool_name,
                            "version": self.tool_version,
                            "informationUri": self.tool_uri,
                            "rules": [rule.to_dict() for rule in self.rules.values()],
                        }
                    },
                    "results": [result.to_dict() for result in self.results],
                }
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, output_path: str | Path) -> None:
        """
        Write SARIF report to file.

        Args:
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json())


def create_sarif_from_issues(
    issues: list[LintIssue],
    output_path: str | Path | None = None,
) -> SARIFReport:
    """
    Create SARIF report from linter issues.

    Args:
        issues: List of issues from static analysis
        output_path: Optional path to write the report

    Returns:
        SARIFReport object
    """
    report = SARIFReport()

    for issue in issues:
        report.add_issue(issue)

    if output_path:
        report.write(output_path)

    return report


def create_sarif_from_verdicts(
    verdicts: list[Verdict],
    output_path: str | Path | None = None,
) -> SARIFReport:
    """
    Create SARIF report from evaluation verdicts.

    Args:
        verdicts: List of verdicts from evaluation
        output_path: Optional path to write the report

    Returns:
        SARIFReport object
    """
    report = SARIFReport()

    for verdict in verdicts:
        report.add_verdict(verdict)

    if output_path:
        report.write(output_path)

    return report


def create_sarif_from_redteam(
    redteam_report: RedTeamReport,
    output_path: str | Path | None = None,
) -> SARIFReport:
    """
    Create SARIF report from red team assessment results.

    Only includes successful attacks (actual vulnerabilities).

    Args:
        redteam_report: Red team assessment report
        output_path: Optional path to write the report

    Returns:
        SARIFReport object
    """
    report = SARIFReport()
    report.add_redteam_report(redteam_report)

    if output_path:
        report.write(output_path)

    return report
