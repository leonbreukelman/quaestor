"""
Evaluation Models for Quaestor.

Provides data models for verdicts, evidence, severity levels, and
evaluation categories. These models are used by QuaestorJudge and
DeepEval integration.

Part of Phase 4: Evaluation & Judgment.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field

# =============================================================================
# Severity Levels
# =============================================================================


class Severity(str, Enum):
    """
    Severity levels for verdicts.

    Follows industry-standard severity classification for
    security and quality findings.
    """

    CRITICAL = "critical"  # Severe issue requiring immediate action
    HIGH = "high"  # Major issue that should be fixed soon
    MEDIUM = "medium"  # Moderate issue that should be addressed
    LOW = "low"  # Minor issue for consideration
    INFO = "info"  # Informational finding, not an issue

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        """
        Convert a numeric score (0-1) to severity level.

        Args:
            score: A score from 0 (worst) to 1 (best)

        Returns:
            Corresponding severity level
        """
        if score < 0.2:
            return cls.CRITICAL
        elif score < 0.4:
            return cls.HIGH
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.LOW
        else:
            return cls.INFO


# =============================================================================
# Evaluation Categories
# =============================================================================


class EvaluationCategory(str, Enum):
    """
    Categories for classifying evaluation findings.

    Maps to governance principles and agent quality dimensions.
    """

    # Functional categories
    CORRECTNESS = "correctness"  # Output accuracy and tool use
    COMPLETENESS = "completeness"  # Task completion
    CONSISTENCY = "consistency"  # Behavior consistency across runs

    # Safety categories
    SAFETY = "safety"  # General safety compliance
    JAILBREAK = "jailbreak"  # Jailbreak vulnerability
    INJECTION = "injection"  # Prompt injection vulnerability
    INFORMATION_LEAK = "information_leak"  # Data leakage

    # Performance categories
    PERFORMANCE = "performance"  # Response time and efficiency
    RELIABILITY = "reliability"  # Error handling and recovery

    # Compliance categories
    COMPLIANCE = "compliance"  # Policy and governance adherence
    ETHICS = "ethics"  # Ethical behavior

    # Quality categories
    HELPFULNESS = "helpfulness"  # User assistance quality
    COHERENCE = "coherence"  # Response coherence and clarity


# =============================================================================
# Evidence Model
# =============================================================================


class Evidence(BaseModel):
    """
    Evidence supporting a verdict.

    Links findings to specific observations, responses, or behaviors
    from the test execution.
    """

    type: str = Field(..., description="Type of evidence (observation, response, tool_call)")
    source: str = Field(..., description="Source identifier (e.g., turn number, observation ID)")
    content: str = Field(..., description="The actual evidence content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "source": self.source,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Verdict Model
# =============================================================================


class Verdict(BaseModel):
    """
    A structured verdict from evaluation.

    Represents a finding from test evaluation with severity,
    category, evidence, and optional remediation guidance.
    """

    id: str = Field(..., description="Unique verdict identifier")
    title: str = Field(..., description="Short title summarizing the finding")
    description: str = Field(..., description="Detailed description of the finding")
    severity: Severity = Field(..., description="Severity level of the finding")
    category: EvaluationCategory = Field(..., description="Category of the finding")

    # Evidence
    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence supporting this verdict",
    )

    # Reasoning
    reasoning: str = Field(
        default="",
        description="Explanation of how the verdict was reached",
    )

    # Remediation
    remediation: str | None = Field(
        default=None,
        description="Suggested remediation action",
    )

    # Scores
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Numeric score (0-1) if applicable",
    )

    # Context
    test_case_id: str | None = Field(default=None, description="Related test case ID")
    agent_id: str | None = Field(default=None, description="Agent being evaluated")
    governance_principle: str | None = Field(
        default=None,
        description="Related governance principle ID",
    )

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "evidence": [e.to_dict() for e in self.evidence],
            "reasoning": self.reasoning,
            "remediation": self.remediation,
            "score": self.score,
            "test_case_id": self.test_case_id,
            "agent_id": self.agent_id,
            "governance_principle": self.governance_principle,
            "created_at": self.created_at.isoformat(),
        }

    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Verdict":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)

    @property
    def is_passing(self) -> bool:
        """Check if this verdict represents a passing result."""
        return self.severity == Severity.INFO

    @property
    def is_failing(self) -> bool:
        """Check if this verdict represents a failing result."""
        return self.severity in (Severity.CRITICAL, Severity.HIGH)


# =============================================================================
# Verdict Summary Model
# =============================================================================


class VerdictSummary(BaseModel):
    """
    Summary of multiple verdicts.

    Provides aggregated statistics and overall assessment
    for a test run or agent evaluation.
    """

    total_verdicts: int = Field(default=0)
    critical_count: int = Field(default=0)
    high_count: int = Field(default=0)
    medium_count: int = Field(default=0)
    low_count: int = Field(default=0)
    info_count: int = Field(default=0)

    # Category breakdown
    category_counts: dict[str, int] = Field(default_factory=dict)

    # Overall assessment
    overall_score: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_status: str = Field(default="unknown")  # pass, fail, warning

    # Verdicts
    verdicts: list[Verdict] = Field(default_factory=list)

    @classmethod
    def from_verdicts(cls, verdicts: list[Verdict]) -> "VerdictSummary":
        """Create a summary from a list of verdicts."""
        summary = cls(total_verdicts=len(verdicts), verdicts=verdicts)

        # Count severities
        for v in verdicts:
            if v.severity == Severity.CRITICAL:
                summary.critical_count += 1
            elif v.severity == Severity.HIGH:
                summary.high_count += 1
            elif v.severity == Severity.MEDIUM:
                summary.medium_count += 1
            elif v.severity == Severity.LOW:
                summary.low_count += 1
            elif v.severity == Severity.INFO:
                summary.info_count += 1

            # Count categories
            cat = v.category.value
            summary.category_counts[cat] = summary.category_counts.get(cat, 0) + 1

        # Calculate overall score
        if verdicts:
            scores = [v.score for v in verdicts if v.score is not None]
            if scores:
                summary.overall_score = sum(scores) / len(scores)

        # Determine overall status
        if summary.critical_count > 0 or summary.high_count > 0:
            summary.overall_status = "fail"
        elif summary.medium_count > 0 or summary.low_count > 0:
            summary.overall_status = "warning"
        else:
            summary.overall_status = "pass"

        return summary

    @property
    def categories_affected(self) -> list[EvaluationCategory]:
        """Get list of affected categories."""
        return [EvaluationCategory(cat) for cat in self.category_counts]

    def filter_by_severity(
        self,
        severities: list[Severity] | Severity,
    ) -> list[Verdict]:
        """
        Filter verdicts by severity level(s).

        Args:
            severities: Single severity or list of severities to filter by

        Returns:
            List of verdicts matching the severity filter
        """
        if isinstance(severities, Severity):
            severities = [severities]
        return [v for v in self.verdicts if v.severity in severities]

    def filter_by_category(
        self,
        categories: list[EvaluationCategory] | EvaluationCategory,
    ) -> list[Verdict]:
        """
        Filter verdicts by category(ies).

        Args:
            categories: Single category or list of categories to filter by

        Returns:
            List of verdicts matching the category filter
        """
        if isinstance(categories, EvaluationCategory):
            categories = [categories]
        return [v for v in self.verdicts if v.category in categories]

    def get_failing_verdicts(self) -> list[Verdict]:
        """
        Get all failing verdicts (CRITICAL or HIGH severity).

        Returns:
            List of failing verdicts
        """
        return [v for v in self.verdicts if v.is_failing]

    def get_governance_violations(self) -> list[Verdict]:
        """
        Get verdicts linked to governance principles.

        Returns:
            List of verdicts with governance_principle set
        """
        return [v for v in self.verdicts if v.governance_principle is not None]

    def query(
        self,
        *,
        severity: Severity | list[Severity] | None = None,
        category: EvaluationCategory | list[EvaluationCategory] | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        test_case_id: str | None = None,
        agent_id: str | None = None,
        governance_principle: str | None = None,
    ) -> list[Verdict]:
        """
        Query verdicts with multiple filter criteria.

        Args:
            severity: Filter by severity level(s)
            category: Filter by category(ies)
            min_score: Minimum score (inclusive)
            max_score: Maximum score (inclusive)
            test_case_id: Filter by test case ID
            agent_id: Filter by agent ID
            governance_principle: Filter by governance principle ID

        Returns:
            List of verdicts matching all specified criteria
        """
        results = self.verdicts

        # Apply severity filter
        if severity is not None:
            sev_list = [severity] if isinstance(severity, Severity) else severity
            results = [v for v in results if v.severity in sev_list]

        # Apply category filter
        if category is not None:
            cat_list = [category] if isinstance(category, EvaluationCategory) else category
            results = [v for v in results if v.category in cat_list]

        # Apply score filters
        if min_score is not None:
            results = [v for v in results if v.score is not None and v.score >= min_score]

        if max_score is not None:
            results = [v for v in results if v.score is not None and v.score <= max_score]

        # Apply test case filter
        if test_case_id is not None:
            results = [v for v in results if v.test_case_id == test_case_id]

        # Apply agent filter
        if agent_id is not None:
            results = [v for v in results if v.agent_id == agent_id]

        # Apply governance principle filter
        if governance_principle is not None:
            results = [v for v in results if v.governance_principle == governance_principle]

        return results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_verdicts": self.total_verdicts,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "info_count": self.info_count,
            "category_counts": self.category_counts,
            "overall_score": self.overall_score,
            "overall_status": self.overall_status,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


# =============================================================================
# Metric Result Model
# =============================================================================


class MetricResult(BaseModel):
    """
    Result from a single metric evaluation.

    Used by DeepEval integration and custom metrics.
    """

    metric_name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=1.0, description="Metric score (0-1)")
    passed: bool = Field(..., description="Whether the metric passed threshold")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Pass threshold")
    reason: str = Field(default="", description="Explanation of the score")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional details")

    def to_verdict(
        self,
        verdict_id: str,
        category: EvaluationCategory = EvaluationCategory.CORRECTNESS,
    ) -> Verdict:
        """
        Convert metric result to a verdict.

        Args:
            verdict_id: Unique ID for the verdict
            category: Category for the verdict

        Returns:
            Verdict based on this metric result
        """
        severity = Severity.from_score(self.score)

        return Verdict(
            id=verdict_id,
            title=f"{self.metric_name}: {'Pass' if self.passed else 'Fail'}",
            description=self.reason or f"Metric {self.metric_name} scored {self.score:.2f}",
            severity=severity,
            category=category,
            score=self.score,
            reasoning=self.reason,
        )


# =============================================================================
# Evaluation Context Model
# =============================================================================


class EvaluationContext(BaseModel):
    """
    Context for evaluation.

    Contains all information needed to evaluate agent behavior
    including inputs, outputs, and observations.
    """

    # Input
    input_messages: list[str] = Field(default_factory=list, description="Input messages sent")
    expected_output: str | None = Field(default=None, description="Expected output if known")

    # Output
    actual_output: str = Field(..., description="Actual agent output")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Tool calls made by agent",
    )

    # Context
    context: list[str] = Field(
        default_factory=list,
        description="Retrieved context or documents",
    )

    # Observations
    observations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Observations from investigator",
    )

    # Metadata
    test_case_id: str | None = Field(default=None)
    agent_id: str | None = Field(default=None)
    response_time_ms: int | None = Field(default=None)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for evaluation context",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_messages": self.input_messages,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "tool_calls": self.tool_calls,
            "context": self.context,
            "observations": self.observations,
            "test_case_id": self.test_case_id,
            "agent_id": self.agent_id,
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata,
        }
