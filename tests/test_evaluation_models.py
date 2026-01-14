"""
Tests for evaluation models.

Part of Phase 4: Evaluation & Judgment.
"""

from quaestor.evaluation.models import (
    EvaluationCategory,
    EvaluationContext,
    Evidence,
    MetricResult,
    Severity,
    Verdict,
    VerdictSummary,
)


class TestSeverity:
    """Test Severity enum."""

    def test_all_severities_exist(self):
        """Test that all expected severities exist."""
        assert Severity.CRITICAL == "critical"
        assert Severity.HIGH == "high"
        assert Severity.MEDIUM == "medium"
        assert Severity.LOW == "low"
        assert Severity.INFO == "info"

    def test_from_score_critical(self):
        """Test from_score for critical severity."""
        assert Severity.from_score(0.0) == Severity.CRITICAL
        assert Severity.from_score(0.1) == Severity.CRITICAL

    def test_from_score_high(self):
        """Test from_score for high severity."""
        assert Severity.from_score(0.2) == Severity.HIGH
        assert Severity.from_score(0.3) == Severity.HIGH

    def test_from_score_medium(self):
        """Test from_score for medium severity."""
        assert Severity.from_score(0.4) == Severity.MEDIUM
        assert Severity.from_score(0.5) == Severity.MEDIUM

    def test_from_score_low(self):
        """Test from_score for low severity."""
        assert Severity.from_score(0.6) == Severity.LOW
        assert Severity.from_score(0.7) == Severity.LOW

    def test_from_score_info(self):
        """Test from_score for info severity."""
        assert Severity.from_score(0.8) == Severity.INFO
        assert Severity.from_score(1.0) == Severity.INFO


class TestEvaluationCategory:
    """Test EvaluationCategory enum."""

    def test_all_categories_defined(self):
        """Test all expected categories exist."""
        categories = [e.value for e in EvaluationCategory]
        assert "correctness" in categories
        assert "safety" in categories
        assert "helpfulness" in categories
        assert "jailbreak" in categories
        assert "injection" in categories
        assert "information_leak" in categories
        assert "reliability" in categories


class TestEvidence:
    """Test Evidence model."""

    def test_create_minimal(self):
        """Test creating evidence with minimal fields."""
        evidence = Evidence(
            type="response",
            source="test",
            content="Test content",
        )
        assert evidence.type == "response"
        assert evidence.source == "test"
        assert evidence.content == "Test content"
        assert evidence.metadata == {}

    def test_create_with_metadata(self):
        """Test creating evidence with metadata."""
        evidence = Evidence(
            type="observation",
            source="investigator",
            content="Found issue",
            metadata={"turn": 1, "severity": "high"},
        )
        assert evidence.metadata["turn"] == 1
        assert evidence.metadata["severity"] == "high"


class TestVerdict:
    """Test Verdict model."""

    def test_create_minimal(self):
        """Test creating verdict with minimal fields."""
        verdict = Verdict(
            id="V-001",
            title="Test Verdict",
            description="A test verdict",
            severity=Severity.MEDIUM,
            category=EvaluationCategory.CORRECTNESS,
        )
        assert verdict.id == "V-001"
        assert verdict.title == "Test Verdict"
        assert verdict.severity == Severity.MEDIUM

    def test_create_full(self):
        """Test creating verdict with all fields."""
        evidence = Evidence(
            type="response",
            source="agent",
            content="Response text",
        )
        verdict = Verdict(
            id="V-002",
            title="Safety Issue",
            description="Agent leaked information",
            severity=Severity.HIGH,
            category=EvaluationCategory.INFORMATION_LEAK,
            evidence=[evidence],
            remediation="Review output filtering",
            reasoning="Detected PII in response",
            score=0.3,
            test_case_id="TC-001",
            agent_id="agent-1",
        )
        assert len(verdict.evidence) == 1
        assert verdict.remediation == "Review output filtering"
        assert verdict.score == 0.3

    def test_is_failing(self):
        """Test is_failing property."""
        critical = Verdict(
            id="V1",
            title="Critical",
            description="Critical issue",
            severity=Severity.CRITICAL,
            category=EvaluationCategory.SAFETY,
        )
        info = Verdict(
            id="V2",
            title="Info",
            description="Info only",
            severity=Severity.INFO,
            category=EvaluationCategory.CORRECTNESS,
        )
        assert critical.is_failing is True
        assert info.is_failing is False

    def test_is_passing(self):
        """Test is_passing property."""
        info = Verdict(
            id="V1",
            title="Info",
            description="Info only",
            severity=Severity.INFO,
            category=EvaluationCategory.CORRECTNESS,
        )
        critical = Verdict(
            id="V2",
            title="Critical",
            description="Critical issue",
            severity=Severity.CRITICAL,
            category=EvaluationCategory.SAFETY,
        )
        assert info.is_passing is True
        assert critical.is_passing is False


class TestVerdictSummary:
    """Test VerdictSummary model."""

    def test_from_verdicts_empty(self):
        """Test creating summary from empty list."""
        summary = VerdictSummary.from_verdicts([])
        assert summary.total_verdicts == 0
        assert summary.overall_status == "pass"

    def test_from_verdicts_all_info(self):
        """Test summary with all info verdicts."""
        verdicts = [
            Verdict(
                id="V1",
                title="V1",
                description="D1",
                severity=Severity.INFO,
                category=EvaluationCategory.CORRECTNESS,
            ),
            Verdict(
                id="V2",
                title="V2",
                description="D2",
                severity=Severity.INFO,
                category=EvaluationCategory.CORRECTNESS,
            ),
        ]
        summary = VerdictSummary.from_verdicts(verdicts)
        assert summary.total_verdicts == 2
        assert summary.info_count == 2
        assert summary.overall_status == "pass"

    def test_from_verdicts_with_failures(self):
        """Test summary with some failures."""
        verdicts = [
            Verdict(
                id="V1",
                title="Info",
                description="OK",
                severity=Severity.INFO,
                category=EvaluationCategory.CORRECTNESS,
            ),
            Verdict(
                id="V2",
                title="Failed",
                description="Issue",
                severity=Severity.HIGH,
                category=EvaluationCategory.SAFETY,
            ),
            Verdict(
                id="V3",
                title="Critical",
                description="Critical issue",
                severity=Severity.CRITICAL,
                category=EvaluationCategory.JAILBREAK,
            ),
        ]
        summary = VerdictSummary.from_verdicts(verdicts)
        assert summary.total_verdicts == 3
        assert summary.critical_count == 1
        assert summary.high_count == 1
        assert summary.overall_status == "fail"

    def test_category_counts(self):
        """Test category_counts in summary."""
        verdicts = [
            Verdict(
                id="V1",
                title="V1",
                description="D1",
                severity=Severity.HIGH,
                category=EvaluationCategory.SAFETY,
            ),
            Verdict(
                id="V2",
                title="V2",
                description="D2",
                severity=Severity.MEDIUM,
                category=EvaluationCategory.JAILBREAK,
            ),
            Verdict(
                id="V3",
                title="V3",
                description="D3",
                severity=Severity.MEDIUM,
                category=EvaluationCategory.SAFETY,  # Duplicate category
            ),
        ]
        summary = VerdictSummary.from_verdicts(verdicts)
        assert summary.category_counts.get("safety") == 2
        assert summary.category_counts.get("jailbreak") == 1
        # Test categories_affected property
        assert len(summary.categories_affected) == 2
        assert EvaluationCategory.SAFETY in summary.categories_affected
        assert EvaluationCategory.JAILBREAK in summary.categories_affected


class TestMetricResult:
    """Test MetricResult model."""

    def test_create(self):
        """Test creating a metric result."""
        result = MetricResult(
            metric_name="exact_match",
            score=0.85,
            passed=True,
            reason="Good match",
        )
        assert result.metric_name == "exact_match"
        assert result.score == 0.85
        assert result.passed is True

    def test_to_verdict(self):
        """Test converting to verdict."""
        result = MetricResult(
            metric_name="safety_check",
            score=0.3,
            passed=False,
            reason="Safety concern detected",
            details={"pattern": "leaked_pii"},
        )
        verdict = result.to_verdict(
            verdict_id="V-001",
            category=EvaluationCategory.SAFETY,
        )
        assert verdict.id == "V-001"
        assert "safety_check" in verdict.title
        assert verdict.score == 0.3
        assert verdict.category == EvaluationCategory.SAFETY


class TestEvaluationContext:
    """Test EvaluationContext model."""

    def test_create_minimal(self):
        """Test creating context with minimal fields."""
        context = EvaluationContext(
            input_messages=["Hello"],
            actual_output="Hi there!",
        )
        assert context.input_messages == ["Hello"]
        assert context.actual_output == "Hi there!"
        assert context.expected_output is None
        assert context.tool_calls == []

    def test_create_full(self):
        """Test creating context with all fields."""
        context = EvaluationContext(
            input_messages=["What is 2+2?", "Thanks"],
            actual_output="2+2 equals 4",
            expected_output="4",
            tool_calls=[{"name": "calculator", "args": {"a": 2, "b": 2}}],
            observations=[{"type": "tool_call", "message": "Used calculator"}],
            test_case_id="TC-001",
            agent_id="agent-1",
            response_time_ms=150.0,
            metadata={"model": "gpt-4"},
        )
        assert len(context.input_messages) == 2
        assert context.expected_output == "4"
        assert len(context.tool_calls) == 1
        assert context.response_time_ms == 150.0
