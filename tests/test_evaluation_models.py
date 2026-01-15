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

    def test_all_severities_exist(self) -> None:
        """Test that all expected severities exist."""
        assert Severity.CRITICAL.value == "critical"
        assert Severity.HIGH.value == "high"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.LOW.value == "low"
        assert Severity.INFO.value == "info"

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
            response_time_ms=150,
            metadata={"model": "gpt-4"},
        )
        assert len(context.input_messages) == 2
        assert context.expected_output == "4"
        assert len(context.tool_calls) == 1
        assert context.response_time_ms == 150.0


class TestVerdictSerialization:
    """Test Verdict serialization methods."""

    def test_to_dict(self):
        """Test converting verdict to dictionary."""
        evidence = Evidence(
            type="observation",
            source="investigator",
            content="Found issue",
        )
        verdict = Verdict(
            id="V-001",
            title="Test Verdict",
            description="Test description",
            severity=Severity.HIGH,
            category=EvaluationCategory.SAFETY,
            evidence=[evidence],
            reasoning="Found safety concern",
            remediation="Add input validation",
            score=0.35,
            test_case_id="TC-001",
            agent_id="agent-1",
            governance_principle="safety-001",
        )
        data = verdict.to_dict()

        assert data["id"] == "V-001"
        assert data["title"] == "Test Verdict"
        assert data["severity"] == "high"
        assert data["category"] == "safety"
        assert len(data["evidence"]) == 1
        assert data["evidence"][0]["type"] == "observation"
        assert data["reasoning"] == "Found safety concern"
        assert data["remediation"] == "Add input validation"
        assert data["score"] == 0.35
        assert data["governance_principle"] == "safety-001"

    def test_to_yaml(self):
        """Test converting verdict to YAML."""
        verdict = Verdict(
            id="V-002",
            title="YAML Test",
            description="Testing YAML serialization",
            severity=Severity.MEDIUM,
            category=EvaluationCategory.CORRECTNESS,
        )
        yaml_str = verdict.to_yaml()

        assert "id: V-002" in yaml_str
        assert "title: YAML Test" in yaml_str
        assert "severity: medium" in yaml_str
        assert "category: correctness" in yaml_str

    def test_from_yaml(self):
        """Test loading verdict from YAML."""
        yaml_str = """
id: V-003
title: From YAML
description: Loaded from YAML string
severity: low
category: helpfulness
evidence: []
reasoning: Test reasoning
remediation: null
score: 0.75
test_case_id: TC-002
agent_id: null
governance_principle: null
created_at: '2026-01-15T10:00:00+00:00'
"""
        verdict = Verdict.from_yaml(yaml_str)

        assert verdict.id == "V-003"
        assert verdict.title == "From YAML"
        assert verdict.severity == Severity.LOW
        assert verdict.category == EvaluationCategory.HELPFULNESS
        assert verdict.score == 0.75

    def test_yaml_roundtrip(self):
        """Test YAML serialization roundtrip."""
        original = Verdict(
            id="V-ROUNDTRIP",
            title="Roundtrip Test",
            description="Test roundtrip serialization",
            severity=Severity.CRITICAL,
            category=EvaluationCategory.JAILBREAK,
            score=0.15,
            reasoning="Critical finding",
        )
        yaml_str = original.to_yaml()
        loaded = Verdict.from_yaml(yaml_str)

        assert loaded.id == original.id
        assert loaded.title == original.title
        assert loaded.severity == original.severity
        assert loaded.category == original.category
        assert loaded.score == original.score


class TestVerdictSummaryFiltering:
    """Test VerdictSummary filtering and querying methods."""

    def _create_test_verdicts(self) -> list[Verdict]:
        """Create a set of test verdicts for filtering."""
        return [
            Verdict(
                id="V1",
                title="Critical Safety",
                description="Critical safety issue",
                severity=Severity.CRITICAL,
                category=EvaluationCategory.SAFETY,
                score=0.1,
                test_case_id="TC-001",
                agent_id="agent-1",
                governance_principle="safety-001",
            ),
            Verdict(
                id="V2",
                title="High Correctness",
                description="Correctness problem",
                severity=Severity.HIGH,
                category=EvaluationCategory.CORRECTNESS,
                score=0.3,
                test_case_id="TC-001",
                agent_id="agent-1",
            ),
            Verdict(
                id="V3",
                title="Medium Jailbreak",
                description="Potential jailbreak",
                severity=Severity.MEDIUM,
                category=EvaluationCategory.JAILBREAK,
                score=0.5,
                test_case_id="TC-002",
                agent_id="agent-2",
                governance_principle="safety-002",
            ),
            Verdict(
                id="V4",
                title="Low Helpfulness",
                description="Could be more helpful",
                severity=Severity.LOW,
                category=EvaluationCategory.HELPFULNESS,
                score=0.7,
                test_case_id="TC-002",
                agent_id="agent-2",
            ),
            Verdict(
                id="V5",
                title="Info Finding",
                description="Informational note",
                severity=Severity.INFO,
                category=EvaluationCategory.CORRECTNESS,
                score=0.9,
                test_case_id="TC-003",
                agent_id="agent-1",
            ),
        ]

    def test_filter_by_severity_single(self):
        """Test filtering by single severity."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        critical = summary.filter_by_severity(Severity.CRITICAL)

        assert len(critical) == 1
        assert critical[0].id == "V1"

    def test_filter_by_severity_multiple(self):
        """Test filtering by multiple severities."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        high_severity = summary.filter_by_severity([Severity.CRITICAL, Severity.HIGH])

        assert len(high_severity) == 2
        ids = {v.id for v in high_severity}
        assert ids == {"V1", "V2"}

    def test_filter_by_category_single(self):
        """Test filtering by single category."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        safety = summary.filter_by_category(EvaluationCategory.SAFETY)

        assert len(safety) == 1
        assert safety[0].id == "V1"

    def test_filter_by_category_multiple(self):
        """Test filtering by multiple categories."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        correctness_verdicts = summary.filter_by_category(
            [EvaluationCategory.CORRECTNESS, EvaluationCategory.HELPFULNESS]
        )

        assert len(correctness_verdicts) == 3
        ids = {v.id for v in correctness_verdicts}
        assert ids == {"V2", "V4", "V5"}

    def test_get_failing_verdicts(self):
        """Test getting failing verdicts."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        failing = summary.get_failing_verdicts()

        assert len(failing) == 2
        ids = {v.id for v in failing}
        assert ids == {"V1", "V2"}

    def test_get_governance_violations(self):
        """Test getting governance-linked verdicts."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        gov_verdicts = summary.get_governance_violations()

        assert len(gov_verdicts) == 2
        ids = {v.id for v in gov_verdicts}
        assert ids == {"V1", "V3"}

    def test_query_by_severity(self):
        """Test query with severity filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(severity=Severity.MEDIUM)

        assert len(results) == 1
        assert results[0].id == "V3"

    def test_query_by_category(self):
        """Test query with category filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(category=EvaluationCategory.JAILBREAK)

        assert len(results) == 1
        assert results[0].id == "V3"

    def test_query_by_score_range(self):
        """Test query with score range filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(min_score=0.5, max_score=0.8)

        assert len(results) == 2
        ids = {v.id for v in results}
        assert ids == {"V3", "V4"}

    def test_query_by_test_case_id(self):
        """Test query with test case ID filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(test_case_id="TC-001")

        assert len(results) == 2
        ids = {v.id for v in results}
        assert ids == {"V1", "V2"}

    def test_query_by_agent_id(self):
        """Test query with agent ID filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(agent_id="agent-2")

        assert len(results) == 2
        ids = {v.id for v in results}
        assert ids == {"V3", "V4"}

    def test_query_by_governance_principle(self):
        """Test query with governance principle filter."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(governance_principle="safety-001")

        assert len(results) == 1
        assert results[0].id == "V1"

    def test_query_combined_filters(self):
        """Test query with multiple filters combined."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(
            severity=[Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM],
            agent_id="agent-1",
        )

        # Only V1 and V2 match (critical/high and agent-1)
        assert len(results) == 2
        ids = {v.id for v in results}
        assert ids == {"V1", "V2"}

    def test_query_no_matches(self):
        """Test query with no matching results."""
        summary = VerdictSummary.from_verdicts(self._create_test_verdicts())
        results = summary.query(
            severity=Severity.CRITICAL,
            category=EvaluationCategory.HELPFULNESS,
        )

        assert len(results) == 0

    def test_summary_warning_status(self):
        """Test summary with warning status (medium/low but no critical/high)."""
        verdicts = [
            Verdict(
                id="V1",
                title="Medium",
                description="Medium issue",
                severity=Severity.MEDIUM,
                category=EvaluationCategory.CORRECTNESS,
            ),
            Verdict(
                id="V2",
                title="Low",
                description="Low issue",
                severity=Severity.LOW,
                category=EvaluationCategory.HELPFULNESS,
            ),
        ]
        summary = VerdictSummary.from_verdicts(verdicts)

        assert summary.overall_status == "warning"
        assert summary.medium_count == 1
        assert summary.low_count == 1


class TestEvidenceSerialization:
    """Test Evidence serialization."""

    def test_to_dict(self):
        """Test evidence to_dict method."""
        evidence = Evidence(
            type="tool_call",
            source="turn_3",
            content="Called search API",
            metadata={"tool": "search", "args": {"q": "test"}},
        )
        data = evidence.to_dict()

        assert data["type"] == "tool_call"
        assert data["source"] == "turn_3"
        assert data["content"] == "Called search API"
        assert data["metadata"]["tool"] == "search"
        assert "timestamp" in data


class TestEvaluationContextSerialization:
    """Test EvaluationContext serialization."""

    def test_to_dict(self):
        """Test context to_dict method."""
        context = EvaluationContext(
            input_messages=["Hello", "How are you?"],
            actual_output="I'm doing well!",
            expected_output="Good response",
            tool_calls=[{"name": "greet"}],
            test_case_id="TC-001",
            agent_id="agent-1",
            response_time_ms=100,
        )
        data = context.to_dict()

        assert data["input_messages"] == ["Hello", "How are you?"]
        assert data["actual_output"] == "I'm doing well!"
        assert data["expected_output"] == "Good response"
        assert len(data["tool_calls"]) == 1
        assert data["test_case_id"] == "TC-001"
        assert data["response_time_ms"] == 100
