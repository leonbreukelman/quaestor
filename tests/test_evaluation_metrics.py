"""
Tests for evaluation metrics.

Part of Phase 4: Evaluation & Judgment.
"""

import pytest

from quaestor.evaluation.metrics import (
    BaseMetric,
    ContainsMetric,
    ExactMatchMetric,
    InformationLeakMetric,
    JailbreakDetectionMetric,
    MetricConfig,
    MetricRegistry,
    RefusalDetectionMetric,
    ResponseLengthMetric,
    ResponseTimeMetric,
    ToolUseMetric,
    create_default_registry,
)
from quaestor.evaluation.models import EvaluationCategory, EvaluationContext


class TestMetricConfig:
    """Test MetricConfig dataclass."""

    def test_defaults(self):
        """Test default configuration."""
        config = MetricConfig()
        assert config.threshold == 0.5
        assert config.model == "gpt-4o-mini"
        assert config.use_mock is False

    def test_custom_values(self):
        """Test custom configuration."""
        config = MetricConfig(threshold=0.8, use_mock=True)
        assert config.threshold == 0.8
        assert config.use_mock is True


class TestExactMatchMetric:
    """Test ExactMatchMetric."""

    @pytest.fixture
    def metric(self):
        return ExactMatchMetric()

    def test_name_and_category(self, metric):
        """Test metric metadata."""
        assert metric.name == "exact_match"
        assert metric.category == EvaluationCategory.CORRECTNESS

    def test_exact_match(self, metric):
        """Test exact match returns 1.0."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello World",
            expected_output="Hello World",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True

    def test_no_match(self, metric):
        """Test no match returns 0.0."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello World",
            expected_output="Goodbye World",
        )
        result = metric.evaluate(context)
        assert result.score == 0.0
        assert result.passed is False

    def test_no_expected_output(self, metric):
        """Test when no expected output is set."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello World",
        )
        result = metric.evaluate(context)
        assert result.score == 0.0  # Fails when no expected output
        assert "No expected output" in result.reason


class TestContainsMetric:
    """Test ContainsMetric."""

    @pytest.fixture
    def metric(self):
        return ContainsMetric(expected_substrings=["hello", "world"])

    def test_all_substrings_present(self, metric):
        """Test when all substrings are present."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello World, how are you?",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True

    def test_some_substrings_missing(self, metric):
        """Test when some substrings are missing."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello there!",
        )
        result = metric.evaluate(context)
        assert result.score == 0.5  # 1 of 2 found
        assert result.passed is False  # Below default threshold

    def test_no_substrings_found(self, metric):
        """Test when no substrings are found."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Goodbye!",
        )
        result = metric.evaluate(context)
        assert result.score == 0.0
        assert result.passed is False


class TestToolUseMetric:
    """Test ToolUseMetric."""

    @pytest.fixture
    def metric(self):
        return ToolUseMetric(expected_tools=["search", "calculator"])

    def test_all_expected_tools_used(self, metric):
        """Test when all required tools are used."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Result",
            tool_calls=[
                {"name": "search", "arguments": {}},
                {"name": "calculator", "arguments": {}},
            ],
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True

    def test_missing_expected_tool(self, metric):
        """Test when expected tool is missing."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Result",
            tool_calls=[
                {"name": "search", "arguments": {}},
            ],
        )
        result = metric.evaluate(context)
        assert result.score < 1.0
        assert result.passed is False


class TestRefusalDetectionMetric:
    """Test RefusalDetectionMetric."""

    @pytest.fixture
    def metric(self):
        return RefusalDetectionMetric()

    def test_refusal_detected(self, metric):
        """Test when refusal is detected."""
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="I cannot help with that request.",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True

    def test_no_refusal(self, metric):
        """Test normal response."""
        context = EvaluationContext(
            input_messages=["What is 2+2?"],
            actual_output="2+2 equals 4.",
        )
        result = metric.evaluate(context)
        assert result.score == 0.0


class TestJailbreakDetectionMetric:
    """Test JailbreakDetectionMetric."""

    @pytest.fixture
    def metric(self):
        return JailbreakDetectionMetric()

    def test_normal_response(self, metric):
        """Test normal response has no jailbreak."""
        context = EvaluationContext(
            input_messages=["What is Python?"],
            actual_output="Python is a programming language.",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True


class TestInformationLeakMetric:
    """Test InformationLeakMetric."""

    @pytest.fixture
    def metric(self):
        return InformationLeakMetric()

    def test_no_leak(self, metric):
        """Test response with no sensitive info."""
        context = EvaluationContext(
            input_messages=["Hello"],
            actual_output="Hello! How can I help you today?",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True


class TestResponseLengthMetric:
    """Test ResponseLengthMetric."""

    def test_within_range(self):
        """Test response within acceptable range."""
        metric = ResponseLengthMetric(min_length=10, max_length=100)
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="This is a good response with adequate length.",
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True

    def test_too_short(self):
        """Test response too short."""
        metric = ResponseLengthMetric(min_length=50)
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Short",
        )
        result = metric.evaluate(context)
        assert result.score < 1.0

    def test_too_long(self):
        """Test response too long."""
        metric = ResponseLengthMetric(max_length=20)
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="This is a very long response that exceeds the maximum allowed length.",
        )
        result = metric.evaluate(context)
        assert result.score < 1.0


class TestResponseTimeMetric:
    """Test ResponseTimeMetric."""

    def test_fast_response(self):
        """Test fast response passes."""
        metric = ResponseTimeMetric(target_ms=5000, max_ms=10000)
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Quick response",
            response_time_ms=100.0,
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert result.passed is True


class TestMetricRegistry:
    """Test MetricRegistry."""

    @pytest.fixture
    def registry(self):
        return MetricRegistry()

    def test_register_and_get(self, registry):
        """Test registering and retrieving metrics."""
        metric = ExactMatchMetric()
        registry.register(metric)

        retrieved = registry.get("exact_match")
        assert retrieved is metric

    def test_get_by_category(self, registry):
        """Test getting metrics by category."""
        metric1 = ExactMatchMetric()
        metric2 = ContainsMetric(expected_substrings=["test"])
        metric3 = JailbreakDetectionMetric()

        registry.register(metric1)
        registry.register(metric2)
        registry.register(metric3)

        correctness_metrics = registry.get_by_category(EvaluationCategory.CORRECTNESS)
        assert len(correctness_metrics) == 2

        safety_metrics = registry.get_by_category(EvaluationCategory.SAFETY)
        assert len(safety_metrics) == 1

    def test_evaluate_all(self, registry):
        """Test evaluating all metrics."""
        registry.register(ExactMatchMetric())
        registry.register(ResponseLengthMetric(min_length=5))

        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Hello World",
            expected_output="Hello World",
        )

        results = registry.evaluate_all(context)
        assert len(results) == 2


class TestCreateDefaultRegistry:
    """Test create_default_registry factory."""

    def test_creates_registry_with_metrics(self):
        """Test that default registry has metrics."""
        registry = create_default_registry()
        metrics = list(registry._metrics.keys())

        # Should have core metrics
        assert len(metrics) >= 4

        # Check for expected metrics
        assert "response_length" in metrics
        assert "response_time" in metrics

    def test_uses_config(self):
        """Test that config is applied to metrics."""
        config = MetricConfig(threshold=0.9)
        registry = create_default_registry(config)

        # Metrics should use the threshold
        metric = registry.get("response_length")
        assert metric is not None
