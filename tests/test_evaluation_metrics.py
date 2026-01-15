"""
Tests for evaluation metrics.

Part of Phase 4: Evaluation & Judgment.
"""

import builtins

import pytest

from quaestor.evaluation.metrics import (
    ContainsMetric,
    DeepEvalConfig,
    DeepEvalMetric,
    ExactMatchMetric,
    InformationLeakMetric,
    JailbreakDetectionMetric,
    MetricConfig,
    MetricRegistry,
    RefusalDetectionMetric,
    ResponseLengthMetric,
    ResponseTimeMetric,
    ToolUseMetric,
    create_deepeval_registry,
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
        # Use threshold > 0.5 so that 50% scores fail
        return ContainsMetric(
            expected_substrings=["hello", "world"],
            config=MetricConfig(threshold=0.6),
        )

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
        # Use threshold > 0.5 so that 50% scores fail
        return ToolUseMetric(
            expected_tools=["search", "calculator"],
            config=MetricConfig(threshold=0.6),
        )

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

    def test_no_expected_tools_reports_usage(self):
        """Test behavior when no expected tools are configured."""
        metric = ToolUseMetric(expected_tools=[])
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Result",
            tool_calls=[{"tool_name": "search", "arguments": {}}],
        )
        result = metric.evaluate(context)
        assert result.score == 1.0
        assert "tools" in result.reason.lower()


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

    def test_no_response_time_recorded(self):
        """Test missing response time returns neutral score."""
        metric = ResponseTimeMetric()
        context = EvaluationContext(
            input_messages=["Test"],
            actual_output="Quick response",
            response_time_ms=None,
        )
        result = metric.evaluate(context)
        assert result.score == 0.5
        assert "not recorded" in result.reason.lower()


class TestDeepEvalMetric:
    """Tests for DeepEval integration wrapper (mock + import-error paths)."""

    def test_mock_mode(self):
        metric = DeepEvalMetric(
            metric_name="faithfulness",
            deepeval_config=DeepEvalConfig(use_mock=True),
        )
        context = EvaluationContext(input_messages=["Test"], actual_output="Hello")
        result = metric.evaluate(context)

        assert metric.name == "deepeval_faithfulness"
        assert result.details.get("mock") is True
        assert result.score == 0.85

    def test_missing_deepeval_dependency_graceful(self, monkeypatch):
        """When deepeval isn't importable, metric should degrade gracefully."""
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if str(name).startswith("deepeval"):
                raise ImportError("deepeval missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        metric = DeepEvalMetric(
            metric_name="answer_relevancy",
            deepeval_config=DeepEvalConfig(use_mock=False),
        )
        context = EvaluationContext(input_messages=["Test"], actual_output="Hello")
        result = metric.evaluate(context)

        assert result.score == 0.0
        assert "not installed" in result.reason.lower()
        assert result.details.get("error") == "import_error"


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

        jailbreak_metrics = registry.get_by_category(EvaluationCategory.JAILBREAK)
        assert len(jailbreak_metrics) == 1

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
        metrics = list(registry.metrics.keys())

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


class TestDeepEvalConfig:
    """Test DeepEvalConfig dataclass."""

    def test_defaults(self):
        """Test default DeepEval configuration."""
        config = DeepEvalConfig()
        assert config.model == "gpt-4o-mini"
        assert config.threshold == 0.5
        assert config.include_reason is True
        assert config.use_mock is False

    def test_custom_values(self):
        """Test custom DeepEval configuration."""
        config = DeepEvalConfig(model="gpt-4", threshold=0.8, use_mock=True)
        assert config.model == "gpt-4"
        assert config.threshold == 0.8
        assert config.use_mock is True


class TestDeepEvalMetricCategories:
    """Test DeepEval metric category mapping."""

    def test_faithfulness_category(self):
        """Test faithfulness maps to correctness."""
        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.CORRECTNESS

    def test_answer_relevancy_category(self):
        """Test answer relevancy maps to helpfulness."""
        metric = DeepEvalMetric("answer_relevancy", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.HELPFULNESS

    def test_hallucination_category(self):
        """Test hallucination maps to correctness."""
        metric = DeepEvalMetric("hallucination", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.CORRECTNESS

    def test_toxicity_category(self):
        """Test toxicity maps to safety."""
        metric = DeepEvalMetric("toxicity", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.SAFETY

    def test_bias_category(self):
        """Test bias maps to ethics."""
        metric = DeepEvalMetric("bias", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.ETHICS

    def test_unknown_metric_defaults_to_correctness(self):
        """Test unknown metric types default to correctness."""
        metric = DeepEvalMetric("unknown_metric", DeepEvalConfig(use_mock=True))
        assert metric.category == EvaluationCategory.CORRECTNESS


class TestDeepEvalMetricNames:
    """Test DeepEval metric naming."""

    def test_metric_name_format(self):
        """Test metrics get deepeval_ prefix."""
        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=True))
        assert metric.name == "deepeval_faithfulness"

    def test_different_metric_names(self):
        """Test different metric types have correct names."""
        metrics = [
            ("faithfulness", "deepeval_faithfulness"),
            ("answer_relevancy", "deepeval_answer_relevancy"),
            ("hallucination", "deepeval_hallucination"),
            ("toxicity", "deepeval_toxicity"),
            ("bias", "deepeval_bias"),
        ]
        for metric_type, expected_name in metrics:
            metric = DeepEvalMetric(metric_type, DeepEvalConfig(use_mock=True))
            assert metric.name == expected_name


class TestDeepEvalMetricMockEvaluation:
    """Test DeepEval metrics in mock mode."""

    def test_mock_evaluation_returns_consistent_score(self):
        """Test mock mode returns 0.85 score."""
        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=True))
        context = EvaluationContext(input_messages=["Test"], actual_output="Response")
        result = metric.evaluate(context)

        assert result.score == 0.85
        assert result.details["mock"] is True

    def test_mock_evaluation_includes_metric_name(self):
        """Test mock evaluation includes metric type in reason."""
        metric = DeepEvalMetric("answer_relevancy", DeepEvalConfig(use_mock=True))
        context = EvaluationContext(input_messages=["Test"], actual_output="Response")
        result = metric.evaluate(context)

        assert "answer_relevancy" in result.reason.lower()

    def test_mock_evaluation_with_threshold(self):
        """Test mock evaluation respects threshold."""
        config = DeepEvalConfig(use_mock=True, threshold=0.9)
        metric = DeepEvalMetric("faithfulness", config)
        context = EvaluationContext(input_messages=["Test"], actual_output="Response")
        result = metric.evaluate(context)

        # Mock score is 0.85, threshold is 0.9
        assert result.score == 0.85
        assert result.passed is False  # 0.85 < 0.9


class TestDeepEvalMetricImportError:
    """Test DeepEval metric error handling."""

    def test_import_error_handling(self, monkeypatch):
        """Test graceful handling when deepeval is not installed."""
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if str(name).startswith("deepeval"):
                raise ImportError("deepeval not found")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=False))
        context = EvaluationContext(input_messages=["Test"], actual_output="Response")
        result = metric.evaluate(context)

        assert result.score == 0.0
        assert "not installed" in result.reason.lower()
        assert result.details["error"] == "import_error"

    def test_unknown_metric_type_error(self, monkeypatch):
        """Test error handling for unknown metric types."""
        # Only test in non-mock mode error path
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            # Allow deepeval import to test the ValueError path
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        metric = DeepEvalMetric("invalid_metric", DeepEvalConfig(use_mock=False))

        # Accessing the metric should raise ValueError for unknown types
        with pytest.raises(ValueError, match="Unknown DeepEval metric"):
            metric._get_deepeval_metric()


class TestCreateDeepEvalRegistry:
    """Test create_deepeval_registry factory."""

    def test_creates_registry_with_deepeval_metrics(self):
        """Test registry includes DeepEval metrics."""
        registry = create_deepeval_registry()
        metrics = list(registry.metrics.keys())

        # Should have default metrics plus DeepEval metrics
        assert "deepeval_faithfulness" in metrics
        assert "deepeval_answer_relevancy" in metrics
        assert "deepeval_hallucination" in metrics
        assert "deepeval_toxicity" in metrics
        assert "deepeval_bias" in metrics

    def test_includes_default_metrics(self):
        """Test registry includes default metrics too."""
        registry = create_deepeval_registry()
        metrics = list(registry.metrics.keys())

        # Should still have default metrics
        assert "exact_match" in metrics
        assert "response_length" in metrics

    def test_uses_deepeval_config(self):
        """Test that DeepEval config is applied."""
        de_config = DeepEvalConfig(threshold=0.9, model="gpt-4", use_mock=True)
        registry = create_deepeval_registry(deepeval_config=de_config)

        metric = registry.get("deepeval_faithfulness")
        assert metric is not None
        assert metric.deepeval_config.threshold == 0.9
        assert metric.deepeval_config.model == "gpt-4"
        assert metric.deepeval_config.use_mock is True

    def test_uses_base_metric_config(self):
        """Test that base metric config is also applied."""
        config = MetricConfig(threshold=0.8)
        de_config = DeepEvalConfig(use_mock=True)
        registry = create_deepeval_registry(config=config, deepeval_config=de_config)

        # Check a DeepEval metric uses the config
        metric = registry.get("deepeval_faithfulness")
        assert metric is not None
        assert metric.config.threshold == 0.8


class TestDeepEvalMetricLazyLoading:
    """Test DeepEval metric lazy loading behavior."""

    def test_metric_not_loaded_until_used(self):
        """Test that DeepEval metric is not loaded on init."""
        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=True))
        # In mock mode, _deepeval_metric stays None
        assert metric._deepeval_metric is None

    def test_metric_loaded_on_first_access(self):
        """Test that metric is loaded on first _get_deepeval_metric call."""
        metric = DeepEvalMetric("faithfulness", DeepEvalConfig(use_mock=True))
        result = metric._get_deepeval_metric()
        # In mock mode, should return None
        assert result is None
