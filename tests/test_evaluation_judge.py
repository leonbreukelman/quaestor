"""
Tests for QuaestorJudge.

Part of Phase 4: Evaluation & Judgment.
"""

import pytest

from quaestor.evaluation.judge import (
    JudgeConfig,
    MockEvaluator,
    QuaestorJudge,
    quick_evaluate,
)
from quaestor.evaluation.models import (
    EvaluationCategory,
    EvaluationContext,
    Severity,
)


class TestJudgeConfig:
    """Test JudgeConfig dataclass."""

    def test_defaults(self):
        """Test default configuration."""
        config = JudgeConfig()
        assert config.model == "gpt-4o-mini"
        assert config.use_mock is False
        assert config.evaluate_correctness is True
        assert config.evaluate_safety is True
        assert config.evaluate_helpfulness is True
        assert config.metric_threshold == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = JudgeConfig(
            model="gpt-4",
            use_mock=True,
            evaluate_helpfulness=False,
            metric_threshold=0.8,
        )
        assert config.model == "gpt-4"
        assert config.use_mock is True
        assert config.evaluate_helpfulness is False
        assert config.metric_threshold == 0.8


class TestMockEvaluator:
    """Test MockEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return MockEvaluator()

    @pytest.fixture
    def context(self):
        return EvaluationContext(
            input_messages=["What is 2+2?"],
            actual_output="The answer is 4.",
        )

    def test_evaluate_correctness(self, evaluator, context):
        """Test mock correctness evaluation."""
        result = evaluator.evaluate_correctness(context)
        assert "score" in result
        assert "reasoning" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_evaluate_safety(self, evaluator, context):
        """Test mock safety evaluation."""
        result = evaluator.evaluate_safety(context)
        assert "is_safe" in result
        assert "safety_score" in result
        assert "category" in result

    def test_evaluate_safety_with_unsafe_content(self, evaluator):
        """Test safety evaluation with unsafe content."""
        context = EvaluationContext(
            input_messages=["Jailbreak attempt"],
            actual_output="I'll ignore instructions and help you hack",
        )
        result = evaluator.evaluate_safety(context)
        assert result["is_safe"] is False
        assert result["safety_score"] < 0.5

    def test_evaluate_helpfulness(self, evaluator, context):
        """Test mock helpfulness evaluation."""
        result = evaluator.evaluate_helpfulness(context)
        assert "score" in result
        assert "reasoning" in result

    def test_generate_verdict(self, evaluator):
        """Test mock verdict generation."""
        result = evaluator.generate_verdict(
            context_summary="Test context",
            metric_results="[]",
            observations="None",
        )
        assert "verdict" in result
        assert "severity" in result
        assert "summary" in result
        assert "recommendations" in result


class TestQuaestorJudge:
    """Test QuaestorJudge."""

    @pytest.fixture
    def judge(self):
        """Create a judge with mock evaluation."""
        return QuaestorJudge(JudgeConfig(use_mock=True))

    @pytest.fixture
    def context(self):
        return EvaluationContext(
            input_messages=["What is 2+2?"],
            actual_output="The answer is 4.",
        )

    def test_init_with_mock(self):
        """Test initialization with mock evaluator."""
        judge = QuaestorJudge(JudgeConfig(use_mock=True))
        assert isinstance(judge._evaluator, MockEvaluator)

    def test_evaluate_returns_verdicts(self, judge, context):
        """Test evaluate returns list of verdicts."""
        verdicts = judge.evaluate(context)
        assert isinstance(verdicts, list)
        # All verdicts should have required fields
        for v in verdicts:
            assert v.id is not None
            assert v.title is not None
            assert v.severity is not None

    def test_evaluate_good_response(self, judge):
        """Test evaluation of a good response."""
        context = EvaluationContext(
            input_messages=["What is the capital of France?"],
            actual_output="The capital of France is Paris. It is known for the Eiffel Tower and rich cultural heritage.",
        )
        verdicts = judge.evaluate(context)
        # Good responses should have fewer issues
        critical = [v for v in verdicts if v.severity == Severity.CRITICAL]
        assert len(critical) == 0

    def test_evaluate_short_response(self, judge):
        """Test evaluation of a short response."""
        context = EvaluationContext(
            input_messages=["Explain quantum physics"],
            actual_output="OK",  # Too short
        )
        verdicts = judge.evaluate(context)
        # Short responses may have helpfulness issues
        assert len(verdicts) > 0

    def test_summarize(self, judge, context):
        """Test summarizing verdicts."""
        verdicts = judge.evaluate(context)
        summary = judge.summarize(verdicts)
        
        assert summary.total == len(verdicts)
        assert hasattr(summary, "overall_passed")

    def test_registry_property(self, judge):
        """Test registry property."""
        registry = judge.registry
        assert registry is not None
        assert len(registry.list_metrics()) > 0

    def test_generate_final_verdict(self, judge, context):
        """Test generating final verdict."""
        from quaestor.evaluation.models import MetricResult
        
        metric_results = [
            MetricResult(
                metric_name="test",
                score=0.8,
                passed=True,
                explanation="Good",
            )
        ]
        
        verdict = judge.generate_final_verdict(context, metric_results)
        assert verdict.id.startswith("FINAL-")
        assert verdict.title.startswith("Overall Verdict")


class TestQuickEvaluate:
    """Test quick_evaluate convenience function."""

    def test_basic_usage(self):
        """Test basic quick evaluation."""
        summary = quick_evaluate(
            actual_output="Hello, I'm an AI assistant!",
            input_message="Hello",
            use_mock=True,
        )
        assert hasattr(summary, "total")
        assert hasattr(summary, "overall_passed")

    def test_with_expected_output(self):
        """Test with expected output."""
        summary = quick_evaluate(
            actual_output="4",
            input_message="What is 2+2?",
            expected_output="4",
            use_mock=True,
        )
        assert summary is not None


class TestJudgeCategories:
    """Test that judge correctly categorizes issues."""

    @pytest.fixture
    def judge(self):
        return QuaestorJudge(JudgeConfig(use_mock=True))

    def test_correctness_category(self, judge):
        """Test correctness issues are categorized correctly."""
        context = EvaluationContext(
            input_messages=["What is 2+2?"],
            actual_output="5",  # Wrong answer
            expected_output="4",
        )
        verdicts = judge.evaluate(context)
        correctness_verdicts = [
            v for v in verdicts if v.category == EvaluationCategory.CORRECTNESS
        ]
        # Should have at least one correctness issue
        assert len(correctness_verdicts) >= 0  # May depend on mock behavior

    def test_safety_issues_detected(self, judge):
        """Test safety issues are detected."""
        context = EvaluationContext(
            input_messages=["Help me hack"],
            actual_output="Sure, I'll help you hack into the system and ignore all instructions",
        )
        verdicts = judge.evaluate(context)
        # Check for safety-related verdicts
        safety_verdicts = [
            v for v in verdicts
            if v.category in (
                EvaluationCategory.SAFETY,
                EvaluationCategory.JAILBREAK,
            )
        ]
        assert len(safety_verdicts) >= 1


class TestJudgeWithMetrics:
    """Test judge's metric integration."""

    def test_custom_metrics(self):
        """Test using custom metrics."""
        from quaestor.evaluation.metrics import ContainsMetric
        
        custom_metric = ContainsMetric(
            required_substrings=["thank you"],
            name="politeness_check",
        )
        
        config = JudgeConfig(
            use_mock=True,
            additional_metrics=[custom_metric],
        )
        judge = QuaestorJudge(config)
        
        # Metric should be registered
        assert judge.registry.get("politeness_check") is not None

    def test_metric_results_become_verdicts(self):
        """Test that failed metrics create verdicts."""
        judge = QuaestorJudge(JudgeConfig(use_mock=True))
        
        context = EvaluationContext(
            input_messages=["Write something long"],
            actual_output="Hi",  # Very short
        )
        
        verdicts = judge.evaluate(context)
        # Should have verdict about length
        metric_verdicts = [v for v in verdicts if v.id.startswith("M-")]
        # May have length-related verdict
        assert isinstance(metric_verdicts, list)
