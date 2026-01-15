"""
Tests for DSPy MIPROv2 optimizer integration.

Part of Phase 6: Self-Optimization.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import dspy
import pytest

from quaestor.evaluation.models import (
    EvaluationCategory,
    Severity,
    Verdict,
)
from quaestor.optimization.optimizer import (
    OptimizerConfig,
    QuaestorOptimizer,
    quick_optimize,
)


# Test DSPy module for optimization
class SimpleModule(dspy.Module):
    """Simple test module."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        return self.predict(question=question)


class TestOptimizerConfig:
    """Test OptimizerConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.num_candidates == 10
        assert config.init_temperature == 1.0
        assert config.num_trials == 30
        assert config.minibatch_size == 25
        assert config.metric_threshold == 0.8
        assert config.verbose is True

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = OptimizerConfig(
            num_candidates=5,
            init_temperature=0.5,
            num_trials=10,
            minibatch_size=10,
            metric_threshold=0.9,
            verbose=False,
        )

        assert config.num_candidates == 5
        assert config.init_temperature == 0.5
        assert config.num_trials == 10
        assert config.minibatch_size == 10
        assert config.metric_threshold == 0.9
        assert config.verbose is False


class TestQuaestorOptimizer:
    """Test QuaestorOptimizer."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default config."""
        optimizer = QuaestorOptimizer()

        assert optimizer.config is not None
        assert optimizer.config.num_candidates == 10
        assert optimizer.teacher_model is None
        assert len(optimizer._optimizers) == 0

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = OptimizerConfig(num_candidates=5, num_trials=10)
        optimizer = QuaestorOptimizer(config=config)

        assert optimizer.config.num_candidates == 5
        assert optimizer.config.num_trials == 10

    def test_init_with_teacher_model(self) -> None:
        """Test initialization with teacher model."""
        optimizer = QuaestorOptimizer(teacher_model="gpt-4")

        assert optimizer.teacher_model == "gpt-4"

    def test_create_metric_from_verdicts(self) -> None:
        """Test verdict-based metric creation."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts(min_score=0.8)

        assert callable(metric)

        # Test with verdict in prediction
        verdict = Verdict(
            id="test-verdict",
            category=EvaluationCategory.CORRECTNESS,
            severity=Severity.INFO,
            title="Test",
            description="Test verdict",
            score=0.9,
            evidence=[],
        )

        prediction = Mock()
        prediction.verdict = verdict
        example = dspy.Example()

        score = metric(example, prediction)
        assert score == 0.9

    def test_metric_with_score_attribute(self) -> None:
        """Test metric with score attribute."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts()

        prediction = Mock()
        prediction.score = 0.85
        example = dspy.Example()

        score = metric(example, prediction)
        assert score == 0.85

    def test_metric_with_output_match(self) -> None:
        """Test metric with output matching."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts()

        prediction = Mock(spec=["output"])
        prediction.output = "expected output"
        example = dspy.Example(expected_output="expected output")

        score = metric(example, prediction)
        assert score == 1.0

    def test_metric_with_partial_match(self) -> None:
        """Test metric with partial output match."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts()

        prediction = Mock(spec=["output"])
        prediction.output = "This contains expected output here"
        example = dspy.Example(expected_output="expected output")

        score = metric(example, prediction)
        assert score == 0.8

    def test_metric_no_match(self) -> None:
        """Test metric with no match."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts()

        prediction = Mock(spec=["output"])
        prediction.output = "different output"
        example = dspy.Example(expected_output="expected output")

        score = metric(example, prediction)
        assert score == 0.5

    def test_metric_fallback(self) -> None:
        """Test metric fallback to neutral score."""
        optimizer = QuaestorOptimizer()
        metric = optimizer.create_metric_from_verdicts()

        prediction = Mock(spec=[])
        example = dspy.Example()

        score = metric(example, prediction)
        assert score == 0.5

    @patch("quaestor.optimization.optimizer.MIPROv2")
    def test_optimize_basic(self, mock_mipro: MagicMock) -> None:
        """Test basic optimization."""
        # Setup mocks
        mock_optimizer = MagicMock()
        mock_mipro.return_value = mock_optimizer

        optimized_module = SimpleModule()
        mock_optimizer.compile.return_value = optimized_module

        # Create training data
        trainset = [
            dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
            dspy.Example(question="What is 3+3?", answer="6").with_inputs("question"),
        ]

        # Run optimization
        optimizer = QuaestorOptimizer()
        module = SimpleModule()
        result = optimizer.optimize(module, trainset)

        # Verify
        assert result is optimized_module
        mock_mipro.assert_called_once()
        mock_optimizer.compile.assert_called_once()

    def test_optimize_empty_trainset(self) -> None:
        """Test optimization with empty trainset."""
        optimizer = QuaestorOptimizer()
        module = SimpleModule()

        with pytest.raises(ValueError, match="Training set cannot be empty"):
            optimizer.optimize(module, [])

    def test_optimize_invalid_module(self) -> None:
        """Test optimization with non-DSPy module."""
        optimizer = QuaestorOptimizer()
        trainset = [dspy.Example(question="test").with_inputs("question")]

        with pytest.raises(ValueError, match="Module must be a DSPy module"):
            optimizer.optimize("not a module", trainset)  # type: ignore

    @patch("quaestor.optimization.optimizer.MIPROv2")
    def test_optimize_with_custom_metric(self, mock_mipro: MagicMock) -> None:
        """Test optimization with custom metric."""
        mock_optimizer = MagicMock()
        mock_mipro.return_value = mock_optimizer
        mock_optimizer.compile.return_value = SimpleModule()

        def custom_metric(example, prediction, trace=None) -> float:
            return 1.0

        trainset = [dspy.Example(question="test").with_inputs("question")]

        optimizer = QuaestorOptimizer()
        module = SimpleModule()
        optimizer.optimize(module, trainset, metric=custom_metric)

        # Verify custom metric was used
        call_kwargs = mock_mipro.call_args[1]
        assert call_kwargs["metric"] == custom_metric

    @patch("quaestor.optimization.optimizer.MIPROv2")
    def test_optimize_with_validation_set(self, mock_mipro: MagicMock) -> None:
        """Test optimization with validation set."""
        mock_optimizer = MagicMock()
        mock_mipro.return_value = mock_optimizer
        mock_optimizer.compile.return_value = SimpleModule()

        trainset = [dspy.Example(question="train").with_inputs("question")]
        valset = [dspy.Example(question="val").with_inputs("question")]

        optimizer = QuaestorOptimizer()
        module = SimpleModule()
        optimizer.optimize(module, trainset, valset=valset)

        # Verify valset was passed
        compile_call = mock_optimizer.compile.call_args[1]
        assert compile_call["valset"] == valset

    @patch("quaestor.optimization.optimizer.MIPROv2")
    def test_optimize_caches_optimizer(self, mock_mipro: MagicMock) -> None:
        """Test that optimizer instances are cached."""
        mock_optimizer = MagicMock()
        mock_mipro.return_value = mock_optimizer
        mock_optimizer.compile.return_value = SimpleModule()

        trainset = [dspy.Example(question="test").with_inputs("question")]

        optimizer = QuaestorOptimizer()
        module1 = SimpleModule()
        module2 = SimpleModule()

        optimizer.optimize(module1, trainset)
        optimizer.optimize(module2, trainset)

        # Should only create one MIPROv2 instance for SimpleModule
        assert mock_mipro.call_count == 1
        assert "SimpleModule" in optimizer._optimizers

    @patch("quaestor.optimization.optimizer.MIPROv2")
    def test_optimize_with_teacher_model(self, mock_mipro: MagicMock) -> None:
        """Test optimization with teacher model."""
        mock_optimizer = MagicMock()
        mock_mipro.return_value = mock_optimizer
        mock_optimizer.compile.return_value = SimpleModule()

        trainset = [dspy.Example(question="test").with_inputs("question")]

        optimizer = QuaestorOptimizer(teacher_model="gpt-4")
        module = SimpleModule()
        optimizer.optimize(module, trainset)

        # Verify teacher settings were passed
        call_kwargs = mock_mipro.call_args[1]
        assert "teacher_settings" in call_kwargs

    def test_save_and_load_optimized_module(self) -> None:
        """Test saving and loading optimized modules."""
        with TemporaryDirectory() as tmpdir:
            module = SimpleModule()
            save_path = Path(tmpdir) / "optimized" / "test_module.json"

            optimizer = QuaestorOptimizer()

            # Test save
            with patch.object(module, "save") as mock_save:
                optimizer.save_optimized_module(module, save_path)
                assert save_path.parent.exists()
                mock_save.assert_called_once_with(str(save_path))

    def test_load_nonexistent_module(self) -> None:
        """Test loading non-existent module."""
        optimizer = QuaestorOptimizer()

        with pytest.raises(FileNotFoundError, match="Optimized module not found"):
            optimizer.load_optimized_module(SimpleModule, "/nonexistent/path.json")


class TestQuickOptimize:
    """Test quick_optimize helper function."""

    @patch("quaestor.optimization.optimizer.QuaestorOptimizer")
    def test_basic_usage(self, mock_optimizer_class: MagicMock) -> None:
        """Test basic quick_optimize usage."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        optimized_module = SimpleModule()
        mock_optimizer.optimize.return_value = optimized_module

        examples = [
            ({"question": "What is 2+2?"}, "4"),
            ({"question": "What is 3+3?"}, "6"),
        ]

        module = SimpleModule()
        result = quick_optimize(module, examples)

        assert result is optimized_module
        mock_optimizer.optimize.assert_called_once()

        # Verify trainset was converted correctly
        call_args = mock_optimizer.optimize.call_args[0]
        trainset = call_args[1]
        assert len(trainset) == 2
        assert all(isinstance(ex, dspy.Example) for ex in trainset)

    @patch("quaestor.optimization.optimizer.QuaestorOptimizer")
    def test_with_custom_config(self, mock_optimizer_class: MagicMock) -> None:
        """Test quick_optimize with custom config."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = SimpleModule()

        examples = [({"question": "test"}, "answer")]
        module = SimpleModule()

        quick_optimize(
            module,
            examples,
            num_candidates=5,
            num_trials=10,
            verbose=False,
        )

        # Verify config was created with custom values
        optimizer_init_call = mock_optimizer_class.call_args
        config = optimizer_init_call[1]["config"]
        assert config.num_candidates == 5
        assert config.num_trials == 10
        assert config.verbose is False

    @patch("quaestor.optimization.optimizer.QuaestorOptimizer")
    def test_with_custom_metric(self, mock_optimizer_class: MagicMock) -> None:
        """Test quick_optimize with custom metric."""
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer
        mock_optimizer.optimize.return_value = SimpleModule()

        def custom_metric(example, prediction, trace=None) -> float:
            return 1.0

        examples = [({"question": "test"}, "answer")]
        module = SimpleModule()

        quick_optimize(module, examples, metric=custom_metric)

        # Verify metric was passed through
        optimize_call = mock_optimizer.optimize.call_args
        assert optimize_call[1]["metric"] == custom_metric
