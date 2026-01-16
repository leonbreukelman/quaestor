"""
DSPy MIPROv2 Optimizer Integration.

Provides automatic prompt optimization for Quaestor DSPy modules using
MIPROv2 (Multi-prompt Instruction Proposal Optimizer).

Part of Phase 6: Self-Optimization.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import dspy
from dspy.teleprompt import MIPROv2

from quaestor.evaluation.models import Verdict


class OptimizerConfig:
    """Configuration for DSPy optimization."""

    def __init__(
        self,
        num_candidates: int = 10,
        init_temperature: float = 1.0,
        num_trials: int = 30,
        minibatch_size: int = 25,
        metric_threshold: float = 0.8,
        verbose: bool = True,
    ):
        """
        Initialize optimizer configuration.

        Args:
            num_candidates: Number of candidate prompts to generate
            init_temperature: Initial temperature for generation
            num_trials: Number of optimization trials
            minibatch_size: Size of minibatch for each trial
            metric_threshold: Success threshold for optimization
            verbose: Enable verbose logging
        """
        self.num_candidates = num_candidates
        self.init_temperature = init_temperature
        self.num_trials = num_trials
        self.minibatch_size = minibatch_size
        self.metric_threshold = metric_threshold
        self.verbose = verbose


class QuaestorOptimizer:
    """
    DSPy optimizer for Quaestor modules.

    Uses MIPROv2 to optimize prompts based on test results and verdicts.
    Supports optimization of WorkflowAnalyzer, TestDesigner, and QuaestorJudge.
    """

    def __init__(
        self,
        config: OptimizerConfig | None = None,
        teacher_model: str | None = None,
    ):
        """
        Initialize Quaestor optimizer.

        Args:
            config: Optimization configuration
            teacher_model: Optional teacher model for MIPROv2 (defaults to current LM)
        """
        self.config = config or OptimizerConfig()
        self.teacher_model = teacher_model
        self._optimizers: dict[str, MIPROv2] = {}

    def create_metric_from_verdicts(self, min_score: float = 0.8) -> Callable:
        """
        Create an optimization metric from verdicts.

        Args:
            min_score: Minimum acceptable score (0-1)

        Returns:
            Metric function for DSPy optimization
        """

        def verdict_metric(example: dspy.Example, prediction: Any, trace=None) -> float:
            """
            Evaluate prediction quality based on verdict scores.

            Args:
                example: Input example with expected output
                prediction: Model prediction
                trace: Optional execution trace

            Returns:
                Score from 0 (worst) to 1 (best)
            """
            # Extract verdict from prediction if present
            if hasattr(prediction, "verdict"):
                verdict = prediction.verdict
                if isinstance(verdict, Verdict):
                    return verdict.score if verdict.score else 0.0

            # Extract score from prediction attributes
            if hasattr(prediction, "score"):
                return float(prediction.score)

            # Default: check if output matches expected
            if hasattr(example, "expected_output") and hasattr(prediction, "output"):
                if prediction.output == example.expected_output:
                    return 1.0
                elif example.expected_output in prediction.output:
                    return 0.8
                else:
                    return 0.5

            # Fallback: neutral score
            return 0.5

        return verdict_metric

    def optimize(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable | None = None,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """
        Optimize a DSPy module using MIPROv2.

        Args:
            module: DSPy module to optimize (e.g., TestDesigner, QuaestorJudge)
            trainset: Training examples with input/output pairs
            metric: Evaluation metric (defaults to verdict-based metric)
            valset: Optional validation set

        Returns:
            Optimized module with improved prompts

        Raises:
            ValueError: If trainset is empty or module is not a DSPy module
        """
        if not trainset:
            raise ValueError("Training set cannot be empty")

        if not isinstance(module, dspy.Module):
            raise ValueError("Module must be a DSPy module")

        # Use verdict metric by default
        if metric is None:
            metric = self.create_metric_from_verdicts(min_score=self.config.metric_threshold)

        # Create or get cached optimizer
        module_name = module.__class__.__name__
        if module_name not in self._optimizers:
            optimizer_kwargs = {
                "metric": metric,
                "num_candidates": self.config.num_candidates,
                "init_temperature": self.config.init_temperature,
                "verbose": self.config.verbose,
            }

            # Add teacher model if specified
            if self.teacher_model:
                optimizer_kwargs["teacher_settings"] = {"lm": dspy.LM(self.teacher_model)}

            self._optimizers[module_name] = MIPROv2(**optimizer_kwargs)

        # Run optimization
        optimizer = self._optimizers[module_name]

        # Prepare kwargs for compile
        compile_kwargs = {
            "student": module,
            "trainset": trainset,
            "num_trials": self.config.num_trials,
            "minibatch_size": min(self.config.minibatch_size, len(trainset)),
        }

        if valset:
            compile_kwargs["valset"] = valset

        optimized_module = optimizer.compile(**compile_kwargs)

        return optimized_module

    def save_optimized_module(self, module: dspy.Module, output_path: str | Path) -> None:
        """
        Save optimized module to disk.

        Args:
            module: Optimized DSPy module
            output_path: Path to save module (e.g., 'optimized_judge.json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        module.save(str(output_path))

    def load_optimized_module(
        self, module_class: type[dspy.Module], input_path: str | Path
    ) -> dspy.Module:
        """
        Load optimized module from disk.

        Args:
            module_class: Class of the module to load
            input_path: Path to saved module

        Returns:
            Loaded optimized module
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Optimized module not found: {input_path}")

        module = module_class()
        module.load(str(input_path))
        return module


def quick_optimize(
    module: dspy.Module,
    examples: list[tuple[dict[str, Any], Any]],
    metric: Callable | None = None,
    **kwargs: Any,
) -> dspy.Module:
    """
    Quick optimization helper for common use cases.

    Args:
        module: DSPy module to optimize
        examples: List of (input_dict, expected_output) tuples
        metric: Optional custom metric
        **kwargs: Additional arguments for OptimizerConfig

    Returns:
        Optimized module

    Example:
        >>> from quaestor.testing.test_designer import TestDesigner
        >>> examples = [
        ...     ({"workflow": "multi-tool agent"}, "test using tool1 and tool2"),
        ...     ({"workflow": "state machine"}, "test state transitions"),
        ... ]
        >>> designer = TestDesigner()
        >>> optimized = quick_optimize(designer, examples)
    """
    # Convert examples to DSPy format
    trainset = []
    for inputs, output in examples:
        example = dspy.Example(**inputs, expected_output=output).with_inputs(*inputs.keys())
        trainset.append(example)

    # Create optimizer and run
    config = OptimizerConfig(**kwargs)
    optimizer = QuaestorOptimizer(config=config)

    return optimizer.optimize(module, trainset, metric=metric)
