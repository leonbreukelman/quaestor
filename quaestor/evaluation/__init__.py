"""
Evaluation module for Quaestor.

Provides LLM-as-Judge evaluation with DeepEval metrics.

Part of Phase 4: Evaluation & Judgment.
"""

from quaestor.evaluation.judge import (
    DSPyEvaluator,
    EvaluateCorrectness,
    EvaluateHelpfulness,
    EvaluateSafety,
    GenerateVerdict,
    JudgeConfig,
    MockEvaluator,
    QuaestorJudge,
    evaluate_probe_result,
    quick_evaluate,
)
from quaestor.evaluation.metrics import (
    BaseMetric,
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
from quaestor.evaluation.models import (
    EvaluationCategory,
    EvaluationContext,
    Evidence,
    MetricResult,
    Severity,
    Verdict,
    VerdictSummary,
)

__all__ = [
    # Models
    "Evidence",
    "EvaluationCategory",
    "EvaluationContext",
    "Severity",
    "Verdict",
    "VerdictSummary",
    # Metrics
    "BaseMetric",
    "ContainsMetric",
    "DeepEvalConfig",
    "DeepEvalMetric",
    "ExactMatchMetric",
    "InformationLeakMetric",
    "JailbreakDetectionMetric",
    "MetricConfig",
    "MetricRegistry",
    "MetricResult",
    "RefusalDetectionMetric",
    "ResponseLengthMetric",
    "ResponseTimeMetric",
    "ToolUseMetric",
    "create_deepeval_registry",
    "create_default_registry",
    # Judge
    "DSPyEvaluator",
    "EvaluateCorrectness",
    "EvaluateHelpfulness",
    "EvaluateSafety",
    "GenerateVerdict",
    "JudgeConfig",
    "MockEvaluator",
    "QuaestorJudge",
    "evaluate_probe_result",
    "quick_evaluate",
]
