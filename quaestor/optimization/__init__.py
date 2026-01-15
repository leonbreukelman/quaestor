"""
Quaestor Optimization Engine.

DSPy-powered self-improvement and learning.
"""

from quaestor.optimization.optimizer import (
    OptimizerConfig,
    QuaestorOptimizer,
    quick_optimize,
)
from quaestor.optimization.patterns import (
    BUILTIN_PATTERNS,
    Pattern,
    PatternCategory,
    PatternExtractor,
    PatternLibrary,
    PatternMatch,
    PatternMatcher,
    PatternType,
    get_builtin_library,
)

__all__ = [
    # Optimizer
    "OptimizerConfig",
    "QuaestorOptimizer",
    "quick_optimize",
    # Pattern Learning
    "Pattern",
    "PatternCategory",
    "PatternExtractor",
    "PatternLibrary",
    "PatternMatch",
    "PatternMatcher",
    "PatternType",
    "BUILTIN_PATTERNS",
    "get_builtin_library",
]
