"""
Quaestor - Self-optimizing agentic testing framework.

Built on Smactorio governance infrastructure for deterministic, compliant agent evaluation.

Usage:
    quaestor analyze <path>    # Analyze agent workflow
    quaestor lint <path>       # Static analysis (no LLM)
    quaestor test <path>       # Run tests
    quaestor coverage <path>   # Generate coverage report
"""

__version__ = "0.1.0"
__author__ = "Leon Breukelman"

# Smactorio integration marker - this package requires Smactorio as its core dependency
SMACTORIO_INTEGRATED = True
