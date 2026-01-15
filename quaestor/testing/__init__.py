"""
Quaestor Testing Engine.

Test generation and execution for AI agents.
"""

from quaestor.testing.fixtures import (
    CyclicDependencyError,
    DuplicateFixtureError,
    FixtureDefinition,
    FixtureNotFoundError,
    FixtureRegistry,
    FixtureResolver,
    FixtureScope,
    FixtureValue,
    ScopedFixtureManager,
)
from quaestor.testing.models import (
    Assertion,
    AssertionResult,
    AssertionType,
    ContainsAssertion,
    EqualsAssertion,
    RegexAssertion,
    SchemaValidAssertion,
    StateReachedAssertion,
    TestCase,
    TestResult,
    TestSuite,
    ToolCalledAssertion,
    parse_assertion,
)
from quaestor.testing.test_designer import (
    DesignerConfig,
    DesignResult,
    TestDesigner,
    TestScenario,
    TestScenarioType,
    design_tests,
)

__all__ = [
    # Assertion types
    "Assertion",
    "AssertionResult",
    "AssertionType",
    "ContainsAssertion",
    "EqualsAssertion",
    "RegexAssertion",
    "SchemaValidAssertion",
    "StateReachedAssertion",
    "ToolCalledAssertion",
    "parse_assertion",
    # Test case models
    "TestCase",
    "TestResult",
    "TestSuite",
    # Test designer
    "DesignerConfig",
    "DesignResult",
    "TestDesigner",
    "TestScenario",
    "TestScenarioType",
    "design_tests",
    # Fixture models and errors
    "CyclicDependencyError",
    "DuplicateFixtureError",
    "FixtureDefinition",
    "FixtureNotFoundError",
    "FixtureRegistry",
    "FixtureResolver",
    "FixtureScope",
    "FixtureValue",
    "ScopedFixtureManager",
]
