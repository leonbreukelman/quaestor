"""
Quaestor Testing Engine.

Test generation and execution for AI agents.
"""

from quaestor.testing.fixtures import (
    FixtureDefinition,
    FixtureScope,
    FixtureValue,
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
    # Fixture models
    "FixtureDefinition",
    "FixtureScope",
    "FixtureValue",
]
