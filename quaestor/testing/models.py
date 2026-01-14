"""
Pydantic models for test generation and execution.

This module provides data models for representing test cases, test suites,
assertions, and test results. Uses discriminated unions for type-safe
assertion handling and supports both JSON and YAML serialization.

Follows patterns established in quaestor.analysis.models.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AssertionType(str, Enum):
    """Types of assertions supported by Quaestor."""

    EQUALS = "equals"
    CONTAINS = "contains"
    REGEX = "regex"
    TOOL_CALLED = "tool_called"
    STATE_REACHED = "state_reached"
    SCHEMA_VALID = "schema_valid"


# =============================================================================
# Assertion Types (Discriminated Union)
# =============================================================================


class AssertionBase(BaseModel):
    """Base class for all assertion types."""

    name: str = Field(..., description="Assertion name for identification")
    description: str = Field(default="", description="Human-readable description")

    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        return yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "AssertionBase":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


class EqualsAssertion(AssertionBase):
    """Assertion that checks exact equality of output."""

    type: Literal[AssertionType.EQUALS] = Field(
        default=AssertionType.EQUALS, description="Assertion type discriminator"
    )
    expected: Any = Field(..., description="Expected value to match exactly")


class ContainsAssertion(AssertionBase):
    """Assertion that checks if output contains a substring."""

    type: Literal[AssertionType.CONTAINS] = Field(
        default=AssertionType.CONTAINS, description="Assertion type discriminator"
    )
    substring: str = Field(..., description="Substring that must be present in output")

    @field_validator("substring")
    @classmethod
    def substring_not_empty(cls, v: str) -> str:
        """Validate that substring is not empty."""
        if not v:
            raise ValueError("Substring must not be empty")
        return v


class RegexAssertion(AssertionBase):
    """Assertion that checks if output matches a regex pattern."""

    type: Literal[AssertionType.REGEX] = Field(
        default=AssertionType.REGEX, description="Assertion type discriminator"
    )
    pattern: str = Field(..., description="Regex pattern to match")
    flags: int = Field(default=0, description="Regex flags (e.g., re.IGNORECASE)")

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate that pattern is a valid regex."""
        try:
            re.compile(v)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e
        return v


class ToolCalledAssertion(AssertionBase):
    """Assertion that verifies a specific tool was called."""

    type: Literal[AssertionType.TOOL_CALLED] = Field(
        default=AssertionType.TOOL_CALLED, description="Assertion type discriminator"
    )
    tool_name: str = Field(..., description="Name of the tool that should be called")
    expected_args: dict[str, Any] | None = Field(
        default=None, description="Expected arguments passed to the tool"
    )

    @field_validator("tool_name")
    @classmethod
    def tool_name_not_empty(cls, v: str) -> str:
        """Validate that tool name is not empty."""
        if not v:
            raise ValueError("Tool name must not be empty")
        return v


class StateReachedAssertion(AssertionBase):
    """Assertion that verifies a specific state was reached."""

    type: Literal[AssertionType.STATE_REACHED] = Field(
        default=AssertionType.STATE_REACHED, description="Assertion type discriminator"
    )
    state_name: str = Field(..., description="Name of the state that should be reached")

    @field_validator("state_name")
    @classmethod
    def state_name_not_empty(cls, v: str) -> str:
        """Validate that state name is not empty."""
        if not v:
            raise ValueError("State name must not be empty")
        return v


class SchemaValidAssertion(AssertionBase):
    """Assertion that validates output against a JSON schema."""

    type: Literal[AssertionType.SCHEMA_VALID] = Field(
        default=AssertionType.SCHEMA_VALID, description="Assertion type discriminator"
    )
    json_schema: dict[str, Any] = Field(
        ..., description="JSON schema to validate against"
    )

    @field_validator("json_schema")
    @classmethod
    def schema_not_empty(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that schema is not empty."""
        if not v:
            raise ValueError("JSON schema must not be empty")
        return v


# Discriminated union type for all assertions
_AssertionUnion = Annotated[
    Union[
        EqualsAssertion,
        ContainsAssertion,
        RegexAssertion,
        ToolCalledAssertion,
        StateReachedAssertion,
        SchemaValidAssertion,
    ],
    Field(discriminator="type"),
]

# Type alias for annotation purposes
Assertion = _AssertionUnion


def parse_assertion(data: dict[str, Any]) -> AssertionBase:
    """
    Parse a dictionary into the appropriate Assertion subtype.

    Uses the 'type' field as a discriminator to determine which
    assertion class to instantiate.

    Args:
        data: Dictionary with assertion data including 'type' field.

    Returns:
        The appropriate Assertion subtype instance.

    Raises:
        ValueError: If the type is unknown or data is invalid.
    """
    from pydantic import TypeAdapter

    adapter: TypeAdapter[AssertionBase] = TypeAdapter(_AssertionUnion)
    result: AssertionBase = adapter.validate_python(data)
    return result


# =============================================================================
# TestCase Model
# =============================================================================


class TestCase(BaseModel):
    """
    A single test case with input, assertions, and optional references.

    Maintains loose coupling to analysis models via optional target_tool
    and target_state fields.
    """

    id: str = Field(..., description="Unique test case identifier")
    name: str = Field(..., description="Human-readable test case name")
    description: str = Field(default="", description="Detailed description of what is tested")
    input: dict[str, Any] = Field(..., description="Input data for the test")
    assertions: list[Assertion] = Field(..., description="Assertions to evaluate")

    # Loose coupling to analysis models
    target_tool: str | None = Field(
        default=None, description="Optional tool being tested (from analysis)"
    )
    target_state: str | None = Field(
        default=None, description="Optional state being tested (from analysis)"
    )

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Tags for filtering/categorization")
    timeout_ms: int | None = Field(default=None, description="Timeout in milliseconds")

    @field_validator("assertions")
    @classmethod
    def at_least_one_assertion(cls, v: list[Assertion]) -> list[Assertion]:
        """Validate that there is at least one assertion."""
        if not v:
            raise ValueError("TestCase must have at least one assertion")
        return v

    @field_validator("timeout_ms")
    @classmethod
    def timeout_positive(cls, v: int | None) -> int | None:
        """Validate that timeout is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        result: str = yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return result

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TestCase":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


# =============================================================================
# TestSuite Model
# =============================================================================


class TestSuite(BaseModel):
    """
    A collection of related test cases.

    Provides organization and batch execution capabilities.
    """

    id: str = Field(..., description="Unique test suite identifier")
    name: str = Field(..., description="Human-readable test suite name")
    description: str = Field(default="", description="Description of the test suite")
    test_cases: list[TestCase] = Field(..., description="Test cases in this suite")

    # Metadata
    tags: list[str] = Field(default_factory=list, description="Tags for filtering/categorization")
    created_at: datetime | None = Field(default=None, description="Creation timestamp")

    @field_validator("test_cases")
    @classmethod
    def at_least_one_test_case(cls, v: list[TestCase]) -> list[TestCase]:
        """Validate that there is at least one test case."""
        if not v:
            raise ValueError("TestSuite must have at least one test case")
        return v

    def add_test_case(self, test_case: TestCase) -> "TestSuite":
        """
        Add a test case to the suite (returns new instance, immutable).

        Args:
            test_case: The test case to add.

        Returns:
            New TestSuite with the added test case.
        """
        return TestSuite(
            id=self.id,
            name=self.name,
            description=self.description,
            test_cases=[*self.test_cases, test_case],
            tags=self.tags,
            created_at=self.created_at,
        )

    def get_test_case(self, test_case_id: str) -> TestCase | None:
        """
        Get a test case by ID.

        Args:
            test_case_id: The ID of the test case to find.

        Returns:
            The test case if found, None otherwise.
        """
        for tc in self.test_cases:
            if tc.id == test_case_id:
                return tc
        return None

    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        result: str = yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return result

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TestSuite":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


# =============================================================================
# TestResult Models
# =============================================================================


class AssertionResult(BaseModel):
    """Result of evaluating a single assertion."""

    assertion_name: str = Field(..., description="Name of the assertion that was evaluated")
    passed: bool = Field(..., description="Whether the assertion passed")
    actual: Any = Field(default=None, description="Actual value that was compared")
    message: str = Field(default="", description="Additional message (e.g., failure reason)")


class TestResult(BaseModel):
    """
    Result of executing a test case.

    The `passed` field is computed from assertion results and error state.
    """

    test_case_id: str = Field(..., description="ID of the test case that was executed")
    assertion_results: list[AssertionResult] = Field(
        default_factory=list, description="Results of individual assertions"
    )
    executed_at: datetime = Field(..., description="When the test was executed")

    # Output and diagnostics
    actual_output: Any = Field(default=None, description="Actual output from the test")
    error_message: str | None = Field(default=None, description="Error message if execution failed")
    duration_ms: int | None = Field(default=None, description="Execution duration in milliseconds")

    @property
    def passed(self) -> bool:
        """
        Compute whether the test passed.

        A test passes if:
        - There was no execution error
        - All assertions passed
        """
        if self.error_message:
            return False
        if not self.assertion_results:
            return False
        return all(ar.passed for ar in self.assertion_results)

    def to_yaml(self) -> str:
        """Serialize to YAML format."""
        # Include computed 'passed' field in output
        data = self.model_dump(mode="json")
        data["passed"] = self.passed
        result: str = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return result

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TestResult":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        # Remove computed field if present (will be recomputed)
        data.pop("passed", None)
        return cls.model_validate(data)
