"""
Tests for testing module Pydantic models.

Tests cover:
- Assertion types (discriminated union)
- TestCase model with validation
- TestSuite model with validation
- TestResult model with computed fields
- JSON/YAML serialization round-trips
"""

import json
import re
from datetime import UTC, datetime

import pytest
import yaml

from quaestor.testing.models import (
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

# =============================================================================
# Assertion Types Tests (Discriminated Union)
# =============================================================================


class TestEqualsAssertion:
    """Tests for EqualsAssertion model."""

    def test_create_equals_assertion(self):
        """Test creating a valid equals assertion."""
        assertion = EqualsAssertion(
            name="check_output",
            expected="hello world",
        )
        assert assertion.type == AssertionType.EQUALS
        assert assertion.name == "check_output"
        assert assertion.expected == "hello world"

    def test_equals_assertion_with_description(self):
        """Test equals assertion with optional description."""
        assertion = EqualsAssertion(
            name="check_status",
            expected=200,
            description="Status code should be 200",
        )
        assert assertion.description == "Status code should be 200"

    def test_equals_assertion_with_complex_expected(self):
        """Test equals assertion with dict/list expected values."""
        assertion = EqualsAssertion(
            name="check_response",
            expected={"status": "ok", "data": [1, 2, 3]},
        )
        assert assertion.expected == {"status": "ok", "data": [1, 2, 3]}


class TestContainsAssertion:
    """Tests for ContainsAssertion model."""

    def test_create_contains_assertion(self):
        """Test creating a valid contains assertion."""
        assertion = ContainsAssertion(
            name="check_message",
            substring="success",
        )
        assert assertion.type == AssertionType.CONTAINS
        assert assertion.substring == "success"

    def test_contains_assertion_empty_substring_fails(self):
        """Test that empty substring is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            ContainsAssertion(name="bad", substring="")


class TestRegexAssertion:
    """Tests for RegexAssertion model."""

    def test_create_regex_assertion(self):
        """Test creating a valid regex assertion."""
        assertion = RegexAssertion(
            name="check_email",
            pattern=r"[\w.]+@[\w.]+\.\w+",
        )
        assert assertion.type == AssertionType.REGEX
        assert assertion.pattern == r"[\w.]+@[\w.]+\.\w+"

    def test_regex_assertion_invalid_pattern_fails(self):
        """Test that invalid regex pattern is rejected."""
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            RegexAssertion(name="bad", pattern=r"[invalid")

    def test_regex_assertion_with_flags(self):
        """Test regex assertion with flags."""
        assertion = RegexAssertion(
            name="case_insensitive",
            pattern=r"hello",
            flags=re.IGNORECASE,
        )
        assert assertion.flags == re.IGNORECASE


class TestToolCalledAssertion:
    """Tests for ToolCalledAssertion model."""

    def test_create_tool_called_assertion(self):
        """Test creating a valid tool_called assertion."""
        assertion = ToolCalledAssertion(
            name="check_search",
            tool_name="search_database",
        )
        assert assertion.type == AssertionType.TOOL_CALLED
        assert assertion.tool_name == "search_database"

    def test_tool_called_with_expected_args(self):
        """Test tool_called assertion with expected arguments."""
        assertion = ToolCalledAssertion(
            name="check_search_args",
            tool_name="search_database",
            expected_args={"query": "test", "limit": 10},
        )
        assert assertion.expected_args == {"query": "test", "limit": 10}

    def test_tool_called_empty_name_fails(self):
        """Test that empty tool name is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            ToolCalledAssertion(name="bad", tool_name="")


class TestStateReachedAssertion:
    """Tests for StateReachedAssertion model."""

    def test_create_state_reached_assertion(self):
        """Test creating a valid state_reached assertion."""
        assertion = StateReachedAssertion(
            name="check_complete",
            state_name="completed",
        )
        assert assertion.type == AssertionType.STATE_REACHED
        assert assertion.state_name == "completed"

    def test_state_reached_empty_name_fails(self):
        """Test that empty state name is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            StateReachedAssertion(name="bad", state_name="")


class TestSchemaValidAssertion:
    """Tests for SchemaValidAssertion model."""

    def test_create_schema_valid_assertion(self):
        """Test creating a valid schema_valid assertion."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        assertion = SchemaValidAssertion(
            name="check_response_schema",
            json_schema=schema,
        )
        assert assertion.type == AssertionType.SCHEMA_VALID
        assert assertion.json_schema == schema

    def test_schema_valid_empty_schema_fails(self):
        """Test that empty schema is rejected."""
        with pytest.raises(ValueError, match="must not be empty"):
            SchemaValidAssertion(name="bad", json_schema={})


class TestAssertionDiscriminatedUnion:
    """Tests for Assertion discriminated union type."""

    def test_parse_equals_assertion(self):
        """Test parsing equals assertion from dict."""
        data = {"type": "equals", "name": "test", "expected": "value"}
        assertion = parse_assertion(data)
        assert isinstance(assertion, EqualsAssertion)

    def test_parse_contains_assertion(self):
        """Test parsing contains assertion from dict."""
        data = {"type": "contains", "name": "test", "substring": "value"}
        assertion = parse_assertion(data)
        assert isinstance(assertion, ContainsAssertion)

    def test_parse_regex_assertion(self):
        """Test parsing regex assertion from dict."""
        data = {"type": "regex", "name": "test", "pattern": r"\d+"}
        assertion = parse_assertion(data)
        assert isinstance(assertion, RegexAssertion)

    def test_parse_tool_called_assertion(self):
        """Test parsing tool_called assertion from dict."""
        data = {"type": "tool_called", "name": "test", "tool_name": "search"}
        assertion = parse_assertion(data)
        assert isinstance(assertion, ToolCalledAssertion)

    def test_parse_state_reached_assertion(self):
        """Test parsing state_reached assertion from dict."""
        data = {"type": "state_reached", "name": "test", "state_name": "done"}
        assertion = parse_assertion(data)
        assert isinstance(assertion, StateReachedAssertion)

    def test_parse_schema_valid_assertion(self):
        """Test parsing schema_valid assertion from dict."""
        data = {
            "type": "schema_valid",
            "name": "test",
            "json_schema": {"type": "string"},
        }
        assertion = parse_assertion(data)
        assert isinstance(assertion, SchemaValidAssertion)

    def test_parse_invalid_type_fails(self):
        """Test that invalid assertion type is rejected."""
        data = {"type": "invalid_type", "name": "test"}
        with pytest.raises(ValueError):
            parse_assertion(data)


# =============================================================================
# TestCase Model Tests
# =============================================================================


class TestTestCase:
    """Tests for TestCase model."""

    def test_create_minimal_test_case(self):
        """Test creating a test case with minimum required fields."""
        test_case = TestCase(
            id="TC-001",
            name="Test basic functionality",
            input={"query": "hello"},
            assertions=[EqualsAssertion(name="check", expected="world")],
        )
        assert test_case.id == "TC-001"
        assert test_case.name == "Test basic functionality"
        assert test_case.input == {"query": "hello"}
        assert len(test_case.assertions) == 1

    def test_create_full_test_case(self):
        """Test creating a test case with all fields."""
        test_case = TestCase(
            id="TC-002",
            name="Full test case",
            description="A comprehensive test case",
            input={"data": "value"},
            assertions=[
                EqualsAssertion(name="equals_check", expected="result"),
                ContainsAssertion(name="contains_check", substring="res"),
            ],
            target_tool="process_data",
            target_state="processing",
            tags=["integration", "slow"],
            timeout_ms=5000,
        )
        assert test_case.description == "A comprehensive test case"
        assert test_case.target_tool == "process_data"
        assert test_case.target_state == "processing"
        assert test_case.tags == ["integration", "slow"]
        assert test_case.timeout_ms == 5000
        assert len(test_case.assertions) == 2

    def test_test_case_empty_assertions_fails(self):
        """Test that empty assertions list is rejected."""
        with pytest.raises(ValueError, match="at least one assertion"):
            TestCase(
                id="TC-BAD",
                name="No assertions",
                input={},
                assertions=[],
            )

    def test_test_case_negative_timeout_fails(self):
        """Test that negative timeout is rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            TestCase(
                id="TC-BAD",
                name="Negative timeout",
                input={},
                assertions=[EqualsAssertion(name="check", expected="x")],
                timeout_ms=-100,
            )


# =============================================================================
# TestSuite Model Tests
# =============================================================================


class TestTestSuite:
    """Tests for TestSuite model."""

    def test_create_minimal_test_suite(self):
        """Test creating a test suite with minimum required fields."""
        tc = TestCase(
            id="TC-001",
            name="Test 1",
            input={},
            assertions=[EqualsAssertion(name="check", expected="x")],
        )
        suite = TestSuite(
            id="TS-001",
            name="Basic Suite",
            test_cases=[tc],
        )
        assert suite.id == "TS-001"
        assert suite.name == "Basic Suite"
        assert len(suite.test_cases) == 1

    def test_create_full_test_suite(self):
        """Test creating a test suite with all fields."""
        tc1 = TestCase(
            id="TC-001",
            name="Test 1",
            input={},
            assertions=[EqualsAssertion(name="check", expected="x")],
        )
        tc2 = TestCase(
            id="TC-002",
            name="Test 2",
            input={},
            assertions=[ContainsAssertion(name="check", substring="y")],
        )
        suite = TestSuite(
            id="TS-002",
            name="Full Suite",
            description="A comprehensive test suite",
            test_cases=[tc1, tc2],
            tags=["smoke", "regression"],
            created_at=datetime(2026, 1, 14, tzinfo=UTC),
        )
        assert suite.description == "A comprehensive test suite"
        assert len(suite.test_cases) == 2
        assert suite.tags == ["smoke", "regression"]
        assert suite.created_at == datetime(2026, 1, 14, tzinfo=UTC)

    def test_test_suite_empty_test_cases_fails(self):
        """Test that empty test_cases list is rejected."""
        with pytest.raises(ValueError, match="at least one test case"):
            TestSuite(
                id="TS-BAD",
                name="Empty Suite",
                test_cases=[],
            )

    def test_test_suite_add_test_case(self):
        """Test adding a test case to a suite."""
        tc1 = TestCase(
            id="TC-001",
            name="Test 1",
            input={},
            assertions=[EqualsAssertion(name="check", expected="x")],
        )
        suite = TestSuite(id="TS-001", name="Suite", test_cases=[tc1])
        tc2 = TestCase(
            id="TC-002",
            name="Test 2",
            input={},
            assertions=[EqualsAssertion(name="check", expected="y")],
        )
        new_suite = suite.add_test_case(tc2)
        assert len(new_suite.test_cases) == 2
        # Original suite unchanged (immutable)
        assert len(suite.test_cases) == 1

    def test_test_suite_get_test_case_by_id(self):
        """Test getting a test case by ID."""
        tc = TestCase(
            id="TC-001",
            name="Test 1",
            input={},
            assertions=[EqualsAssertion(name="check", expected="x")],
        )
        suite = TestSuite(id="TS-001", name="Suite", test_cases=[tc])
        found = suite.get_test_case("TC-001")
        assert found is not None
        assert found.id == "TC-001"

    def test_test_suite_get_test_case_not_found(self):
        """Test getting a non-existent test case returns None."""
        tc = TestCase(
            id="TC-001",
            name="Test 1",
            input={},
            assertions=[EqualsAssertion(name="check", expected="x")],
        )
        suite = TestSuite(id="TS-001", name="Suite", test_cases=[tc])
        assert suite.get_test_case("TC-999") is None


# =============================================================================
# TestResult Model Tests
# =============================================================================


class TestTestResult:
    """Tests for TestResult model."""

    def test_create_passing_result(self):
        """Test creating a passing test result."""
        result = TestResult(
            test_case_id="TC-001",
            assertion_results=[
                AssertionResult(assertion_name="check1", passed=True),
                AssertionResult(assertion_name="check2", passed=True),
            ],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
        )
        assert result.passed is True
        assert result.test_case_id == "TC-001"
        assert len(result.assertion_results) == 2

    def test_create_failing_result(self):
        """Test creating a failing test result."""
        result = TestResult(
            test_case_id="TC-002",
            assertion_results=[
                AssertionResult(assertion_name="check1", passed=True),
                AssertionResult(
                    assertion_name="check2",
                    passed=False,
                    actual="wrong",
                    message="Expected 'right' but got 'wrong'",
                ),
            ],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
        )
        assert result.passed is False

    def test_create_result_with_error(self):
        """Test creating a result with execution error."""
        result = TestResult(
            test_case_id="TC-003",
            assertion_results=[],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
            error_message="Connection timeout",
        )
        assert result.passed is False
        assert result.error_message == "Connection timeout"

    def test_result_with_actual_output(self):
        """Test result with actual output captured."""
        result = TestResult(
            test_case_id="TC-004",
            assertion_results=[AssertionResult(assertion_name="check", passed=True)],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
            actual_output={"response": "success"},
            duration_ms=150,
        )
        assert result.actual_output == {"response": "success"}
        assert result.duration_ms == 150


# =============================================================================
# Serialization Tests (JSON & YAML)
# =============================================================================


class TestJsonSerialization:
    """Tests for JSON serialization/deserialization."""

    def test_assertion_json_roundtrip(self):
        """Test JSON roundtrip for assertion."""
        assertion = EqualsAssertion(name="test", expected={"key": "value"})
        json_str = assertion.model_dump_json()
        parsed = json.loads(json_str)
        restored = parse_assertion(parsed)
        assert isinstance(restored, EqualsAssertion)
        assert restored.expected == {"key": "value"}

    def test_test_case_json_roundtrip(self):
        """Test JSON roundtrip for TestCase."""
        tc = TestCase(
            id="TC-001",
            name="Test",
            input={"query": "hello"},
            assertions=[
                EqualsAssertion(name="eq", expected="world"),
                ContainsAssertion(name="cont", substring="wor"),
            ],
            target_tool="search",
        )
        json_str = tc.model_dump_json()
        parsed = json.loads(json_str)
        restored = TestCase.model_validate(parsed)
        assert restored.id == "TC-001"
        assert len(restored.assertions) == 2
        assert restored.target_tool == "search"

    def test_test_suite_json_roundtrip(self):
        """Test JSON roundtrip for TestSuite."""
        tc = TestCase(
            id="TC-001",
            name="Test",
            input={},
            assertions=[EqualsAssertion(name="eq", expected="x")],
        )
        suite = TestSuite(
            id="TS-001",
            name="Suite",
            test_cases=[tc],
            tags=["smoke"],
        )
        json_str = suite.model_dump_json()
        parsed = json.loads(json_str)
        restored = TestSuite.model_validate(parsed)
        assert restored.id == "TS-001"
        assert len(restored.test_cases) == 1

    def test_test_result_json_roundtrip(self):
        """Test JSON roundtrip for TestResult."""
        result = TestResult(
            test_case_id="TC-001",
            assertion_results=[AssertionResult(assertion_name="check", passed=True)],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
            duration_ms=100,
        )
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        restored = TestResult.model_validate(parsed)
        assert restored.passed is True
        assert restored.duration_ms == 100


class TestYamlSerialization:
    """Tests for YAML serialization/deserialization."""

    def test_assertion_yaml_roundtrip(self):
        """Test YAML roundtrip for assertion."""
        assertion = EqualsAssertion(name="test", expected={"key": "value"})
        yaml_str = assertion.to_yaml()
        assert "type: equals" in yaml_str
        restored = EqualsAssertion.from_yaml(yaml_str)
        assert restored.expected == {"key": "value"}

    def test_test_case_yaml_roundtrip(self):
        """Test YAML roundtrip for TestCase."""
        tc = TestCase(
            id="TC-001",
            name="Test Case",
            description="A test case in YAML",
            input={"query": "hello"},
            assertions=[EqualsAssertion(name="eq", expected="world")],
        )
        yaml_str = tc.to_yaml()
        assert "id: TC-001" in yaml_str
        assert "query: hello" in yaml_str
        restored = TestCase.from_yaml(yaml_str)
        assert restored.id == "TC-001"
        assert restored.description == "A test case in YAML"

    def test_test_suite_yaml_roundtrip(self):
        """Test YAML roundtrip for TestSuite."""
        tc = TestCase(
            id="TC-001",
            name="Test",
            input={},
            assertions=[EqualsAssertion(name="eq", expected="x")],
        )
        suite = TestSuite(
            id="TS-001",
            name="YAML Suite",
            description="Test suite in YAML format",
            test_cases=[tc],
        )
        yaml_str = suite.to_yaml()
        assert "id: TS-001" in yaml_str
        assert "YAML Suite" in yaml_str
        restored = TestSuite.from_yaml(yaml_str)
        assert restored.id == "TS-001"
        assert len(restored.test_cases) == 1

    def test_test_result_yaml_roundtrip(self):
        """Test YAML roundtrip for TestResult."""
        result = TestResult(
            test_case_id="TC-001",
            assertion_results=[AssertionResult(assertion_name="check", passed=True)],
            executed_at=datetime(2026, 1, 14, 12, 0, 0, tzinfo=UTC),
        )
        yaml_str = result.to_yaml()
        assert "test_case_id: TC-001" in yaml_str
        restored = TestResult.from_yaml(yaml_str)
        assert restored.passed is True

    def test_yaml_is_human_readable(self):
        """Test that YAML output is properly formatted and readable."""
        tc = TestCase(
            id="TC-001",
            name="Readable Test",
            input={"nested": {"key": "value"}},
            assertions=[EqualsAssertion(name="check", expected="result")],
        )
        yaml_str = tc.to_yaml()
        # Should have proper indentation (not inline JSON)
        lines = yaml_str.strip().split("\n")
        assert len(lines) > 5  # Multi-line, not compact
        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert parsed["id"] == "TC-001"
