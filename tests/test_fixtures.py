"""
Tests for quaestor.testing.fixtures module.

Comprehensive tests for FixtureScope, FixtureDefinition, FixtureValue,
and FixtureRegistry. Tests validation, serialization, and model behavior.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quaestor.testing.fixtures import (
    DuplicateFixtureError,
    FixtureDefinition,
    FixtureNotFoundError,
    FixtureRegistry,
    FixtureScope,
    FixtureValue,
)

# =============================================================================
# FixtureScope Tests
# =============================================================================


class TestFixtureScope:
    """Tests for FixtureScope enum."""

    def test_has_three_values(self) -> None:
        """FixtureScope should have exactly three values."""
        assert len(FixtureScope) == 3

    def test_test_scope_exists(self) -> None:
        """TEST scope should exist with value 'test'."""
        assert FixtureScope.TEST.value == "test"

    def test_suite_scope_exists(self) -> None:
        """SUITE scope should exist with value 'suite'."""
        assert FixtureScope.SUITE.value == "suite"

    def test_session_scope_exists(self) -> None:
        """SESSION scope should exist with value 'session'."""
        assert FixtureScope.SESSION.value == "session"

    def test_is_string_backed(self) -> None:
        """FixtureScope should be string-backed for serialization."""
        assert isinstance(FixtureScope.TEST.value, str)
        assert FixtureScope.TEST == "test"  # String comparison should work

    def test_scope_comparison(self) -> None:
        """Scope values should be comparable for equality."""
        scope1 = FixtureScope.TEST
        scope2 = FixtureScope.TEST
        scope3 = FixtureScope.SUITE
        assert scope1 == scope2
        assert scope1 != scope3


# =============================================================================
# FixtureDefinition Tests
# =============================================================================


class TestFixtureDefinition:
    """Tests for FixtureDefinition model."""

    def test_creation_with_minimal_fields(self) -> None:
        """FixtureDefinition should work with just a name."""
        fixture = FixtureDefinition(name="test_fixture")
        assert fixture.name == "test_fixture"
        assert fixture.scope == FixtureScope.TEST
        assert fixture.description == ""
        assert fixture.dependencies == []
        assert fixture.is_async is False

    def test_creation_with_all_fields(self) -> None:
        """FixtureDefinition should accept all fields."""
        fixture = FixtureDefinition(
            name="db_session",
            scope=FixtureScope.SUITE,
            description="Database session fixture",
            dependencies=["db_engine", "config"],
            is_async=True,
        )
        assert fixture.name == "db_session"
        assert fixture.scope == FixtureScope.SUITE
        assert fixture.description == "Database session fixture"
        assert fixture.dependencies == ["db_engine", "config"]
        assert fixture.is_async is True

    def test_name_is_required(self) -> None:
        """Name field should be required."""
        with pytest.raises(ValidationError) as exc_info:
            FixtureDefinition()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("name",) for e in errors)

    def test_name_cannot_be_empty(self) -> None:
        """Name cannot be an empty string."""
        with pytest.raises(ValidationError) as exc_info:
            FixtureDefinition(name="")
        errors = exc_info.value.errors()
        assert any("name" in str(e["loc"]) for e in errors)

    def test_name_cannot_be_whitespace_only(self) -> None:
        """Name cannot be whitespace-only."""
        with pytest.raises(ValidationError) as exc_info:
            FixtureDefinition(name="   ")
        errors = exc_info.value.errors()
        assert any("name" in str(e["loc"]) or "whitespace" in str(e) for e in errors)

    def test_scope_defaults_to_test(self) -> None:
        """Scope should default to TEST."""
        fixture = FixtureDefinition(name="test")
        assert fixture.scope == FixtureScope.TEST

    def test_scope_accepts_string_value(self) -> None:
        """Scope should accept string value for deserialization."""
        fixture = FixtureDefinition(name="test", scope="suite")  # type: ignore[arg-type]
        assert fixture.scope == FixtureScope.SUITE

    def test_invalid_scope_raises_error(self) -> None:
        """Invalid scope value should raise ValidationError."""
        with pytest.raises(ValidationError):
            FixtureDefinition(name="test", scope="invalid")  # type: ignore[arg-type]

    def test_dependencies_default_to_empty_list(self) -> None:
        """Dependencies should default to empty list."""
        fixture = FixtureDefinition(name="test")
        assert fixture.dependencies == []

    def test_dependencies_empty_name_rejected(self) -> None:
        """Empty dependency names should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            FixtureDefinition(name="test", dependencies=["valid", ""])
        assert "empty" in str(exc_info.value).lower()

    def test_is_async_defaults_to_false(self) -> None:
        """is_async should default to False."""
        fixture = FixtureDefinition(name="test")
        assert fixture.is_async is False

    def test_model_is_frozen(self) -> None:
        """Model should be immutable after creation."""
        fixture = FixtureDefinition(name="test")
        with pytest.raises(ValidationError):
            fixture.name = "new_name"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            FixtureDefinition(name="test", unknown_field="value")  # type: ignore[call-arg]


class TestFixtureDefinitionSerialization:
    """Tests for FixtureDefinition serialization."""

    def test_json_serialization(self) -> None:
        """Model should serialize to JSON."""
        fixture = FixtureDefinition(
            name="test_fixture",
            scope=FixtureScope.SUITE,
            description="A test fixture",
            dependencies=["dep1"],
            is_async=True,
        )
        json_str = fixture.model_dump_json()
        data = json.loads(json_str)

        assert data["name"] == "test_fixture"
        assert data["scope"] == "suite"
        assert data["description"] == "A test fixture"
        assert data["dependencies"] == ["dep1"]
        assert data["is_async"] is True

    def test_json_deserialization(self) -> None:
        """Model should deserialize from JSON."""
        json_str = json.dumps(
            {
                "name": "test_fixture",
                "scope": "session",
                "description": "Deserialized fixture",
                "dependencies": ["a", "b"],
                "is_async": True,
            }
        )
        fixture = FixtureDefinition.model_validate_json(json_str)

        assert fixture.name == "test_fixture"
        assert fixture.scope == FixtureScope.SESSION
        assert fixture.description == "Deserialized fixture"
        assert fixture.dependencies == ["a", "b"]
        assert fixture.is_async is True

    def test_json_round_trip(self) -> None:
        """Serialization roundtrip should preserve all fields."""
        original = FixtureDefinition(
            name="round_trip_test",
            scope=FixtureScope.SESSION,
            description="Testing round trip",
            dependencies=["x", "y", "z"],
            is_async=True,
        )

        json_str = original.model_dump_json()
        restored = FixtureDefinition.model_validate_json(json_str)

        assert restored.name == original.name
        assert restored.scope == original.scope
        assert restored.description == original.description
        assert restored.dependencies == original.dependencies
        assert restored.is_async == original.is_async

    def test_yaml_serialization(self) -> None:
        """Model should serialize to YAML."""
        fixture = FixtureDefinition(
            name="yaml_fixture",
            scope=FixtureScope.SUITE,
        )
        yaml_str = fixture.to_yaml()

        assert "name: yaml_fixture" in yaml_str
        assert "scope: suite" in yaml_str

    def test_yaml_deserialization(self) -> None:
        """Model should deserialize from YAML."""
        yaml_str = """
name: yaml_test
scope: session
description: From YAML
dependencies:
  - dep1
  - dep2
is_async: true
"""
        fixture = FixtureDefinition.from_yaml(yaml_str)

        assert fixture.name == "yaml_test"
        assert fixture.scope == FixtureScope.SESSION
        assert fixture.description == "From YAML"
        assert fixture.dependencies == ["dep1", "dep2"]
        assert fixture.is_async is True

    def test_yaml_round_trip(self) -> None:
        """YAML roundtrip should preserve all fields."""
        original = FixtureDefinition(
            name="yaml_round_trip",
            scope=FixtureScope.TEST,
            description="YAML test",
            dependencies=["a"],
            is_async=False,
        )

        yaml_str = original.to_yaml()
        restored = FixtureDefinition.from_yaml(yaml_str)

        assert restored.name == original.name
        assert restored.scope == original.scope
        assert restored.description == original.description
        assert restored.dependencies == original.dependencies
        assert restored.is_async == original.is_async


# =============================================================================
# FixtureValue Tests
# =============================================================================


class TestFixtureValue:
    """Tests for FixtureValue model."""

    def test_creation_with_value_and_definition(self) -> None:
        """FixtureValue should wrap a value with its definition."""
        definition = FixtureDefinition(name="test_client")
        fixture_value = FixtureValue(
            value={"client": "mock"},
            definition=definition,
        )

        assert fixture_value.value == {"client": "mock"}
        assert fixture_value.definition == definition
        assert fixture_value.error is None

    def test_created_at_auto_populated(self) -> None:
        """created_at should be auto-populated with current time."""
        before = datetime.now(UTC)
        definition = FixtureDefinition(name="test")
        fixture_value = FixtureValue(value="test_value", definition=definition)
        after = datetime.now(UTC)

        assert before <= fixture_value.created_at <= after

    def test_value_can_be_none(self) -> None:
        """None should be a valid fixture value."""
        definition = FixtureDefinition(name="null_fixture")
        fixture_value = FixtureValue(value=None, definition=definition)

        assert fixture_value.value is None

    def test_value_can_be_any_type(self) -> None:
        """Value should accept any type."""
        definition = FixtureDefinition(name="any_fixture")

        # String
        fv_str = FixtureValue(value="string", definition=definition)
        assert fv_str.value == "string"

        # Integer
        fv_int = FixtureValue(value=42, definition=definition)
        assert fv_int.value == 42

        # List
        fv_list = FixtureValue(value=[1, 2, 3], definition=definition)
        assert fv_list.value == [1, 2, 3]

        # Dict
        fv_dict = FixtureValue(value={"key": "value"}, definition=definition)
        assert fv_dict.value == {"key": "value"}

        # Object
        class CustomObject:
            pass

        obj = CustomObject()
        fv_obj = FixtureValue(value=obj, definition=definition)
        assert fv_obj.value is obj

    def test_error_field_optional(self) -> None:
        """Error field should be optional and default to None."""
        definition = FixtureDefinition(name="test")
        fixture_value = FixtureValue(value="ok", definition=definition)

        assert fixture_value.error is None

    def test_error_field_captures_failure(self) -> None:
        """Error field should capture setup failure messages."""
        definition = FixtureDefinition(name="failing_fixture")
        fixture_value = FixtureValue(
            value=None,
            definition=definition,
            error="Database connection refused",
        )

        assert fixture_value.error == "Database connection refused"

    def test_is_error_property(self) -> None:
        """is_error property should indicate setup failure."""
        definition = FixtureDefinition(name="test")

        success = FixtureValue(value="ok", definition=definition)
        assert success.is_error is False

        failure = FixtureValue(value=None, definition=definition, error="Failed")
        assert failure.is_error is True

    def test_name_property(self) -> None:
        """name property should return definition name."""
        definition = FixtureDefinition(name="my_fixture")
        fixture_value = FixtureValue(value="test", definition=definition)

        assert fixture_value.name == "my_fixture"

    def test_scope_property(self) -> None:
        """scope property should return definition scope."""
        definition = FixtureDefinition(name="test", scope=FixtureScope.SESSION)
        fixture_value = FixtureValue(value="test", definition=definition)

        assert fixture_value.scope == FixtureScope.SESSION

    def test_model_is_frozen(self) -> None:
        """Model should be immutable after creation."""
        definition = FixtureDefinition(name="test")
        fixture_value = FixtureValue(value="test", definition=definition)

        with pytest.raises(ValidationError):
            fixture_value.value = "new_value"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Extra fields should raise ValidationError."""
        definition = FixtureDefinition(name="test")
        with pytest.raises(ValidationError):
            FixtureValue(
                value="test",
                definition=definition,
                unknown="extra",  # type: ignore[call-arg]
            )


class TestFixtureValueSerialization:
    """Tests for FixtureValue serialization."""

    def test_json_serialization(self) -> None:
        """Model should serialize to JSON."""
        definition = FixtureDefinition(name="json_test", scope=FixtureScope.SUITE)
        fixture_value = FixtureValue(
            value={"key": "value"},
            definition=definition,
        )

        json_str = fixture_value.model_dump_json()
        data = json.loads(json_str)

        assert data["value"] == {"key": "value"}
        assert data["definition"]["name"] == "json_test"
        assert data["definition"]["scope"] == "suite"
        assert data["error"] is None
        assert "created_at" in data

    def test_json_deserialization(self) -> None:
        """Model should deserialize from JSON."""
        json_str = json.dumps(
            {
                "value": "test_value",
                "definition": {
                    "name": "deserialized",
                    "scope": "test",
                    "description": "",
                    "dependencies": [],
                    "is_async": False,
                },
                "created_at": "2026-01-14T12:00:00+00:00",
                "error": None,
            }
        )

        fixture_value = FixtureValue.model_validate_json(json_str)

        assert fixture_value.value == "test_value"
        assert fixture_value.definition.name == "deserialized"

    def test_json_round_trip(self) -> None:
        """JSON roundtrip should preserve all fields."""
        definition = FixtureDefinition(
            name="round_trip",
            scope=FixtureScope.SESSION,
            dependencies=["a", "b"],
        )
        original = FixtureValue(
            value={"nested": {"data": [1, 2, 3]}},
            definition=definition,
            error=None,
        )

        json_str = original.model_dump_json()
        restored = FixtureValue.model_validate_json(json_str)

        assert restored.value == original.value
        assert restored.definition.name == original.definition.name
        assert restored.definition.scope == original.definition.scope
        assert restored.definition.dependencies == original.definition.dependencies
        assert restored.error == original.error

    def test_yaml_serialization(self) -> None:
        """Model should serialize to YAML."""
        definition = FixtureDefinition(name="yaml_fixture")
        fixture_value = FixtureValue(value="test", definition=definition)

        yaml_str = fixture_value.to_yaml()

        assert "value: test" in yaml_str
        assert "name: yaml_fixture" in yaml_str

    def test_yaml_deserialization(self) -> None:
        """Model should deserialize from YAML."""
        yaml_str = """
value: yaml_value
definition:
  name: yaml_def
  scope: suite
  description: ""
  dependencies: []
  is_async: false
created_at: "2026-01-14T12:00:00+00:00"
error: null
"""
        fixture_value = FixtureValue.from_yaml(yaml_str)

        assert fixture_value.value == "yaml_value"
        assert fixture_value.definition.name == "yaml_def"
        assert fixture_value.definition.scope == FixtureScope.SUITE

    def test_datetime_serialization(self) -> None:
        """Datetime should serialize correctly in ISO format."""
        definition = FixtureDefinition(name="dt_test")
        fixture_value = FixtureValue(value="test", definition=definition)

        json_str = fixture_value.model_dump_json()
        data = json.loads(json_str)

        # Should be ISO 8601 format
        created_at_str = data["created_at"]
        assert "T" in created_at_str
        # Should parse back correctly
        restored = FixtureValue.model_validate_json(json_str)
        assert abs((restored.created_at - fixture_value.created_at).total_seconds()) < 1


# =============================================================================
# FixtureRegistry Tests
# =============================================================================


class TestFixtureRegistry:
    """Tests for FixtureRegistry class."""

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    @pytest.fixture
    def sample_fixture(self) -> FixtureDefinition:
        """Create a sample fixture definition."""
        return FixtureDefinition(
            name="test_fixture",
            scope=FixtureScope.TEST,
            description="A test fixture",
        )

    def test_empty_registry(self, registry: FixtureRegistry) -> None:
        """New registry should be empty."""
        assert len(registry) == 0
        assert registry.list_all() == []

    def test_register_fixture(
        self, registry: FixtureRegistry, sample_fixture: FixtureDefinition
    ) -> None:
        """Should register a fixture successfully."""
        registry.register(sample_fixture)
        assert registry.has("test_fixture") is True
        assert len(registry) == 1

    def test_register_multiple_fixtures(self, registry: FixtureRegistry) -> None:
        """Should register multiple fixtures."""
        f1 = FixtureDefinition(name="fixture1")
        f2 = FixtureDefinition(name="fixture2")
        f3 = FixtureDefinition(name="fixture3")

        registry.register(f1)
        registry.register(f2)
        registry.register(f3)

        assert len(registry) == 3
        assert registry.has("fixture1")
        assert registry.has("fixture2")
        assert registry.has("fixture3")

    def test_get_fixture(
        self, registry: FixtureRegistry, sample_fixture: FixtureDefinition
    ) -> None:
        """Should retrieve a registered fixture by name."""
        registry.register(sample_fixture)
        retrieved = registry.get("test_fixture")

        assert retrieved.name == "test_fixture"
        assert retrieved.scope == FixtureScope.TEST
        assert retrieved.description == "A test fixture"

    def test_get_nonexistent_raises_error(self, registry: FixtureRegistry) -> None:
        """Should raise FixtureNotFoundError for non-existent fixture."""
        with pytest.raises(FixtureNotFoundError) as exc_info:
            registry.get("nonexistent")

        assert exc_info.value.name == "nonexistent"
        assert "nonexistent" in str(exc_info.value)

    def test_has_returns_true_for_registered(
        self, registry: FixtureRegistry, sample_fixture: FixtureDefinition
    ) -> None:
        """has() should return True for registered fixtures."""
        registry.register(sample_fixture)
        assert registry.has("test_fixture") is True

    def test_has_returns_false_for_unregistered(self, registry: FixtureRegistry) -> None:
        """has() should return False for non-existent fixtures."""
        assert registry.has("nonexistent") is False

    def test_has_no_exception_for_nonexistent(self, registry: FixtureRegistry) -> None:
        """has() should not raise exception for non-existent fixtures."""
        # Should not raise
        result = registry.has("definitely_not_there")
        assert result is False

    def test_list_all_empty(self, registry: FixtureRegistry) -> None:
        """list_all() should return empty list for empty registry."""
        assert registry.list_all() == []

    def test_list_all_returns_all(self, registry: FixtureRegistry) -> None:
        """list_all() should return all registered fixtures."""
        f1 = FixtureDefinition(name="fixture1")
        f2 = FixtureDefinition(name="fixture2")

        registry.register(f1)
        registry.register(f2)

        all_fixtures = registry.list_all()
        assert len(all_fixtures) == 2
        names = [f.name for f in all_fixtures]
        assert "fixture1" in names
        assert "fixture2" in names

    def test_list_all_returns_copy(self, registry: FixtureRegistry) -> None:
        """list_all() should return a copy, not internal state."""
        f1 = FixtureDefinition(name="fixture1")
        registry.register(f1)

        # Modify the returned list
        all_fixtures = registry.list_all()
        all_fixtures.clear()

        # Registry should be unaffected
        assert len(registry) == 1
        assert registry.has("fixture1")

    def test_clear_removes_all(
        self, registry: FixtureRegistry, sample_fixture: FixtureDefinition
    ) -> None:
        """clear() should remove all registered fixtures."""
        registry.register(sample_fixture)
        registry.register(FixtureDefinition(name="another"))

        assert len(registry) == 2

        registry.clear()

        assert len(registry) == 0
        assert registry.has("test_fixture") is False
        assert registry.has("another") is False
        assert registry.list_all() == []

    def test_clear_on_empty_registry(self, registry: FixtureRegistry) -> None:
        """clear() should work on empty registry without error."""
        # Should not raise
        registry.clear()
        assert len(registry) == 0


class TestFixtureRegistryDuplicates:
    """Tests for duplicate registration handling."""

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    def test_duplicate_raises_error(self, registry: FixtureRegistry) -> None:
        """Registering duplicate name should raise DuplicateFixtureError."""
        f1 = FixtureDefinition(name="duplicate")
        f2 = FixtureDefinition(name="duplicate", description="different")

        registry.register(f1)

        with pytest.raises(DuplicateFixtureError) as exc_info:
            registry.register(f2)

        assert exc_info.value.name == "duplicate"
        assert "duplicate" in str(exc_info.value)

    def test_duplicate_error_message(self, registry: FixtureRegistry) -> None:
        """DuplicateFixtureError should have descriptive message."""
        f1 = FixtureDefinition(name="my_fixture")
        registry.register(f1)

        with pytest.raises(DuplicateFixtureError) as exc_info:
            registry.register(FixtureDefinition(name="my_fixture"))

        error_msg = str(exc_info.value)
        assert "my_fixture" in error_msg
        assert "already registered" in error_msg.lower()

    def test_original_unchanged_after_duplicate_attempt(self, registry: FixtureRegistry) -> None:
        """Original fixture should remain after failed duplicate registration."""
        original = FixtureDefinition(name="keeper", description="I should stay")
        duplicate = FixtureDefinition(name="keeper", description="I should not replace")

        registry.register(original)

        with pytest.raises(DuplicateFixtureError):
            registry.register(duplicate)

        # Original should be unchanged
        retrieved = registry.get("keeper")
        assert retrieved.description == "I should stay"

    def test_case_sensitive_names(self, registry: FixtureRegistry) -> None:
        """Fixture names should be case-sensitive."""
        lower = FixtureDefinition(name="fixture")
        upper = FixtureDefinition(name="FIXTURE")
        mixed = FixtureDefinition(name="Fixture")

        # All should register successfully (different names)
        registry.register(lower)
        registry.register(upper)
        registry.register(mixed)

        assert len(registry) == 3
        assert registry.has("fixture")
        assert registry.has("FIXTURE")
        assert registry.has("Fixture")


class TestFixtureRegistryDunderMethods:
    """Tests for dunder methods (__len__, __contains__, etc.)."""

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    def test_len(self, registry: FixtureRegistry) -> None:
        """__len__ should return fixture count."""
        assert len(registry) == 0

        registry.register(FixtureDefinition(name="one"))
        assert len(registry) == 1

        registry.register(FixtureDefinition(name="two"))
        assert len(registry) == 2

    def test_contains(self, registry: FixtureRegistry) -> None:
        """__contains__ should support 'in' operator."""
        registry.register(FixtureDefinition(name="exists"))

        assert "exists" in registry
        assert "not_exists" not in registry

    def test_iter(self, registry: FixtureRegistry) -> None:
        """__iter__ should iterate over fixture names."""
        registry.register(FixtureDefinition(name="a"))
        registry.register(FixtureDefinition(name="b"))
        registry.register(FixtureDefinition(name="c"))

        names = list(registry)
        assert len(names) == 3
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_repr(self, registry: FixtureRegistry) -> None:
        """__repr__ should show fixture count."""
        assert "0 fixtures" in repr(registry)

        registry.register(FixtureDefinition(name="test"))
        assert "1 fixtures" in repr(registry)

        registry.register(FixtureDefinition(name="test2"))
        assert "2 fixtures" in repr(registry)
