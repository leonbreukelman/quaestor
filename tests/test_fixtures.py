"""
Tests for quaestor.testing.fixtures module.

Comprehensive tests for FixtureScope, FixtureDefinition, FixtureValue,
FixtureRegistry, and FixtureResolver. Tests validation, serialization,
dependency resolution, and model behavior.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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


# =============================================================================
# FixtureResolver Tests
# =============================================================================


class TestFixtureResolver:
    """Tests for FixtureResolver class."""

    @pytest.fixture
    def resolver(self) -> FixtureResolver:
        """Create a fresh resolver for each test."""
        return FixtureResolver()

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    def test_resolve_fixture_no_dependencies(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Fixture with no dependencies should return single-element list."""
        registry.register(FixtureDefinition(name="standalone"))

        result = resolver.resolve("standalone", registry)

        assert result == ["standalone"]

    def test_resolve_single_dependency(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Fixture with one dependency should return both in order."""
        registry.register(FixtureDefinition(name="base"))
        registry.register(FixtureDefinition(name="dependent", dependencies=["base"]))

        result = resolver.resolve("dependent", registry)

        assert result == ["base", "dependent"]

    def test_resolve_chain_of_dependencies(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Chain A->B->C should resolve to [C, B, A]."""
        registry.register(FixtureDefinition(name="c"))
        registry.register(FixtureDefinition(name="b", dependencies=["c"]))
        registry.register(FixtureDefinition(name="a", dependencies=["b"]))

        result = resolver.resolve("a", registry)

        assert result == ["c", "b", "a"]

    def test_resolve_multiple_dependencies(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Fixture with multiple dependencies should include all."""
        registry.register(FixtureDefinition(name="dep1"))
        registry.register(FixtureDefinition(name="dep2"))
        registry.register(FixtureDefinition(name="main", dependencies=["dep1", "dep2"]))

        result = resolver.resolve("main", registry)

        # main should be last, deps should come first
        assert result[-1] == "main"
        assert "dep1" in result
        assert "dep2" in result
        assert result.index("dep1") < result.index("main")
        assert result.index("dep2") < result.index("main")

    def test_resolve_diamond_dependency(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Diamond pattern (A->B,C; B->D; C->D) should include D only once."""
        registry.register(FixtureDefinition(name="d"))
        registry.register(FixtureDefinition(name="b", dependencies=["d"]))
        registry.register(FixtureDefinition(name="c", dependencies=["d"]))
        registry.register(FixtureDefinition(name="a", dependencies=["b", "c"]))

        result = resolver.resolve("a", registry)

        # d should appear only once
        assert result.count("d") == 1
        # a should be last
        assert result[-1] == "a"
        # d should come before b and c
        assert result.index("d") < result.index("b")
        assert result.index("d") < result.index("c")

    def test_resolve_nonexistent_fixture(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Resolving non-existent fixture should raise FixtureNotFoundError."""
        with pytest.raises(FixtureNotFoundError) as exc_info:
            resolver.resolve("nonexistent", registry)

        assert exc_info.value.name == "nonexistent"

    def test_resolve_missing_dependency(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Missing dependency should raise FixtureNotFoundError."""
        registry.register(FixtureDefinition(name="dependent", dependencies=["missing"]))

        with pytest.raises(FixtureNotFoundError) as exc_info:
            resolver.resolve("dependent", registry)

        assert exc_info.value.name == "missing"


class TestFixtureResolverCycles:
    """Tests for cycle detection in FixtureResolver."""

    @pytest.fixture
    def resolver(self) -> FixtureResolver:
        """Create a fresh resolver for each test."""
        return FixtureResolver()

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    def test_detect_direct_cycle(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Direct cycle A->B->A should raise CyclicDependencyError."""
        registry.register(FixtureDefinition(name="a", dependencies=["b"]))
        registry.register(FixtureDefinition(name="b", dependencies=["a"]))

        with pytest.raises(CyclicDependencyError) as exc_info:
            resolver.resolve("a", registry)

        assert "a" in exc_info.value.cycle
        assert "b" in exc_info.value.cycle

    def test_detect_indirect_cycle(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Indirect cycle A->B->C->A should raise CyclicDependencyError."""
        registry.register(FixtureDefinition(name="a", dependencies=["b"]))
        registry.register(FixtureDefinition(name="b", dependencies=["c"]))
        registry.register(FixtureDefinition(name="c", dependencies=["a"]))

        with pytest.raises(CyclicDependencyError) as exc_info:
            resolver.resolve("a", registry)

        # All three should be in the cycle
        assert "a" in exc_info.value.cycle
        assert "b" in exc_info.value.cycle
        assert "c" in exc_info.value.cycle

    def test_detect_self_referential(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Self-referential A->A should raise CyclicDependencyError."""
        registry.register(FixtureDefinition(name="a", dependencies=["a"]))

        with pytest.raises(CyclicDependencyError) as exc_info:
            resolver.resolve("a", registry)

        assert "a" in exc_info.value.cycle

    def test_cycle_error_message(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """CyclicDependencyError message should identify the cycle."""
        registry.register(FixtureDefinition(name="x", dependencies=["y"]))
        registry.register(FixtureDefinition(name="y", dependencies=["x"]))

        with pytest.raises(CyclicDependencyError) as exc_info:
            resolver.resolve("x", registry)

        error_msg = str(exc_info.value)
        assert "x" in error_msg
        assert "y" in error_msg
        assert "Circular dependency" in error_msg


class TestFixtureResolverValidation:
    """Tests for validate_dependencies method."""

    @pytest.fixture
    def resolver(self) -> FixtureResolver:
        """Create a fresh resolver for each test."""
        return FixtureResolver()

    @pytest.fixture
    def registry(self) -> FixtureRegistry:
        """Create a fresh registry for each test."""
        return FixtureRegistry()

    def test_validate_empty_registry(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Empty registry should return no errors."""
        errors = resolver.validate_dependencies(registry)
        assert errors == []

    def test_validate_valid_dependencies(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Valid dependencies should return no errors."""
        registry.register(FixtureDefinition(name="base"))
        registry.register(FixtureDefinition(name="dependent", dependencies=["base"]))

        errors = resolver.validate_dependencies(registry)
        assert errors == []

    def test_validate_missing_dependency(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Missing dependency should be reported."""
        registry.register(FixtureDefinition(name="broken", dependencies=["missing"]))

        errors = resolver.validate_dependencies(registry)

        assert len(errors) == 1
        assert "broken" in errors[0]
        assert "missing" in errors[0]

    def test_validate_multiple_missing(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Multiple missing dependencies should all be reported."""
        registry.register(FixtureDefinition(name="a", dependencies=["missing1"]))
        registry.register(FixtureDefinition(name="b", dependencies=["missing2"]))

        errors = resolver.validate_dependencies(registry)

        assert len(errors) == 2
        assert any("missing1" in e for e in errors)
        assert any("missing2" in e for e in errors)

    def test_validate_circular_dependency(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Circular dependencies should be reported."""
        registry.register(FixtureDefinition(name="x", dependencies=["y"]))
        registry.register(FixtureDefinition(name="y", dependencies=["x"]))

        errors = resolver.validate_dependencies(registry)

        assert len(errors) >= 1
        assert any("Circular" in e for e in errors)

    def test_validate_no_false_positives(
        self, resolver: FixtureResolver, registry: FixtureRegistry
    ) -> None:
        """Complex valid graph should pass validation."""
        # Diamond pattern - valid
        registry.register(FixtureDefinition(name="d"))
        registry.register(FixtureDefinition(name="b", dependencies=["d"]))
        registry.register(FixtureDefinition(name="c", dependencies=["d"]))
        registry.register(FixtureDefinition(name="a", dependencies=["b", "c"]))

        errors = resolver.validate_dependencies(registry)
        assert errors == []


# =============================================================================
# ScopedFixtureManager Tests
# =============================================================================


class TestScopedFixtureManagerInit:
    """Tests for ScopedFixtureManager initialization."""

    def test_init_creates_empty_caches(self) -> None:
        """Manager should initialize with empty caches for all scopes."""
        manager = ScopedFixtureManager()
        for scope in FixtureScope:
            assert not manager.is_cached("any", scope)

    def test_init_creates_empty_cleanup_handlers(self) -> None:
        """Manager should initialize with empty cleanup handlers."""
        manager = ScopedFixtureManager()
        # Verify by calling teardown - should not raise
        for scope in FixtureScope:
            manager.teardown_scope(scope)


class TestScopedFixtureManagerGetOrCreate:
    """Tests for get_or_create method."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_creates_fixture_with_factory(self, manager: ScopedFixtureManager) -> None:
        """Should create fixture using provided factory."""
        value = manager.get_or_create("test_fixture", lambda: "created_value")
        assert value == "created_value"

    def test_caches_fixture_value(self, manager: ScopedFixtureManager) -> None:
        """Should cache fixture value for subsequent calls."""
        call_count = 0

        def factory() -> str:
            nonlocal call_count
            call_count += 1
            return f"value_{call_count}"

        first = manager.get_or_create("test_fixture", factory)
        second = manager.get_or_create("test_fixture", factory)

        assert first == "value_1"
        assert second == "value_1"  # Same cached value
        assert call_count == 1  # Factory called only once

    def test_scope_isolation(self, manager: ScopedFixtureManager) -> None:
        """Different scopes should have separate caches."""
        manager.get_or_create("fixture", lambda: "test_value", FixtureScope.TEST)
        manager.get_or_create("fixture", lambda: "suite_value", FixtureScope.SUITE)

        # Each scope has its own value
        assert manager.get_cached("fixture", FixtureScope.TEST).value == "test_value"
        assert manager.get_cached("fixture", FixtureScope.SUITE).value == "suite_value"

    def test_default_scope_is_test(self, manager: ScopedFixtureManager) -> None:
        """Default scope should be TEST."""
        manager.get_or_create("fixture", lambda: "value")
        assert manager.is_cached("fixture", FixtureScope.TEST)
        assert not manager.is_cached("fixture", FixtureScope.SUITE)
        assert not manager.is_cached("fixture", FixtureScope.SESSION)

    def test_registers_cleanup_handler(self, manager: ScopedFixtureManager) -> None:
        """Should register cleanup handler."""
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager.get_or_create("fixture", lambda: "value", cleanup=cleanup)
        manager.teardown_scope(FixtureScope.TEST)

        assert cleanup_called

    def test_raises_for_unknown_fixture_without_factory(
        self, manager: ScopedFixtureManager
    ) -> None:
        """Should raise ValueError for unknown fixture without factory."""
        with pytest.raises(ValueError, match="No factory provided"):
            manager.get_or_create("unknown_fixture")

    def test_factory_can_return_none(self, manager: ScopedFixtureManager) -> None:
        """Factory returning None should be valid."""
        value = manager.get_or_create("null_fixture", lambda: None)
        assert value is None


class TestScopedFixtureManagerBuiltinTestId:
    """Tests for built-in test_id fixture."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_test_id_without_factory(self, manager: ScopedFixtureManager) -> None:
        """test_id should work without providing a factory."""
        test_id = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEST_ID)
        assert test_id is not None
        assert isinstance(test_id, str)

    def test_test_id_format(self, manager: ScopedFixtureManager) -> None:
        """test_id should have expected format: test-NNNN-XXXXXXXX."""
        test_id = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEST_ID)
        parts = test_id.split("-")
        assert len(parts) == 3
        assert parts[0] == "test"
        assert len(parts[1]) == 4  # Counter part
        assert len(parts[2]) == 8  # Short UUID

    def test_test_id_uniqueness(self, manager: ScopedFixtureManager) -> None:
        """Each test_id call (different scope) should generate unique IDs."""
        # Clear cache between calls to generate new IDs
        id1 = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEST_ID, scope=FixtureScope.TEST)
        manager.teardown_scope(FixtureScope.TEST)
        id2 = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEST_ID, scope=FixtureScope.TEST)

        assert id1 != id2

    def test_test_id_counter_increments(self, manager: ScopedFixtureManager) -> None:
        """Counter should increment for each new test_id."""
        # Generate multiple IDs by tearing down between each
        ids = []
        for _ in range(3):
            test_id = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEST_ID)
            ids.append(test_id)
            manager.teardown_scope(FixtureScope.TEST)

        # Extract counters
        counters = [int(id_.split("-")[1]) for id_ in ids]
        assert counters == [1, 2, 3]


class TestScopedFixtureManagerBuiltinTempDir:
    """Tests for built-in temp_dir fixture."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_temp_dir_without_factory(self, manager: ScopedFixtureManager) -> None:
        """temp_dir should work without providing a factory."""
        from pathlib import Path

        temp_dir = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEMP_DIR)
        assert isinstance(temp_dir, Path)
        manager.teardown_scope(FixtureScope.TEST)

    def test_temp_dir_exists(self, manager: ScopedFixtureManager) -> None:
        """temp_dir should create an actual directory."""
        temp_dir = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEMP_DIR)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        manager.teardown_scope(FixtureScope.TEST)

    def test_temp_dir_prefix(self, manager: ScopedFixtureManager) -> None:
        """temp_dir should have quaestor_test_ prefix."""
        temp_dir = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEMP_DIR)
        assert "quaestor_test_" in temp_dir.name
        manager.teardown_scope(FixtureScope.TEST)

    def test_temp_dir_cleanup_on_teardown(self, manager: ScopedFixtureManager) -> None:
        """temp_dir should be cleaned up on scope teardown."""
        temp_dir = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEMP_DIR)
        assert temp_dir.exists()

        manager.teardown_scope(FixtureScope.TEST)
        assert not temp_dir.exists()

    def test_temp_dir_with_content(self, manager: ScopedFixtureManager) -> None:
        """temp_dir cleanup should work even with content."""
        temp_dir = manager.get_or_create(ScopedFixtureManager.BUILTIN_TEMP_DIR)

        # Create files in the temp directory
        (temp_dir / "test_file.txt").write_text("hello")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "nested.txt").write_text("nested")

        manager.teardown_scope(FixtureScope.TEST)
        assert not temp_dir.exists()


class TestScopedFixtureManagerTeardown:
    """Tests for teardown_scope method."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_teardown_clears_cache(self, manager: ScopedFixtureManager) -> None:
        """Teardown should clear the cache for the scope."""
        manager.get_or_create("fixture", lambda: "value", FixtureScope.TEST)
        assert manager.is_cached("fixture", FixtureScope.TEST)

        manager.teardown_scope(FixtureScope.TEST)
        assert not manager.is_cached("fixture", FixtureScope.TEST)

    def test_teardown_only_affects_specified_scope(self, manager: ScopedFixtureManager) -> None:
        """Teardown should only clear the specified scope."""
        manager.get_or_create("test_f", lambda: "t", FixtureScope.TEST)
        manager.get_or_create("suite_f", lambda: "s", FixtureScope.SUITE)
        manager.get_or_create("session_f", lambda: "ss", FixtureScope.SESSION)

        manager.teardown_scope(FixtureScope.TEST)

        assert not manager.is_cached("test_f", FixtureScope.TEST)
        assert manager.is_cached("suite_f", FixtureScope.SUITE)
        assert manager.is_cached("session_f", FixtureScope.SESSION)

    def test_cleanup_handlers_run_in_lifo_order(self, manager: ScopedFixtureManager) -> None:
        """Cleanup handlers should run in LIFO order."""
        order: list[int] = []

        manager.get_or_create("f1", lambda: 1, cleanup=lambda: order.append(1))
        manager.get_or_create("f2", lambda: 2, cleanup=lambda: order.append(2))
        manager.get_or_create("f3", lambda: 3, cleanup=lambda: order.append(3))

        manager.teardown_scope(FixtureScope.TEST)

        assert order == [3, 2, 1]  # LIFO order

    def test_cleanup_exception_doesnt_stop_other_handlers(
        self, manager: ScopedFixtureManager
    ) -> None:
        """Exception in cleanup handler shouldn't stop others."""
        handler2_called = False

        def failing_cleanup() -> None:
            raise RuntimeError("Cleanup failed")

        def handler2() -> None:
            nonlocal handler2_called
            handler2_called = True

        manager.get_or_create("f1", lambda: 1, cleanup=handler2)
        manager.get_or_create("f2", lambda: 2, cleanup=failing_cleanup)

        manager.teardown_scope(FixtureScope.TEST)

        assert handler2_called  # Second handler still ran

    def test_cleanup_handlers_cleared_after_teardown(self, manager: ScopedFixtureManager) -> None:
        """Cleanup handlers should be cleared after teardown."""
        call_count = 0

        def cleanup() -> None:
            nonlocal call_count
            call_count += 1

        manager.get_or_create("fixture", lambda: "v", cleanup=cleanup)
        manager.teardown_scope(FixtureScope.TEST)
        manager.teardown_scope(FixtureScope.TEST)  # Second teardown

        assert call_count == 1  # Handler only called once


class TestScopedFixtureManagerGetCached:
    """Tests for get_cached method."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_get_cached_returns_fixture_value(self, manager: ScopedFixtureManager) -> None:
        """get_cached should return FixtureValue wrapper."""
        manager.get_or_create("fixture", lambda: "value")
        cached = manager.get_cached("fixture", FixtureScope.TEST)

        assert isinstance(cached, FixtureValue)
        assert cached.value == "value"

    def test_get_cached_returns_none_for_uncached(self, manager: ScopedFixtureManager) -> None:
        """get_cached should return None for uncached fixtures."""
        cached = manager.get_cached("nonexistent", FixtureScope.TEST)
        assert cached is None


class TestScopedFixtureManagerIsCached:
    """Tests for is_cached method."""

    @pytest.fixture
    def manager(self) -> ScopedFixtureManager:
        """Provide a fresh ScopedFixtureManager."""
        return ScopedFixtureManager()

    def test_is_cached_true_for_cached(self, manager: ScopedFixtureManager) -> None:
        """is_cached should return True for cached fixtures."""
        manager.get_or_create("fixture", lambda: "value")
        assert manager.is_cached("fixture", FixtureScope.TEST) is True

    def test_is_cached_false_for_uncached(self, manager: ScopedFixtureManager) -> None:
        """is_cached should return False for uncached fixtures."""
        assert manager.is_cached("fixture", FixtureScope.TEST) is False

    def test_is_cached_scope_specific(self, manager: ScopedFixtureManager) -> None:
        """is_cached should be scope-specific."""
        manager.get_or_create("fixture", lambda: "v", FixtureScope.SUITE)

        assert manager.is_cached("fixture", FixtureScope.SUITE) is True
        assert manager.is_cached("fixture", FixtureScope.TEST) is False
        assert manager.is_cached("fixture", FixtureScope.SESSION) is False
