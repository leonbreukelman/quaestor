"""
Pydantic models for the fixture system.

This module provides data models for representing test fixtures, including
fixture definitions (metadata and configuration), fixture values (resolved
instances), and fixture scopes (lifetime management).

Follows patterns established in quaestor.testing.models.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class FixtureScope(str, Enum):
    """Fixture lifetime scopes defining when fixtures are created/destroyed.

    - TEST: Created fresh for each test, destroyed after the test completes
    - SUITE: Created once per test suite, shared across tests in the suite
    - SESSION: Created once per session, shared across all suites and tests
    """

    TEST = "test"
    SUITE = "suite"
    SESSION = "session"


class FixtureDefinition(BaseModel):
    """Pydantic model representing a fixture's configuration and metadata.

    Defines the static properties of a fixture including its name, lifetime
    scope, dependencies on other fixtures, and whether it's async.

    Example:
        >>> fixture = FixtureDefinition(
        ...     name="db_session",
        ...     scope=FixtureScope.SUITE,
        ...     description="Database session for integration tests",
        ...     dependencies=["db_engine"],
        ...     is_async=True,
        ... )
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = Field(
        ...,
        description="Unique name identifying this fixture",
        min_length=1,
    )
    scope: FixtureScope = Field(
        default=FixtureScope.TEST,
        description="Fixture lifetime scope (test, suite, or session)",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the fixture's purpose",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Names of other fixtures this fixture depends on",
    )
    is_async: bool = Field(
        default=False,
        description="Whether the fixture factory is an async function",
    )

    @field_validator("name")
    @classmethod
    def name_is_valid(cls, v: str) -> str:
        """Validate that name is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Fixture name must not be empty or whitespace-only")
        return v

    @field_validator("dependencies")
    @classmethod
    def dependencies_no_empty(cls, v: list[str]) -> list[str]:
        """Validate that dependency names are not empty."""
        for dep in v:
            if not dep.strip():
                raise ValueError("Dependency names must not be empty")
        return v

    def to_yaml(self) -> str:
        """Serialize to YAML format for OSCAL compatibility."""
        result: str = yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return result

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FixtureDefinition":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


class FixtureValue(BaseModel):
    """Pydantic model representing a resolved fixture value with metadata.

    Wraps the actual fixture value along with its definition and lifecycle
    metadata. Supports tracking creation time and any errors during setup.

    Example:
        >>> definition = FixtureDefinition(name="test_client")
        >>> fixture_value = FixtureValue(
        ...     value={"client": mock_client},
        ...     definition=definition,
        ... )
    """

    model_config = {"frozen": True, "extra": "forbid"}

    value: Any = Field(
        ...,
        description="The resolved fixture value (can be any type, including None)",
    )
    definition: FixtureDefinition = Field(
        ...,
        description="The fixture definition that produced this value",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the fixture was created",
    )
    error: str | None = Field(
        default=None,
        description="Error message if fixture setup failed",
    )

    @property
    def is_error(self) -> bool:
        """Check if this fixture value represents a failed setup."""
        return self.error is not None

    @property
    def name(self) -> str:
        """Convenience accessor for the fixture name."""
        return self.definition.name

    @property
    def scope(self) -> FixtureScope:
        """Convenience accessor for the fixture scope."""
        return self.definition.scope

    def to_yaml(self) -> str:
        """Serialize to YAML format for OSCAL compatibility.

        Note: Complex values may not serialize cleanly; use for simple types.
        """
        result: str = yaml.dump(
            self.model_dump(mode="json"),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        return result

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FixtureValue":
        """Deserialize from YAML format."""
        data = yaml.safe_load(yaml_str)
        return cls.model_validate(data)


# =============================================================================
# Custom Exceptions
# =============================================================================


class DuplicateFixtureError(ValueError):
    """Raised when attempting to register a fixture with an existing name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Fixture '{name}' is already registered")


class FixtureNotFoundError(KeyError):
    """Raised when attempting to retrieve a fixture that does not exist."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Fixture '{name}' not found in registry")


# =============================================================================
# Fixture Registry
# =============================================================================


class FixtureRegistry:
    """Central registry for managing FixtureDefinition instances.

    Provides a centralized mechanism for registering, looking up, and managing
    fixture definitions. Enforces uniqueness constraints to prevent duplicate
    registrations.

    Note: This registry does NOT implement dependency resolution or scope
    management. Those concerns are handled by separate components.

    Example:
        >>> registry = FixtureRegistry()
        >>> fixture_def = FixtureDefinition(name="db_session", scope=FixtureScope.SUITE)
        >>> registry.register(fixture_def)
        >>> registry.get("db_session")
        FixtureDefinition(name='db_session', ...)
    """

    def __init__(self) -> None:
        """Initialize an empty fixture registry."""
        self._fixtures: dict[str, FixtureDefinition] = {}

    def register(self, definition: FixtureDefinition) -> None:
        """Register a fixture definition in the registry.

        Args:
            definition: The FixtureDefinition to register.

        Raises:
            DuplicateFixtureError: If a fixture with the same name is already registered.
        """
        if definition.name in self._fixtures:
            raise DuplicateFixtureError(definition.name)
        self._fixtures[definition.name] = definition

    def get(self, name: str) -> FixtureDefinition:
        """Retrieve a fixture definition by name.

        Args:
            name: The name of the fixture to retrieve.

        Returns:
            The FixtureDefinition with the given name.

        Raises:
            FixtureNotFoundError: If no fixture with the given name exists.
        """
        if name not in self._fixtures:
            raise FixtureNotFoundError(name)
        return self._fixtures[name]

    def has(self, name: str) -> bool:
        """Check if a fixture is registered.

        Args:
            name: The name of the fixture to check.

        Returns:
            True if the fixture is registered, False otherwise.
        """
        return name in self._fixtures

    def list_all(self) -> list[FixtureDefinition]:
        """List all registered fixture definitions.

        Returns:
            A copy of the list of all registered FixtureDefinitions.
            Modifications to this list do not affect registry state.
        """
        return list(self._fixtures.values())

    def clear(self) -> None:
        """Remove all registered fixtures from the registry.

        This operation is destructive and removes all fixtures.
        Use with caution, typically for test isolation purposes.
        """
        self._fixtures.clear()

    def __len__(self) -> int:
        """Return the number of registered fixtures."""
        return len(self._fixtures)

    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for checking fixture existence."""
        return self.has(name)

    def __iter__(self) -> "Iterator[str]":
        """Iterate over fixture names."""
        return iter(self._fixtures)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        return f"FixtureRegistry({len(self._fixtures)} fixtures)"


# =============================================================================
# Custom Exceptions for Dependency Resolution
# =============================================================================


class CyclicDependencyError(ValueError):
    """Raised when circular dependencies are detected in fixtures."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")


# =============================================================================
# Fixture Resolver
# =============================================================================


class FixtureResolver:
    """Resolves fixture dependencies and validates dependency graphs.

    Performs topological sorting to determine the correct fixture initialization
    order and validates that all fixture dependencies exist and are free of
    circular references.

    Note: This resolver does NOT implement scope management or value caching.
    Those concerns are handled by separate components.

    Example:
        >>> registry = FixtureRegistry()
        >>> registry.register(FixtureDefinition(name="db", scope=FixtureScope.SESSION))
        >>> registry.register(FixtureDefinition(name="session", dependencies=["db"]))
        >>> resolver = FixtureResolver()
        >>> resolver.resolve("session", registry)
        ['db', 'session']
    """

    def resolve(self, name: str, registry: FixtureRegistry) -> list[str]:
        """Resolve a fixture's complete dependency chain in topological order.

        Performs depth-first search to build a topologically sorted list of
        fixture names, where dependencies appear before dependents.

        Args:
            name: The name of the fixture to resolve.
            registry: The fixture registry to look up definitions.

        Returns:
            List of fixture names in topologically sorted order (dependencies first).
            The requested fixture name appears last.

        Raises:
            FixtureNotFoundError: If the fixture or any dependency doesn't exist.
            CyclicDependencyError: If circular dependencies are detected.
        """
        if not registry.has(name):
            raise FixtureNotFoundError(name)

        result: list[str] = []
        visiting: set[str] = set()  # Currently in recursion stack (for cycle detection)
        visited: set[str] = set()  # Completely processed

        self._resolve_recursive(name, registry, result, visiting, visited, [name])
        return result

    def _resolve_recursive(
        self,
        name: str,
        registry: FixtureRegistry,
        result: list[str],
        visiting: set[str],
        visited: set[str],
        path: list[str],
    ) -> None:
        """Recursively resolve dependencies using DFS.

        Args:
            name: Current fixture name being resolved.
            registry: The fixture registry.
            result: Accumulator for the topologically sorted result.
            visiting: Set of fixtures currently in the recursion stack.
            visited: Set of fixtures completely processed.
            path: Current path for cycle detection error messages.
        """
        if name in visited:
            return

        if name in visiting:
            # Found a cycle - extract the cycle path for error message
            cycle_start = path.index(name)
            cycle = path[cycle_start:] + [name]
            raise CyclicDependencyError(cycle)

        visiting.add(name)

        definition = registry.get(name)
        for dep in definition.dependencies:
            if not registry.has(dep):
                raise FixtureNotFoundError(dep)
            self._resolve_recursive(dep, registry, result, visiting, visited, path + [dep])

        visiting.remove(name)
        visited.add(name)
        result.append(name)

    def validate_dependencies(self, registry: FixtureRegistry) -> list[str]:
        """Validate that all fixtures in a registry have valid dependencies.

        Checks all fixtures for missing dependencies and circular references.
        Collects all validation errors rather than stopping at the first one.

        Args:
            registry: The fixture registry to validate.

        Returns:
            List of validation error messages. Empty list if all valid.
        """
        errors: list[str] = []

        # Check for missing dependencies
        for fixture_name in registry:
            definition = registry.get(fixture_name)
            for dep in definition.dependencies:
                if not registry.has(dep):
                    errors.append(
                        f"Fixture '{fixture_name}' depends on " f"non-existent fixture '{dep}'"
                    )

        # Check for circular dependencies (only if no missing deps)
        if not errors:
            for fixture_name in registry:
                try:
                    self.resolve(fixture_name, registry)
                except CyclicDependencyError as e:
                    cycle_str = " -> ".join(e.cycle)
                    error_msg = f"Circular dependency: {cycle_str}"
                    if error_msg not in errors:  # Avoid duplicate cycle reports
                        errors.append(error_msg)

        return errors
