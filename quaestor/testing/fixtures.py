"""
Pydantic models for the fixture system.

This module provides data models for representing test fixtures, including
fixture definitions (metadata and configuration), fixture values (resolved
instances), and fixture scopes (lifetime management).

Follows patterns established in quaestor.testing.models.
"""

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
