"""
Domain models for agent workflow analysis.

These models represent the complete understanding of an agent's workflow,
extracted via the WorkflowAnalyzer DSPy module.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Categories of agent tools."""

    RETRIEVAL = "retrieval"
    ACTION = "action"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    STATE_MUTATION = "state_mutation"
    EXTERNAL_API = "external_api"


class StateType(str, Enum):
    """Types of agent states."""

    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    TERMINAL = "terminal"
    ERROR = "error"


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (Python type annotation)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not required")


class ToolDefinition(BaseModel):
    """Definition of an agent tool/function."""

    name: str = Field(..., description="Tool function name")
    description: str = Field(..., description="What the tool does")
    category: ToolCategory = Field(..., description="Tool category")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool parameters")
    return_type: str = Field(default="Any", description="Return type annotation")
    side_effects: list[str] = Field(
        default_factory=list, description="Side effects (e.g., 'modifies_database', 'sends_email')"
    )
    requires_confirmation: bool = Field(
        default=False, description="Whether tool requires user confirmation"
    )

    # Source location for traceability
    source_file: str | None = Field(default=None, description="Source file path")
    source_line: int | None = Field(default=None, description="Line number in source")


class StateVariable(BaseModel):
    """A variable in the agent's state."""

    name: str = Field(..., description="Variable name")
    type: str = Field(..., description="Variable type")
    description: str = Field(default="", description="What this variable represents")
    initial_value: Any = Field(default=None, description="Initial value")


class StateDefinition(BaseModel):
    """Definition of an agent state."""

    name: str = Field(..., description="State name")
    description: str = Field(..., description="What this state represents")
    state_type: StateType = Field(..., description="Type of state")
    variables: list[StateVariable] = Field(
        default_factory=list, description="Variables in this state"
    )
    allowed_tools: list[str] = Field(
        default_factory=list, description="Tools that can be called in this state"
    )


class TransitionCondition(BaseModel):
    """Condition for a state transition."""

    expression: str = Field(..., description="Condition expression (Python-like)")
    description: str = Field(default="", description="Human-readable description")


class Transition(BaseModel):
    """A transition between agent states."""

    from_state: str = Field(..., description="Source state name")
    to_state: str = Field(..., description="Target state name")
    trigger: str = Field(..., description="What triggers this transition")
    conditions: list[TransitionCondition] = Field(
        default_factory=list, description="Conditions that must be met"
    )
    actions: list[str] = Field(
        default_factory=list, description="Actions performed during transition"
    )


class Invariant(BaseModel):
    """An invariant that must hold throughout agent execution."""

    name: str = Field(..., description="Invariant name")
    description: str = Field(..., description="What this invariant ensures")
    expression: str = Field(..., description="Invariant expression")
    severity: str = Field(default="error", description="Severity if violated: error, warning, info")
    category: str = Field(
        default="correctness", description="Category: correctness, safety, security, performance"
    )


class EntryPoint(BaseModel):
    """An entry point into the agent."""

    name: str = Field(..., description="Entry point name")
    description: str = Field(..., description="What this entry point does")
    method: str = Field(..., description="HTTP method or invocation type")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="Expected input schema")
    initial_state: str = Field(..., description="State after entry")


class AgentWorkflow(BaseModel):
    """
    Complete understanding of an agent's workflow.

    This is the primary output of the WorkflowAnalyzer,
    representing everything Quaestor knows about an agent.
    """

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent purpose and capabilities")
    version: str = Field(default="1.0.0", description="Agent version")

    # Core workflow components
    tools: list[ToolDefinition] = Field(
        default_factory=list, description="Tools available to the agent"
    )
    states: list[StateDefinition] = Field(default_factory=list, description="Possible agent states")
    transitions: list[Transition] = Field(default_factory=list, description="State transitions")
    invariants: list[Invariant] = Field(
        default_factory=list, description="Invariants that must hold"
    )
    entry_points: list[EntryPoint] = Field(
        default_factory=list, description="Ways to invoke the agent"
    )

    # Metadata
    source_files: list[str] = Field(default_factory=list, description="Source files analyzed")
    analysis_timestamp: str | None = Field(default=None, description="When analysis was performed")
    analyzer_version: str | None = Field(default=None, description="Version of analyzer used")

    # Governance integration
    governance_rules: list[str] = Field(
        default_factory=list, description="Applicable governance rule IDs from Smactorio"
    )

    def get_initial_state(self) -> StateDefinition | None:
        """Get the initial state of the agent."""
        for state in self.states:
            if state.state_type == StateType.INITIAL:
                return state
        return None

    def get_terminal_states(self) -> list[StateDefinition]:
        """Get all terminal states."""
        return [s for s in self.states if s.state_type == StateType.TERMINAL]

    def get_tool_by_name(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_transitions_from(self, state_name: str) -> list[Transition]:
        """Get all transitions from a given state."""
        return [t for t in self.transitions if t.from_state == state_name]
