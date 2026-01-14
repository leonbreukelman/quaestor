"""Tests for Quaestor domain models."""

import pytest

from quaestor.analysis.models import (
    AgentWorkflow,
    Invariant,
    StateDefinition,
    StateType,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    Transition,
)


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_basic_tool_creation(self):
        """Test creating a basic tool definition."""
        tool = ToolDefinition(
            name="search_documents",
            description="Search for documents in the knowledge base",
            category=ToolCategory.RETRIEVAL,
            parameters=[
                ToolParameter(
                    name="query",
                    type="str",
                    description="Search query",
                    required=True,
                )
            ],
            return_type="list[Document]",
        )

        assert tool.name == "search_documents"
        assert tool.category == ToolCategory.RETRIEVAL
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "query"

    def test_tool_with_side_effects(self):
        """Test tool with side effects marked."""
        tool = ToolDefinition(
            name="send_email",
            description="Send an email",
            category=ToolCategory.COMMUNICATION,
            side_effects=["sends_email", "logs_action"],
            requires_confirmation=True,
        )

        assert "sends_email" in tool.side_effects
        assert tool.requires_confirmation is True


class TestStateDefinition:
    """Tests for StateDefinition model."""

    def test_initial_state(self):
        """Test creating an initial state."""
        state = StateDefinition(
            name="awaiting_input",
            description="Waiting for user input",
            state_type=StateType.INITIAL,
            allowed_tools=["greet_user", "get_context"],
        )

        assert state.state_type == StateType.INITIAL
        assert "greet_user" in state.allowed_tools

    def test_terminal_state(self):
        """Test creating a terminal state."""
        state = StateDefinition(
            name="task_completed",
            description="Task has been successfully completed",
            state_type=StateType.TERMINAL,
        )

        assert state.state_type == StateType.TERMINAL


class TestAgentWorkflow:
    """Tests for AgentWorkflow model."""

    @pytest.fixture
    def sample_workflow(self) -> AgentWorkflow:
        """Create a sample workflow for testing."""
        return AgentWorkflow(
            name="CustomerSupportAgent",
            description="Handles customer support inquiries",
            tools=[
                ToolDefinition(
                    name="search_kb",
                    description="Search knowledge base",
                    category=ToolCategory.RETRIEVAL,
                ),
                ToolDefinition(
                    name="create_ticket",
                    description="Create support ticket",
                    category=ToolCategory.ACTION,
                ),
            ],
            states=[
                StateDefinition(
                    name="greeting",
                    description="Initial greeting",
                    state_type=StateType.INITIAL,
                ),
                StateDefinition(
                    name="investigating",
                    description="Investigating issue",
                    state_type=StateType.INTERMEDIATE,
                ),
                StateDefinition(
                    name="resolved",
                    description="Issue resolved",
                    state_type=StateType.TERMINAL,
                ),
            ],
            transitions=[
                Transition(
                    from_state="greeting",
                    to_state="investigating",
                    trigger="user_describes_issue",
                ),
                Transition(
                    from_state="investigating",
                    to_state="resolved",
                    trigger="solution_provided",
                ),
            ],
            invariants=[
                Invariant(
                    name="no_pii_exposure",
                    description="Never expose customer PII",
                    expression="not contains_pii(response)",
                    severity="error",
                    category="security",
                ),
            ],
        )

    def test_get_initial_state(self, sample_workflow: AgentWorkflow):
        """Test getting the initial state."""
        initial = sample_workflow.get_initial_state()
        assert initial is not None
        assert initial.name == "greeting"

    def test_get_terminal_states(self, sample_workflow: AgentWorkflow):
        """Test getting terminal states."""
        terminals = sample_workflow.get_terminal_states()
        assert len(terminals) == 1
        assert terminals[0].name == "resolved"

    def test_get_tool_by_name(self, sample_workflow: AgentWorkflow):
        """Test looking up a tool by name."""
        tool = sample_workflow.get_tool_by_name("search_kb")
        assert tool is not None
        assert tool.category == ToolCategory.RETRIEVAL

        missing = sample_workflow.get_tool_by_name("nonexistent")
        assert missing is None

    def test_get_transitions_from(self, sample_workflow: AgentWorkflow):
        """Test getting transitions from a state."""
        transitions = sample_workflow.get_transitions_from("greeting")
        assert len(transitions) == 1
        assert transitions[0].to_state == "investigating"

        # No transitions from terminal state
        transitions = sample_workflow.get_transitions_from("resolved")
        assert len(transitions) == 0
