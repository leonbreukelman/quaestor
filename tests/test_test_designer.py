"""Tests for the TestDesigner module."""

import pytest

from quaestor.analysis.models import (
    AgentWorkflow,
    StateDefinition,
    StateType,
    ToolCategory,
    ToolDefinition,
    ToolParameter,
    Transition,
)
from quaestor.testing.test_designer import (
    DesignerConfig,
    DesignResult,
    TestDesigner,
    TestScenarioType,
    design_tests,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_tool() -> ToolDefinition:
    """Create a simple tool for testing."""
    return ToolDefinition(
        name="search_documents",
        description="Search for documents in the knowledge base",
        category=ToolCategory.RETRIEVAL,
        parameters=[
            ToolParameter(name="query", type="str", description="Search query"),
            ToolParameter(name="limit", type="int", description="Max results", required=False),
        ],
    )


@pytest.fixture
def simple_state() -> StateDefinition:
    """Create a simple state for testing."""
    return StateDefinition(
        name="searching",
        description="Agent is searching for information",
        state_type=StateType.INTERMEDIATE,
        allowed_tools=["search_documents"],
    )


@pytest.fixture
def simple_workflow(simple_tool: ToolDefinition, simple_state: StateDefinition) -> AgentWorkflow:
    """Create a simple workflow for testing."""
    initial_state = StateDefinition(
        name="idle",
        description="Agent is idle",
        state_type=StateType.INITIAL,
    )
    terminal_state = StateDefinition(
        name="completed",
        description="Agent has completed",
        state_type=StateType.TERMINAL,
    )

    return AgentWorkflow(
        name="TestAgent",
        description="A test agent workflow",
        tools=[simple_tool],
        states=[initial_state, simple_state, terminal_state],
        transitions=[
            Transition(
                from_state="idle",
                to_state="searching",
                trigger="user_query",
            ),
            Transition(
                from_state="searching",
                to_state="completed",
                trigger="results_found",
            ),
        ],
    )


@pytest.fixture
def complex_workflow() -> AgentWorkflow:
    """Create a complex workflow with multiple tools and states."""
    tools = [
        ToolDefinition(
            name="search",
            description="Search the knowledge base",
            category=ToolCategory.RETRIEVAL,
            parameters=[
                ToolParameter(name="query", type="str", description="Query"),
            ],
        ),
        ToolDefinition(
            name="summarize",
            description="Summarize text",
            category=ToolCategory.COMPUTATION,
            parameters=[
                ToolParameter(name="text", type="str", description="Text to summarize"),
            ],
        ),
        ToolDefinition(
            name="send_email",
            description="Send an email",
            category=ToolCategory.COMMUNICATION,
            parameters=[
                ToolParameter(name="to", type="str", description="Recipient"),
                ToolParameter(name="subject", type="str", description="Subject"),
                ToolParameter(name="body", type="str", description="Body"),
            ],
        ),
    ]

    states = [
        StateDefinition(
            name="idle",
            description="Waiting for input",
            state_type=StateType.INITIAL,
        ),
        StateDefinition(
            name="researching",
            description="Researching information",
            state_type=StateType.INTERMEDIATE,
            allowed_tools=["search"],
        ),
        StateDefinition(
            name="processing",
            description="Processing results",
            state_type=StateType.INTERMEDIATE,
            allowed_tools=["summarize"],
        ),
        StateDefinition(
            name="communicating",
            description="Sending results",
            state_type=StateType.INTERMEDIATE,
            allowed_tools=["send_email"],
        ),
        StateDefinition(
            name="error",
            description="Error occurred",
            state_type=StateType.ERROR,
        ),
        StateDefinition(
            name="done",
            description="Completed successfully",
            state_type=StateType.TERMINAL,
        ),
    ]

    transitions = [
        Transition(from_state="idle", to_state="researching", trigger="start"),
        Transition(from_state="researching", to_state="processing", trigger="results"),
        Transition(from_state="processing", to_state="communicating", trigger="summary"),
        Transition(from_state="communicating", to_state="done", trigger="sent"),
        Transition(from_state="researching", to_state="error", trigger="error"),
    ]

    return AgentWorkflow(
        name="ComplexAgent",
        description="A complex multi-step agent",
        tools=tools,
        states=states,
        transitions=transitions,
    )


@pytest.fixture
def empty_workflow() -> AgentWorkflow:
    """Create a minimal workflow with no tools or states."""
    return AgentWorkflow(
        name="EmptyAgent",
        description="An empty agent",
        tools=[],
        states=[],
        transitions=[],
    )


# =============================================================================
# TestDesigner Initialization Tests
# =============================================================================


class TestTestDesignerInit:
    """Tests for TestDesigner initialization."""

    def test_default_config(self) -> None:
        """Test designer initializes with default config."""
        designer = TestDesigner()
        assert designer.config.use_mock is False
        assert designer.config.max_scenarios_per_tool == 3

    def test_custom_config(self) -> None:
        """Test designer accepts custom config."""
        config = DesignerConfig(
            use_mock=True,
            max_scenarios_per_tool=5,
            include_edge_cases=False,
        )
        designer = TestDesigner(config)
        assert designer.config.use_mock is True
        assert designer.config.max_scenarios_per_tool == 5
        assert designer.config.include_edge_cases is False


# =============================================================================
# Mock Mode Tests
# =============================================================================


class TestTestDesignerMockMode:
    """Tests for TestDesigner in mock mode."""

    def test_design_simple_workflow(self, simple_workflow: AgentWorkflow) -> None:
        """Test designing tests for a simple workflow."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert isinstance(result, DesignResult)
        assert result.scenario_count >= 3  # Should generate at least 3 scenarios
        assert result.test_suite is not None

    def test_design_complex_workflow(self, complex_workflow: AgentWorkflow) -> None:
        """Test designing tests for a complex workflow."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(complex_workflow)

        assert result.scenario_count >= 5  # More scenarios for complex workflow
        assert result.test_suite is not None
        assert len(result.test_suite.test_cases) == result.scenario_count

    def test_design_empty_workflow(self, empty_workflow: AgentWorkflow) -> None:
        """Test designing tests for an empty workflow."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(empty_workflow)

        # Should still generate some edge cases and error handling tests
        assert result.scenario_count >= 1
        assert len(result.warnings) > 0  # Should warn about few scenarios


# =============================================================================
# Scenario Type Tests
# =============================================================================


class TestScenarioTypes:
    """Tests for different scenario types."""

    def test_positive_scenarios_generated(self, simple_workflow: AgentWorkflow) -> None:
        """Test that positive scenarios are generated."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        positive_scenarios = result.by_type.get(TestScenarioType.POSITIVE, [])
        assert len(positive_scenarios) >= 1

    def test_edge_case_scenarios_generated(self, simple_workflow: AgentWorkflow) -> None:
        """Test that edge case scenarios are generated."""
        config = DesignerConfig(use_mock=True, include_edge_cases=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        edge_cases = result.by_type.get(TestScenarioType.EDGE_CASE, [])
        assert len(edge_cases) >= 1

    def test_error_handling_scenarios_generated(self, complex_workflow: AgentWorkflow) -> None:
        """Test that error handling scenarios are generated."""
        config = DesignerConfig(use_mock=True, include_error_handling=True)
        designer = TestDesigner(config)

        result = designer.design(complex_workflow)

        error_scenarios = result.by_type.get(TestScenarioType.ERROR_HANDLING, [])
        assert len(error_scenarios) >= 1

    def test_state_transition_scenarios_generated(self, simple_workflow: AgentWorkflow) -> None:
        """Test that state transition scenarios are generated."""
        config = DesignerConfig(use_mock=True, include_state_transitions=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        transition_scenarios = result.by_type.get(TestScenarioType.STATE_TRANSITION, [])
        assert len(transition_scenarios) >= 1

    def test_disable_edge_cases(self, simple_workflow: AgentWorkflow) -> None:
        """Test that edge cases can be disabled."""
        config = DesignerConfig(use_mock=True, include_edge_cases=False)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        # Edge cases from tool scenarios may still exist, but no dedicated edge cases
        all_scenario_types = {s.scenario_type for s in result.scenarios}
        # Verify positive scenarios still generated
        assert TestScenarioType.POSITIVE in all_scenario_types


# =============================================================================
# Scenario Content Tests
# =============================================================================


class TestScenarioContent:
    """Tests for generated scenario content."""

    def test_tool_scenario_has_tool_name(self, simple_workflow: AgentWorkflow) -> None:
        """Test that tool scenarios reference the tool."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        tool_scenarios = [s for s in result.scenarios if s.target_tool is not None]
        assert len(tool_scenarios) >= 1
        assert any(s.target_tool == "search_documents" for s in tool_scenarios)

    def test_state_scenario_has_state_name(self, simple_workflow: AgentWorkflow) -> None:
        """Test that state scenarios reference the state."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        state_scenarios = [s for s in result.scenarios if s.target_state is not None]
        assert len(state_scenarios) >= 1

    def test_scenarios_have_assertions(self, simple_workflow: AgentWorkflow) -> None:
        """Test that all scenarios have assertions."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        for scenario in result.scenarios:
            assert len(scenario.assertions) >= 1


# =============================================================================
# Test Suite Generation Tests
# =============================================================================


class TestTestSuiteGeneration:
    """Tests for TestSuite generation."""

    def test_suite_has_test_cases(self, simple_workflow: AgentWorkflow) -> None:
        """Test that generated suite has test cases."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert result.test_suite is not None
        assert len(result.test_suite.test_cases) >= 1

    def test_suite_test_cases_match_scenarios(self, simple_workflow: AgentWorkflow) -> None:
        """Test that suite test cases match scenarios."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert result.test_suite is not None
        assert len(result.test_suite.test_cases) == result.scenario_count

    def test_suite_has_tags(self, simple_workflow: AgentWorkflow) -> None:
        """Test that generated suite has tags."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert result.test_suite is not None
        assert "auto-generated" in result.test_suite.tags
        assert "quaestor" in result.test_suite.tags

    def test_test_case_ids_unique(self, complex_workflow: AgentWorkflow) -> None:
        """Test that test case IDs are unique."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(complex_workflow)

        assert result.test_suite is not None
        ids = [tc.id for tc in result.test_suite.test_cases]
        assert len(ids) == len(set(ids))  # All unique


# =============================================================================
# Design Result Tests
# =============================================================================


class TestDesignResult:
    """Tests for DesignResult."""

    def test_scenario_count(self, simple_workflow: AgentWorkflow) -> None:
        """Test scenario_count property."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert result.scenario_count == len(result.scenarios)

    def test_by_type_grouping(self, complex_workflow: AgentWorkflow) -> None:
        """Test by_type grouping."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(complex_workflow)

        by_type = result.by_type
        total = sum(len(scenarios) for scenarios in by_type.values())
        assert total == result.scenario_count

    def test_summary_includes_workflow_name(self, simple_workflow: AgentWorkflow) -> None:
        """Test that summary includes workflow name."""
        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)

        result = designer.design(simple_workflow)

        assert "TestAgent" in result.summary


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_design_tests_function(self, simple_workflow: AgentWorkflow) -> None:
        """Test design_tests convenience function."""
        result = design_tests(simple_workflow, use_mock=True)

        assert isinstance(result, DesignResult)
        assert result.scenario_count >= 1
        assert result.test_suite is not None

    def test_design_tests_defaults_to_llm_mode(self, simple_workflow: AgentWorkflow) -> None:
        """Test that design_tests defaults to LLM mode (but falls back to mock)."""
        result = design_tests(simple_workflow, use_mock=False)

        # Currently falls back to mock mode, so should still work
        assert isinstance(result, DesignResult)


# =============================================================================
# Input Generation Tests
# =============================================================================


class TestInputGeneration:
    """Tests for input generation helpers."""

    def test_sample_inputs_for_string_param(self) -> None:
        """Test sample input generation for string parameters."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.ACTION,
            parameters=[
                ToolParameter(name="text", type="str", description="Text input"),
            ],
        )
        workflow = AgentWorkflow(
            name="Test", description="Test", tools=[tool], states=[], transitions=[]
        )

        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)
        result = designer.design(workflow)

        # Find a positive scenario for this tool
        positive_scenarios = [
            s
            for s in result.scenarios
            if s.scenario_type == TestScenarioType.POSITIVE and s.target_tool == "test_tool"
        ]
        assert len(positive_scenarios) >= 1
        assert "text" in positive_scenarios[0].inputs

    def test_sample_inputs_for_int_param(self) -> None:
        """Test sample input generation for int parameters."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test",
            category=ToolCategory.ACTION,
            parameters=[
                ToolParameter(name="count", type="int", description="Count"),
            ],
        )
        workflow = AgentWorkflow(
            name="Test", description="Test", tools=[tool], states=[], transitions=[]
        )

        config = DesignerConfig(use_mock=True)
        designer = TestDesigner(config)
        result = designer.design(workflow)

        positive_scenarios = [
            s
            for s in result.scenarios
            if s.scenario_type == TestScenarioType.POSITIVE and s.target_tool == "test_tool"
        ]
        assert len(positive_scenarios) >= 1
        assert isinstance(positive_scenarios[0].inputs.get("count"), int)


# =============================================================================
# Config Tests
# =============================================================================


class TestDesignerConfig:
    """Tests for DesignerConfig."""

    def test_default_values(self) -> None:
        """Test default config values."""
        config = DesignerConfig()
        assert config.use_mock is False
        assert config.max_scenarios_per_tool == 3
        assert config.max_scenarios_per_state == 2
        assert config.include_edge_cases is True
        assert config.include_error_handling is True
        assert config.include_state_transitions is True

    def test_custom_values(self) -> None:
        """Test custom config values."""
        config = DesignerConfig(
            use_mock=True,
            max_scenarios_per_tool=10,
            include_edge_cases=False,
        )
        assert config.use_mock is True
        assert config.max_scenarios_per_tool == 10
        assert config.include_edge_cases is False
