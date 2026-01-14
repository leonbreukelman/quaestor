"""
TestDesigner - DSPy-based Test Scenario Generator.

Generates test scenarios from AgentWorkflow analysis results.
Produces positive path tests, edge case tests, and error handling tests.

Part of Phase 2: Test Generation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from quaestor.analysis.models import (
    AgentWorkflow,
    StateDefinition,
    StateType,
    ToolDefinition,
)
from quaestor.testing.models import (
    Assertion,
    ContainsAssertion,
    StateReachedAssertion,
    TestCase,
    TestSuite,
    ToolCalledAssertion,
)


class TestScenarioType(str, Enum):
    """Types of test scenarios that can be generated."""

    POSITIVE = "positive"  # Happy path tests
    EDGE_CASE = "edge_case"  # Boundary conditions
    ERROR_HANDLING = "error_handling"  # Error conditions
    STATE_TRANSITION = "state_transition"  # State machine tests
    TOOL_INTERACTION = "tool_interaction"  # Tool usage tests


@dataclass
class TestScenario:
    """
    A generated test scenario with metadata.

    Intermediate representation before converting to TestCase.
    """

    name: str
    description: str
    scenario_type: TestScenarioType
    inputs: dict[str, Any]
    expected_behavior: str
    assertions: list[Assertion] = field(default_factory=list)
    target_tool: str | None = None
    target_state: str | None = None
    priority: int = 1  # 1 = highest priority


@dataclass
class DesignerConfig:
    """Configuration for the test designer."""

    use_mock: bool = False
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    temperature: float = 0.0

    # Generation settings
    max_scenarios_per_tool: int = 3
    max_scenarios_per_state: int = 2
    include_edge_cases: bool = True
    include_error_handling: bool = True
    include_state_transitions: bool = True


@dataclass
class DesignResult:
    """Result of test design generation."""

    scenarios: list[TestScenario] = field(default_factory=list)
    test_suite: TestSuite | None = None
    summary: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def scenario_count(self) -> int:
        """Total number of scenarios generated."""
        return len(self.scenarios)

    @property
    def by_type(self) -> dict[TestScenarioType, list[TestScenario]]:
        """Group scenarios by type."""
        result: dict[TestScenarioType, list[TestScenario]] = {}
        for scenario in self.scenarios:
            if scenario.scenario_type not in result:
                result[scenario.scenario_type] = []
            result[scenario.scenario_type].append(scenario)
        return result


class TestDesigner:
    """
    DSPy-based test designer for agent workflows.

    Generates test scenarios from workflow analysis:
    - Positive path tests for each tool
    - Edge case tests for boundary conditions
    - Error handling tests for failure modes
    - State transition tests for state machines

    MODES:
    - Mock mode (use_mock=True): Fast generation without LLM calls
    - Full mode: Uses DSPy for intelligent test design

    Usage:
        designer = TestDesigner()
        workflow = analyzer.to_agent_workflow(analysis)
        result = designer.design(workflow)
        test_suite = result.test_suite
    """

    def __init__(self, config: DesignerConfig | None = None):
        """Initialize the test designer."""
        self.config = config or DesignerConfig()
        self._dspy_module = None

    def design(self, workflow: AgentWorkflow) -> DesignResult:
        """
        Design test scenarios for an agent workflow.

        Args:
            workflow: AgentWorkflow from WorkflowAnalyzer

        Returns:
            DesignResult with generated scenarios and test suite
        """
        if self.config.use_mock:
            return self._design_mock(workflow)

        return self._design_with_dspy(workflow)

    def _design_mock(self, workflow: AgentWorkflow) -> DesignResult:
        """
        Generate test scenarios without LLM calls.

        Uses heuristics to generate basic test scenarios.
        Fast but less creative than DSPy-based design.
        """
        scenarios: list[TestScenario] = []
        warnings: list[str] = []

        # Generate tool-based tests
        for tool in workflow.tools:
            tool_scenarios = self._generate_tool_scenarios(tool)
            scenarios.extend(tool_scenarios[: self.config.max_scenarios_per_tool])

        # Generate state-based tests
        if self.config.include_state_transitions:
            for state in workflow.states:
                state_scenarios = self._generate_state_scenarios(state, workflow)
                scenarios.extend(state_scenarios[: self.config.max_scenarios_per_state])

        # Generate transition tests
        if self.config.include_state_transitions and workflow.transitions:
            transition_scenarios = self._generate_transition_scenarios(workflow)
            scenarios.extend(transition_scenarios)

        # Generate edge cases
        if self.config.include_edge_cases:
            edge_cases = self._generate_edge_case_scenarios(workflow)
            scenarios.extend(edge_cases)

        # Generate error handling tests
        if self.config.include_error_handling:
            error_scenarios = self._generate_error_scenarios(workflow)
            scenarios.extend(error_scenarios)

        # Validate and warn if few scenarios
        if len(scenarios) < 3:
            warnings.append(
                f"Only {len(scenarios)} scenarios generated. "
                "Consider providing more workflow detail."
            )

        # Build summary
        summary = self._build_summary(workflow, scenarios)

        # Convert to test suite
        test_suite = self._scenarios_to_test_suite(workflow, scenarios)

        return DesignResult(
            scenarios=scenarios,
            test_suite=test_suite,
            summary=summary,
            warnings=warnings,
        )

    def _design_with_dspy(self, workflow: AgentWorkflow) -> DesignResult:
        """
        Generate test scenarios using DSPy.

        Uses language models for intelligent test design.
        """
        # For now, fall back to mock mode
        # TODO: Implement full DSPy integration
        return self._design_mock(workflow)

    def _generate_tool_scenarios(self, tool: ToolDefinition) -> list[TestScenario]:
        """Generate test scenarios for a single tool."""
        scenarios: list[TestScenario] = []

        # Positive path test
        positive_inputs = self._generate_sample_inputs(tool)
        scenarios.append(
            TestScenario(
                name=f"test_{tool.name}_positive",
                description=f"Verify {tool.name} works correctly with valid inputs",
                scenario_type=TestScenarioType.POSITIVE,
                inputs=positive_inputs,
                expected_behavior=f"Tool {tool.name} executes successfully",
                target_tool=tool.name,
                assertions=[
                    ToolCalledAssertion(
                        name=f"assert_{tool.name}_called",
                        description=f"Verify {tool.name} was called",
                        tool_name=tool.name,
                    ),
                ],
                priority=1,
            )
        )

        # Edge case: empty/minimal inputs
        if tool.parameters:
            minimal_inputs = self._generate_minimal_inputs(tool)
            scenarios.append(
                TestScenario(
                    name=f"test_{tool.name}_minimal_inputs",
                    description=f"Verify {tool.name} handles minimal inputs",
                    scenario_type=TestScenarioType.EDGE_CASE,
                    inputs=minimal_inputs,
                    expected_behavior=f"Tool {tool.name} handles minimal inputs gracefully",
                    target_tool=tool.name,
                    assertions=[
                        ToolCalledAssertion(
                            name=f"assert_{tool.name}_called_minimal",
                            description=f"Verify {tool.name} was called with minimal inputs",
                            tool_name=tool.name,
                        ),
                    ],
                    priority=2,
                )
            )

        # Error handling: invalid inputs
        invalid_inputs = self._generate_invalid_inputs(tool)
        if invalid_inputs:
            scenarios.append(
                TestScenario(
                    name=f"test_{tool.name}_invalid_inputs",
                    description=f"Verify {tool.name} handles invalid inputs",
                    scenario_type=TestScenarioType.ERROR_HANDLING,
                    inputs=invalid_inputs,
                    expected_behavior=f"Tool {tool.name} returns appropriate error",
                    target_tool=tool.name,
                    assertions=[
                        ContainsAssertion(
                            name=f"assert_{tool.name}_error_message",
                            description="Verify error message is returned",
                            substring="error",
                        ),
                    ],
                    priority=3,
                )
            )

        return scenarios

    def _generate_state_scenarios(
        self,
        state: StateDefinition,
        workflow: AgentWorkflow,  # noqa: ARG002
    ) -> list[TestScenario]:
        """Generate test scenarios for a state."""
        scenarios: list[TestScenario] = []

        # Test reaching this state
        scenarios.append(
            TestScenario(
                name=f"test_reach_{state.name}",
                description=f"Verify agent can reach {state.name} state",
                scenario_type=TestScenarioType.STATE_TRANSITION,
                inputs={"target_state": state.name},
                expected_behavior=f"Agent transitions to {state.name}",
                target_state=state.name,
                assertions=[
                    StateReachedAssertion(
                        name=f"assert_{state.name}_reached",
                        description=f"Verify {state.name} state is reached",
                        state_name=state.name,
                    ),
                ],
                priority=1 if state.state_type == StateType.INITIAL else 2,
            )
        )

        # Test tools allowed in this state
        if state.allowed_tools:
            for tool_name in state.allowed_tools[:2]:  # Limit to first 2 tools
                scenarios.append(
                    TestScenario(
                        name=f"test_{state.name}_{tool_name}",
                        description=f"Verify {tool_name} can be called in {state.name}",
                        scenario_type=TestScenarioType.TOOL_INTERACTION,
                        inputs={"state": state.name, "tool": tool_name},
                        expected_behavior=f"Tool {tool_name} executes in {state.name}",
                        target_state=state.name,
                        target_tool=tool_name,
                        assertions=[
                            StateReachedAssertion(
                                name=f"assert_in_{state.name}",
                                description=f"Verify agent is in {state.name}",
                                state_name=state.name,
                            ),
                            ToolCalledAssertion(
                                name=f"assert_{tool_name}_called",
                                description=f"Verify {tool_name} was called",
                                tool_name=tool_name,
                            ),
                        ],
                        priority=2,
                    )
                )

        return scenarios

    def _generate_transition_scenarios(self, workflow: AgentWorkflow) -> list[TestScenario]:
        """Generate test scenarios for state transitions."""
        scenarios: list[TestScenario] = []

        for transition in workflow.transitions[:5]:  # Limit to first 5 transitions
            scenarios.append(
                TestScenario(
                    name=f"test_transition_{transition.from_state}_to_{transition.to_state}",
                    description=f"Verify transition from {transition.from_state} to {transition.to_state}",
                    scenario_type=TestScenarioType.STATE_TRANSITION,
                    inputs={
                        "from_state": transition.from_state,
                        "trigger": transition.trigger,
                    },
                    expected_behavior=f"Agent transitions from {transition.from_state} to {transition.to_state}",
                    target_state=transition.to_state,
                    assertions=[
                        StateReachedAssertion(
                            name=f"assert_reached_{transition.to_state}",
                            description=f"Verify {transition.to_state} is reached",
                            state_name=transition.to_state,
                        ),
                    ],
                    priority=2,
                )
            )

        return scenarios

    def _generate_edge_case_scenarios(self, workflow: AgentWorkflow) -> list[TestScenario]:
        """Generate edge case test scenarios."""
        scenarios: list[TestScenario] = []

        # Empty input test
        scenarios.append(
            TestScenario(
                name="test_empty_input",
                description="Verify agent handles empty input gracefully",
                scenario_type=TestScenarioType.EDGE_CASE,
                inputs={},
                expected_behavior="Agent responds appropriately to empty input",
                assertions=[
                    ContainsAssertion(
                        name="assert_response_not_empty",
                        description="Verify agent provides a response",
                        substring=" ",  # At least some response
                    ),
                ],
                priority=2,
            )
        )

        # Large input test (if tools have text parameters)
        text_tools = [
            t
            for t in workflow.tools
            if any(p.type in ("str", "string", "text") for p in t.parameters)
        ]
        if text_tools:
            tool = text_tools[0]
            scenarios.append(
                TestScenario(
                    name=f"test_{tool.name}_large_input",
                    description=f"Verify {tool.name} handles large input",
                    scenario_type=TestScenarioType.EDGE_CASE,
                    inputs={"text": "x" * 10000},  # Large text input
                    expected_behavior=f"Tool {tool.name} handles large input without crash",
                    target_tool=tool.name,
                    assertions=[
                        ToolCalledAssertion(
                            name=f"assert_{tool.name}_called_large",
                            description=f"Verify {tool.name} was called with large input",
                            tool_name=tool.name,
                        ),
                    ],
                    priority=3,
                )
            )

        return scenarios

    def _generate_error_scenarios(self, workflow: AgentWorkflow) -> list[TestScenario]:
        """Generate error handling test scenarios."""
        scenarios: list[TestScenario] = []

        # Timeout scenario
        scenarios.append(
            TestScenario(
                name="test_timeout_handling",
                description="Verify agent handles timeout gracefully",
                scenario_type=TestScenarioType.ERROR_HANDLING,
                inputs={"simulate_timeout": True},
                expected_behavior="Agent handles timeout with appropriate error message",
                assertions=[
                    ContainsAssertion(
                        name="assert_timeout_handled",
                        description="Verify timeout is handled",
                        substring="timeout",
                    ),
                ],
                priority=3,
            )
        )

        # Invalid state transition (if states exist)
        error_states = [s for s in workflow.states if s.state_type == StateType.ERROR]
        if error_states:
            error_state = error_states[0]
            scenarios.append(
                TestScenario(
                    name=f"test_error_state_{error_state.name}",
                    description=f"Verify agent enters {error_state.name} on error",
                    scenario_type=TestScenarioType.ERROR_HANDLING,
                    inputs={"trigger_error": True},
                    expected_behavior=f"Agent transitions to {error_state.name}",
                    target_state=error_state.name,
                    assertions=[
                        StateReachedAssertion(
                            name=f"assert_{error_state.name}_reached",
                            description=f"Verify {error_state.name} is reached",
                            state_name=error_state.name,
                        ),
                    ],
                    priority=2,
                )
            )

        return scenarios

    def _generate_sample_inputs(self, tool: ToolDefinition) -> dict[str, Any]:
        """Generate sample inputs for a tool based on parameter types."""
        inputs: dict[str, Any] = {}
        for param in tool.parameters:
            inputs[param.name] = self._sample_value_for_type(param.type, param.name)
        return inputs

    def _generate_minimal_inputs(self, tool: ToolDefinition) -> dict[str, Any]:
        """Generate minimal required inputs for a tool."""
        inputs: dict[str, Any] = {}
        for param in tool.parameters:
            if param.required:
                inputs[param.name] = self._minimal_value_for_type(param.type)
        return inputs

    def _generate_invalid_inputs(self, tool: ToolDefinition) -> dict[str, Any]:
        """Generate invalid inputs for testing error handling."""
        if not tool.parameters:
            return {}

        inputs: dict[str, Any] = {}
        for param in tool.parameters:
            inputs[param.name] = self._invalid_value_for_type(param.type)
        return inputs

    def _sample_value_for_type(self, type_str: str, param_name: str) -> Any:
        """Generate a sample value based on type string."""
        type_lower = type_str.lower()

        if "int" in type_lower:
            return 42
        if "float" in type_lower or "double" in type_lower:
            return 3.14
        if "bool" in type_lower:
            return True
        if "list" in type_lower or "array" in type_lower:
            return ["item1", "item2"]
        if "dict" in type_lower or "object" in type_lower:
            return {"key": "value"}

        # Default to string with meaningful content
        return f"sample_{param_name}"

    def _minimal_value_for_type(self, type_str: str) -> Any:
        """Generate minimal value for a type."""
        type_lower = type_str.lower()

        if "int" in type_lower:
            return 0
        if "float" in type_lower:
            return 0.0
        if "bool" in type_lower:
            return False
        if "list" in type_lower:
            return []
        if "dict" in type_lower:
            return {}

        return ""

    def _invalid_value_for_type(self, type_str: str) -> Any:
        """Generate an invalid value for testing error handling."""
        type_lower = type_str.lower()

        if "int" in type_lower:
            return "not_an_int"
        if "float" in type_lower:
            return "not_a_float"
        if "bool" in type_lower:
            return "not_a_bool"
        if "list" in type_lower:
            return "not_a_list"
        if "dict" in type_lower:
            return "not_a_dict"

        return None  # None might be invalid for required string params

    def _build_summary(
        self,
        workflow: AgentWorkflow,
        scenarios: list[TestScenario],
    ) -> str:
        """Build a summary of the generated test design."""
        type_counts: dict[str, int] = {}
        for scenario in scenarios:
            t = scenario.scenario_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        parts = [f"Generated {len(scenarios)} test scenarios for {workflow.name}:"]
        for stype, count in type_counts.items():
            parts.append(f"  - {stype}: {count}")

        return "\n".join(parts)

    def _scenarios_to_test_suite(
        self,
        workflow: AgentWorkflow,
        scenarios: list[TestScenario],
    ) -> TestSuite:
        """Convert scenarios to a TestSuite."""
        test_cases = []

        for scenario in scenarios:
            test_case = TestCase(
                id=f"tc_{uuid4().hex[:8]}",
                name=scenario.name,
                description=scenario.description,
                input=scenario.inputs,
                assertions=scenario.assertions,
                target_tool=scenario.target_tool,
                target_state=scenario.target_state,
                tags=[scenario.scenario_type.value],
            )
            test_cases.append(test_case)

        return TestSuite(
            id=f"ts_{uuid4().hex[:8]}",
            name=f"Generated Tests for {workflow.name}",
            description=f"Automatically generated test suite for {workflow.name}",
            test_cases=test_cases,
            tags=["auto-generated", "quaestor"],
        )


# Convenience functions


def design_tests(
    workflow: AgentWorkflow,
    use_mock: bool = False,
) -> DesignResult:
    """
    Design tests for an agent workflow.

    Args:
        workflow: AgentWorkflow from WorkflowAnalyzer
        use_mock: Whether to use mock mode (no LLM calls)

    Returns:
        DesignResult with generated test scenarios
    """
    config = DesignerConfig(use_mock=use_mock)
    designer = TestDesigner(config)
    return designer.design(workflow)
