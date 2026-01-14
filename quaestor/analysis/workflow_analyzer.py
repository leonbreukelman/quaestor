"""
Workflow Analyzer - DSPy-based Semantic Analysis.

Performs deep semantic analysis of agent code using DSPy.
Extracts workflow patterns, tools, states, and execution paths.

Part of Phase 1: Core Analysis Engine.
"""

from dataclasses import dataclass, field

from quaestor.analysis.models import (
    AgentWorkflow,
    EntryPoint,
    StateDefinition,
    StateType,
    ToolDefinition,
    ToolParameter,
)
from quaestor.analysis.parser import FunctionDef, ParsedCode


@dataclass
class DetectedTool:
    """A tool detected in the agent code."""

    name: str
    description: str
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None
    is_async: bool = False


@dataclass
class DetectedState:
    """A state detected in the agent code."""

    name: str
    type: str  # "initial", "intermediate", "terminal"
    transitions_from: list[str] = field(default_factory=list)
    transitions_to: list[str] = field(default_factory=list)


@dataclass
class WorkflowAnalysis:
    """Complete workflow analysis result."""

    summary: str
    complexity_score: float
    tools_detected: list[DetectedTool] = field(default_factory=list)
    states_detected: list[DetectedState] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    execution_paths: list[list[str]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AnalyzerConfig:
    """Configuration for the workflow analyzer."""

    use_mock: bool = False
    model: str = "gpt-4o-mini"
    max_retries: int = 3
    temperature: float = 0.0

    # Analysis settings
    detect_tools: bool = True
    detect_states: bool = True
    detect_entry_points: bool = True
    generate_recommendations: bool = True


class WorkflowAnalyzer:
    """
    DSPy-based workflow analyzer for agent code.

    Performs semantic analysis to extract:
    - Tool definitions and capabilities
    - State machine patterns
    - Entry points and execution paths
    - Complexity metrics
    - Improvement recommendations

    MODES:
    - Mock mode (use_mock=True): Fast analysis without LLM calls
    - Full mode: Uses DSPy for deep semantic understanding

    Usage:
        analyzer = WorkflowAnalyzer()
        parsed = parse_python_file("agent.py")
        analysis = analyzer.analyze(parsed)
    """

    # Common tool decorator names
    TOOL_DECORATORS = {"tool", "function_tool", "agent_tool", "action"}

    # Common entry point method names
    ENTRY_POINT_NAMES = {"run", "execute", "__call__", "invoke", "arun", "ainvoke"}

    def __init__(self, config: AnalyzerConfig | None = None):
        """Initialize the analyzer."""
        self.config = config or AnalyzerConfig()
        self._dspy_module = None

    def analyze(self, parsed: ParsedCode) -> WorkflowAnalysis:
        """
        Analyze parsed code and extract workflow information.

        Args:
            parsed: ParsedCode from PythonParser

        Returns:
            WorkflowAnalysis with extracted workflow information
        """
        if self.config.use_mock:
            return self._analyze_mock(parsed)

        return self._analyze_with_dspy(parsed)

    def _analyze_mock(self, parsed: ParsedCode) -> WorkflowAnalysis:
        """
        Perform mock analysis without LLM calls.

        Uses heuristics to extract basic workflow information.
        Fast but less accurate than DSPy analysis.
        """
        tools = self._extract_tools(parsed) if self.config.detect_tools else []
        states = self._extract_states(parsed) if self.config.detect_states else []
        entry_points = self._extract_entry_points(parsed) if self.config.detect_entry_points else []

        # Calculate complexity
        complexity = self._calculate_complexity(parsed, tools, states)

        # Generate recommendations
        recommendations = []
        if self.config.generate_recommendations:
            recommendations = self._generate_recommendations(parsed, tools, states)

        # Build summary
        summary = self._build_summary(parsed, tools, states, entry_points)

        return WorkflowAnalysis(
            summary=summary,
            complexity_score=complexity,
            tools_detected=tools,
            states_detected=states,
            entry_points=entry_points,
            execution_paths=[],  # Requires deeper analysis
            recommendations=recommendations,
        )

    def _analyze_with_dspy(self, parsed: ParsedCode) -> WorkflowAnalysis:
        """
        Perform full analysis using DSPy.

        Uses LLM for deep semantic understanding.
        """
        # For now, fall back to mock analysis
        # Full DSPy integration to be implemented in Phase 2
        return self._analyze_mock(parsed)

    def _extract_tools(self, parsed: ParsedCode) -> list[DetectedTool]:
        """Extract tools from parsed code."""
        tools = []

        # Check top-level functions
        for func in parsed.functions:
            if self._is_tool(func):
                tools.append(self._function_to_tool(func))

        # Check class methods
        for cls in parsed.classes:
            for method in cls.methods:
                if self._is_tool(method):
                    tools.append(self._function_to_tool(method))

        return tools

    def _is_tool(self, func: FunctionDef) -> bool:
        """Check if a function is a tool."""
        # Check decorators
        return any(decorator.name in self.TOOL_DECORATORS for decorator in func.decorators)

    def _function_to_tool(self, func: FunctionDef) -> DetectedTool:
        """Convert a FunctionDef to a DetectedTool."""
        params = [p.name for p in func.parameters if p.name != "self"]

        return DetectedTool(
            name=func.name,
            description=func.docstring or f"Tool: {func.name}",
            parameters=params,
            return_type=func.return_type,
            is_async=func.is_async,
        )

    def _extract_states(self, parsed: ParsedCode) -> list[DetectedState]:
        """Extract states from parsed code."""
        states = []

        for cls in parsed.classes:
            # Check if it's an Enum class (likely a state enum)
              if any(base in ["Enum", "str, Enum", "IntEnum"] for base in cls.bases) and cls.body_text:
                  # Extract enum members as states
                  # This is a simplified extraction - full implementation would parse the class body
                    lines = cls.body_text.split("\n")
                    for line in lines:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            name = line.split("=")[0].strip()
                            if name and name.isupper():
                                state_type = self._infer_state_type(name)
                                states.append(
                                    DetectedState(
                                        name=name,
                                        type=state_type,
                                    )
                                )

        return states

    def _infer_state_type(self, name: str) -> str:
        """Infer the type of a state from its name."""
        name_lower = name.lower()

        if any(s in name_lower for s in ["idle", "init", "start", "ready"]):
            return "initial"
        elif any(s in name_lower for s in ["done", "complete", "finish", "end", "error", "fail"]):
            return "terminal"
        else:
            return "intermediate"

    def _extract_entry_points(self, parsed: ParsedCode) -> list[str]:
        """Extract entry points from parsed code."""
        entry_points = []

        # Check top-level functions
        for func in parsed.functions:
            if func.name in self.ENTRY_POINT_NAMES:
                entry_points.append(func.name)

        # Check class methods
        for cls in parsed.classes:
            for method in cls.methods:
                if method.name in self.ENTRY_POINT_NAMES:
                    entry_points.append(method.name)

        return list(set(entry_points))  # Deduplicate

    def _calculate_complexity(
        self,
        parsed: ParsedCode,
        tools: list[DetectedTool],
        states: list[DetectedState],
    ) -> float:
        """Calculate workflow complexity score."""
        score = 1.0  # Base complexity

        # Add complexity for tools
        score += len(tools) * 0.5

        # Add complexity for states
        score += len(states) * 0.3

        # Add complexity for classes
        score += len(parsed.classes) * 0.5

        # Add complexity for async operations
        async_funcs = sum(1 for f in parsed.functions if f.is_async)
        for cls in parsed.classes:
            async_funcs += sum(1 for m in cls.methods if m.is_async)
        score += async_funcs * 0.2

        return round(score, 2)

    def _generate_recommendations(
        self,
        parsed: ParsedCode,
        tools: list[DetectedTool],
        states: list[DetectedState],
    ) -> list[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Check for undocumented tools
        undocumented = [t for t in tools if not t.description or t.description.startswith("Tool:")]
        if undocumented:
            recommendations.append(f"Add docstrings to {len(undocumented)} undocumented tool(s)")

        # Check for missing error handling in async methods
        for cls in parsed.classes:
            for method in cls.methods:
                  if method.is_async and method.body_text and "await " in method.body_text and "try:" not in method.body_text:
                        recommendations.append(
                            f"Add error handling to async method '{method.name}'"
                        )

        # Check for state machine patterns without complete coverage
        if states:
            terminal_states = [s for s in states if s.type == "terminal"]
            if not terminal_states:
                recommendations.append("Add terminal state(s) to state machine")

            initial_states = [s for s in states if s.type == "initial"]
            if not initial_states:
                recommendations.append("Add initial state to state machine")

        return recommendations

    def _build_summary(
        self,
        parsed: ParsedCode,
        tools: list[DetectedTool],
        states: list[DetectedState],
        entry_points: list[str],
    ) -> str:
        """Build a summary of the workflow."""
        parts = []

        # Module info
        if parsed.module_docstring:
            parts.append(parsed.module_docstring.split("\n")[0])

        # Component counts
        counts = []
        if tools:
            counts.append(f"{len(tools)} tool(s)")
        if states:
            counts.append(f"{len(states)} state(s)")
        if parsed.classes:
            counts.append(f"{len(parsed.classes)} class(es)")
        if entry_points:
            counts.append(f"{len(entry_points)} entry point(s)")

        if counts:
            parts.append(f"Contains: {', '.join(counts)}")

        return " | ".join(parts) if parts else "Agent workflow analysis"

    def to_agent_workflow(self, analysis: WorkflowAnalysis) -> AgentWorkflow:
        """
        Convert WorkflowAnalysis to AgentWorkflow domain model.

        Args:
            analysis: WorkflowAnalysis from analyze()

        Returns:
            AgentWorkflow model for further processing
        """
        # Import ToolCategory for categorization
        from quaestor.analysis.models import ToolCategory

        # Convert tools
        tools = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                category=ToolCategory.COMPUTATION,  # Default category
                parameters=[ToolParameter(name=p, type="Any", required=True) for p in t.parameters],
                return_type=t.return_type or "Any",
            )
            for t in analysis.tools_detected
        ]

        # Convert states
        states = [
            StateDefinition(
                name=s.name,
                state_type=StateType(s.type)
                if s.type in ["initial", "intermediate", "terminal"]
                else StateType.INTERMEDIATE,
                description=f"State: {s.name}",
            )
            for s in analysis.states_detected
        ]

        # Create entry points
        entry_points = [
            EntryPoint(
                name=ep,
                description=f"Entry point: {ep}",
                method="invoke",  # Default method
                initial_state=states[0].name if states else "UNKNOWN",
            )
            for ep in analysis.entry_points
        ]

        return AgentWorkflow(
            name="AnalyzedWorkflow",
            description=analysis.summary,
            tools=tools,
            states=states,
            entry_points=entry_points,
        )
