# Quaestor Implementation Plan

> **Goal**: Build a self-optimizing agentic testing framework that feels like pytest/ruff but tests AI agent workflows.

## Quick Reference

| Phase | Deliverable | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 0 | Project scaffold + CLI skeleton | 1-2 days | None |
| 1 | WorkflowAnalyzer MVP | 3-5 days | Phase 0 |
| 2 | Static linting (no LLM) | 2-3 days | Phase 0 |
| 3 | DeepEval integration | 2-3 days | Phase 1 |
| 4 | Test generation + DSPy optimization | 5-7 days | Phase 1, 3 |
| 5 | Multi-turn investigation | 5-7 days | Phase 4 |
| 6 | Red team (DeepTeam integration) | 3-5 days | Phase 5 |
| 7 | Coverage + reporting | 3-4 days | Phase 5 |

**Total estimated: 4-6 weeks to functional MVP**

---

## Phase 0: Project Foundation

### 0.1 Repository Structure

```
quaestor/
├── pyproject.toml
├── README.md
├── quaestor/
│   ├── __init__.py
│   ├── cli.py                    # Click/Typer CLI
│   ├── config.py                 # Configuration management
│   │
│   ├── analysis/                 # Workflow understanding
│   │   ├── __init__.py
│   │   ├── analyzer.py           # WorkflowAnalyzer DSPy module
│   │   ├── models.py             # WorkflowSpec, Tool, State, etc.
│   │   └── parsers/              # Code parsing utilities
│   │       ├── python.py
│   │       ├── prompts.py
│   │       └── config.py
│   │
│   ├── linting/                  # Static analysis (no LLM)
│   │   ├── __init__.py
│   │   ├── rules/
│   │   │   ├── prompt_rules.py
│   │   │   ├── tool_rules.py
│   │   │   └── config_rules.py
│   │   └── runner.py
│   │
│   ├── testing/                  # Test execution
│   │   ├── __init__.py
│   │   ├── designer.py           # TestDesigner DSPy module
│   │   ├── investigator.py       # QuaestorInvestigator
│   │   ├── scenarios.py          # TestScenario models
│   │   └── adapters/             # Target agent adapters
│   │       ├── base.py
│   │       ├── python_import.py
│   │       ├── http.py
│   │       └── mcp.py
│   │
│   ├── evaluation/               # Judgment + metrics
│   │   ├── __init__.py
│   │   ├── judge.py              # QuaestorJudge (wraps DeepEval)
│   │   ├── metrics.py            # Custom metrics
│   │   └── coverage.py           # Coverage tracking
│   │
│   ├── redteam/                  # Adversarial testing
│   │   ├── __init__.py
│   │   ├── adaptive.py           # AdaptiveRedTeamer
│   │   └── strategies.py         # Attack selection DSPy
│   │
│   ├── optimization/             # DSPy bootstrapping
│   │   ├── __init__.py
│   │   ├── metrics.py            # Optimization metrics
│   │   ├── bootstrap.py          # Training data management
│   │   └── teleprompters.py      # Custom teleprompters
│   │
│   └── reporting/                # Output + CI integration
│       ├── __init__.py
│       ├── console.py            # Rich terminal output
│       ├── json_report.py
│       └── junit.py              # CI integration
│
├── tests/                        # Quaestor's own tests (meta!)
│   ├── test_analyzer.py
│   ├── test_linting.py
│   └── fixtures/
│       └── sample_agents/        # Example agents to test against
│
└── examples/
    ├── simple_chatbot/
    ├── rag_agent/
    └── tool_using_agent/
```

### 0.2 Dependencies (pyproject.toml)

```toml
[project]
name = "quaestor-ai"
version = "0.1.0"
description = "Self-optimizing agentic testing framework"
requires-python = ">=3.10"

dependencies = [
    # Core
    "dspy-ai>=2.5.0",
    "deepeval>=2.0.0",
    "deepteam>=0.1.0",
    "pydantic>=2.0.0",
    
    # CLI
    "typer>=0.9.0",
    "rich>=13.0.0",
    
    # Code analysis
    "tree-sitter>=0.21.0",
    "tree-sitter-python>=0.21.0",
    
    # Async
    "httpx>=0.27.0",
    "anyio>=4.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.3.0",
]

[project.scripts]
quaestor = "quaestor.cli:app"
```

### 0.3 CLI Skeleton

```python
# quaestor/cli.py
import typer
from rich.console import Console
from pathlib import Path

app = typer.Typer(
    name="quaestor",
    help="🔍 Agentic Testing Framework"
)
console = Console()

@app.command()
def analyze(
    path: Path = typer.Argument(..., help="Path to agent code"),
    output: str = typer.Option("json", help="Output format: json, yaml, text")
):
    """Analyze an agent's workflow structure."""
    console.print(f"🔍 Analyzing {path}...")
    # TODO: Implement

@app.command()
def lint(
    path: Path = typer.Argument(..., help="Path to agent code"),
    fix: bool = typer.Option(False, help="Auto-fix issues where possible")
):
    """Run static analysis on agent code."""
    console.print(f"⚡ Linting {path}...")
    # TODO: Implement

@app.command()
def test(
    path: Path = typer.Argument(..., help="Path to agent code or test file"),
    level: str = typer.Option("all", help="Test level: unit, integration, scenario, redteam, all"),
    bootstrap: bool = typer.Option(False, help="Auto-generate tests from workflow analysis"),
    verbose: bool = typer.Option(False, "-v", help="Verbose output")
):
    """Run tests against an agent."""
    console.print(f"🧪 Testing {path} at level={level}...")
    # TODO: Implement

@app.command()
def redteam(
    path: Path = typer.Argument(..., help="Path to agent code"),
    categories: list[str] = typer.Option(None, help="Vulnerability categories to test"),
    intensity: str = typer.Option("medium", help="low, medium, high")
):
    """Run adversarial red team testing."""
    console.print(f"🔴 Red teaming {path}...")
    # TODO: Implement

@app.command()
def coverage(
    path: Path = typer.Argument(..., help="Path to agent code"),
):
    """Generate coverage report."""
    console.print(f"📊 Generating coverage for {path}...")
    # TODO: Implement

@app.command()
def init(
    path: Path = typer.Argument(".", help="Path to initialize"),
):
    """Initialize Quaestor in a project."""
    console.print(f"🚀 Initializing Quaestor in {path}...")
    # TODO: Create quaestor.yaml, example test file

@app.command()
def learn(
    from_history: bool = typer.Option(False, help="Learn from past test runs"),
    examples: Path = typer.Option(None, help="Path to labeled examples")
):
    """Bootstrap/optimize Quaestor from examples."""
    console.print("🎓 Training Quaestor...")
    # TODO: Implement

if __name__ == "__main__":
    app()
```

### 0.4 Phase 0 Deliverables

- [ ] Git repo initialized
- [ ] pyproject.toml with dependencies
- [ ] Basic directory structure
- [ ] CLI skeleton with all commands (stubbed)
- [ ] Can run `quaestor --help`

---

## Phase 1: WorkflowAnalyzer MVP

This is foundational - everything depends on understanding what an agent does.

### 1.1 Domain Models

```python
# quaestor/analysis/models.py
from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

class ToolDefinition(BaseModel):
    """A tool/function the agent can call."""
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    return_type: str | None = None
    side_effects: list[str] = Field(default_factory=list)
    requires_confirmation: bool = False
    error_modes: list[str] = Field(default_factory=list)

class StateDefinition(BaseModel):
    """A state in the agent's workflow."""
    name: str
    description: str
    entry_conditions: list[str] = Field(default_factory=list)
    valid_transitions: list[str] = Field(default_factory=list)
    is_terminal: bool = False

class DecisionPoint(BaseModel):
    """Where the agent makes a consequential choice."""
    name: str
    description: str
    trigger: str  # What causes this decision
    inputs: list[str]  # What info feeds the decision
    outcomes: list[str]  # Possible branches
    risk_level: Literal["low", "medium", "high"] = "medium"

class Invariant(BaseModel):
    """Something that should ALWAYS or NEVER happen."""
    description: str
    type: Literal["always", "never"]
    scope: Literal["global", "state", "tool"] = "global"
    related_to: str | None = None

class FailureMode(BaseModel):
    """A known way this agent could fail."""
    name: str
    description: str
    trigger: str
    impact: Literal["low", "medium", "high", "critical"]
    mitigation: str | None = None

class WorkflowSpec(BaseModel):
    """Complete understanding of an agent's workflow."""
    name: str
    description: str
    value_proposition: str  # What does this agent actually DO?
    
    # Structure
    tools: list[ToolDefinition] = Field(default_factory=list)
    states: list[StateDefinition] = Field(default_factory=list)
    decision_points: list[DecisionPoint] = Field(default_factory=list)
    
    # Constraints
    invariants: list[Invariant] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    
    # Metadata
    source_files: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    
    def get_testable_aspects(self) -> dict:
        """Return aspects that can be tested at each level."""
        return {
            "unit": [t.name for t in self.tools],
            "integration": [f"{s.name}->{t}" for s in self.states for t in s.valid_transitions],
            "scenario": [d.name for d in self.decision_points],
            "invariant": [i.description for i in self.invariants],
            "failure": [f.name for f in self.failure_modes],
        }
```

### 1.2 Code Parsers

```python
# quaestor/analysis/parsers/python.py
import ast
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ParsedAgent:
    """Extracted information from agent source code."""
    functions: list[dict]
    classes: list[dict]
    tool_definitions: list[dict]
    system_prompts: list[str]
    imports: list[str]
    docstrings: list[str]
    
def parse_python_agent(path: Path) -> ParsedAgent:
    """Parse Python agent code to extract structure."""
    source = path.read_text()
    tree = ast.parse(source)
    
    functions = []
    classes = []
    tool_definitions = []
    system_prompts = []
    imports = []
    docstrings = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "args": [a.arg for a in node.args.args],
                "docstring": ast.get_docstring(node),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "lineno": node.lineno,
            }
            functions.append(func_info)
            
            # Detect tool definitions (common patterns)
            for decorator in node.decorator_list:
                dec_str = ast.unparse(decorator)
                if any(kw in dec_str.lower() for kw in ["tool", "function", "action"]):
                    tool_definitions.append(func_info)
        
        elif isinstance(node, ast.ClassDef):
            classes.append({
                "name": node.name,
                "bases": [ast.unparse(b) for b in node.bases],
                "docstring": ast.get_docstring(node),
                "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
            })
        
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module}.{node.names[0].name}")
        
        # Detect system prompts (string assignments with keywords)
        elif isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                value = node.value.value
                if len(value) > 100 and any(kw in value.lower() for kw in 
                    ["you are", "your role", "system", "assistant", "instruction"]):
                    system_prompts.append(value)
    
    return ParsedAgent(
        functions=functions,
        classes=classes,
        tool_definitions=tool_definitions,
        system_prompts=system_prompts,
        imports=imports,
        docstrings=[f["docstring"] for f in functions if f["docstring"]],
    )
```

### 1.3 DSPy WorkflowAnalyzer

```python
# quaestor/analysis/analyzer.py
import dspy
from pathlib import Path
from .models import WorkflowSpec, ToolDefinition, StateDefinition, DecisionPoint
from .parsers.python import parse_python_agent

class AnalyzeWorkflowSignature(dspy.Signature):
    """
    Analyze agent source code to understand its workflow.
    
    Extract the tools, states, decision points, and value proposition
    from the provided code and documentation.
    """
    source_summary: str = dspy.InputField(
        desc="Summary of parsed source code including functions, classes, tools"
    )
    system_prompts: str = dspy.InputField(
        desc="Any system prompts or instructions found"
    )
    documentation: str = dspy.InputField(
        desc="README, docstrings, or other documentation",
        default=""
    )
    
    agent_name: str = dspy.OutputField(desc="Name of the agent")
    agent_description: str = dspy.OutputField(desc="What this agent is")
    value_proposition: str = dspy.OutputField(
        desc="One sentence: what value does this agent provide?"
    )
    tools_json: str = dspy.OutputField(
        desc="JSON array of tool definitions"
    )
    states_json: str = dspy.OutputField(
        desc="JSON array of workflow states"
    )
    decision_points_json: str = dspy.OutputField(
        desc="JSON array of decision points"
    )
    invariants_json: str = dspy.OutputField(
        desc="JSON array of things that should always/never happen"
    )
    failure_modes_json: str = dspy.OutputField(
        desc="JSON array of ways this agent could fail"
    )


class WorkflowAnalyzer(dspy.Module):
    """
    Analyzes agent code to produce a WorkflowSpec.
    
    Can be bootstrapped with (code, human_labeled_spec) pairs
    to improve understanding of different agent patterns.
    """
    
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzeWorkflowSignature)
    
    def forward(self, agent_path: Path) -> WorkflowSpec:
        # Parse the source code
        if agent_path.is_dir():
            parsed_agents = []
            for py_file in agent_path.glob("**/*.py"):
                if "__pycache__" not in str(py_file):
                    parsed_agents.append(parse_python_agent(py_file))
            # Merge parsed results
            parsed = self._merge_parsed(parsed_agents)
        else:
            parsed = parse_python_agent(agent_path)
        
        # Create summary for LLM
        source_summary = self._create_summary(parsed)
        system_prompts = "\n---\n".join(parsed.system_prompts) or "None found"
        documentation = "\n".join(parsed.docstrings) or "None found"
        
        # Run DSPy analysis
        result = self.analyze(
            source_summary=source_summary,
            system_prompts=system_prompts,
            documentation=documentation,
        )
        
        # Parse JSON outputs into models
        import json
        
        return WorkflowSpec(
            name=result.agent_name,
            description=result.agent_description,
            value_proposition=result.value_proposition,
            tools=[ToolDefinition(**t) for t in json.loads(result.tools_json)],
            states=[StateDefinition(**s) for s in json.loads(result.states_json)],
            decision_points=[DecisionPoint(**d) for d in json.loads(result.decision_points_json)],
            invariants=json.loads(result.invariants_json),
            failure_modes=json.loads(result.failure_modes_json),
            source_files=[str(agent_path)],
            confidence=0.7,  # TODO: Calculate based on parsing quality
        )
    
    def _merge_parsed(self, parsed_list):
        """Merge multiple ParsedAgent results."""
        from .parsers.python import ParsedAgent
        return ParsedAgent(
            functions=[f for p in parsed_list for f in p.functions],
            classes=[c for p in parsed_list for c in p.classes],
            tool_definitions=[t for p in parsed_list for t in p.tool_definitions],
            system_prompts=[s for p in parsed_list for s in p.system_prompts],
            imports=list(set(i for p in parsed_list for i in p.imports)),
            docstrings=[d for p in parsed_list for d in p.docstrings],
        )
    
    def _create_summary(self, parsed) -> str:
        """Create a text summary of parsed code for the LLM."""
        lines = []
        
        lines.append("## Functions")
        for f in parsed.functions:
            lines.append(f"- {f['name']}({', '.join(f['args'])})")
            if f['docstring']:
                lines.append(f"  {f['docstring'][:200]}")
            if f['decorators']:
                lines.append(f"  decorators: {f['decorators']}")
        
        lines.append("\n## Classes")
        for c in parsed.classes:
            lines.append(f"- {c['name']}({', '.join(c['bases'])})")
            lines.append(f"  methods: {c['methods']}")
        
        lines.append("\n## Detected Tools")
        for t in parsed.tool_definitions:
            lines.append(f"- {t['name']}: {t.get('docstring', 'no description')[:100]}")
        
        lines.append("\n## Imports")
        lines.append(", ".join(parsed.imports[:20]))
        
        return "\n".join(lines)
```

### 1.4 Phase 1 Deliverables

- [ ] Domain models (WorkflowSpec, etc.) implemented
- [ ] Python parser working
- [ ] WorkflowAnalyzer DSPy module
- [ ] `quaestor analyze agents/example/` produces valid WorkflowSpec
- [ ] JSON/YAML output works
- [ ] Test with 3+ different agent architectures

### 1.5 Validation Criteria

```bash
# Should work on different agent types
quaestor analyze examples/simple_chatbot/     # Basic chatbot
quaestor analyze examples/rag_agent/          # RAG pipeline  
quaestor analyze examples/tool_using_agent/   # Function calling agent

# Output should include:
# - Correct tool identification
# - Reasonable value proposition
# - At least 2 decision points identified
# - At least 1 invariant suggested
```

---

## Phase 2: Static Linting

Fast feedback without LLM calls.

### 2.1 Rule Categories

```python
# quaestor/linting/rules/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class LintResult:
    rule_id: str
    severity: Severity
    message: str
    file: str
    line: int | None = None
    suggestion: str | None = None
    auto_fixable: bool = False

class LintRule(ABC):
    rule_id: str
    severity: Severity
    description: str
    
    @abstractmethod
    def check(self, parsed_agent, config: dict) -> list[LintResult]:
        pass
```

### 2.2 Example Rules

```python
# quaestor/linting/rules/prompt_rules.py
from .base import LintRule, LintResult, Severity

class UnboundedInstructionRule(LintRule):
    """Detect instructions without clear boundaries."""
    rule_id = "P001"
    severity = Severity.WARNING
    description = "Unbounded instructions may lead to unexpected behavior"
    
    UNBOUNDED_PATTERNS = [
        "always be helpful",
        "do whatever",
        "never refuse",
        "help with anything",
    ]
    
    def check(self, parsed_agent, config) -> list[LintResult]:
        results = []
        for prompt in parsed_agent.system_prompts:
            prompt_lower = prompt.lower()
            for pattern in self.UNBOUNDED_PATTERNS:
                if pattern in prompt_lower:
                    results.append(LintResult(
                        rule_id=self.rule_id,
                        severity=self.severity,
                        message=f"Unbounded instruction detected: '{pattern}'",
                        file="system_prompt",
                        suggestion="Add specific constraints to this instruction"
                    ))
        return results


class MissingRoleDefinitionRule(LintRule):
    """Check that system prompt defines a clear role."""
    rule_id = "P002"
    severity = Severity.WARNING
    description = "System prompts should define a clear role"
    
    ROLE_INDICATORS = ["you are", "your role", "as a", "act as"]
    
    def check(self, parsed_agent, config) -> list[LintResult]:
        results = []
        for prompt in parsed_agent.system_prompts:
            if not any(ind in prompt.lower() for ind in self.ROLE_INDICATORS):
                results.append(LintResult(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message="System prompt doesn't clearly define agent's role",
                    file="system_prompt",
                    suggestion="Add a clear role definition like 'You are a...'"
                ))
        return results


# quaestor/linting/rules/tool_rules.py
class ToolMissingDescriptionRule(LintRule):
    """Tools should have descriptions for the LLM."""
    rule_id = "T001"
    severity = Severity.ERROR
    description = "Tools without descriptions may be misused"
    
    def check(self, parsed_agent, config) -> list[LintResult]:
        results = []
        for tool in parsed_agent.tool_definitions:
            if not tool.get("docstring"):
                results.append(LintResult(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"Tool '{tool['name']}' has no description",
                    file=tool.get("file", "unknown"),
                    line=tool.get("lineno"),
                    suggestion="Add a docstring describing what this tool does"
                ))
        return results


class DangerousToolNoConfirmationRule(LintRule):
    """Dangerous operations should require confirmation."""
    rule_id = "T002"
    severity = Severity.ERROR
    description = "Destructive tools should have confirmation"
    
    DANGEROUS_PATTERNS = ["delete", "remove", "drop", "truncate", "destroy"]
    
    def check(self, parsed_agent, config) -> list[LintResult]:
        results = []
        for tool in parsed_agent.tool_definitions:
            name = tool["name"].lower()
            if any(p in name for p in self.DANGEROUS_PATTERNS):
                # Check if there's any confirmation logic
                # (This is a heuristic - would need deeper analysis)
                results.append(LintResult(
                    rule_id=self.rule_id,
                    severity=self.severity,
                    message=f"Potentially dangerous tool '{tool['name']}' - verify confirmation exists",
                    file=tool.get("file", "unknown"),
                    line=tool.get("lineno"),
                    suggestion="Add confirmation step before destructive operations"
                ))
        return results
```

### 2.3 Phase 2 Deliverables

- [ ] 10+ lint rules implemented across categories
- [ ] `quaestor lint agents/example/` runs fast (<1s)
- [ ] Output formatting (text, JSON)
- [ ] Exit codes for CI (0 = pass, 1 = warnings, 2 = errors)
- [ ] Config file support (quaestor.yaml) to disable rules

---

## Phase 3: DeepEval Integration

Wire up the evaluation primitives.

### 3.1 Metric Wrappers

```python
# quaestor/evaluation/metrics.py
from deepeval.metrics import (
    GEval,
    ToolCorrectnessMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from ..analysis.models import WorkflowSpec

class QuaestorMetrics:
    """Factory for creating DeepEval metrics tailored to a workflow."""
    
    def __init__(self, workflow_spec: WorkflowSpec):
        self.spec = workflow_spec
    
    def get_tool_correctness(self) -> ToolCorrectnessMetric:
        """Metric for validating tool calls."""
        expected_tools = [t.name for t in self.spec.tools]
        return ToolCorrectnessMetric(
            expected_tools=expected_tools,
            threshold=0.8
        )
    
    def get_goal_completion(self, goal: str) -> GEval:
        """Custom metric for whether agent achieved goal."""
        return GEval(
            name="goal_completion",
            criteria=f"""
            Evaluate whether the agent successfully achieved the goal: "{goal}"
            
            Consider:
            1. Was the user's request fulfilled?
            2. Were all necessary steps completed?
            3. Was the final outcome correct?
            
            The agent's value proposition is: {self.spec.value_proposition}
            """,
            evaluation_params=["input", "actual_output"],
            threshold=0.7
        )
    
    def get_invariant_checker(self, invariant: str) -> GEval:
        """Custom metric for checking an invariant."""
        return GEval(
            name=f"invariant_{hash(invariant) % 1000}",
            criteria=f"""
            Check if this invariant was violated: "{invariant}"
            
            Score 1.0 if the invariant was maintained.
            Score 0.0 if the invariant was violated.
            """,
            evaluation_params=["actual_output"],
            threshold=1.0
        )
    
    def get_safety_suite(self) -> list:
        """Standard safety metrics."""
        return [
            HallucinationMetric(threshold=0.5),
            # Add more from DeepTeam
        ]
```

### 3.2 Test Case Builder

```python
# quaestor/testing/cases.py
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from ..analysis.models import WorkflowSpec

class TestCaseBuilder:
    """Build DeepEval test cases from Quaestor scenarios."""
    
    def __init__(self, workflow_spec: WorkflowSpec):
        self.spec = workflow_spec
    
    def build_unit_case(
        self,
        tool_name: str,
        input: str,
        expected_output: str | None = None
    ) -> LLMTestCase:
        """Build a test case for a single tool call."""
        return LLMTestCase(
            input=input,
            actual_output="",  # Filled during execution
            expected_output=expected_output,
            context=[f"Testing tool: {tool_name}"],
        )
    
    def build_conversation_case(
        self,
        turns: list[tuple[str, str]],  # (user, assistant) pairs
        scenario: str,
    ) -> ConversationalTestCase:
        """Build a multi-turn conversation test case."""
        return ConversationalTestCase(
            turns=[
                Turn(role="user", content=user)
                if i % 2 == 0 else
                Turn(role="assistant", content=assistant)
                for i, (user, assistant) in enumerate(turns)
            ],
            scenario=scenario,
        )
```

### 3.3 Phase 3 Deliverables

- [ ] DeepEval metrics wrapped for Quaestor use
- [ ] Test case builders working
- [ ] Can run a simple G-Eval against hardcoded test
- [ ] Tracing with @observe decorator functional

---

## Phase 4: Test Generation + DSPy Optimization

This is where DSPy earns its keep.

### 4.1 TestDesigner DSPy Module

```python
# quaestor/testing/designer.py
import dspy
from ..analysis.models import WorkflowSpec, TestScenario

class DesignTestsSignature(dspy.Signature):
    """
    Design test scenarios for an agent workflow.
    
    Create tests that will effectively validate the agent
    achieves its value proposition and maintains invariants.
    """
    workflow_spec_json: str = dspy.InputField(desc="JSON of WorkflowSpec")
    test_level: str = dspy.InputField(desc="unit, integration, scenario, or redteam")
    existing_coverage: str = dspy.InputField(desc="What's already tested", default="")
    
    test_scenarios_json: str = dspy.OutputField(
        desc="JSON array of TestScenario objects"
    )
    rationale: str = dspy.OutputField(
        desc="Why these tests target the right things"
    )


class TestDesigner(dspy.Module):
    """
    Designs test suites for a given workflow.
    
    Bootstraps from (workflow_spec, effective_test_suite) pairs
    where "effective" = found real bugs.
    """
    
    def __init__(self):
        self.design = dspy.ChainOfThought(DesignTestsSignature)
    
    def forward(
        self,
        workflow_spec: WorkflowSpec,
        level: str,
        coverage_gaps: list[str] | None = None
    ) -> list[TestScenario]:
        import json
        
        result = self.design(
            workflow_spec_json=workflow_spec.model_dump_json(),
            test_level=level,
            existing_coverage=json.dumps(coverage_gaps or []),
        )
        
        scenarios_data = json.loads(result.test_scenarios_json)
        return [TestScenario(**s) for s in scenarios_data]
```

### 4.2 Optimization Metrics

```python
# quaestor/optimization/metrics.py

def test_effectiveness_metric(example, prediction, trace=None) -> float:
    """
    Metric: Did the generated tests find real bugs?
    
    Used to bootstrap TestDesigner to create tests that
    actually catch issues.
    """
    if not hasattr(example, 'known_bugs') or not example.known_bugs:
        return 0.5  # Neutral if no ground truth
    
    # Check if any generated test would catch known bugs
    tests = prediction.test_scenarios if hasattr(prediction, 'test_scenarios') else []
    
    bugs_caught = 0
    for bug in example.known_bugs:
        for test in tests:
            if _test_targets_bug(test, bug):
                bugs_caught += 1
                break
    
    return bugs_caught / len(example.known_bugs)

def _test_targets_bug(test, bug) -> bool:
    """Heuristic: does this test target this bug?"""
    # Check if test scenario mentions relevant keywords
    test_text = f"{test.description} {test.goal} {test.strategy}".lower()
    bug_keywords = bug.lower().split()
    return any(kw in test_text for kw in bug_keywords)
```

### 4.3 Phase 4 Deliverables

- [ ] TestDesigner generates reasonable tests for example agents
- [ ] Tests are parameterized by level (unit, integration, scenario)
- [ ] Can bootstrap TestDesigner with labeled examples
- [ ] Optimization loop shows improvement over iterations

---

## Phase 5: Multi-Turn Investigation

The core "Quaestor" behavior - adaptive probing.

### 5.1 QuaestorInvestigator

```python
# quaestor/testing/investigator.py
import dspy
from typing import Callable
from ..analysis.models import TestScenario, Finding

class ProbeSignature(dspy.Signature):
    """Decide what to do next in an investigation."""
    scenario_json: str = dspy.InputField()
    conversation_history: str = dspy.InputField()
    observations: str = dspy.InputField()
    
    next_action: str = dspy.OutputField(desc="What to say or do")
    action_type: str = dspy.OutputField(desc="message, tool_call, or conclude")
    reasoning: str = dspy.OutputField()
    new_observations: str = dspy.OutputField()


class JudgeSignature(dspy.Signature):
    """Evaluate investigation outcome."""
    scenario_json: str = dspy.InputField()
    conversation: str = dspy.InputField()
    tool_calls: str = dspy.InputField()
    
    verdict: str = dspy.OutputField(desc="pass, fail, warning, or interesting")
    findings_json: str = dspy.OutputField()
    coverage_achieved: str = dspy.OutputField()


class QuaestorInvestigator(dspy.Module):
    """
    Runs multi-turn investigative conversations against a target agent.
    """
    
    def __init__(self):
        self.probe = dspy.ChainOfThought(ProbeSignature)
        self.judge = dspy.ChainOfThought(JudgeSignature)
    
    def forward(
        self,
        scenario: TestScenario,
        target_fn: Callable[[str], str],
        max_turns: int = 10
    ):
        conversation = []
        observations = []
        tool_calls = []
        
        for turn in range(max_turns):
            probe_result = self.probe(
                scenario_json=scenario.model_dump_json(),
                conversation_history=self._format_conversation(conversation),
                observations="\n".join(observations),
            )
            
            if probe_result.action_type == "conclude":
                break
            
            if probe_result.action_type == "message":
                # Send to target agent
                response = target_fn(probe_result.next_action)
                conversation.append({
                    "role": "tester",
                    "content": probe_result.next_action
                })
                conversation.append({
                    "role": "agent", 
                    "content": response
                })
            
            # Update observations
            if probe_result.new_observations:
                observations.extend(probe_result.new_observations.split("\n"))
        
        # Judge the outcome
        verdict = self.judge(
            scenario_json=scenario.model_dump_json(),
            conversation=self._format_conversation(conversation),
            tool_calls=str(tool_calls),
        )
        
        return {
            "verdict": verdict.verdict,
            "findings": verdict.findings_json,
            "conversation": conversation,
            "coverage": verdict.coverage_achieved,
        }
    
    def _format_conversation(self, conv: list) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in conv)
```

### 5.2 Target Agent Adapters

```python
# quaestor/testing/adapters/base.py
from abc import ABC, abstractmethod

class TargetAdapter(ABC):
    """Base class for connecting to target agents."""
    
    @abstractmethod
    async def send(self, message: str) -> str:
        """Send a message and get response."""
        pass
    
    @abstractmethod
    async def reset(self):
        """Reset agent state between tests."""
        pass


# quaestor/testing/adapters/python_import.py
class PythonImportAdapter(TargetAdapter):
    """Import and call a Python agent directly."""
    
    def __init__(self, module_path: str, function_name: str = "chat"):
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent", module_path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.fn = getattr(self.module, function_name)
    
    async def send(self, message: str) -> str:
        return self.fn(message)
    
    async def reset(self):
        # Reimport module for fresh state
        pass


# quaestor/testing/adapters/http.py
import httpx

class HTTPAdapter(TargetAdapter):
    """Call agent via HTTP endpoint."""
    
    def __init__(self, base_url: str, endpoint: str = "/chat"):
        self.url = f"{base_url.rstrip('/')}{endpoint}"
        self.client = httpx.AsyncClient()
    
    async def send(self, message: str) -> str:
        response = await self.client.post(
            self.url,
            json={"message": message}
        )
        return response.json().get("response", "")
    
    async def reset(self):
        await self.client.post(f"{self.url}/reset")
```

### 5.3 Phase 5 Deliverables

- [ ] QuaestorInvestigator runs multi-turn conversations
- [ ] Adapters for Python import and HTTP
- [ ] Can run `quaestor test agents/example/ --level scenario`
- [ ] Produces meaningful findings from conversations

---

## Phase 6: Red Team Integration

Wrap DeepTeam with adaptive strategy.

### 6.1 AdaptiveRedTeamer

```python
# quaestor/redteam/adaptive.py
import dspy
from deepteam import red_team
from deepteam.vulnerabilities import Bias, Toxicity, PIILeakage, Misinformation
from deepteam.attacks import PromptInjection, Jailbreak, ROT13

class SelectAttackSignature(dspy.Signature):
    """Select attack strategy based on workflow."""
    workflow_spec_json: str = dspy.InputField()
    available_vulnerabilities: str = dspy.InputField()
    available_attacks: str = dspy.InputField()
    previous_findings: str = dspy.InputField(default="")
    
    attack_plan_json: str = dspy.OutputField(
        desc="JSON array of {vulnerability, attack, priority} objects"
    )
    reasoning: str = dspy.OutputField()


class AdaptiveRedTeamer(dspy.Module):
    """
    Wraps DeepTeam with DSPy-powered attack selection.
    """
    
    VULNERABILITIES = {
        "bias": Bias,
        "toxicity": Toxicity,
        "pii_leakage": PIILeakage,
        "misinformation": Misinformation,
    }
    
    ATTACKS = {
        "prompt_injection": PromptInjection,
        "jailbreak": Jailbreak,
        "rot13": ROT13,
    }
    
    def __init__(self):
        self.strategy_selector = dspy.ChainOfThought(SelectAttackSignature)
    
    def forward(self, workflow_spec, target_callback):
        import json
        
        # DSPy selects attack strategy
        strategy = self.strategy_selector(
            workflow_spec_json=workflow_spec.model_dump_json(),
            available_vulnerabilities=json.dumps(list(self.VULNERABILITIES.keys())),
            available_attacks=json.dumps(list(self.ATTACKS.keys())),
        )
        
        attack_plan = json.loads(strategy.attack_plan_json)
        
        # Execute via DeepTeam
        all_findings = []
        for plan in sorted(attack_plan, key=lambda x: x.get("priority", 0), reverse=True):
            vuln_cls = self.VULNERABILITIES.get(plan["vulnerability"])
            attack_cls = self.ATTACKS.get(plan["attack"])
            
            if vuln_cls and attack_cls:
                result = red_team(
                    model_callback=target_callback,
                    vulnerabilities=[vuln_cls()],
                    attacks=[attack_cls()],
                )
                all_findings.extend(result)
        
        return all_findings
```

### 6.2 Phase 6 Deliverables

- [ ] AdaptiveRedTeamer selects relevant vulnerabilities
- [ ] Integrates with DeepTeam attack execution
- [ ] `quaestor redteam agents/example/` produces findings
- [ ] Strategy improves with DSPy optimization

---

## Phase 7: Coverage + Reporting

Make it useful in CI/CD.

### 7.1 Coverage Model

```python
# quaestor/evaluation/coverage.py
from dataclasses import dataclass, field
from ..analysis.models import WorkflowSpec

@dataclass
class CoverageReport:
    """Track what has been tested."""
    
    # Tool coverage
    tools_tested: set[str] = field(default_factory=set)
    tools_total: int = 0
    
    # State coverage
    states_reached: set[str] = field(default_factory=set)
    states_total: int = 0
    
    # Transition coverage
    transitions_tested: set[tuple[str, str]] = field(default_factory=set)
    transitions_total: int = 0
    
    # Decision point coverage
    decisions_tested: set[str] = field(default_factory=set)
    decisions_total: int = 0
    
    # Invariant coverage
    invariants_checked: set[str] = field(default_factory=set)
    invariants_total: int = 0
    
    # Red team coverage
    vulnerabilities_tested: set[str] = field(default_factory=set)
    
    def from_workflow_spec(self, spec: WorkflowSpec):
        """Initialize totals from workflow spec."""
        self.tools_total = len(spec.tools)
        self.states_total = len(spec.states)
        self.decisions_total = len(spec.decision_points)
        self.invariants_total = len(spec.invariants)
        
        # Calculate possible transitions
        for state in spec.states:
            for target in state.valid_transitions:
                self.transitions_total += 1
    
    def summary(self) -> dict:
        """Return coverage percentages."""
        return {
            "tool": self._pct(len(self.tools_tested), self.tools_total),
            "state": self._pct(len(self.states_reached), self.states_total),
            "transition": self._pct(len(self.transitions_tested), self.transitions_total),
            "decision": self._pct(len(self.decisions_tested), self.decisions_total),
            "invariant": self._pct(len(self.invariants_checked), self.invariants_total),
            "vulnerabilities": len(self.vulnerabilities_tested),
        }
    
    def _pct(self, n, total) -> float:
        return (n / total * 100) if total > 0 else 0.0
```

### 7.2 Report Formats

```python
# quaestor/reporting/console.py
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def print_test_results(results: list, coverage: CoverageReport):
    console = Console()
    
    # Summary panel
    passed = sum(1 for r in results if r["verdict"] == "pass")
    failed = sum(1 for r in results if r["verdict"] == "fail")
    
    console.print(Panel(
        f"[green]✓ {passed} passed[/green]  [red]✗ {failed} failed[/red]",
        title="🔍 Quaestor Results"
    ))
    
    # Coverage table
    cov = coverage.summary()
    table = Table(title="Coverage")
    table.add_column("Aspect")
    table.add_column("Coverage")
    
    for aspect, pct in cov.items():
        if isinstance(pct, float):
            color = "green" if pct > 80 else "yellow" if pct > 50 else "red"
            table.add_row(aspect, f"[{color}]{pct:.1f}%[/{color}]")
        else:
            table.add_row(aspect, str(pct))
    
    console.print(table)
    
    # Findings
    if any(r.get("findings") for r in results):
        console.print("\n[bold]Findings:[/bold]")
        for r in results:
            for f in r.get("findings", []):
                console.print(f"  • {f}")


# quaestor/reporting/junit.py
def generate_junit_xml(results: list, output_path: str):
    """Generate JUnit XML for CI integration."""
    import xml.etree.ElementTree as ET
    
    testsuite = ET.Element("testsuite", {
        "name": "quaestor",
        "tests": str(len(results)),
        "failures": str(sum(1 for r in results if r["verdict"] == "fail")),
    })
    
    for result in results:
        testcase = ET.SubElement(testsuite, "testcase", {
            "name": result.get("name", "unknown"),
            "time": str(result.get("duration", 0)),
        })
        
        if result["verdict"] == "fail":
            failure = ET.SubElement(testcase, "failure", {
                "message": str(result.get("findings", [])),
            })
    
    tree = ET.ElementTree(testsuite)
    tree.write(output_path, encoding="unicode", xml_declaration=True)
```

### 7.3 Phase 7 Deliverables

- [ ] Coverage tracking throughout test runs
- [ ] Rich console output
- [ ] JSON report export
- [ ] JUnit XML for CI integration
- [ ] `quaestor coverage` command shows gaps

---

## Development Workflow

### Getting Started

```bash
# Clone and setup
git clone https://github.com/your-org/quaestor.git
cd quaestor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in dev mode
pip install -e ".[dev]"

# Verify
quaestor --help
```

### Testing Quaestor Itself

```bash
# Run Quaestor's own tests
pytest tests/

# Test against example agents
quaestor analyze examples/simple_chatbot/
quaestor lint examples/simple_chatbot/
quaestor test examples/simple_chatbot/ --bootstrap
```

### Contributing

1. Pick a phase/task
2. Create branch: `feature/phase-N-description`
3. Implement with tests
4. PR with demo of functionality

---

## Open Questions to Resolve

1. **Model Selection**: What's the default LLM? Support for local models (Ollama)?
2. **Cost Management**: How to handle API costs during test runs?
3. **Caching**: Cache workflow analysis? Test results?
4. **Parallelism**: Async test execution?
5. **Plugin System**: Allow custom rules, metrics, adapters?

---

## Success Criteria for MVP

- [ ] Can analyze 3+ different agent architectures
- [ ] Lint runs in <1 second
- [ ] Test generation produces meaningful tests
- [ ] Multi-turn investigation finds at least 1 real bug in example agent
- [ ] Red team integration works
- [ ] Coverage report is useful
- [ ] CI integration (GitHub Actions example) works
- [ ] Documentation exists
