# Quaestor - Architecture Documentation

> Built on Smactorio Governance Infrastructure

## Overview

Quaestor is a self-optimizing agentic testing framework that leverages Smactorio's governance-as-a-service capabilities to provide deterministic, compliant agent evaluation. Think of it as "pytest for AI agents" with built-in best practices enforcement.

## Core Principles

1. **Governance-First**: All testing follows Smactorio's governance rules
2. **Deterministic Execution**: Reproducible test runs with full audit trail
3. **Self-Optimization**: DSPy-powered learning from test history
4. **Compliance by Default**: Built-in security and safety checks

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUAESTOR CLI                                    │
│  quaestor analyze | quaestor lint | quaestor test | quaestor coverage       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SMACTORIO GOVERNANCE LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  LLMClient      │  │  Governance     │  │  Agent Infrastructure      │  │
│  │  - Multi-model  │  │  Engine         │  │  - BaseAgent               │  │
│  │  - Caching      │  │  - OSCAL Rules  │  │  - AgentFactory            │  │
│  │  - Cost Track   │  │  - Constitution │  │  - State Management        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUAESTOR CORE MODULES                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      ANALYSIS ENGINE                                 │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │ Python Parser  │  │ Workflow       │  │ Static Linter          │ │    │
│  │  │ (tree-sitter)  │→ │ Analyzer (DSPy)│→ │ (No LLM)               │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      TEST GENERATION                                 │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │ TestDesigner   │  │ Scenario       │  │ Fixture                │ │    │
│  │  │ (DSPy)         │→ │ Generator      │→ │ System                 │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      RUNTIME TESTING                                 │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │ Investigator   │  │ Target         │  │ Observation            │ │    │
│  │  │ (Multi-turn)   │↔ │ Adapters       │→ │ Collector              │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      EVALUATION                                      │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │ QuaestorJudge  │  │ DeepEval       │  │ Verdict                │ │    │
│  │  │ (LLM-as-Judge) │↔ │ Metrics        │→ │ Generator              │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      REPORTING                                       │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐ │    │
│  │  │ Coverage       │  │ Report         │  │ CI                     │ │    │
│  │  │ Tracker        │→ │ Generator      │→ │ Integration            │ │    │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Smactorio Integration Points

### 1. LLM Infrastructure

Quaestor uses Smactorio's `LLMClient` for all LLM operations:

```python
from smactorio.llm import LLMClient

class WorkflowAnalyzer:
    def __init__(self):
        self.llm = LLMClient()  # Multi-model, cached, cost-tracked
```

**Benefits:**
- Consistent model selection across the system
- Built-in circuit breaker for resilience
- Cost tracking and budgeting
- Response caching

### 2. Governance Engine

Test rules and compliance checks flow through Smactorio's governance:

```python
from smactorio.governance import GovernanceEngine

class QuaestorJudge:
    def __init__(self):
        self.governance = GovernanceEngine()

    def validate_verdict(self, verdict: Verdict) -> bool:
        # Ensure verdict follows governance rules
        return self.governance.validate(verdict)
```

**Benefits:**
- OSCAL-based rule definitions
- Auditable decision trail
- Consistent policy enforcement

### 3. Agent Patterns

Quaestor's modules follow Smactorio's agent architecture:

```python
from smactorio.agents import BaseAgent

class QuaestorInvestigator(BaseAgent):
    """Multi-turn conversational prober following Smactorio patterns."""

    def __init__(self):
        super().__init__(
            name="investigator",
            description="Probes agent under test with adaptive strategy"
        )
```

### 4. State Management

Test state flows through Smactorio's state infrastructure:

```python
from smactorio.state import FeatureState

class TestExecutionState(FeatureState):
    agent_workflow: AgentWorkflow
    test_cases: list[TestCase]
    observations: list[Observation]
    verdicts: list[Verdict]
```

---

## Directory Structure

```
quaestor/
├── __init__.py
├── cli.py                    # Typer CLI entry point
├── config.py                 # Configuration management
│
├── analysis/                 # Code analysis and workflow extraction
│   ├── __init__.py
│   ├── models.py            # AgentWorkflow, ToolDefinition, StateDefinition
│   ├── parser.py            # tree-sitter Python parser
│   ├── analyzer.py          # WorkflowAnalyzer DSPy module
│   └── linter.py            # Static analysis (no LLM)
│
├── testing/                  # Test generation and execution
│   ├── __init__.py
│   ├── models.py            # TestCase, TestSuite, TestResult
│   ├── designer.py          # TestDesigner DSPy module
│   ├── investigator.py      # QuaestorInvestigator multi-turn prober
│   └── adapters/            # Target agent adapters
│       ├── __init__.py
│       ├── http.py          # HTTP/REST adapter
│       ├── python.py        # Python import adapter
│       └── mcp.py           # Model Context Protocol adapter
│
├── evaluation/               # Verdict generation
│   ├── __init__.py
│   ├── judge.py             # QuaestorJudge LLM-as-judge
│   ├── metrics.py           # DeepEval metric wrappers
│   └── verdicts.py          # Verdict models and classification
│
├── coverage/                 # Coverage tracking
│   ├── __init__.py
│   ├── tracker.py           # Coverage data collection
│   └── models.py            # Coverage models
│
├── reporting/                # Output generation
│   ├── __init__.py
│   ├── html.py              # HTML report generator
│   ├── sarif.py             # SARIF for CI integration
│   └── console.py           # Rich console output
│
└── optimization/             # Self-improvement
    ├── __init__.py
    ├── learner.py           # DSPy optimization
    └── history.py           # Test history management
```

---

## Data Flow

### 1. Analysis Flow

```
Agent Code → Python Parser → AST → WorkflowAnalyzer → AgentWorkflow
                                        ↓
                                   Static Linter
                                        ↓
                                   Lint Results
```

### 2. Test Generation Flow

```
AgentWorkflow → TestDesigner → TestCases → Fixture Injection → Executable Tests
                    ↓
              DSPy Optimization
                    ↓
              Improved Prompts
```

### 3. Execution Flow

```
TestCase → Investigator → Target Adapter → Agent Under Test
              ↓                                  ↓
         Strategy                           Response
         Adaptation                              ↓
              ↓                           Observation
              ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

### 4. Evaluation Flow

```
Observations → QuaestorJudge → DeepEval Metrics → Verdict
                    ↓                                ↓
              Governance                      Coverage Update
              Validation                             ↓
                                              Report Generation
```

---

## Key Domain Models

### AgentWorkflow (Analysis Output)

```python
class AgentWorkflow(BaseModel):
    """Complete understanding of an agent's workflow."""
    name: str
    description: str
    tools: list[ToolDefinition]
    states: list[StateDefinition]
    transitions: list[Transition]
    invariants: list[Invariant]
    entry_points: list[EntryPoint]
```

### TestCase (Test Definition)

```python
class TestCase(BaseModel):
    """Single test scenario."""
    id: str
    name: str
    description: str
    category: TestCategory
    inputs: list[TestInput]
    expected_behaviors: list[ExpectedBehavior]
    assertions: list[Assertion]
```

### Verdict (Evaluation Output)

```python
class Verdict(BaseModel):
    """Test result with full context."""
    test_case_id: str
    passed: bool
    severity: Severity
    category: VerdictCategory
    evidence: list[Evidence]
    observations: list[Observation]
    recommendations: list[str]
```

---

## Configuration

### Smactorio Configuration (`.smactorio/config.yaml`)

```yaml
# Auto-generated by `smactorio init`
llm:
  default_model: gpt-4o
  fallback_model: claude-3-sonnet
  cache_enabled: true
  cost_limit_daily: 10.00

governance:
  catalog: oscal/catalog.yaml
  strict_mode: true
```

### Quaestor Configuration (`quaestor.yaml`)

```yaml
analysis:
  parser: tree-sitter
  include_patterns:
    - "**/*_agent.py"
    - "**/agents/**/*.py"

testing:
  default_level: integration
  timeout_seconds: 60
  max_turns: 10

evaluation:
  judge_model: gpt-4o
  metrics:
    - correctness
    - safety
    - helpfulness

coverage:
  targets:
    tools: 80%
    states: 90%
    transitions: 75%

reporting:
  format: html
  output_dir: ./quaestor-reports
```

---

## Security Considerations

1. **API Key Management**: Keys via environment variables, never in config files
2. **Agent Isolation**: Probed agents run in sandboxed environments
3. **Data Privacy**: Test inputs/outputs can be redacted in reports
4. **Governance Audit**: All decisions logged via Smactorio

---

## Future: Red Team Integration

> ⚠️ Deferred pending DeepTeam availability

```python
# Planned architecture
class RedTeamer:
    def __init__(self):
        self.strategies = [
            JailbreakStrategy(),
            PromptInjectionStrategy(),
            BoundaryProbeStrategy(),
        ]

    async def attack(self, workflow: AgentWorkflow) -> list[Vulnerability]:
        ...
```
