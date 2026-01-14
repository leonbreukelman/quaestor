# Quaestor - Development Roadmap & TODO

> Self-optimizing agentic testing framework built on Smactorio governance infrastructure

## Project Status: Phase 0 - Foundation

**Last Updated:** 2026-01-13

---

## 🎯 Current Sprint: Project Setup & Core Infrastructure

### ✅ Completed
- [x] Project scaffolding and directory structure
- [x] `pyproject.toml` with Smactorio as core dependency
- [x] Python >=3.12 constraint set
- [x] Documentation structure created
- [x] Smactorio installation and configuration

### 🔄 In Progress
- [ ] Initialize Smactorio configuration (`.smactorio/`)
- [ ] Validate LLM API key configuration
- [ ] Create base domain models

### 📋 Next Up
- [ ] Implement `quaestor/analysis/models.py` (AgentWorkflow, ToolDefinition, StateDefinition)
- [ ] Implement Python AST parser for agent code analysis
- [ ] Create CLI skeleton with Typer

---

## 📅 Phase Roadmap

### Phase 1: Core Analysis Engine (Target: Week 1-2)
- [ ] **WorkflowAnalyzer** - DSPy module for agent code analysis
  - Extract tools, states, transitions from agent code
  - Generate `AgentWorkflow` models
- [ ] **Python Parser** - tree-sitter based code analysis
  - Function extraction
  - Decorator detection (@tool, @agent)
  - State machine inference
- [ ] **Static Linting** - No-LLM fast analysis
  - Common anti-patterns
  - Security checks
  - Best practice validation

### Phase 2: Test Generation (Target: Week 2-3)
- [ ] **TestDesigner** - DSPy module for test scenario generation
  - Positive path tests
  - Edge case generation
  - Error handling tests
- [ ] **TestCase Models** - Pydantic schemas
- [ ] **Fixture System** - Reusable test components

### Phase 3: Runtime Testing (Target: Week 3-4)
- [ ] **QuaestorInvestigator** - Multi-turn conversational prober
  - Adaptive strategy based on observations
  - Tool call verification
  - State transition validation
- [ ] **Target Adapters**
  - HTTP adapter for deployed agents
  - Python import adapter for local agents
  - MCP adapter for protocol-compliant agents

### Phase 4: Evaluation & Judgment (Target: Week 4-5)
- [ ] **QuaestorJudge** - LLM-as-judge with DeepEval
  - Verdict generation with evidence
  - Severity classification
  - Category mapping
- [ ] **DeepEval Integration**
  - Custom metrics wrapping
  - Assertion library
  - Scoring infrastructure

### Phase 5: Coverage & Reporting (Target: Week 5-6)
- [ ] **Coverage Tracker**
  - Tool coverage
  - State coverage
  - Transition coverage
  - Invariant coverage
- [ ] **Report Generator**
  - HTML reports
  - JSON/SARIF for CI integration
  - Coverage visualization

### Phase 6: Optimization Engine (Target: Week 6-7)
- [ ] **DSPy Optimization** - Self-improving test generation
  - MIPROv2 integration
  - Few-shot learning from test history
  - Prompt optimization
- [ ] **Learning System**
  - Store successful test patterns
  - Failure pattern recognition
  - Adaptive difficulty scaling

### Phase 7: Red Team Capabilities (Target: Week 7-8)
> ⚠️ **DEFERRED**: DeepTeam package availability TBD

- [ ] **DeepTeam Integration** - Adversarial testing
  - Jailbreak attempts
  - Prompt injection testing
  - Safety boundary probing
- [ ] **RedTeamer Module**
  - Attack strategy selection
  - Vulnerability classification
  - Remediation suggestions

---

## 🔧 Technical Debt & Improvements

### Infrastructure
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Pre-commit hooks
- [ ] Automated testing on PR
- [ ] Documentation site (MkDocs)

### Code Quality
- [ ] 80%+ test coverage target
- [ ] Type hints throughout (mypy strict)
- [ ] Docstrings for public API

### Performance
- [ ] Async execution for parallel tests
- [ ] Caching for repeated analyses
- [ ] Streaming for long-running probes

---

## 🚧 Known Blockers & Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| DeepTeam availability | Red team phase delayed | Use manual adversarial patterns initially |
| DSPy version conflicts | Build failures | Pin compatible versions, test matrix |
| Tree-sitter Python bindings | Code analysis limited | Fallback to AST module |
| LLM rate limits | Slow test runs | Implement caching, batching |

---

## 📝 Design Decisions Log

### 2026-01-13: Smactorio Integration Strategy
**Decision:** Tight coupling with Smactorio as core dependency
**Rationale:** 
- Smactorio provides governance-as-a-service needed for compliant agent testing
- Shared LLM infrastructure reduces complexity
- Agent patterns (BaseAgent, AgentFactory) directly applicable
- DSPy integration already battle-tested
- Enables deterministic, auditable testing workflows

### 2026-01-13: Python Version Constraint
**Decision:** Require Python >=3.12
**Rationale:**
- Smactorio requires 3.12+
- Modern typing features (TypedDict, Self, etc.)
- Performance improvements in 3.12

### 2026-01-13: DeepTeam Deferral
**Decision:** Place red team capabilities on roadmap, not initial release
**Rationale:**
- Package availability uncertain
- Core functionality (analysis, testing, evaluation) more critical
- Can implement basic adversarial testing manually first

---

## 📚 References

- [Quaestor Implementation Plan](docs/QUAESTOR_IMPLEMENTATION_PLAN.md)
- [Quaestor Architecture](docs/quaestor_architecture.py)
- [Smactorio Repository](https://github.com/leonbreukelman/smactorio)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DeepEval Documentation](https://docs.confident-ai.com/)
