# Quaestor - Development History & Decisions

> Self-optimizing agentic testing framework built on Smactorio governance infrastructure

## ⚠️ Roadmap Migration Notice

**This file is now archival.** Active development tracking has been migrated to GitHub Issues.

| Resource | Purpose |
|----------|---------|
| [GitHub Issues](https://github.com/leonbreukelman/quaestor/issues) | Active task tracking, roadmap, feature requests |
| [PR #1](https://github.com/leonbreukelman/quaestor/pull/1) | Phase 1 Core Analysis Engine |
| This file | Historical decisions and archived roadmap |

See governance principle `quality-004` (GitHub-Native Project Management).

---

## Completed Phases

### ✅ Phase 0: Foundation (Completed 2026-01-13)
- [x] Project scaffolding and directory structure
- [x] `pyproject.toml` with Smactorio as core dependency
- [x] Python >=3.12 constraint set
- [x] Documentation structure created
- [x] Smactorio installation and configuration
- [x] CI/CD pipeline (GitHub Actions)
- [x] Pre-commit hooks (ruff, mypy, detect-secrets)
- [x] CLI skeleton with 7 commands

### ✅ Phase 1: Core Analysis Engine (Completed 2026-01-14, PR #1)
- [x] Domain models (`analysis/models.py`) - 100% coverage
- [x] Python parser (`analysis/parser.py`) - Tree-sitter based, ~83% coverage
- [x] Workflow analyzer (`analysis/workflow_analyzer.py`) - DSPy integration, ~96% coverage
- [x] Static linter (`analysis/linter.py`) - 12 rules Q001-Q051, ~95% coverage
- [x] Analysis pipeline (`analysis/pipeline.py`) - Unified API, ~83% coverage
- [x] 73 passing tests, 90% overall coverage

### 🔄 Phase 2: Test Generation (In Progress)
- [x] TestCase/TestSuite Pydantic models (`testing/models.py`) - Issue #2, 2026-01-14
  - Discriminated union Assertion types (6 variants)
  - JSON/YAML serialization for OSCAL compatibility
  - 45 tests, 98.94% module coverage
- [ ] Fixture system for reusable test components - Issue #3
- [ ] TestDesigner DSPy module

---

## 📅 Archived Roadmap

> **Note:** This roadmap is archived. See [GitHub Issues](https://github.com/leonbreukelman/quaestor/issues) for active tracking.

### Phase 2: Test Generation (Target: Week 2-3)
See [Issue #2](https://github.com/leonbreukelman/quaestor/issues/2) - Epic
- **TestDesigner** - DSPy module for test scenario generation
- **TestCase Models** - Pydantic schemas
- **Fixture System** - Reusable test components

### Phase 3: Runtime Testing (Target: Week 3-4)
See [Issue #6](https://github.com/leonbreukelman/quaestor/issues/6) - Epic
- **QuaestorInvestigator** - Multi-turn conversational prober
- **Target Adapters** - HTTP, Python import, MCP

### Phase 4: Evaluation & Judgment (Target: Week 4-5)
See [Issue #11](https://github.com/leonbreukelman/quaestor/issues/11) - Epic
- **QuaestorJudge** - LLM-as-judge with DeepEval
- **DeepEval Integration** - Custom metrics wrapping
  - Assertion library
  - Scoring infrastructure

### Phase 5: Coverage & Reporting (Target: Week 5-6)
See [Issue #15](https://github.com/leonbreukelman/quaestor/issues/15) - Epic
- **Coverage Tracker** - Tool, state, transition, invariant coverage
- **Report Generator** - HTML reports, JSON/SARIF for CI

### Phase 6: Optimization Engine (Target: Week 6-7)
See [Issue #20](https://github.com/leonbreukelman/quaestor/issues/20) - Epic
- **DSPy Optimization** - MIPROv2, self-improving test generation
- **Learning System** - Pattern recognition, adaptive scaling

### Phase 7: Red Team Capabilities (DEFERRED)
See [Issue #22](https://github.com/leonbreukelman/quaestor/issues/22) - Epic
> ⚠️ **DEFERRED**: DeepTeam package availability TBD

---

## 🔧 Technical Debt & Improvements

### Infrastructure (Completed ✅)
- [x] CI/CD pipeline (GitHub Actions)
- [x] Pre-commit hooks
- [x] Automated testing on PR
- [ ] Documentation site (MkDocs)

### Code Quality (Achieved ✅)
- [x] 90% test coverage (target was 80%)
- [x] Type hints throughout (mypy)
- [ ] Docstrings for public API

### Performance (Future)
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
