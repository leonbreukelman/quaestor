# Quaestor

> Self-optimizing agentic testing framework – pytest for AI agents

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Built on Smactorio](https://img.shields.io/badge/built%20on-Smactorio-green.svg)](https://github.com/leonbreukelman/smactorio)

## Overview

Quaestor is a comprehensive testing framework for AI agents that combines:

- **DSPy-powered analysis** – Understand agent workflows, tools, and state machines
- **Automated test generation** – Create test cases from workflow analysis
- **Multi-turn probing** – Adaptive conversational testing
- **LLM-as-Judge evaluation** – Verdict generation with DeepEval metrics
- **Governance integration** – Built on Smactorio for compliant, deterministic testing

## Quick Start

```bash
# Install with uv
uv sync

# Verify installation
uv run quaestor --version

# Analyze an agent
uv run quaestor analyze path/to/agent.py

# Run tests
uv run quaestor test path/to/agent.py --level integration
```

## Project Structure

```
quaestor/
├── analysis/      # Code analysis and workflow extraction
├── testing/       # Test generation and execution
├── evaluation/    # LLM-as-judge verdict generation
├── coverage/      # Coverage tracking
├── reporting/     # HTML/SARIF report generation
└── optimization/  # DSPy self-improvement

.specify/          # Smactorio governance configuration
├── memory/
│   ├── governance-catalog.yaml   # OSCAL catalog (source of truth)
│   └── constitution.md           # Rendered governance rules
```

## Governance

Quaestor is tightly integrated with [Smactorio](https://github.com/leonbreukelman/smactorio) for governance-as-a-service:

```bash
# View governance principles
uv run smactorio constitution list

# Check compliance
uv run smactorio constitution check path/to/spec.md

# Run full spec-driven development workflow
uv run smactorio workflow run --feature "Your feature description"
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check .

# Type check
uv run mypy quaestor/
```

## Roadmap

See [TODO.md](TODO.md) for the complete development roadmap including:

- ✅ **Phase 0**: Project setup and Smactorio integration
- 🔄 **Phase 1**: Core analysis engine (WorkflowAnalyzer, Python parser)
- 📋 **Phase 2-6**: Test generation, runtime testing, evaluation, coverage, optimization
- ⏳ **Phase 7**: Red team capabilities (pending DeepTeam availability)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) – System design and Smactorio integration
- [TODO.md](TODO.md) – Development roadmap and task tracking
- [docs/QUAESTOR_IMPLEMENTATION_PLAN.md](docs/QUAESTOR_IMPLEMENTATION_PLAN.md) – Original implementation plan
- [docs/quaestor_architecture.py](docs/quaestor_architecture.py) – Architecture code sketch

## License

MIT
