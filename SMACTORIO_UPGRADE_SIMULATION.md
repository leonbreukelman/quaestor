# Smactorio Upgrade Simulation Report

**Date**: 2026-01-14  
**Current Branch**: `feature/phase-1-core-analysis-engine`  
**Upgrade Target**: Latest `main` from https://github.com/leonbreukelman/smactorio

---

## Executive Summary

✅ **SAFE TO UPGRADE** - The upgrade from smactorio appears to be **low-risk** with **no breaking changes** expected for the quaestor repository.

### Key Findings

- **9 new commits** available (minor enhancements and fixes)
- **No breaking API changes** detected
- **No direct code dependencies** on smactorio Python modules (only CLI usage)
- **Minimal dependency changes** (one new pytest marker)
- **Current test suite**: 389 passing, 1 failing (unrelated to smactorio)

---

## Current State

### Installed Version
```
Name: smactorio
Version: 0.1.0
Current Commit: 652587aa53ef8b989ffc281c6e50416abb475cfb
Location: .venv/lib/python3.12/site-packages
```

### Remote Version
```
Latest Commit: b9d31c8dc9273d409c7c2b5f8fd6c6a9aaed4b24
Commits Behind: 9
```

---

## Changes in Smactorio (9 commits)

### 1. `b9d31c8` - fix(tests): resolve interceptor upstream test failures
- **Impact**: None - Test fixes only
- **Risk**: Low

### 2. `1162f4d` - feat(agents): add ContractGeneratorAgent for API contracts (#90)
- **Impact**: New agent available
- **Risk**: None - Additive feature
- **Benefit**: Can be used for future contract generation

### 3. `1dd4330` - refactor(tests): remove unused imports in graceful_degradation tests
- **Impact**: None - Code cleanup
- **Risk**: Low

### 4. `6aaed4d` - chore: update work-queue with completed tasks
- **Impact**: None - Documentation
- **Risk**: None

### 5. `149a2d4` - feat(dx): add skip_verification option to simulation harness
- **Impact**: New optional feature for DX tools
- **Risk**: None - Additive feature

### 6. `359e349` - fix(tests): improve CI compatibility and pytest-asyncio fixtures
- **Impact**: Better test infrastructure
- **Risk**: Low - Improves stability

### 7. `053a0fc` - feat(interceptor): implement upstream forwarding with circuit breaker
- **Impact**: Experimental interceptor enhancement
- **Risk**: None - Feature is marked experimental

### 8. `bc43d23` - feat(agents): add ChecklistGeneratorAgent for checklists/ directory
- **Impact**: New agent available
- **Risk**: None - Additive feature
- **Benefit**: Can generate checklists automatically

### 9. `c239c15` - feat: Enhanced SDD Artifact Generation (spec-kit parity) (#87)
- **Impact**: Enhanced artifact generation capabilities
- **Risk**: Low - Core enhancement
- **Benefit**: Better spec-kit integration

---

## Dependency Analysis

### Python Module Imports
```bash
grep -r "from smactorio" --include="*.py" .
Result: No direct smactorio imports found
```

**Analysis**: Quaestor **does not import** any Python modules from smactorio. All integration is via:
- CLI commands (`uv run smactorio constitution list`)
- YAML file reading (`.specify/memory/governance-catalog.yaml`)
- No programmatic API dependencies

### CLI Usage in Quaestor
Smactorio is used via CLI in these contexts:
1. **Documentation** (AGENTS.md, README.md, copilot-instructions.md)
   - `smactorio constitution list`
   - `smactorio constitution check`
   - `smactorio workflow run`
   
2. **Dependency declaration** (pyproject.toml)
   - `smactorio[agentic] @ git+https://...` (main deps)
   - `smactorio @ git+https://...` (dev deps)

**CLI Stability**: All documented CLI commands remain stable across the upgrade.

### Dependency Changes
```diff
# Only change detected in pyproject.toml:
+ markers = [..., "network: marks tests that require network access to external services"]
```

**Impact**: Minor - adds a new pytest marker, doesn't affect existing functionality.

---

## Dry-Run Results

```bash
uv sync --dry-run --upgrade-package smactorio
```

### Would Update
```
- smactorio==0.1.0 (from git+...@652587aa)
+ smactorio @ git+...@b9d31c8dc9
```

### Would Download
- 1 package

### Would Uninstall/Install
- 47 packages (dev dependencies like pre-commit, mypy, ruff, bandit, etc.)
- These are **transitive dev dependencies** being refreshed, not breaking changes

---

## Risk Assessment

### Breaking Change Analysis

| Category | Risk Level | Reasoning |
|----------|-----------|-----------|
| **API Changes** | ✅ None | No Python imports; CLI-only usage |
| **CLI Interface** | ✅ Stable | All documented commands unchanged |
| **Dependencies** | ✅ Minor | One pytest marker added |
| **Test Compatibility** | ✅ Safe | Fixes improve test stability |
| **Governance Integration** | ✅ Safe | OSCAL catalog format unchanged |

### Current Test Status (Baseline)
```
389 passed, 1 failed, 6 warnings in 3.57s
Coverage: 81.30% (target: 85%)

Failed test: tests/test_linter.py::TestStaticLinter::test_detect_potential_infinite_loop
Status: UNRELATED to smactorio (linter test issue)
```

---

## Benefits of Upgrading

1. **New Agents Available**
   - `ContractGeneratorAgent` - Generate API contracts automatically
   - `ChecklistGeneratorAgent` - Generate implementation checklists
   
2. **Enhanced DX Tools**
   - `skip_verification` option for faster iteration
   - Better CI compatibility
   
3. **Improved Stability**
   - Test fixtures improvements
   - Circuit breaker for interceptor
   
4. **Bug Fixes**
   - Interceptor upstream test failures resolved
   - pytest-asyncio fixture improvements

---

## Recommended Upgrade Path

### Option 1: Safe Upgrade (Recommended)
```bash
# 1. Ensure clean git state
git status

# 2. Create backup branch
git branch backup/pre-smactorio-upgrade

# 3. Upgrade smactorio
uv sync --upgrade-package smactorio

# 4. Run full test suite
uv run pytest tests/ -v

# 5. Verify governance still works
uv run smactorio constitution list
uv run smactorio constitution validate

# 6. Commit if successful
git add uv.lock
git commit -m "chore: upgrade smactorio to b9d31c8"
```

### Option 2: Incremental Testing
```bash
# Test specific smactorio features before full upgrade
uv run --with smactorio@git+https://github.com/leonbreukelman/smactorio.git@b9d31c8dc9 \
  smactorio constitution list

# If successful, proceed with Option 1
```

---

## Post-Upgrade Verification Checklist

- [ ] Run full test suite: `uv run pytest tests/ -v`
- [ ] Verify governance commands:
  - [ ] `uv run smactorio constitution list`
  - [ ] `uv run smactorio constitution validate`
  - [ ] `uv run smactorio constitution check specs/*/spec.md`
- [ ] Check CLI help: `uv run smactorio --help`
- [ ] Verify workflow command: `uv run smactorio workflow --help`
- [ ] Test agent listing: `uv run smactorio agent list`
- [ ] Run pre-commit hooks: `uv run pre-commit run --all-files`
- [ ] Verify coverage: `uv run pytest --cov`

---

## Rollback Plan (If Needed)

```bash
# If issues arise after upgrade:

# Option A: Restore from backup branch
git checkout backup/pre-smactorio-upgrade
git branch -D feature/phase-1-core-analysis-engine
git checkout -b feature/phase-1-core-analysis-engine

# Option B: Pin to previous commit
# Edit pyproject.toml:
# Change: smactorio[agentic] @ git+https://...
# To: smactorio[agentic] @ git+https://...@652587aa53ef8b989ffc281c6e50416abb475cfb
uv sync
```

---

## Conclusion

**Recommendation**: ✅ **PROCEED WITH UPGRADE**

The upgrade is **safe** and **low-risk** because:
1. No breaking changes in the 9 new commits
2. No direct Python API dependencies
3. CLI interface remains stable
4. Only additive features and bug fixes
5. Better test infrastructure and stability

**Expected Outcome**: Seamless upgrade with access to new agents (ContractGenerator, ChecklistGenerator) and improved stability.

**Action Item**: Execute **Option 1: Safe Upgrade** during Phase 4 work or as a standalone task.

---

## Additional Notes

- The failing test (`test_detect_potential_infinite_loop`) is unrelated to smactorio and should be fixed separately
- Coverage is at 81.30% (below 85% target) - consider addressing during Phase 4
- Consider using new `ContractGeneratorAgent` for generating API contracts in Phase 4 evaluation system
