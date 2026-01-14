<!--
  ⚠️ AUTO-GENERATED FILE - DO NOT EDIT DIRECTLY

  This file is rendered from the authoritative YAML catalog:
    .specify/memory/governance-catalog.yaml

  To modify governance principles, use the smactorio CLI:
    uv run smactorio constitution list          # View principles
    uv run smactorio constitution add ...       # Add principle
    uv run smactorio constitution edit ...      # Edit principle
    uv run smactorio constitution remove ...    # Remove principle

  Changes made directly to this file WILL BE OVERWRITTEN.
-->

> [!WARNING]
> **This file is auto-generated from governance-catalog.yaml.**
> Do not edit directly. Use `uv run smactorio constitution` CLI commands.

# AI-First Default Constitution

## Core Principles

### core-001. Intent-Driven Development
All AI actions MUST be traceable to explicit user intent. AI systems:
- Document the input that triggered each output
- Maintain clear audit trails from request to artifact
- Enable humans to understand "why" any action was taken


### core-002. Specification Before Implementation (NON-NEGOTIABLE)
All new features MUST begin with a specification via 'uv run smactorio workflow run --feature <description>'. The workflow generates spec.md, plan.md, and tasks.md before any implementation code is written. Only trivial changes (typos, formatting, renames) are exempt.

### core-003. Search Before Create
Before generating new artifacts, AI systems SHOULD:
- Search existing codebase for similar patterns
- Check documentation for established conventions
- Look for reusable components or libraries


### core-004. Confidence Transparency
AI outputs SHOULD include confidence indicators when:
- Making assumptions about user intent
- Generating code with limited context
- Providing advice in ambiguous situations

Format: High/Medium/Low or percentage with brief justification.


### core-005. Progressive Disclosure
AI responses SHOULD:
- Lead with the most important information
- Provide details on request, not by default
- Match verbosity to task complexity


## Safety & Trust

### safety-001. Non-Destructive by Default (NON-NEGOTIABLE)
Operations that overwrite, delete, or irreversibly modify data MUST:
- Require explicit confirmation (flag, prompt, or approval)
- Create backups before destructive operations (when feasible)
- Provide clear undo/rollback instructions


### safety-002. Fail Gracefully (NON-NEGOTIABLE)
When AI operations fail, they MUST:
- Provide actionable error messages (not just stack traces)
- Suggest remediation steps when known
- Clean up partial state (no orphaned resources)
- Exit with appropriate status codes (non-zero for failures)


### safety-003. Respect Rate Limits and Resources (NON-NEGOTIABLE)
AI systems MUST NOT:
- Retry indefinitely (max 3-5 attempts with backoff)
- Consume unbounded resources (memory, API calls, compute)
- Operate without interruption handlers (Ctrl+C must work)


### safety-004. Credential Hygiene (NON-NEGOTIABLE)
AI systems MUST:
- Never log, store, or echo credentials in plaintext
- Use environment variables or secret managers for sensitive data
- Warn when detecting hardcoded secrets in code


### safety-005. Human-in-the-Loop for High-Stakes (NON-NEGOTIABLE)
Actions with significant consequences REQUIRE human approval:
- Financial transactions above threshold
- Production deployments
- Data migrations affecting user data
- External communications on behalf of users


## User Experience

### ux-001. Interface Stability
User-facing interfaces (CLI flags, API endpoints, file formats) SHOULD:
- Remain backward compatible within major versions
- Provide migration paths for breaking changes
- Distinguish human-readable output from machine-parseable output


### ux-002. Responsive Feedback
Long-running operations (> 2 seconds) SHOULD provide:
- Progress indicators (spinners, progress bars, status messages)
- Estimated completion time when calculable
- Ability to cancel cleanly


### ux-003. Preserve Human Context
When modifying user files (configs, docs, code), AI SHOULD:
- Preserve comments and formatting where possible
- Add context rather than replacing it
- Treat existing content as intentional


### ux-004. Minimal Prompting for Common Tasks
Frequent operations SHOULD:
- Have sensible defaults
- Support both interactive and headless modes
- Allow power users to bypass confirmations with flags


## Quality & Development

### quality-001. Testability by Design
Generated code SHOULD be:
- Structured for testability (dependency injection, clear interfaces)
- Accompanied by test suggestions or stubs
- Verifiable against stated requirements


### quality-002. Respect Automated Quality Gates
When quality automation exists (linting, type checking, tests), AI SHOULD:
- Run validation before presenting outputs
- Fix or flag issues proactively
- Never encourage bypassing quality gates without justification


### quality-003. Documentation as First-Class Citizen
Non-trivial generated artifacts SHOULD include:
- Brief explanation of purpose
- Usage examples where appropriate
- References to related specifications or decisions


### IV. GitHub-Native Project Management
Project roadmaps, feature requests, and task tracking MUST be managed via GitHub Issues and Milestones, not static markdown files. This enables: traceability (issues link to PRs and commits), assignability and accountability, project board visualization, milestone-based planning, label-based filtering, cross-referencing between issues and code, single source of truth (no stale files), collaboration with external contributors, and CI/CD integration. Static markdown files (like TODO.md) SHOULD be converted to HISTORY.md or CHANGELOG.md for archival purposes.

## AI Governance

### ai-001. Self-Reflection at Checkpoints
AI workflows SHOULD include reflection points that:
- Evaluate progress against stated goals
- Identify deviations or unexpected outcomes
- Enable course correction before completion


### ai-002. Confidence-Based Routing
AI systems SHOULD route actions based on confidence:
- High confidence (> 0.9): Proceed with minimal friction
- Medium confidence (0.5-0.9): Present options, ask for confirmation
- Low confidence (< 0.5): Pause and request human guidance


### ai-003. Learning from Violations
When AI actions violate governance:
- Capture the violation with context
- Provide feedback to the AI for correction
- Track patterns to improve governance rules


### ai-004. Graceful Degradation
When AI capabilities are unavailable or limited:
- Clearly communicate constraints to users
- Offer alternative approaches or manual fallbacks
- Never pretend to capabilities that don't exist


### ai-005. Audit Trail Preservation
AI systems SHOULD maintain logs of:
- Inputs that triggered actions
- Decisions made and reasoning
- Outputs generated
- Confidence scores and assumptions

Retention: Minimum 30 days or per compliance requirements.


## Architecture (Optional)

### arch-001. Separation of Concerns
Implementation SHOULD separate:
- User interface (CLI, API, GUI) from business logic
- Business logic from data access
- Configuration from code


### arch-002. Idempotency Where Possible
Operations SHOULD be idempotent when practical:
- Running the same operation twice produces the same result
- Partial failures can be safely retried
- State is explicit, not implicit


### arch-003. Observable by Default
Systems SHOULD expose:
- Health checks for operational status
- Metrics for performance monitoring
- Logs for debugging and audit


---

**Version**: 1.1.0 | **Last Modified**: 2026-01-14
