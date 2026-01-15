#!/usr/bin/env python3
"""
AI State Management - Machine-Readable Project State

Provides CLI and Python API for AI agents to read/write project state.
Enables autonomous operation, accountability, and state recovery.

Usage:
    # CLI
    python scripts/ai_state.py status
    python scripts/ai_state.py next
    python scripts/ai_state.py complete w004       # Auto-checks commit readiness
    python scripts/ai_state.py verify
    python scripts/ai_state.py commit-reminder     # Detailed commit instructions
    python scripts/ai_state.py export-session

    # Python API
    from scripts.ai_state import AIState
    state = AIState.load()
    next_work = state.get_next_work()
    state.mark_complete('w003')
"""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

STATE_FILE = Path(__file__).parent.parent / ".specify" / "memory" / "state.yaml"


class AIState:
    """AI-readable project state manager."""

    def __init__(self, data: dict[str, Any]):
        """Initialize with state data."""
        self.data = data

    @classmethod
    def load(cls) -> "AIState":
        """Load state from YAML file."""
        if not STATE_FILE.exists():
            raise FileNotFoundError(f"State file not found: {STATE_FILE}")

        with open(STATE_FILE) as f:
            data = yaml.safe_load(f)

        return cls(data)

    def save(self) -> None:
        """Save state to YAML file."""
        self.data["metadata"]["last_updated"] = datetime.now(UTC).isoformat()

        with open(STATE_FILE, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

    def get_status(self) -> dict[str, Any]:
        """Get current project status."""
        gates = self.data["quality_gates"]
        tests_total = int(gates["tests_total"])
        tests_passing = int(round(float(gates["tests_passing_ratio"]) * tests_total))
        return {
            "project": self.data["project"]["name"],
            "branch": self.data["project"]["current_branch"],
            "current_phase": self._get_current_phase(),
            "coverage": gates["current_coverage"],
            "tests": f"{tests_passing}/{tests_total}",
            "can_commit": gates["can_commit"],
            "blockers": gates["blockers"],
            "work_queue_size": len(self.data["work_queue"]),
        }

    def _get_current_phase(self) -> str:
        """Determine current phase."""
        for phase_id, phase_data in self.data["phases"].items():
            if phase_data["status"] in ("in-progress", "ready"):
                return f"{phase_id}: {phase_data['name']}"
        return "No active phase"

    def get_next_work(self) -> dict[str, Any] | None:
        """Get next work item from queue."""
        queue = cast(list[dict[str, Any]], self.data.get("work_queue", []))
        if not queue:
            return None

        # Return first non-blocked item
        for item in queue:
            if not item.get("blocked", False):
                return item

        return None

    def mark_complete(self, work_id: str, **kwargs: Any) -> None:
        """
        Mark work item as complete.

        Args:
            work_id: Work item ID (e.g., 'w004')
            **kwargs: Additional completion data (tests_passing, coverage_delta, etc.)
        """
        # Find in queue
        queue = self.data.get("work_queue", [])
        work_item = None
        for i, item in enumerate(queue):
            if item["id"] == work_id:
                work_item = queue.pop(i)
                break

        if not work_item:
            raise ValueError(f"Work item {work_id} not found in queue")

        # Add to completed with timestamp
        completion = {
            **work_item,
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "complete",
            **kwargs,
        }

        self.data["work_completed"].append(completion)
        self.save()

    def add_work(self, **kwargs: Any) -> str:
        """
        Add new work item to queue.

        Args:
            module: Module name
            action: Action description
            description: Detailed description
            priority: high/medium/low
            estimated_effort: large/medium/small

        Returns:
            Work item ID
        """
        # Generate new ID
        existing_ids = [w["id"] for w in self.data["work_queue"]]
        existing_ids.extend([w["id"] for w in self.data["work_completed"]])

        max_id = 0
        for work_id in existing_ids:
            if work_id.startswith("w"):
                try:
                    num = int(work_id[1:])
                    max_id = max(max_id, num)
                except ValueError:
                    # Ignore non-numeric work IDs (e.g., legacy or malformed entries) when computing max_id
                    continue

        new_id = f"w{max_id + 1:03d}"

        work_item = {
            "id": new_id,
            "blocked": False,
            "dependencies": [],
            **kwargs,
        }

        self.data["work_queue"].append(work_item)
        self.save()

        return new_id

    def verify_quality_gates(self) -> dict[str, Any]:
        """Verify quality gates and update can_commit/can_merge flags.

        Notes:
            - `can_commit` is based only on automated quality gates (tests + coverage).
            - `can_merge` additionally requires no in-progress phases.
        """
        gates = self.data["quality_gates"]
        quality_blockers: list[str] = []

        # Check coverage
        if gates["current_coverage"] < gates["coverage_threshold"]:
            quality_blockers.append(
                f"Coverage {gates['current_coverage']:.2f}% < threshold {gates['coverage_threshold']:.2f}%"
            )

        # Check tests
        tests_passing = int(gates["tests_passing_ratio"] * gates["tests_total"])
        if tests_passing < gates["tests_total"]:
            quality_blockers.append(f"{gates['tests_total'] - tests_passing} test(s) failing")

        # Check phase completeness
        merge_blockers: list[str] = []
        incomplete_phases: list[str] = []
        for phase_id, phase_data in self.data["phases"].items():
            if phase_data["status"] == "in-progress":
                incomplete_phases.append(f"{phase_id}: {phase_data['name']}")

        if incomplete_phases:
            merge_blockers.append(f"Incomplete phases: {', '.join(incomplete_phases)}")

        # Update gates
        gates["blockers"] = quality_blockers
        gates["can_commit"] = len(quality_blockers) == 0
        gates["can_merge"] = gates["can_commit"] and not incomplete_phases

        self.save()

        return {
            "can_commit": gates["can_commit"],
            "can_merge": gates["can_merge"],
            "blockers": quality_blockers,
            "merge_blockers": merge_blockers,
        }

    def export_session(self) -> str:
        """Export current session as markdown for HISTORY.md."""
        session = self.data["current_session"]
        completed = self.data["work_completed"]

        # Filter to current session
        session_work = [
            w for w in completed if w.get("timestamp", "").startswith(session["started"][:10])
        ]

        md = f"""
### Session: {session["id"]} ({session["started"][:10]})

**Objective**: {session["objective"]}
**Agent**: {session["agent"]}
**Duration**: {len(session_work)} work items completed

#### Work Completed
"""

        for work in session_work:
            md += f"""
- **{work["id"]}**: {work["description"]}
  - Module: `{work["module"]}`
  - Tests: {work.get("tests_passing", "N/A")}
  - Coverage Δ: {work.get("coverage_delta", 0):+.2f}%
"""

        # Add quality status
        gates = self.data["quality_gates"]
        md += f"""
#### Quality Status
- Coverage: {gates["current_coverage"]:.2f}%
- Tests: {int(gates["tests_passing_ratio"] * gates["tests_total"])}/{gates["tests_total"]} passing
- Can Commit: {"✅" if gates["can_commit"] else "❌"}
"""

        if gates["blockers"]:
            md += "\n**Blockers**:\n"
            for blocker in gates["blockers"]:
                md += f"- {blocker}\n"

        return md


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: ai_state.py <command>")
        print("Commands: status, next, complete, verify, export-session")
        sys.exit(1)

    command = sys.argv[1]
    state = AIState.load()

    if command == "status":
        status = state.get_status()
        print(f"Project: {status['project']}")
        print(f"Branch: {status['branch']}")
        print(f"Phase: {status['current_phase']}")
        print(f"Coverage: {status['coverage']:.2f}%")
        print(f"Tests: {status['tests']}")
        print(f"Can Commit: {'✅' if status['can_commit'] else '❌'}")
        print(f"Work Queue: {status['work_queue_size']} items")
        if status["blockers"]:
            print("\nBlockers:")
            for blocker in status["blockers"]:
                print(f"  - {blocker}")

    elif command == "next":
        next_work = state.get_next_work()
        if next_work:
            print(f"Next Work: {next_work['id']}")
            print(f"Module: {next_work['module']}")
            print(f"Action: {next_work['action']}")
            print(f"Description: {next_work['description']}")
            print(f"Priority: {next_work['priority']}")
        else:
            print("No work in queue")

    elif command == "complete":
        if len(sys.argv) < 3:
            print("Usage: ai_state.py complete <work_id>")
            sys.exit(1)
        work_id = sys.argv[2]
        state.mark_complete(work_id)
        print(f"Marked {work_id} as complete")

        # Auto-check if ready to commit
        print("\n🔍 Checking if ready to commit...")
        result = state.verify_quality_gates()

        if result["can_commit"]:
            print("\n✅ READY TO COMMIT!")
            print("\n📝 Next steps:")
            print("   1. uv run pre-commit run --all-files")
            print("   2. git add tests/ .specify/memory/state.yaml scripts/")
            print(f"   3. git commit -m 'feat: Complete {work_id}'")
            print("   4. git push origin $(git branch --show-current)")
            print(
                "\n💡 Tip: Run 'uv run python scripts/ai_state.py commit-reminder' for full instructions"
            )
        else:
            print("\n⚠️  Not ready to commit yet")
            print("\nBlockers:")
            for blocker in result.get("blockers", []):
                print(f"  - {blocker}")

    elif command == "verify":
        result = state.verify_quality_gates()
        print(f"Can Commit: {'✅' if result['can_commit'] else '❌'}")
        print(f"Can Merge: {'✅' if result['can_merge'] else '❌'}")
        blockers = result.get("blockers", [])
        merge_blockers = result.get("merge_blockers", [])
        if blockers or merge_blockers:
            print("\nBlockers:")
            for blocker in blockers:
                print(f"  - {blocker}")
            for blocker in merge_blockers:
                print(f"  - {blocker}")

    elif command == "export-session":
        markdown = state.export_session()
        print(markdown)

    elif command == "commit-reminder":
        """Show detailed commit instructions."""
        result = state.verify_quality_gates()
        status = state.get_status()

        print("=" * 70)
        print("📦 COMMIT READINESS CHECK")
        print("=" * 70)
        print(f"\nProject: {status['project']}")
        print(f"Branch: {status['branch']}")
        print(f"Phase: {status['current_phase']}")
        print(f"\n✓ Coverage: {status['coverage']:.2f}% (threshold: ≥85%)")
        print(f"✓ Tests: {status['tests']} passing")
        print(f"✓ Quality Gates: {'✅ PASS' if result['can_commit'] else '❌ FAIL'}")

        if result["can_commit"]:
            print("\n" + "=" * 70)
            print("✅ READY TO COMMIT - Follow these steps:")
            print("=" * 70)

            # Get recent completed work
            completed = state.data["work_completed"]
            recent_work = completed[-3:] if len(completed) > 0 else []

            if recent_work:
                print("\n📋 Recent Work Completed:")
                for work in recent_work:
                    print(f"   • {work['id']}: {work['description'][:60]}...")

            print("\n🔧 Step 1: Run Pre-commit Hooks")
            print("   $ uv run pre-commit run --all-files")

            print("\n📁 Step 2: Stage Your Changes")
            print("   $ git status  # Review changes")
            print("   $ git add tests/")
            print("   $ git add .specify/memory/state.yaml")
            print("   $ git add scripts/  # if modified")

            print("\n💬 Step 3: Commit with Descriptive Message")
            latest_work = completed[-1] if completed else None
            if latest_work:
                work_id = latest_work["id"]
                desc = latest_work["description"]
                print(f"   $ git commit -m 'feat: {desc}")
                print("")
                print(f"   - Coverage: {status['coverage']:.2f}%")
                print(f"   - Tests: {status['tests']} passing")
                print("")
                print(f"   Closes: {work_id}'")
            else:
                print("   $ git commit -m 'feat: <your description here>'")

            print("\n🚀 Step 4: Push to Remote")
            print(f"   $ git push origin {status['branch']}")

            print("\n📝 Step 5: Export Session Log")
            print("   $ uv run python scripts/ai_state.py export-session >> HISTORY.md")

            print("\n" + "=" * 70)

        else:
            print("\n" + "=" * 70)
            print("⚠️  NOT READY TO COMMIT")
            print("=" * 70)
            print("\n❌ Blockers:")
            for blocker in result.get("blockers", []):
                print(f"   • {blocker}")
            print("\n💡 Fix the blockers above, then re-run this command")
            print("=" * 70)

    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  status          - Show project status")
        print("  next            - Show next work item")
        print("  complete <id>   - Mark work item complete (auto-checks commit readiness)")
        print("  verify          - Check quality gates")
        print("  commit-reminder - Detailed commit instructions")
        print("  export-session  - Export session log for HISTORY.md")
        sys.exit(1)


if __name__ == "__main__":
    main()
