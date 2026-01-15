"""
CoverageTracker - Multi-dimensional coverage tracking for agent testing.

Tracks coverage across four dimensions:
- Tool coverage: Which tools/functions were called
- State coverage: Which agent states were visited
- Transition coverage: Which state transitions occurred
- Invariant coverage: Which invariants were tested

Part of Phase 5: Coverage & Reporting.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class CoverageDimension(str, Enum):
    """Coverage tracking dimensions."""

    TOOL = "tool"
    STATE = "state"
    TRANSITION = "transition"
    INVARIANT = "invariant"


@dataclass
class DimensionCoverage:
    """Coverage data for a single dimension."""

    dimension: CoverageDimension
    total_items: int = 0
    covered_items: set[str] = field(default_factory=set)
    uncovered_items: set[str] = field(default_factory=set)

    @property
    def coverage_count(self) -> int:
        """Number of covered items."""
        return len(self.covered_items)

    @property
    def coverage_percentage(self) -> float:
        """Coverage percentage (0.0 to 100.0)."""
        if self.total_items == 0:
            return 100.0
        return (self.coverage_count / self.total_items) * 100.0

    @property
    def gap_count(self) -> int:
        """Number of uncovered items."""
        return len(self.uncovered_items)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "total_items": self.total_items,
            "covered_count": self.coverage_count,
            "uncovered_count": self.gap_count,
            "coverage_percentage": round(self.coverage_percentage, 2),
            "covered_items": sorted(self.covered_items),
            "uncovered_items": sorted(self.uncovered_items),
        }


@dataclass
class CoverageReport:
    """Complete coverage report across all dimensions."""

    session_id: str
    timestamp: datetime
    dimensions: dict[CoverageDimension, DimensionCoverage]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_coverage(self) -> float:
        """Average coverage across all dimensions."""
        if not self.dimensions:
            return 100.0
        percentages = [d.coverage_percentage for d in self.dimensions.values()]
        return sum(percentages) / len(percentages)

    @property
    def has_gaps(self) -> bool:
        """Check if there are any coverage gaps."""
        return any(d.gap_count > 0 for d in self.dimensions.values())

    def get_dimension(self, dimension: CoverageDimension) -> DimensionCoverage | None:
        """Get coverage for a specific dimension."""
        return self.dimensions.get(dimension)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_coverage": round(self.overall_coverage, 2),
            "has_gaps": self.has_gaps,
            "dimensions": {
                dim.value: cov.to_dict() for dim, cov in self.dimensions.items()
            },
            "metadata": self.metadata,
        }


class CoverageTracker:
    """
    Track multi-dimensional coverage for agent testing.

    Supports four coverage dimensions:
    - Tool coverage: Functions/tools called
    - State coverage: Agent states visited
    - Transition coverage: State transitions
    - Invariant coverage: Tested invariants
    """

    def __init__(
        self,
        session_id: str | None = None,
        tools: list[str] | None = None,
        states: list[str] | None = None,
        transitions: list[tuple[str, str]] | None = None,
        invariants: list[str] | None = None,
    ):
        """
        Initialize coverage tracker.

        Args:
            session_id: Unique session identifier (generated if None)
            tools: List of all available tools
            states: List of all possible states
            transitions: List of all valid transitions (from_state, to_state)
            invariants: List of all invariants to check
        """
        self.session_id = session_id or f"cov-{uuid4().hex[:8]}"
        self.start_time = datetime.now(UTC)

        # Initialize dimensions
        self.dimensions: dict[CoverageDimension, DimensionCoverage] = {}

        # Tool coverage
        tools = tools or []
        self.dimensions[CoverageDimension.TOOL] = DimensionCoverage(
            dimension=CoverageDimension.TOOL,
            total_items=len(tools),
            uncovered_items=set(tools),
        )

        # State coverage
        states = states or []
        self.dimensions[CoverageDimension.STATE] = DimensionCoverage(
            dimension=CoverageDimension.STATE,
            total_items=len(states),
            uncovered_items=set(states),
        )

        # Transition coverage
        transitions = transitions or []
        transition_keys = [f"{from_s}->{to_s}" for from_s, to_s in transitions]
        self.dimensions[CoverageDimension.TRANSITION] = DimensionCoverage(
            dimension=CoverageDimension.TRANSITION,
            total_items=len(transition_keys),
            uncovered_items=set(transition_keys),
        )

        # Invariant coverage
        invariants = invariants or []
        self.dimensions[CoverageDimension.INVARIANT] = DimensionCoverage(
            dimension=CoverageDimension.INVARIANT,
            total_items=len(invariants),
            uncovered_items=set(invariants),
        )

    def record_tool(self, tool_name: str) -> None:
        """Record a tool call."""
        dim = self.dimensions[CoverageDimension.TOOL]
        dim.covered_items.add(tool_name)
        dim.uncovered_items.discard(tool_name)

    def record_state(self, state_name: str) -> None:
        """Record a state visit."""
        dim = self.dimensions[CoverageDimension.STATE]
        dim.covered_items.add(state_name)
        dim.uncovered_items.discard(state_name)

    def record_transition(self, from_state: str, to_state: str) -> None:
        """Record a state transition."""
        transition_key = f"{from_state}->{to_state}"
        dim = self.dimensions[CoverageDimension.TRANSITION]
        dim.covered_items.add(transition_key)
        dim.uncovered_items.discard(transition_key)

    def record_invariant(self, invariant_name: str) -> None:
        """Record an invariant test."""
        dim = self.dimensions[CoverageDimension.INVARIANT]
        dim.covered_items.add(invariant_name)
        dim.uncovered_items.discard(invariant_name)

    def get_coverage(self, dimension: CoverageDimension) -> DimensionCoverage:
        """Get coverage for a specific dimension."""
        return self.dimensions[dimension]

    def get_gaps(self, dimension: CoverageDimension) -> set[str]:
        """Get uncovered items for a dimension."""
        return self.dimensions[dimension].uncovered_items.copy()

    def generate_report(self, metadata: dict[str, Any] | None = None) -> CoverageReport:
        """
        Generate a coverage report.

        Args:
            metadata: Optional metadata to include in the report

        Returns:
            CoverageReport with current coverage state
        """
        return CoverageReport(
            session_id=self.session_id,
            timestamp=datetime.now(UTC),
            dimensions=self.dimensions.copy(),
            metadata=metadata or {},
        )

    def merge(self, other: "CoverageTracker") -> "CoverageTracker":
        """
        Merge coverage from another tracker.

        Creates a new tracker with combined coverage from both.

        Args:
            other: Another CoverageTracker to merge

        Returns:
            New CoverageTracker with merged coverage
        """
        # Collect all items from both trackers
        all_tools = set()
        all_states = set()
        all_transitions = set()
        all_invariants = set()

        for tracker in [self, other]:
            all_tools.update(tracker.dimensions[CoverageDimension.TOOL].covered_items)
            all_tools.update(tracker.dimensions[CoverageDimension.TOOL].uncovered_items)

            all_states.update(tracker.dimensions[CoverageDimension.STATE].covered_items)
            all_states.update(tracker.dimensions[CoverageDimension.STATE].uncovered_items)

            all_transitions.update(tracker.dimensions[CoverageDimension.TRANSITION].covered_items)
            all_transitions.update(tracker.dimensions[CoverageDimension.TRANSITION].uncovered_items)

            all_invariants.update(tracker.dimensions[CoverageDimension.INVARIANT].covered_items)
            all_invariants.update(tracker.dimensions[CoverageDimension.INVARIANT].uncovered_items)

        # Parse transitions back to tuples
        transition_tuples = []
        for trans in all_transitions:
            if "->" in trans:
                from_s, to_s = trans.split("->", 1)
                transition_tuples.append((from_s, to_s))

        # Create new merged tracker
        merged = CoverageTracker(
            session_id=f"merged-{uuid4().hex[:8]}",
            tools=list(all_tools),
            states=list(all_states),
            transitions=transition_tuples,
            invariants=list(all_invariants),
        )

        # Mark items as covered if covered in either tracker
        for tracker in [self, other]:
            for tool in tracker.dimensions[CoverageDimension.TOOL].covered_items:
                merged.record_tool(tool)
            for state in tracker.dimensions[CoverageDimension.STATE].covered_items:
                merged.record_state(state)
            for trans in tracker.dimensions[CoverageDimension.TRANSITION].covered_items:
                if "->" in trans:
                    from_s, to_s = trans.split("->", 1)
                    merged.record_transition(from_s, to_s)
            for inv in tracker.dimensions[CoverageDimension.INVARIANT].covered_items:
                merged.record_invariant(inv)

        return merged

    def reset(self) -> None:
        """Reset all coverage counters."""
        for dim_cov in self.dimensions.values():
            # Move all covered items back to uncovered
            dim_cov.uncovered_items.update(dim_cov.covered_items)
            dim_cov.covered_items.clear()
