"""
Tests for CoverageTracker.

Part of Phase 5: Coverage & Reporting.
"""

import pytest

from quaestor.coverage import (
    CoverageDimension,
    CoverageReport,
    CoverageTracker,
    DimensionCoverage,
)


class TestCoverageDimension:
    """Test CoverageDimension enum."""

    def test_all_dimensions_defined(self):
        """Test that all expected dimensions are defined."""
        assert CoverageDimension.TOOL == "tool"
        assert CoverageDimension.STATE == "state"
        assert CoverageDimension.TRANSITION == "transition"
        assert CoverageDimension.INVARIANT == "invariant"


class TestDimensionCoverage:
    """Test DimensionCoverage dataclass."""

    def test_create_empty(self):
        """Test creating empty dimension coverage."""
        dim = DimensionCoverage(
            dimension=CoverageDimension.TOOL,
            total_items=10,
        )
        assert dim.coverage_count == 0
        assert dim.coverage_percentage == 0.0
        assert dim.gap_count == 0

    def test_coverage_percentage_calculation(self):
        """Test coverage percentage calculation."""
        dim = DimensionCoverage(
            dimension=CoverageDimension.STATE,
            total_items=10,
            covered_items={"state1", "state2", "state3"},
        )
        assert dim.coverage_count == 3
        assert dim.coverage_percentage == 30.0

    def test_coverage_percentage_zero_items(self):
        """Test coverage percentage with zero total items."""
        dim = DimensionCoverage(
            dimension=CoverageDimension.TOOL,
            total_items=0,
        )
        assert dim.coverage_percentage == 100.0

    def test_gap_count(self):
        """Test gap counting."""
        dim = DimensionCoverage(
            dimension=CoverageDimension.TOOL,
            total_items=5,
            uncovered_items={"tool1", "tool2"},
        )
        assert dim.gap_count == 2

    def test_to_dict(self):
        """Test converting to dictionary."""
        dim = DimensionCoverage(
            dimension=CoverageDimension.TRANSITION,
            total_items=4,
            covered_items={"a->b", "b->c"},
            uncovered_items={"c->d", "d->a"},
        )
        result = dim.to_dict()

        assert result["dimension"] == "transition"
        assert result["total_items"] == 4
        assert result["covered_count"] == 2
        assert result["uncovered_count"] == 2
        assert result["coverage_percentage"] == 50.0
        assert set(result["covered_items"]) == {"a->b", "b->c"}
        assert set(result["uncovered_items"]) == {"c->d", "d->a"}


class TestCoverageReport:
    """Test CoverageReport."""

    @pytest.fixture
    def sample_report(self):
        """Create a sample coverage report."""
        dims = {
            CoverageDimension.TOOL: DimensionCoverage(
                dimension=CoverageDimension.TOOL,
                total_items=10,
                covered_items={"tool1", "tool2"},
                uncovered_items={"tool3", "tool4", "tool5", "tool6", "tool7", "tool8", "tool9", "tool10"},
            ),
            CoverageDimension.STATE: DimensionCoverage(
                dimension=CoverageDimension.STATE,
                total_items=5,
                covered_items={"state1", "state2", "state3"},
                uncovered_items={"state4", "state5"},
            ),
        }
        from datetime import UTC, datetime

        return CoverageReport(
            session_id="test-123",
            timestamp=datetime.now(UTC),
            dimensions=dims,
            metadata={"test": "value"},
        )

    def test_overall_coverage_calculation(self, sample_report):
        """Test overall coverage calculation."""
        # Tool: 2/10 = 20%, State: 3/5 = 60%
        # Average: (20 + 60) / 2 = 40%
        assert sample_report.overall_coverage == 40.0

    def test_has_gaps(self, sample_report):
        """Test gap detection."""
        # Both dimensions have gaps
        assert sample_report.has_gaps is True

    def test_has_no_gaps(self):
        """Test when there are no gaps."""
        dims = {
            CoverageDimension.TOOL: DimensionCoverage(
                dimension=CoverageDimension.TOOL,
                total_items=2,
                covered_items={"tool1", "tool2"},
            ),
        }
        from datetime import UTC, datetime

        report = CoverageReport(
            session_id="test-456",
            timestamp=datetime.now(UTC),
            dimensions=dims,
        )
        assert report.has_gaps is False

    def test_get_dimension(self, sample_report):
        """Test getting specific dimension."""
        tool_cov = sample_report.get_dimension(CoverageDimension.TOOL)
        assert tool_cov is not None
        assert tool_cov.dimension == CoverageDimension.TOOL
        assert tool_cov.coverage_count == 2

    def test_get_missing_dimension(self, sample_report):
        """Test getting dimension that doesn't exist."""
        inv_cov = sample_report.get_dimension(CoverageDimension.INVARIANT)
        assert inv_cov is None

    def test_to_dict(self, sample_report):
        """Test converting report to dictionary."""
        result = sample_report.to_dict()

        assert result["session_id"] == "test-123"
        assert "timestamp" in result
        assert result["overall_coverage"] == 40.0
        assert result["has_gaps"] is True
        assert "dimensions" in result
        assert "tool" in result["dimensions"]
        assert "state" in result["dimensions"]
        assert result["metadata"]["test"] == "value"


class TestCoverageTracker:
    """Test CoverageTracker main class."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        tracker = CoverageTracker()

        assert tracker.session_id.startswith("cov-")
        assert tracker.start_time is not None
        assert len(tracker.dimensions) == 4

        # All dimensions should be empty
        for dim in CoverageDimension:
            dim_cov = tracker.dimensions[dim]
            assert dim_cov.total_items == 0
            assert dim_cov.coverage_count == 0

    def test_init_with_tools(self):
        """Test initialization with tools."""
        tools = ["tool1", "tool2", "tool3"]
        tracker = CoverageTracker(tools=tools)

        tool_dim = tracker.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.total_items == 3
        assert tool_dim.gap_count == 3
        assert tool_dim.uncovered_items == set(tools)

    def test_init_with_states(self):
        """Test initialization with states."""
        states = ["idle", "processing", "done"]
        tracker = CoverageTracker(states=states)

        state_dim = tracker.get_coverage(CoverageDimension.STATE)
        assert state_dim.total_items == 3
        assert state_dim.uncovered_items == set(states)

    def test_init_with_transitions(self):
        """Test initialization with transitions."""
        transitions = [("idle", "processing"), ("processing", "done")]
        tracker = CoverageTracker(transitions=transitions)

        trans_dim = tracker.get_coverage(CoverageDimension.TRANSITION)
        assert trans_dim.total_items == 2
        assert "idle->processing" in trans_dim.uncovered_items
        assert "processing->done" in trans_dim.uncovered_items

    def test_init_with_invariants(self):
        """Test initialization with invariants."""
        invariants = ["inv1", "inv2"]
        tracker = CoverageTracker(invariants=invariants)

        inv_dim = tracker.get_coverage(CoverageDimension.INVARIANT)
        assert inv_dim.total_items == 2
        assert inv_dim.uncovered_items == set(invariants)

    def test_custom_session_id(self):
        """Test custom session ID."""
        tracker = CoverageTracker(session_id="custom-session")
        assert tracker.session_id == "custom-session"

    def test_record_tool(self):
        """Test recording tool calls."""
        tracker = CoverageTracker(tools=["tool1", "tool2", "tool3"])

        tracker.record_tool("tool1")
        tool_dim = tracker.get_coverage(CoverageDimension.TOOL)

        assert "tool1" in tool_dim.covered_items
        assert "tool1" not in tool_dim.uncovered_items
        assert tool_dim.coverage_count == 1
        assert tool_dim.gap_count == 2

    def test_record_multiple_tools(self):
        """Test recording multiple tool calls."""
        tracker = CoverageTracker(tools=["tool1", "tool2", "tool3"])

        tracker.record_tool("tool1")
        tracker.record_tool("tool2")
        tracker.record_tool("tool1")  # Duplicate should not matter

        tool_dim = tracker.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.coverage_count == 2
        assert tool_dim.coverage_percentage == pytest.approx(66.67, rel=0.01)

    def test_record_state(self):
        """Test recording state visits."""
        tracker = CoverageTracker(states=["idle", "active", "done"])

        tracker.record_state("idle")
        tracker.record_state("active")

        state_dim = tracker.get_coverage(CoverageDimension.STATE)
        assert state_dim.coverage_count == 2
        assert "done" in state_dim.uncovered_items

    def test_record_transition(self):
        """Test recording state transitions."""
        transitions = [("idle", "active"), ("active", "done")]
        tracker = CoverageTracker(transitions=transitions)

        tracker.record_transition("idle", "active")

        trans_dim = tracker.get_coverage(CoverageDimension.TRANSITION)
        assert "idle->active" in trans_dim.covered_items
        assert "active->done" in trans_dim.uncovered_items
        assert trans_dim.coverage_percentage == 50.0

    def test_record_invariant(self):
        """Test recording invariant tests."""
        tracker = CoverageTracker(invariants=["inv1", "inv2", "inv3"])

        tracker.record_invariant("inv1")
        tracker.record_invariant("inv3")

        inv_dim = tracker.get_coverage(CoverageDimension.INVARIANT)
        assert inv_dim.coverage_count == 2
        assert "inv2" in inv_dim.uncovered_items

    def test_get_gaps(self):
        """Test getting coverage gaps."""
        tracker = CoverageTracker(tools=["tool1", "tool2", "tool3"])
        tracker.record_tool("tool1")

        gaps = tracker.get_gaps(CoverageDimension.TOOL)
        assert gaps == {"tool2", "tool3"}

    def test_get_gaps_returns_copy(self):
        """Test that get_gaps returns a copy, not the original set."""
        tracker = CoverageTracker(tools=["tool1", "tool2"])
        gaps = tracker.get_gaps(CoverageDimension.TOOL)

        gaps.add("tool3")  # Modify the returned set

        # Original should be unchanged
        original_gaps = tracker.get_gaps(CoverageDimension.TOOL)
        assert "tool3" not in original_gaps

    def test_generate_report(self):
        """Test generating coverage report."""
        tracker = CoverageTracker(
            tools=["tool1", "tool2"],
            states=["state1", "state2"],
        )
        tracker.record_tool("tool1")
        tracker.record_state("state1")

        report = tracker.generate_report(metadata={"run": "test-1"})

        assert report.session_id == tracker.session_id
        assert report.metadata["run"] == "test-1"
        assert report.has_gaps is True
        # Overall includes all 4 dimensions: Tool 50%, State 50%, Transition 100% (0 items), Invariant 100% (0 items)
        # Average: (50 + 50 + 100 + 100) / 4 = 75%
        assert report.overall_coverage == 75.0

    def test_reset(self):
        """Test resetting coverage counters."""
        tracker = CoverageTracker(tools=["tool1", "tool2"])
        tracker.record_tool("tool1")

        # Should have coverage
        assert tracker.get_coverage(CoverageDimension.TOOL).coverage_count == 1

        tracker.reset()

        # After reset, should be back to zero
        tool_dim = tracker.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.coverage_count == 0
        assert tool_dim.gap_count == 2


class TestCoverageTrackerMerge:
    """Test CoverageTracker merging functionality."""

    def test_merge_disjoint_coverage(self):
        """Test merging trackers with completely different coverage."""
        tracker1 = CoverageTracker(tools=["tool1", "tool2"])
        tracker1.record_tool("tool1")

        tracker2 = CoverageTracker(tools=["tool1", "tool2"])
        tracker2.record_tool("tool2")

        merged = tracker1.merge(tracker2)

        # Merged should have both tools covered
        tool_dim = merged.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.coverage_count == 2
        assert "tool1" in tool_dim.covered_items
        assert "tool2" in tool_dim.covered_items
        assert tool_dim.coverage_percentage == 100.0

    def test_merge_overlapping_coverage(self):
        """Test merging trackers with overlapping coverage."""
        tracker1 = CoverageTracker(tools=["tool1", "tool2", "tool3"])
        tracker1.record_tool("tool1")
        tracker1.record_tool("tool2")

        tracker2 = CoverageTracker(tools=["tool1", "tool2", "tool3"])
        tracker2.record_tool("tool2")  # Overlaps with tracker1
        tracker2.record_tool("tool3")

        merged = tracker1.merge(tracker2)

        tool_dim = merged.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.coverage_count == 3
        assert tool_dim.coverage_percentage == 100.0

    def test_merge_different_item_sets(self):
        """Test merging trackers with different item sets."""
        tracker1 = CoverageTracker(tools=["tool1", "tool2"])
        tracker1.record_tool("tool1")

        tracker2 = CoverageTracker(tools=["tool2", "tool3"])
        tracker2.record_tool("tool3")

        merged = tracker1.merge(tracker2)

        # Merged should know about all three tools
        tool_dim = merged.get_coverage(CoverageDimension.TOOL)
        assert tool_dim.total_items == 3
        assert tool_dim.coverage_count == 2
        assert "tool1" in tool_dim.covered_items
        assert "tool3" in tool_dim.covered_items
        assert "tool2" in tool_dim.uncovered_items

    def test_merge_all_dimensions(self):
        """Test merging across all dimensions."""
        tracker1 = CoverageTracker(
            tools=["tool1"],
            states=["state1"],
            transitions=[("a", "b")],
            invariants=["inv1"],
        )
        tracker1.record_tool("tool1")
        tracker1.record_state("state1")

        tracker2 = CoverageTracker(
            tools=["tool2"],
            states=["state2"],
            transitions=[("b", "c")],
            invariants=["inv2"],
        )
        tracker2.record_transition("b", "c")
        tracker2.record_invariant("inv2")

        merged = tracker1.merge(tracker2)

        # Check each dimension
        assert merged.get_coverage(CoverageDimension.TOOL).coverage_count == 1
        assert merged.get_coverage(CoverageDimension.STATE).coverage_count == 1
        assert merged.get_coverage(CoverageDimension.TRANSITION).coverage_count == 1
        assert merged.get_coverage(CoverageDimension.INVARIANT).coverage_count == 1

    def test_merge_creates_new_session(self):
        """Test that merge creates a new session ID."""
        tracker1 = CoverageTracker(session_id="session-1")
        tracker2 = CoverageTracker(session_id="session-2")

        merged = tracker1.merge(tracker2)

        assert merged.session_id != "session-1"
        assert merged.session_id != "session-2"
        assert merged.session_id.startswith("merged-")


class TestCoverageTrackerIntegration:
    """Integration tests for CoverageTracker."""

    def test_full_agent_test_scenario(self):
        """Test a complete agent testing scenario."""
        # Set up tracker for an agent with known capabilities
        tracker = CoverageTracker(
            tools=["search", "analyze", "summarize"],
            states=["idle", "thinking", "responding"],
            transitions=[
                ("idle", "thinking"),
                ("thinking", "responding"),
                ("responding", "idle"),
            ],
            invariants=["response_not_empty", "tool_called_correctly"],
        )

        # Simulate a test run
        tracker.record_state("idle")
        tracker.record_transition("idle", "thinking")
        tracker.record_tool("search")
        tracker.record_invariant("tool_called_correctly")
        tracker.record_state("thinking")
        tracker.record_transition("thinking", "responding")
        tracker.record_state("responding")
        tracker.record_invariant("response_not_empty")
        tracker.record_transition("responding", "idle")

        # Generate report
        report = tracker.generate_report(metadata={"test": "agent_flow"})

        # Verify results
        assert report.get_dimension(CoverageDimension.TOOL).coverage_percentage == pytest.approx(33.33, rel=0.01)
        assert report.get_dimension(CoverageDimension.STATE).coverage_percentage == 100.0
        assert report.get_dimension(CoverageDimension.TRANSITION).coverage_percentage == 100.0
        assert report.get_dimension(CoverageDimension.INVARIANT).coverage_percentage == 100.0

        # Check gaps
        tool_gaps = tracker.get_gaps(CoverageDimension.TOOL)
        assert tool_gaps == {"analyze", "summarize"}
