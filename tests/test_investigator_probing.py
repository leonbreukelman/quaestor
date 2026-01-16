import pytest
from unittest.mock import AsyncMock
from quaestor.runtime.investigator import QuaestorInvestigator, ProbeType, TargetAdapter

class MockAdapter(TargetAdapter):
    def __init__(self):
        self.turn_count = 0

    async def send_probe(self, probe: str) -> str:
        self.turn_count += 1
        print(f"MockAdapter: Turn {self.turn_count}, Received probe: {probe}")  # Debugging
        if self.turn_count == 2:
            print("MockAdapter: Simulating failure on turn 2")  # Debugging
            raise Exception("Simulated failure on turn 2")
        if "STOP" in probe:
            print("MockAdapter: Returning STOP response")  # Debugging
            return "STOP"
        return f"Response to: {probe}"

@pytest.mark.asyncio
async def test_probe_history_tracking():
    adapter = MockAdapter()
    investigator = QuaestorInvestigator(adapter, max_turns=3)

    await investigator.run_session("Initial context")

    assert len(investigator.probe_history.entries) == 3
    assert all(entry.status == "success" for entry in investigator.probe_history.entries)

@pytest.mark.asyncio
async def test_probe_type_selection():
    adapter = MockAdapter()
    investigator = QuaestorInvestigator(
        adapter,
        probe_type_weights={
            ProbeType.POSITIVE.value: 0.7,
            ProbeType.ADVERSARIAL.value: 0.2,
            ProbeType.EDGE_CASE.value: 0.1,
        },
    )

    probe_types = [investigator._select_probe_type() for _ in range(100)]
    positive_count = sum(1 for pt in probe_types if pt == ProbeType.POSITIVE)
    assert positive_count > 60  # At least 60% should be positive

@pytest.mark.asyncio
async def test_adapter_failure_handling():
    adapter = MockAdapter()
    investigator = QuaestorInvestigator(adapter, max_turns=3)

    await investigator.run_session("fail on second turn")

    entries = investigator.probe_history.entries
    assert len(entries) == 3
    assert entries[1].status.startswith("failure")
    assert entries[2].status == "success"

@pytest.mark.asyncio
async def test_termination_criteria():
    def termination_criteria(response: str) -> bool:
        return "STOP" in response

    adapter = MockAdapter()
    investigator = QuaestorInvestigator(adapter, max_turns=5, termination_criteria=termination_criteria)

    await investigator.run_session("STOP after second turn")

    entries = investigator.probe_history.entries
    assert len(entries) == 2  # Session should terminate after 2 turns