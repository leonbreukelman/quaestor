import pytest

from quaestor.runtime.adapters import (
    AdapterState,
    AgentMessage,
    AgentResponse,
)
from quaestor.runtime.investigator import ProbeType, QuaestorInvestigator


class MockAdapterForProbing:
    """Mock adapter that implements the TargetAdapter protocol for testing."""

    def __init__(self, fail_on_turn: int = 2, stop_on_turn: int | None = None) -> None:
        self.turn_count = 0
        self._state = AdapterState.DISCONNECTED
        self.conversation_history: list[tuple[AgentMessage, AgentResponse]] = []
        self.fail_on_turn = fail_on_turn
        self.stop_on_turn = stop_on_turn

    @property
    def state(self) -> AdapterState:
        return self._state

    async def connect(self) -> None:
        self._state = AdapterState.CONNECTED

    async def disconnect(self) -> None:
        self._state = AdapterState.DISCONNECTED

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        self.turn_count += 1
        probe = message.content
        print(f"MockAdapter: Turn {self.turn_count}, Received probe: {probe}")  # Debugging

        if self.turn_count == self.fail_on_turn:
            print(f"MockAdapter: Simulating failure on turn {self.turn_count}")  # Debugging
            raise Exception(f"Simulated failure on turn {self.turn_count}")

        if self.stop_on_turn and self.turn_count == self.stop_on_turn:
            print("MockAdapter: Returning STOP response")  # Debugging
            response = AgentResponse(content="STOP")
        elif "STOP" in probe:
            print("MockAdapter: Returning STOP response (from probe)")  # Debugging
            response = AgentResponse(content="STOP")
        else:
            response = AgentResponse(content=f"Response to: {probe}")
        self.conversation_history.append((message, response))
        return response

    async def get_available_tools(self) -> list[dict]:
        return []


@pytest.mark.asyncio
async def test_probe_history_tracking():
    """Test that probe history is correctly recorded."""
    adapter = MockAdapterForProbing()
    investigator = QuaestorInvestigator(adapter, max_turns=3)

    await investigator.run_session("Initial context")

    entries = investigator.probe_history.entries
    assert len(entries) == 3
    # Turn 1: success, Turn 2: failure (MockAdapter raises exception), Turn 3: success
    assert entries[0].status == "success"
    assert entries[1].status.startswith("failure")  # Exception on turn 2
    assert entries[2].status == "success"


@pytest.mark.asyncio
async def test_probe_type_selection():
    adapter = MockAdapterForProbing()
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
    adapter = MockAdapterForProbing()
    investigator = QuaestorInvestigator(adapter, max_turns=3)

    await investigator.run_session("fail on second turn")

    entries = investigator.probe_history.entries
    assert len(entries) == 3
    assert entries[1].status.startswith("failure")
    assert entries[2].status == "success"


@pytest.mark.asyncio
async def test_termination_criteria():
    """Test that session terminates when criteria is met."""

    def termination_criteria(response: str) -> bool:
        return "STOP" in response

    # Use adapter that returns STOP on turn 2 and doesn't fail
    adapter = MockAdapterForProbing(fail_on_turn=0, stop_on_turn=2)
    investigator = QuaestorInvestigator(
        adapter, max_turns=5, termination_criteria=termination_criteria
    )

    await investigator.run_session("Initial context")

    entries = investigator.probe_history.entries
    assert len(entries) == 2  # Session should terminate after 2 turns (STOP on turn 2)
