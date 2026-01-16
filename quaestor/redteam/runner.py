"""
Red Team Runner - Integrates red team testing with agent adapters.

Provides the glue between DeepTeamAdapter and the runtime adapters
(HTTPAdapter, PythonImportAdapter, etc.) for executing adversarial
tests against real agents.

Part of Phase 7: Red Team Capabilities.
"""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from quaestor.redteam.adapter import DeepTeamAdapter, MockRedTeamAdapter
from quaestor.redteam.config import RedTeamConfigLoader
from quaestor.redteam.models import RedTeamConfig, RedTeamReport
from quaestor.runtime.adapters import (
    AdapterConfig,
    AgentMessage,
    AgentResponse,
    BaseAdapter,
    HTTPAdapter,
    MockAdapter,
)

# Type alias for agent callback
AgentCallback = Callable[[str], Awaitable[str]]


class RedTeamRunner:
    """
    Orchestrates red team testing against agents via adapters.

    Bridges the gap between DeepTeam's red_team() API (which expects
    an async callback) and Quaestor's adapter-based agent communication.
    """

    def __init__(
        self,
        config: RedTeamConfig | None = None,
        use_mock: bool = False,
    ):
        """
        Initialize the red team runner.

        Args:
            config: Red team configuration
            use_mock: Use mock adapter (no DeepTeam required)
        """
        self.config = config or RedTeamConfig()
        self.use_mock = use_mock

        if use_mock:
            self.adapter = MockRedTeamAdapter(config=self.config)
        else:
            self.adapter = DeepTeamAdapter(config=self.config)

    @classmethod
    def from_playbook(cls, playbook: str, use_mock: bool = False) -> "RedTeamRunner":
        """
        Create a runner from a named playbook.

        Args:
            playbook: Playbook name (quick, standard, comprehensive, owasp-llm)
            use_mock: Use mock adapter

        Returns:
            Configured RedTeamRunner
        """
        config = RedTeamConfigLoader.from_playbook(playbook)
        return cls(config=config, use_mock=use_mock)

    @classmethod
    def from_yaml(cls, path: str | Path, use_mock: bool = False) -> "RedTeamRunner":
        """
        Create a runner from a YAML config file.

        Args:
            path: Path to YAML config
            use_mock: Use mock adapter

        Returns:
            Configured RedTeamRunner
        """
        config = RedTeamConfigLoader.from_yaml(path)
        return cls(config=config, use_mock=use_mock)

    async def run_against_http(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        target_name: str | None = None,
    ) -> RedTeamReport:
        """
        Run red team testing against an HTTP endpoint.

        Args:
            url: Base URL of the agent API
            headers: Optional HTTP headers (e.g., auth tokens)
            target_name: Name for the target in reports

        Returns:
            RedTeamReport with results
        """
        adapter_config = AdapterConfig(
            timeout_seconds=self.config.timeout_seconds,
        )

        http_adapter = HTTPAdapter(
            base_url=url,
            config=adapter_config,
        )

        # Add custom headers if provided
        if headers:
            http_adapter._headers.update(headers)

        return await self._run_with_adapter(
            adapter=http_adapter,
            target_name=target_name or url,
        )

    async def run_against_callback(
        self,
        callback: AgentCallback,
        target_name: str = "callback-agent",
    ) -> RedTeamReport:
        """
        Run red team testing against a callback function.

        This is the most flexible option - provide any async function
        that takes a string input and returns a string response.

        Args:
            callback: Async function (input: str) -> str
            target_name: Name for the target in reports

        Returns:
            RedTeamReport with results
        """
        return await self.adapter.run_red_team(
            agent_callback=callback,
            target_name=target_name,
        )

    async def run_against_mock(
        self,
        responses: dict[str, str] | None = None,
        target_name: str = "mock-agent",
    ) -> RedTeamReport:
        """
        Run red team testing against a mock agent.

        Useful for testing the red team infrastructure itself.

        Args:
            responses: Optional mapping of patterns to responses
            target_name: Name for the target in reports

        Returns:
            RedTeamReport with results
        """
        mock_adapter = MockAdapter(responses=responses)

        return await self._run_with_adapter(
            adapter=mock_adapter,
            target_name=target_name,
        )

    async def _run_with_adapter(
        self,
        adapter: BaseAdapter,
        target_name: str,
    ) -> RedTeamReport:
        """
        Run red team testing using a Quaestor adapter.

        Args:
            adapter: The adapter to use for agent communication
            target_name: Name for the target in reports

        Returns:
            RedTeamReport with results
        """
        # Connect to the agent
        await adapter.connect()

        try:
            # Create a callback that uses the adapter
            async def adapter_callback(input_text: str) -> str:
                message = AgentMessage(content=input_text)
                response: AgentResponse = await adapter.send_message(message)
                return response.content

            # Run the red team assessment
            report = await self.adapter.run_red_team(
                agent_callback=adapter_callback,
                target_name=target_name,
            )

            return report

        finally:
            # Always disconnect
            await adapter.disconnect()

    def results_to_verdicts(self, report: RedTeamReport):
        """
        Convert red team results to evaluation verdicts.

        Args:
            report: RedTeamReport to convert

        Returns:
            List of Verdict objects
        """
        return self.adapter.results_to_verdicts(report)


async def quick_red_team(
    target: str,
    playbook: str = "quick",
    use_mock: bool = False,
) -> RedTeamReport:
    """
    Convenience function for quick red team testing.

    Args:
        target: URL or path to agent
        playbook: Playbook name
        use_mock: Use mock mode

    Returns:
        RedTeamReport with results
    """
    runner = RedTeamRunner.from_playbook(playbook, use_mock=use_mock)

    if target.startswith("http://") or target.startswith("https://"):
        return await runner.run_against_http(url=target)
    else:
        # For now, use mock for file paths
        return await runner.run_against_mock(target_name=target)
