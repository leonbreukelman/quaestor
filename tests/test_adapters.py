"""
Tests for the runtime adapters module.

Part of Phase 3: Runtime Testing.
"""

import textwrap
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quaestor.runtime.adapters import (
    AdapterConfig,
    AdapterState,
    AgentMessage,
    AgentResponse,
    AuthType,
    BaseAdapter,
    HTTPAdapter,
    HTTPAdapterConfig,
    MCPAdapter,
    MCPAdapterConfig,
    MockAdapter,
    MockResponse,
    PythonImportAdapter,
    PythonImportConfig,
    TargetAdapter,
    ToolCall,
    ToolResult,
)

# =============================================================================
# Test Models
# =============================================================================


class TestToolCall:
    """Tests for ToolCall model."""

    def test_creates_with_defaults(self):
        """Test creating a tool call with default values."""
        tc = ToolCall(tool_name="search", arguments={"query": "test"})

        assert tc.tool_name == "search"
        assert tc.arguments == {"query": "test"}
        assert tc.call_id  # Auto-generated
        assert tc.timestamp  # Auto-generated

    def test_creates_with_custom_id(self):
        """Test creating with custom call ID."""
        tc = ToolCall(
            tool_name="search",
            arguments={},
            call_id="custom-123",
        )

        assert tc.call_id == "custom-123"


class TestToolResult:
    """Tests for ToolResult model."""

    def test_creates_success_result(self):
        """Test creating a successful tool result."""
        result = ToolResult(call_id="abc", result={"data": "value"})

        assert result.call_id == "abc"
        assert result.result == {"data": "value"}
        assert result.error is None

    def test_creates_error_result(self):
        """Test creating an error tool result."""
        result = ToolResult(
            call_id="abc",
            result=None,
            error="Tool not found",
        )

        assert result.error == "Tool not found"


class TestAgentMessage:
    """Tests for AgentMessage model."""

    def test_creates_with_defaults(self):
        """Test creating a message with defaults."""
        msg = AgentMessage(content="Hello")

        assert msg.content == "Hello"
        assert msg.role == "user"
        assert msg.metadata == {}

    def test_creates_with_custom_role(self):
        """Test creating with custom role."""
        msg = AgentMessage(content="Hi", role="system")

        assert msg.role == "system"


class TestAgentResponse:
    """Tests for AgentResponse model."""

    def test_creates_with_defaults(self):
        """Test creating a response with defaults."""
        resp = AgentResponse(content="Hello there")

        assert resp.content == "Hello there"
        assert resp.role == "assistant"
        assert resp.tool_calls == []
        assert resp.has_tool_calls is False

    def test_has_tool_calls_property(self):
        """Test the has_tool_calls property."""
        tc = ToolCall(tool_name="test", arguments={})
        resp = AgentResponse(content="Let me search", tool_calls=[tc])

        assert resp.has_tool_calls is True
        assert len(resp.tool_calls) == 1


# =============================================================================
# Test TargetAdapter Protocol
# =============================================================================


class TestTargetAdapterProtocol:
    """Tests for the TargetAdapter protocol."""

    def test_mock_adapter_is_target_adapter(self):
        """Test that MockAdapter satisfies the protocol."""
        adapter = MockAdapter()

        # Check that it's a valid TargetAdapter
        assert isinstance(adapter, TargetAdapter)


# =============================================================================
# Test MockAdapter
# =============================================================================


class TestMockAdapter:
    """Tests for the MockAdapter."""

    @pytest.fixture
    def adapter(self) -> MockAdapter:
        """Create a mock adapter."""
        return MockAdapter()

    @pytest.fixture
    def adapter_with_responses(self) -> MockAdapter:
        """Create a mock adapter with pre-configured responses."""
        responses = [
            MockResponse(content="First response"),
            MockResponse(content="Second response"),
            MockResponse(
                content="With tools",
                tool_calls=[ToolCall(tool_name="search", arguments={})],
            ),
        ]
        return MockAdapter(responses=responses)

    @pytest.mark.asyncio
    async def test_initial_state_is_disconnected(self, adapter: MockAdapter):
        """Test adapter starts disconnected."""
        assert adapter.state == AdapterState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_sets_connected_state(self, adapter: MockAdapter):
        """Test connecting sets state to connected."""
        await adapter.connect()

        assert adapter.state == AdapterState.CONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_sets_disconnected_state(self, adapter: MockAdapter):
        """Test disconnecting sets state to disconnected."""
        await adapter.connect()
        await adapter.disconnect()

        assert adapter.state == AdapterState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_message_requires_connection(self, adapter: MockAdapter):
        """Test that sending a message without connection raises."""
        msg = AgentMessage(content="Hello")

        with pytest.raises(ConnectionError, match="Not connected"):
            await adapter.send_message(msg)

    @pytest.mark.asyncio
    async def test_send_message_returns_default_response(
        self,
        adapter: MockAdapter,
    ):
        """Test sending a message gets a default response."""
        await adapter.connect()
        msg = AgentMessage(content="Hello")

        response = await adapter.send_message(msg)

        assert response.content == "Mock response #1"

    @pytest.mark.asyncio
    async def test_send_message_returns_configured_responses(
        self,
        adapter_with_responses: MockAdapter,
    ):
        """Test responses are returned in order."""
        await adapter_with_responses.connect()

        r1 = await adapter_with_responses.send_message(AgentMessage(content="1"))
        r2 = await adapter_with_responses.send_message(AgentMessage(content="2"))
        r3 = await adapter_with_responses.send_message(AgentMessage(content="3"))

        assert r1.content == "First response"
        assert r2.content == "Second response"
        assert r3.content == "With tools"
        assert r3.has_tool_calls

    @pytest.mark.asyncio
    async def test_conversation_history_is_tracked(self, adapter: MockAdapter):
        """Test that conversation history is maintained."""
        await adapter.connect()

        await adapter.send_message(AgentMessage(content="Hi"))
        await adapter.send_message(AgentMessage(content="How are you?"))

        history = adapter.conversation_history
        assert len(history) == 4  # 2 messages + 2 responses

    @pytest.mark.asyncio
    async def test_clear_history(self, adapter: MockAdapter):
        """Test clearing conversation history."""
        await adapter.connect()
        await adapter.send_message(AgentMessage(content="Hi"))

        adapter.clear_history()

        assert len(adapter.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_add_response_dynamically(self, adapter: MockAdapter):
        """Test adding responses after creation."""
        adapter.add_response(MockResponse(content="Dynamic response"))
        await adapter.connect()

        response = await adapter.send_message(AgentMessage(content="Test"))

        assert response.content == "Dynamic response"

    @pytest.mark.asyncio
    async def test_reset_adapter(self, adapter: MockAdapter):
        """Test resetting adapter state."""
        adapter.add_response(MockResponse(content="Test"))
        await adapter.connect()
        await adapter.send_message(AgentMessage(content="Hi"))

        adapter.reset()

        assert adapter._message_count == 0
        assert len(adapter.conversation_history) == 0

    @pytest.mark.asyncio
    async def test_get_available_tools(self):
        """Test getting tool definitions."""
        tools = [{"name": "search", "description": "Search tool"}]
        adapter_with_tools = MockAdapter(tools=tools)

        result = await adapter_with_tools.get_available_tools()

        assert result == tools


# =============================================================================
# Test HTTPAdapter
# =============================================================================


class TestHTTPAdapter:
    """Tests for the HTTPAdapter."""

    @pytest.fixture
    def config(self) -> HTTPAdapterConfig:
        """Create HTTP adapter config."""
        return HTTPAdapterConfig(
            endpoint="http://localhost:8000/chat",
            timeout_seconds=10.0,
        )

    @pytest.fixture
    def adapter(self, config: HTTPAdapterConfig) -> HTTPAdapter:
        """Create HTTP adapter."""
        return HTTPAdapter(config)

    def test_creates_with_config(self, adapter: HTTPAdapter):
        """Test creating adapter with config."""
        assert adapter.http_config.endpoint == "http://localhost:8000/chat"
        assert adapter.state == AdapterState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_requires_endpoint(self):
        """Test that connect requires an endpoint."""
        adapter = HTTPAdapter(HTTPAdapterConfig())

        with pytest.raises(ValueError, match="Endpoint URL is required"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_creates_client(self, adapter: HTTPAdapter):
        """Test that connect creates an HTTP client."""
        await adapter.connect()

        assert adapter._client is not None
        assert adapter.state == AdapterState.CONNECTED

        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_closes_client(self, adapter: HTTPAdapter):
        """Test that disconnect closes the client."""
        await adapter.connect()
        await adapter.disconnect()

        assert adapter._client is None
        assert adapter.state == AdapterState.DISCONNECTED

    def test_auth_type_api_key(self):
        """Test API key auth configuration."""
        config = HTTPAdapterConfig(
            endpoint="http://localhost:8000/chat",
            auth_type=AuthType.API_KEY,
            api_key="secret-key",  # pragma: allowlist secret
            api_key_header="X-Custom-Key",  # pragma: allowlist secret
        )

        assert config.auth_type == AuthType.API_KEY
        assert config.api_key == "secret-key"  # pragma: allowlist secret

    def test_auth_type_bearer(self):
        """Test Bearer auth configuration."""
        config = HTTPAdapterConfig(
            endpoint="http://localhost:8000/chat",
            auth_type=AuthType.BEARER,
            bearer_token="my-token",  # pragma: allowlist secret
        )

        assert config.auth_type == AuthType.BEARER

    @pytest.mark.asyncio
    async def test_send_message_requires_connection(self, adapter: HTTPAdapter):
        """Test send_message requires connection."""
        with pytest.raises(ConnectionError, match="Not connected"):
            await adapter.send_message(AgentMessage(content="Hi"))

    @pytest.mark.asyncio
    async def test_send_message_with_mock_response(self, adapter: HTTPAdapter):
        """Test send_message with mocked HTTP response."""
        await adapter.connect()

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            adapter._client,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = await adapter.send_message(AgentMessage(content="Hi"))

            assert response.content == "Hello!"
            assert response.response_time_ms is not None

        await adapter.disconnect()


# =============================================================================
# Test PythonImportAdapter
# =============================================================================


class TestPythonImportAdapter:
    """Tests for the PythonImportAdapter."""

    @pytest.fixture
    def config(self) -> PythonImportConfig:
        """Create Python import config."""
        return PythonImportConfig(
            module_path="tests.fixtures.mock_agent",
            class_name="MockAgent",
        )

    def test_creates_with_config(self, config: PythonImportConfig):
        """Test creating adapter with config."""
        adapter = PythonImportAdapter(config)

        assert adapter.import_config.module_path == "tests.fixtures.mock_agent"
        assert adapter.state == AdapterState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_requires_module_path(self):
        """Test connect requires module path."""
        adapter = PythonImportAdapter(PythonImportConfig())

        with pytest.raises(ValueError, match="Module path is required"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_requires_class_or_function(self):
        """Test connect requires class or function name."""
        adapter = PythonImportAdapter(PythonImportConfig(module_path="os"))

        with pytest.raises(
            ConnectionError,
            match="class_name or function_name is required",
        ):
            await adapter.connect()


# =============================================================================
# Test MCPAdapter
# =============================================================================


class TestMCPAdapter:
    """Tests for the MCPAdapter."""

    @pytest.fixture
    def stdio_config(self) -> MCPAdapterConfig:
        """Create stdio MCP config."""
        return MCPAdapterConfig(
            transport="stdio",
            command="python -m mock_mcp_server",
        )

    @pytest.fixture
    def http_config(self) -> MCPAdapterConfig:
        """Create HTTP MCP config."""
        return MCPAdapterConfig(
            transport="http",
            endpoint="http://localhost:3000/mcp",
        )

    def test_creates_with_stdio_config(self, stdio_config: MCPAdapterConfig):
        """Test creating with stdio transport."""
        adapter = MCPAdapter(stdio_config)

        assert adapter.mcp_config.transport == "stdio"
        assert adapter.mcp_config.command == "python -m mock_mcp_server"

    def test_creates_with_http_config(self, http_config: MCPAdapterConfig):
        """Test creating with http transport."""
        adapter = MCPAdapter(http_config)

        assert adapter.mcp_config.transport == "http"

    @pytest.mark.asyncio
    async def test_connect_stdio_requires_command(self):
        """Test stdio connect requires command."""
        adapter = MCPAdapter(MCPAdapterConfig(transport="stdio"))

        with pytest.raises(
            ConnectionError,
            match="Command is required for stdio transport",
        ):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_http_requires_endpoint(self):
        """Test http connect requires endpoint."""
        adapter = MCPAdapter(MCPAdapterConfig(transport="http"))

        with pytest.raises(
            ConnectionError,
            match="Endpoint is required for HTTP transport",
        ):
            await adapter.connect()


# =============================================================================
# Test AdapterConfig
# =============================================================================


class TestAdapterConfig:
    """Tests for AdapterConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AdapterConfig()

        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.debug is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AdapterConfig(
            timeout_seconds=60.0,
            max_retries=5,
            debug=True,
        )

        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5
        assert config.debug is True


# =============================================================================
# Test AdapterState
# =============================================================================


class TestAdapterState:
    """Tests for AdapterState enum."""

    def test_states_are_strings(self):
        """Test that states are string values."""
        assert AdapterState.DISCONNECTED == "disconnected"
        assert AdapterState.CONNECTING == "connecting"
        assert AdapterState.CONNECTED == "connected"
        assert AdapterState.ERROR == "error"


# =============================================================================
# Test BaseAdapter
# =============================================================================


class TestBaseAdapter:
    """Tests for the BaseAdapter abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseAdapter cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore

    def test_conversation_history_returns_copy(self):
        """Test conversation_history returns a copy."""
        adapter = MockAdapter()

        history1 = adapter.conversation_history
        history2 = adapter.conversation_history

        assert history1 is not history2


# =============================================================================
# Test PythonImportAdapter
# =============================================================================


class TestPythonImportAdapterExtra:
    """Tests for PythonImportAdapter code paths not covered elsewhere."""

    @pytest.mark.asyncio
    async def test_imports_file_and_calls_function(self, tmp_path) -> None:
        module_path = tmp_path / "agent_func.py"
        module_path.write_text(
            textwrap.dedent(
                """
                def agent(message: str) -> str:
                    return f"echo: {message}"
                """
            )
        )

        adapter = PythonImportAdapter(
            PythonImportConfig(
                module_path=str(module_path),
                function_name="agent",
                capture_output=False,
            )
        )

        await adapter.connect()
        resp = await adapter.send_message(AgentMessage(content="hi"))
        await adapter.disconnect()

        assert resp.content == "echo: hi"

    @pytest.mark.asyncio
    async def test_parses_tool_calls_from_dict_result(self, tmp_path) -> None:
        module_path = tmp_path / "agent_toolcalls.py"
        module_path.write_text(
            textwrap.dedent(
                """
                def agent(message: str):
                    return {
                        "content": "ok",
                        "tool_calls": [
                            {"name": "search", "arguments": {"query": message}},
                        ],
                    }
                """
            )
        )

        adapter = PythonImportAdapter(
            PythonImportConfig(
                module_path=str(module_path),
                function_name="agent",
                capture_output=False,
            )
        )

        await adapter.connect()
        resp = await adapter.send_message(AgentMessage(content="hello"))
        await adapter.disconnect()

        assert resp.content == "ok"
        assert resp.tool_calls
        assert resp.tool_calls[0].tool_name == "search"
        assert resp.tool_calls[0].arguments == {"query": "hello"}

    @pytest.mark.asyncio
    async def test_captures_stdout_in_response(self, tmp_path) -> None:
        module_path = tmp_path / "agent_stdout.py"
        module_path.write_text(
            textwrap.dedent(
                """
                def agent(message: str) -> str:
                    print("printed:", message)
                    return "done"
                """
            )
        )

        adapter = PythonImportAdapter(
            PythonImportConfig(
                module_path=str(module_path),
                function_name="agent",
                capture_output=True,
            )
        )

        await adapter.connect()
        resp = await adapter.send_message(AgentMessage(content="xyz"))
        await adapter.disconnect()

        assert "done" in resp.content
        assert "[stdout]:" in resp.content
        assert "printed: xyz" in resp.content

    @pytest.mark.asyncio
    async def test_imports_class_and_calls_callable(self, tmp_path) -> None:
        module_path = tmp_path / "agent_class.py"
        module_path.write_text(
            textwrap.dedent(
                """
                class Agent:
                    def __init__(self, prefix: str = ""):
                        self.prefix = prefix

                    def __call__(self, message: str) -> str:
                        return f"{self.prefix}{message}"
                """
            )
        )

        adapter = PythonImportAdapter(
            PythonImportConfig(
                module_path=str(module_path),
                class_name="Agent",
                init_kwargs={"prefix": "p:"},
                capture_output=False,
            )
        )

        await adapter.connect()
        resp = await adapter.send_message(AgentMessage(content="abc"))
        await adapter.disconnect()

        assert resp.content == "p:abc"
