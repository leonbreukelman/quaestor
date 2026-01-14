"""
Target Adapters for Runtime Testing.

Provides adapters for communicating with different types of agents:
- HTTPAdapter: For agents deployed behind HTTP endpoints
- PythonImportAdapter: For locally imported Python agents
- MCPAdapter: For MCP-compliant agents
- MockAdapter: For testing without real agents

Part of Phase 3: Runtime Testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import httpx


# =============================================================================
# Core Models
# =============================================================================


@dataclass
class ToolCall:
    """Represents a tool call made by an agent."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid4().hex[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ToolResult:
    """Represents the result of a tool call."""

    call_id: str
    result: Any
    error: str | None = None
    duration_ms: int | None = None


@dataclass
class AgentMessage:
    """A message to send to an agent."""

    content: str
    role: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Response from an agent."""

    content: str
    role: str = "assistant"
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: Any = None

    # Timing
    response_time_ms: int | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response includes tool calls."""
        return len(self.tool_calls) > 0


class AdapterState(str, Enum):
    """State of an adapter connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


# =============================================================================
# Base Adapter Protocol
# =============================================================================


@dataclass
class AdapterConfig:
    """Base configuration for adapters."""

    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    debug: bool = False


@runtime_checkable
class TargetAdapter(Protocol):
    """
    Protocol defining the interface for target adapters.

    All adapters must implement this interface to be used with
    QuaestorInvestigator for runtime testing.
    """

    @property
    def state(self) -> AdapterState:
        """Get the current adapter state."""
        ...

    @property
    def conversation_history(self) -> list[AgentMessage | AgentResponse]:
        """Get the conversation history."""
        ...

    async def connect(self) -> None:
        """
        Establish connection to the target agent.

        Raises:
            ConnectionError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """Close connection to the target agent."""
        ...

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """
        Send a message to the agent and get response.

        Args:
            message: The message to send

        Returns:
            AgentResponse from the agent

        Raises:
            ConnectionError: If not connected
            TimeoutError: If request times out
        """
        ...

    async def send_tool_result(self, result: ToolResult) -> AgentResponse | None:
        """
        Send a tool result back to the agent.

        Args:
            result: The tool execution result

        Returns:
            AgentResponse if agent continues, None if done
        """
        ...

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """
        Get list of tools available from the agent.

        Returns:
            List of tool definitions
        """
        ...


# =============================================================================
# Abstract Base Adapter
# =============================================================================


class BaseAdapter(ABC):
    """Abstract base class for adapters with common functionality."""

    def __init__(self, config: AdapterConfig | None = None):
        """Initialize the adapter."""
        self.config = config or AdapterConfig()
        self._state = AdapterState.DISCONNECTED
        self._conversation_history: list[AgentMessage | AgentResponse] = []

    @property
    def state(self) -> AdapterState:
        """Get the current adapter state."""
        return self._state

    @property
    def conversation_history(self) -> list[AgentMessage | AgentResponse]:
        """Get the conversation history."""
        return self._conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()

    def _record_message(self, message: AgentMessage) -> None:
        """Record a sent message in history."""
        self._conversation_history.append(message)

    def _record_response(self, response: AgentResponse) -> None:
        """Record a received response in history."""
        self._conversation_history.append(response)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the target agent."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the target agent."""
        ...

    @abstractmethod
    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message to the agent and get response."""
        ...

    async def send_tool_result(self, result: ToolResult) -> AgentResponse | None:
        """Send a tool result back to the agent."""
        # Default implementation - subclasses can override
        return None

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get list of tools available from the agent."""
        # Default implementation - subclasses can override
        return []


# =============================================================================
# Mock Adapter (for testing)
# =============================================================================


@dataclass
class MockResponse:
    """A mock response configuration."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class MockAdapter(BaseAdapter):
    """
    Mock adapter for testing without real agents.

    Provides configurable responses for testing the investigator
    and other runtime components.
    """

    def __init__(
        self,
        config: AdapterConfig | None = None,
        responses: list[MockResponse] | None = None,
        tools: list[dict[str, Any]] | None = None,
    ):
        """
        Initialize the mock adapter.

        Args:
            config: Adapter configuration
            responses: List of mock responses to return in order
            tools: List of tool definitions to report
        """
        super().__init__(config)
        self._responses = responses or []
        self._response_index = 0
        self._tools = tools or []
        self._message_count = 0

    async def connect(self) -> None:
        """Establish mock connection."""
        self._state = AdapterState.CONNECTED

    async def disconnect(self) -> None:
        """Close mock connection."""
        self._state = AdapterState.DISCONNECTED

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message and get a mock response."""
        if self._state != AdapterState.CONNECTED:
            raise ConnectionError("Not connected")

        self._record_message(message)
        self._message_count += 1

        # Get the next mock response
        if self._response_index < len(self._responses):
            mock = self._responses[self._response_index]
            self._response_index += 1
            response = AgentResponse(
                content=mock.content,
                tool_calls=mock.tool_calls,
            )
        else:
            # Default response when no more mocks
            response = AgentResponse(
                content=f"Mock response #{self._message_count}",
            )

        self._record_response(response)
        return response

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get mock tool definitions."""
        return self._tools.copy()

    def add_response(self, response: MockResponse) -> None:
        """Add a mock response to the queue."""
        self._responses.append(response)

    def reset(self) -> None:
        """Reset the mock adapter state."""
        self._response_index = 0
        self._message_count = 0
        self.clear_history()


# =============================================================================
# HTTP Adapter
# =============================================================================


class AuthType(str, Enum):
    """Authentication types for HTTP adapter."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    CUSTOM_HEADER = "custom_header"


@dataclass
class HTTPAdapterConfig(AdapterConfig):
    """Configuration for HTTP adapter."""

    endpoint: str = ""
    auth_type: AuthType = AuthType.NONE
    api_key: str | None = None
    api_key_header: str = "X-API-Key"
    bearer_token: str | None = None
    basic_username: str | None = None
    basic_password: str | None = None
    custom_headers: dict[str, str] = field(default_factory=dict)

    # Request format
    message_field: str = "message"
    response_field: str = "response"

    # Streaming
    stream: bool = False


class HTTPAdapter(BaseAdapter):
    """
    Adapter for testing agents behind HTTP endpoints.

    Supports various authentication schemes and both
    synchronous and streaming responses.
    """

    def __init__(self, config: HTTPAdapterConfig):
        """Initialize the HTTP adapter."""
        super().__init__(config)
        self.http_config = config
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        """Establish HTTP connection."""
        if not self.http_config.endpoint:
            raise ValueError("Endpoint URL is required")

        self._state = AdapterState.CONNECTING

        try:
            # Build headers
            headers = dict(self.http_config.custom_headers)
            headers["Content-Type"] = "application/json"

            # Add authentication
            if self.http_config.auth_type == AuthType.API_KEY:
                if self.http_config.api_key:
                    headers[self.http_config.api_key_header] = self.http_config.api_key
            elif self.http_config.auth_type == AuthType.BEARER:
                if self.http_config.bearer_token:
                    headers["Authorization"] = f"Bearer {self.http_config.bearer_token}"

            # Create client
            self._client = httpx.AsyncClient(
                timeout=self.http_config.timeout_seconds,
                headers=headers,
            )

            # Test connection with a simple request (if endpoint supports it)
            # For now, just mark as connected
            self._state = AdapterState.CONNECTED

        except Exception as e:
            self._state = AdapterState.ERROR
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def disconnect(self) -> None:
        """Close HTTP connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._state = AdapterState.DISCONNECTED

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message via HTTP."""
        if self._state != AdapterState.CONNECTED or self._client is None:
            raise ConnectionError("Not connected")

        self._record_message(message)

        start_time = datetime.now(UTC)

        # Build request payload
        payload = {
            self.http_config.message_field: message.content,
            **message.metadata,
        }

        # Send request with retries
        last_error: Exception | None = None
        for attempt in range(self.http_config.max_retries):
            try:
                response = await self._client.post(
                    self.http_config.endpoint,
                    json=payload,
                )
                response.raise_for_status()

                # Parse response
                data = response.json()
                end_time = datetime.now(UTC)
                duration_ms = int((end_time - start_time).total_seconds() * 1000)

                # Extract content
                content = data.get(self.http_config.response_field, str(data))

                # Extract tool calls if present
                tool_calls = []
                if "tool_calls" in data:
                    for tc in data["tool_calls"]:
                        tool_calls.append(
                            ToolCall(
                                tool_name=tc.get("name", ""),
                                arguments=tc.get("arguments", {}),
                            )
                        )

                agent_response = AgentResponse(
                    content=content,
                    tool_calls=tool_calls,
                    raw_response=data,
                    response_time_ms=duration_ms,
                )

                self._record_response(agent_response)
                return agent_response

            except httpx.TimeoutException as e:
                last_error = TimeoutError(f"Request timed out: {e}")
            except httpx.HTTPStatusError as e:
                last_error = ConnectionError(f"HTTP error {e.response.status_code}: {e}")
            except Exception as e:
                last_error = e

            # Wait before retry (except on last attempt)
            if attempt < self.http_config.max_retries - 1:
                import asyncio

                await asyncio.sleep(self.http_config.retry_delay_seconds)

        raise last_error or ConnectionError("Unknown error")

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get tools from the endpoint if supported."""
        # Many HTTP APIs don't support tool discovery
        # Subclasses can implement specific discovery protocols
        return []


# =============================================================================
# Python Import Adapter
# =============================================================================


@dataclass
class PythonImportConfig(AdapterConfig):
    """Configuration for Python import adapter."""

    module_path: str = ""
    class_name: str | None = None
    function_name: str | None = None
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    capture_output: bool = True


class PythonImportAdapter(BaseAdapter):
    """
    Adapter for testing locally imported Python agents.

    Enables testing of agent code directly without network overhead.
    Supports various agent patterns including classes with __call__,
    async functions, and standard method invocation.
    """

    def __init__(self, config: PythonImportConfig):
        """Initialize the Python import adapter."""
        super().__init__(config)
        self.import_config = config
        self._agent: Any = None
        self._invoke_method: str | None = None

    async def connect(self) -> None:
        """Import and instantiate the agent."""
        if not self.import_config.module_path:
            raise ValueError("Module path is required")

        self._state = AdapterState.CONNECTING

        try:
            import importlib.util
            import sys
            from pathlib import Path

            # Handle both module names and file paths
            module_path = self.import_config.module_path

            if module_path.endswith(".py") or "/" in module_path:
                # File path
                path = Path(module_path)
                if not path.exists():
                    raise FileNotFoundError(f"Module not found: {module_path}")

                spec = importlib.util.spec_from_file_location(
                    path.stem,
                    path,
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load module: {module_path}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[path.stem] = module
                spec.loader.exec_module(module)
            else:
                # Module name
                module = importlib.import_module(module_path)

            # Get the agent
            if self.import_config.class_name:
                cls = getattr(module, self.import_config.class_name)
                self._agent = cls(**self.import_config.init_kwargs)
                # Determine invoke method
                if hasattr(self._agent, "__call__"):
                    self._invoke_method = "__call__"
                elif hasattr(self._agent, "invoke"):
                    self._invoke_method = "invoke"
                elif hasattr(self._agent, "run"):
                    self._invoke_method = "run"
                else:
                    raise ValueError(
                        f"Agent class {self.import_config.class_name} has no "
                        "callable method (__call__, invoke, or run)"
                    )
            elif self.import_config.function_name:
                self._agent = getattr(module, self.import_config.function_name)
                self._invoke_method = None  # Direct call
            else:
                raise ValueError("Either class_name or function_name is required")

            self._state = AdapterState.CONNECTED

        except Exception as e:
            self._state = AdapterState.ERROR
            raise ConnectionError(f"Failed to import agent: {e}") from e

    async def disconnect(self) -> None:
        """Clean up agent instance."""
        self._agent = None
        self._invoke_method = None
        self._state = AdapterState.DISCONNECTED

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message to the Python agent."""
        if self._state != AdapterState.CONNECTED or self._agent is None:
            raise ConnectionError("Not connected")

        self._record_message(message)

        import asyncio
        import io
        import sys

        start_time = datetime.now(UTC)

        # Capture stdout if requested
        captured_output = ""
        if self.import_config.capture_output:
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        try:
            # Invoke the agent
            if self._invoke_method:
                method = getattr(self._agent, self._invoke_method)
                if asyncio.iscoroutinefunction(method):
                    result = await method(message.content)
                else:
                    result = method(message.content)
            else:
                # Direct function call
                if asyncio.iscoroutinefunction(self._agent):
                    result = await self._agent(message.content)
                else:
                    result = self._agent(message.content)

            end_time = datetime.now(UTC)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Capture output
            if self.import_config.capture_output:
                captured_output = sys.stdout.getvalue()  # type: ignore
                sys.stdout = old_stdout

            # Parse result
            content: str
            tool_calls: list[ToolCall] = []

            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                raw_content = result.get("content", result.get("output", str(result)))
                content = str(raw_content) if raw_content is not None else str(result)
                if "tool_calls" in result:
                    for tc in result["tool_calls"]:
                        tool_calls.append(
                            ToolCall(
                                tool_name=tc.get("name", ""),
                                arguments=tc.get("arguments", {}),
                            )
                        )
            else:
                content = str(result)

            # Include captured output if any
            if captured_output:
                content = f"{content}\n\n[stdout]: {captured_output}"

            response = AgentResponse(
                content=content,
                tool_calls=tool_calls,
                raw_response=result,
                response_time_ms=duration_ms,
            )

            self._record_response(response)
            return response

        except Exception as e:
            if self.import_config.capture_output:
                sys.stdout = old_stdout
            raise RuntimeError(f"Agent execution failed: {e}") from e


# =============================================================================
# MCP Adapter
# =============================================================================


@dataclass
class MCPAdapterConfig(AdapterConfig):
    """Configuration for MCP adapter."""

    # Transport configuration
    transport: str = "stdio"  # "stdio" or "http"
    command: str = ""  # For stdio transport
    endpoint: str = ""  # For HTTP transport

    # Connection settings
    server_name: str = "agent"


class MCPAdapter(BaseAdapter):
    """
    Adapter for testing MCP (Model Context Protocol) compliant agents.

    Supports both stdio and HTTP transports as defined in the MCP spec.
    Reference: https://modelcontextprotocol.io/
    """

    def __init__(self, config: MCPAdapterConfig):
        """Initialize the MCP adapter."""
        super().__init__(config)
        self.mcp_config = config
        self._tools: list[dict[str, Any]] = []
        self._process: Any = None  # subprocess for stdio

    async def connect(self) -> None:
        """Establish MCP connection."""
        self._state = AdapterState.CONNECTING

        try:
            if self.mcp_config.transport == "stdio":
                await self._connect_stdio()
            elif self.mcp_config.transport == "http":
                await self._connect_http()
            else:
                raise ValueError(f"Unknown transport: {self.mcp_config.transport}")

            # Discover available tools
            await self._discover_tools()

            self._state = AdapterState.CONNECTED

        except Exception as e:
            self._state = AdapterState.ERROR
            raise ConnectionError(f"Failed to connect: {e}") from e

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        import asyncio
        import shlex

        if not self.mcp_config.command:
            raise ValueError("Command is required for stdio transport")

        args = shlex.split(self.mcp_config.command)
        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _connect_http(self) -> None:
        """Connect via HTTP transport."""
        if not self.mcp_config.endpoint:
            raise ValueError("Endpoint is required for HTTP transport")
        # HTTP connection is stateless, just validate endpoint
        # Actual connection happens on first request

    async def _discover_tools(self) -> None:
        """Discover available tools from MCP server."""
        # Send tools/list request
        try:
            response = await self._send_mcp_request("tools/list", {})
            if "tools" in response:
                self._tools = response["tools"]
        except Exception:
            # Tool discovery is optional
            self._tools = []

    async def _send_mcp_request(
        self,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send an MCP JSON-RPC request."""
        import json

        request_id = uuid4().hex[:8]
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        if self.mcp_config.transport == "stdio" and self._process:
            # Send via stdin
            request_bytes = json.dumps(request).encode() + b"\n"
            assert self._process.stdin is not None
            self._process.stdin.write(request_bytes)
            await self._process.stdin.drain()

            # Read response from stdout
            assert self._process.stdout is not None
            response_line = await self._process.stdout.readline()
            response_data = json.loads(response_line.decode())
            result: dict[str, Any] = response_data.get("result", {})
            return result

        elif self.mcp_config.transport == "http":
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.mcp_config.endpoint,
                    json=request,
                    timeout=self.mcp_config.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                http_result: dict[str, Any] = data.get("result", {})
                return http_result

        return {}

    async def disconnect(self) -> None:
        """Close MCP connection."""
        if self._process:
            self._process.terminate()
            await self._process.wait()
            self._process = None
        self._tools = []
        self._state = AdapterState.DISCONNECTED

    async def send_message(self, message: AgentMessage) -> AgentResponse:
        """Send a message via MCP."""
        if self._state != AdapterState.CONNECTED:
            raise ConnectionError("Not connected")

        self._record_message(message)

        start_time = datetime.now(UTC)

        # MCP uses completion/complete for messages
        response_data = await self._send_mcp_request(
            "completion/complete",
            {
                "messages": [
                    {"role": message.role, "content": message.content}
                ],
            },
        )

        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Parse response
        content = response_data.get("content", "")
        tool_calls = []

        if "toolCalls" in response_data:
            for tc in response_data["toolCalls"]:
                tool_calls.append(
                    ToolCall(
                        tool_name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}),
                    )
                )

        response = AgentResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response_data,
            response_time_ms=duration_ms,
        )

        self._record_response(response)
        return response

    async def send_tool_result(self, result: ToolResult) -> AgentResponse | None:
        """Send tool result back to MCP server."""
        if self._state != AdapterState.CONNECTED:
            raise ConnectionError("Not connected")

        response_data = await self._send_mcp_request(
            "tools/result",
            {
                "call_id": result.call_id,
                "result": result.result,
                "error": result.error,
            },
        )

        if "content" in response_data:
            return AgentResponse(
                content=response_data["content"],
                raw_response=response_data,
            )

        return None

    async def get_available_tools(self) -> list[dict[str, Any]]:
        """Get discovered MCP tools."""
        return self._tools.copy()
