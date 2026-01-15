"""
Quaestor Runtime Testing Infrastructure.

Provides adapters and investigators for testing AI agents at runtime.
"""

from quaestor.runtime.adapters import (
    AdapterConfig,
    AdapterState,
    AgentMessage,
    AgentResponse,
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
from quaestor.runtime.investigator import (
    InvestigatorConfig,
    Observation,
    ObservationType,
    ProbeResult,
    ProbeStrategy,
    QuaestorInvestigator,
)

__all__ = [
    # Adapters
    "AdapterConfig",
    "AdapterState",
    "AgentMessage",
    "AgentResponse",
    "HTTPAdapter",
    "HTTPAdapterConfig",
    "MCPAdapter",
    "MCPAdapterConfig",
    "MockAdapter",
    "MockResponse",
    "PythonImportAdapter",
    "PythonImportConfig",
    "TargetAdapter",
    "ToolCall",
    "ToolResult",
    # Investigator
    "InvestigatorConfig",
    "Observation",
    "ObservationType",
    "ProbeResult",
    "ProbeStrategy",
    "QuaestorInvestigator",
]
