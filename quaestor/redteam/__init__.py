"""
Red Team module for adversarial testing of AI agents.

Provides DeepTeam integration for:
- Vulnerability scanning (bias, PII leakage, toxicity, etc.)
- Adversarial attacks (prompt injection, jailbreaking)
- Risk assessment and remediation suggestions

Part of Phase 7: Red Team Capabilities.
"""

from quaestor.redteam.models import (
    AttackMethod,
    AttackResult,
    RedTeamConfig,
    RedTeamReport,
    VulnerabilityType,
)
from quaestor.redteam.runner import RedTeamRunner, quick_red_team

__all__ = [
    "AttackMethod",
    "AttackResult",
    "RedTeamConfig",
    "RedTeamReport",
    "RedTeamRunner",
    "VulnerabilityType",
    "quick_red_team",
]
