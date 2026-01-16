"""
Pydantic models for Red Team capabilities.

Defines data structures for attacks, vulnerabilities, and risk assessments.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class VulnerabilityType(str, Enum):
    """Types of vulnerabilities to test for."""

    # Bias vulnerabilities
    BIAS_GENDER = "bias_gender"
    BIAS_RACE = "bias_race"
    BIAS_POLITICAL = "bias_political"
    BIAS_RELIGION = "bias_religion"

    # PII Leakage
    PII_DIRECT = "pii_direct"
    PII_SESSION = "pii_session"
    PII_DATABASE = "pii_database"

    # Toxicity
    TOXICITY_PROFANITY = "toxicity_profanity"
    TOXICITY_INSULTS = "toxicity_insults"
    TOXICITY_THREATS = "toxicity_threats"

    # Misinformation
    MISINFORMATION_FACTUAL = "misinformation_factual"
    MISINFORMATION_UNSUPPORTED = "misinformation_unsupported"

    # Robustness
    ROBUSTNESS_OVERRELIANCE = "robustness_overreliance"
    ROBUSTNESS_HIJACKING = "robustness_hijacking"

    # Custom
    CUSTOM = "custom"


class AttackMethod(str, Enum):
    """Attack methods for adversarial testing."""

    # Single-turn attacks
    PROMPT_INJECTION = "prompt_injection"
    LEETSPEAK = "leetspeak"
    ROT13 = "rot13"
    MATH_PROBLEM = "math_problem"
    BASE64 = "base64"

    # Multi-turn attacks
    LINEAR_JAILBREAK = "linear_jailbreak"
    TREE_JAILBREAK = "tree_jailbreak"
    CRESCENDO_JAILBREAK = "crescendo_jailbreak"

    # Custom
    CUSTOM = "custom"


class AttackSeverity(str, Enum):
    """Severity of a successful attack."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackResult(BaseModel):
    """Result of a single adversarial attack."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    attack_method: AttackMethod
    vulnerability_type: VulnerabilityType
    input_prompt: str
    agent_response: str
    success: bool = Field(description="Whether the attack succeeded")
    severity: AttackSeverity = AttackSeverity.MEDIUM
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: str | None = None
    remediation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(use_enum_values=True)


class RedTeamConfig(BaseModel):
    """Configuration for red team testing."""

    # Vulnerability selection
    vulnerabilities: list[VulnerabilityType] = Field(
        default_factory=lambda: [
            VulnerabilityType.BIAS_GENDER,
            VulnerabilityType.BIAS_RACE,
            VulnerabilityType.PII_DIRECT,
            VulnerabilityType.TOXICITY_PROFANITY,
        ]
    )

    # Attack methods
    attacks: list[AttackMethod] = Field(
        default_factory=lambda: [
            AttackMethod.PROMPT_INJECTION,
            AttackMethod.LEETSPEAK,
        ]
    )

    # Execution settings
    attacks_per_vulnerability: int = Field(default=3, ge=1, le=100)
    max_concurrent: int = Field(default=5, ge=1, le=50)
    timeout_seconds: float = Field(default=30.0, ge=1.0)

    # Target configuration
    target_purpose: str = Field(
        default="A helpful AI assistant",
        description="Description of the target agent's purpose",
    )

    # Output settings
    output_dir: str | None = None
    save_results: bool = True

    model_config = ConfigDict(use_enum_values=True)


class RedTeamReport(BaseModel):
    """Complete red team assessment report."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    target_name: str
    target_purpose: str
    config: RedTeamConfig

    # Results
    results: list[AttackResult] = Field(default_factory=list)

    # Summary statistics
    total_attacks: int = 0
    successful_attacks: int = 0
    vulnerabilities_found: list[VulnerabilityType] = Field(default_factory=list)

    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Percentage of successful attacks (vulnerability rate)."""
        if self.total_attacks == 0:
            return 0.0
        return (self.successful_attacks / self.total_attacks) * 100.0

    @property
    def is_vulnerable(self) -> bool:
        """Whether any vulnerabilities were found."""
        return self.successful_attacks > 0

    def add_result(self, result: AttackResult) -> None:
        """Add an attack result to the report."""
        self.results.append(result)
        self.total_attacks += 1
        if result.success:
            self.successful_attacks += 1
            if result.vulnerability_type not in self.vulnerabilities_found:
                self.vulnerabilities_found.append(result.vulnerability_type)

    def complete(self) -> None:
        """Mark the report as complete."""
        self.completed_at = datetime.now(UTC)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_seconds = delta.total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "target_name": self.target_name,
            "target_purpose": self.target_purpose,
            "total_attacks": self.total_attacks,
            "successful_attacks": self.successful_attacks,
            "success_rate": round(self.success_rate, 2),
            "is_vulnerable": self.is_vulnerable,
            "vulnerabilities_found": list(self.vulnerabilities_found),
            "duration_seconds": self.duration_seconds,
            "results": [r.model_dump() for r in self.results],
            "metadata": self.metadata,
        }

    model_config = ConfigDict(use_enum_values=True)
