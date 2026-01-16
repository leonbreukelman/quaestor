"""
Configuration loading and validation for Red Team testing.

Supports YAML-based attack playbook definitions.
"""

from pathlib import Path
from typing import Any

import yaml

from quaestor.redteam.models import AttackMethod, RedTeamConfig, VulnerabilityType


class RedTeamConfigLoader:
    """Load and validate red team configurations from YAML files."""

    # Default attack playbooks
    PLAYBOOKS: dict[str, dict[str, Any]] = {
        "quick": {
            "description": "Quick scan with minimal attacks",
            "vulnerabilities": ["bias_gender", "pii_direct"],
            "attacks": ["prompt_injection"],
            "attacks_per_vulnerability": 2,
        },
        "standard": {
            "description": "Standard security assessment",
            "vulnerabilities": [
                "bias_gender",
                "bias_race",
                "pii_direct",
                "pii_session",
                "toxicity_profanity",
            ],
            "attacks": ["prompt_injection", "leetspeak"],
            "attacks_per_vulnerability": 3,
        },
        "comprehensive": {
            "description": "Full security assessment with all attack types",
            "vulnerabilities": [
                v.value for v in VulnerabilityType if v != VulnerabilityType.CUSTOM
            ],
            "attacks": [a.value for a in AttackMethod if a != AttackMethod.CUSTOM],
            "attacks_per_vulnerability": 5,
        },
        "owasp-llm": {
            "description": "OWASP LLM Top 10 focused assessment",
            "vulnerabilities": [
                "pii_direct",
                "pii_database",
                "misinformation_factual",
                "robustness_hijacking",
            ],
            "attacks": ["prompt_injection", "linear_jailbreak"],
            "attacks_per_vulnerability": 5,
        },
    }

    @classmethod
    def from_playbook(cls, name: str) -> RedTeamConfig:
        """
        Load a predefined playbook by name.

        Args:
            name: Playbook name (quick, standard, comprehensive, owasp-llm)

        Returns:
            RedTeamConfig configured for the playbook
        """
        if name not in cls.PLAYBOOKS:
            available = ", ".join(cls.PLAYBOOKS.keys())
            msg = f"Unknown playbook: {name}. Available: {available}"
            raise ValueError(msg)

        playbook = cls.PLAYBOOKS[name]
        return RedTeamConfig(
            vulnerabilities=[VulnerabilityType(v) for v in playbook["vulnerabilities"]],
            attacks=[AttackMethod(a) for a in playbook["attacks"]],
            attacks_per_vulnerability=playbook.get("attacks_per_vulnerability", 3),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RedTeamConfig:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RedTeamConfig loaded from file
        """
        path = Path(path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            data = yaml.safe_load(f)

        return cls._parse_config(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RedTeamConfig:
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            RedTeamConfig from dictionary
        """
        return cls._parse_config(data)

    @classmethod
    def _parse_config(cls, data: dict[str, Any]) -> RedTeamConfig:
        """Parse configuration dictionary into RedTeamConfig."""
        # Parse vulnerabilities
        vulns = data.get("vulnerabilities", [])
        vulnerabilities = [VulnerabilityType(v) for v in vulns] if isinstance(vulns, list) else None

        # Parse attacks
        attacks_data = data.get("attacks", [])
        if isinstance(attacks_data, list):
            attacks = [AttackMethod(a) for a in attacks_data]
        else:
            attacks = None

        return RedTeamConfig(
            vulnerabilities=vulnerabilities or RedTeamConfig().vulnerabilities,
            attacks=attacks or RedTeamConfig().attacks,
            attacks_per_vulnerability=data.get("attacks_per_vulnerability", 3),
            max_concurrent=data.get("max_concurrent", 5),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            target_purpose=data.get("target_purpose", "A helpful AI assistant"),
            output_dir=data.get("output_dir"),
            save_results=data.get("save_results", True),
        )

    @classmethod
    def list_playbooks(cls) -> dict[str, str]:
        """
        List available playbooks with descriptions.

        Returns:
            Dict mapping playbook names to descriptions
        """
        return {name: pb["description"] for name, pb in cls.PLAYBOOKS.items()}

    @classmethod
    def generate_sample_config(cls) -> str:
        """
        Generate a sample YAML configuration file.

        Returns:
            YAML string for sample configuration
        """
        sample = {
            "# Quaestor Red Team Configuration": None,
            "target_purpose": "A customer service chatbot for an e-commerce platform",
            "vulnerabilities": [
                "bias_gender",
                "bias_race",
                "pii_direct",
                "pii_session",
                "toxicity_profanity",
            ],
            "attacks": [
                "prompt_injection",
                "leetspeak",
                "linear_jailbreak",
            ],
            "attacks_per_vulnerability": 3,
            "max_concurrent": 10,
            "timeout_seconds": 30.0,
            "output_dir": "./redteam-results",
            "save_results": True,
        }
        return yaml.dump(sample, default_flow_style=False, sort_keys=False)
