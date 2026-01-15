"""
Pattern Learning System for Quaestor.

Extracts, stores, and matches patterns from test history to improve
test generation over time.

Part of Phase 6: Self-Optimization.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class PatternType(str, Enum):
    """Types of patterns that can be learned."""

    # Success patterns
    SUCCESSFUL_TOOL_SEQUENCE = "successful_tool_sequence"
    SUCCESSFUL_STATE_TRANSITION = "successful_state_transition"
    SUCCESSFUL_TEST_TEMPLATE = "successful_test_template"

    # Failure patterns
    FAILURE_MODE = "failure_mode"
    VULNERABILITY_SIGNATURE = "vulnerability_signature"
    EDGE_CASE = "edge_case"

    # Behavioral patterns
    AGENT_BEHAVIOR = "agent_behavior"
    ERROR_RECOVERY = "error_recovery"
    INPUT_SENSITIVITY = "input_sensitivity"


class PatternCategory(str, Enum):
    """Categories for organizing patterns."""

    CORRECTNESS = "correctness"
    SAFETY = "safety"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"


@dataclass
class Pattern:
    """
    A learned pattern from test history.

    Patterns are human-reviewable and can be shared across projects.
    """

    id: str
    name: str
    type: PatternType
    category: PatternCategory
    description: str

    # Pattern content
    signature: dict[str, Any]
    examples: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    source_agent: str | None = None
    source_test: str | None = None
    confidence: float = 0.5
    match_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Tags for filtering
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "category": self.category.value,
            "description": self.description,
            "signature": self.signature,
            "examples": self.examples,
            "source_agent": self.source_agent,
            "source_test": self.source_test,
            "confidence": self.confidence,
            "match_count": self.match_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            type=PatternType(data["type"]),
            category=PatternCategory(data["category"]),
            description=data["description"],
            signature=data["signature"],
            examples=data.get("examples", []),
            source_agent=data.get("source_agent"),
            source_test=data.get("source_test"),
            confidence=data.get("confidence", 0.5),
            match_count=data.get("match_count", 0),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(UTC),
            tags=data.get("tags", []),
        )


@dataclass
class PatternMatch:
    """Result of pattern matching against an agent or test."""

    pattern: Pattern
    similarity: float
    matched_elements: list[str]
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_strong_match(self) -> bool:
        """Check if this is a strong match (>= 0.8 similarity)."""
        return self.similarity >= 0.8

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern.id,
            "pattern_name": self.pattern.name,
            "similarity": self.similarity,
            "matched_elements": self.matched_elements,
            "context": self.context,
            "is_strong_match": self.is_strong_match,
        }


class PatternLibrary:
    """
    Library of learned patterns with persistence.

    Stores patterns in `.quaestor/patterns/` for human review
    and cross-project sharing.
    """

    DEFAULT_PATH = Path(".quaestor/patterns")

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize pattern library.

        Args:
            storage_path: Path to store patterns (defaults to .quaestor/patterns/)
        """
        self.storage_path = storage_path or self.DEFAULT_PATH
        self._patterns: dict[str, Pattern] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load patterns from storage."""
        if not self.storage_path.exists():
            return

        for pattern_file in self.storage_path.glob("*.json"):
            try:
                data = json.loads(pattern_file.read_text())
                pattern = Pattern.from_dict(data)
                self._patterns[pattern.id] = pattern
            except (json.JSONDecodeError, KeyError):
                continue  # Skip invalid files

    def _save_pattern(self, pattern: Pattern) -> None:
        """Save a single pattern to storage."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        pattern_file = self.storage_path / f"{pattern.id}.json"
        pattern_file.write_text(json.dumps(pattern.to_dict(), indent=2))

    def add(self, pattern: Pattern) -> None:
        """Add a pattern to the library."""
        self._patterns[pattern.id] = pattern
        self._save_pattern(pattern)

    def get(self, pattern_id: str) -> Pattern | None:
        """Get a pattern by ID."""
        return self._patterns.get(pattern_id)

    def remove(self, pattern_id: str) -> bool:
        """Remove a pattern by ID."""
        if pattern_id not in self._patterns:
            return False

        del self._patterns[pattern_id]
        pattern_file = self.storage_path / f"{pattern_id}.json"
        if pattern_file.exists():
            pattern_file.unlink()
        return True

    def list_all(self) -> list[Pattern]:
        """List all patterns."""
        return list(self._patterns.values())

    def filter_by_type(self, pattern_type: PatternType) -> list[Pattern]:
        """Filter patterns by type."""
        return [p for p in self._patterns.values() if p.type == pattern_type]

    def filter_by_category(self, category: PatternCategory) -> list[Pattern]:
        """Filter patterns by category."""
        return [p for p in self._patterns.values() if p.category == category]

    def filter_by_tags(self, tags: list[str]) -> list[Pattern]:
        """Filter patterns by tags (any match)."""
        tag_set = set(tags)
        return [p for p in self._patterns.values() if tag_set & set(p.tags)]

    def search(self, query: str) -> list[Pattern]:
        """Search patterns by name or description."""
        query_lower = query.lower()
        results = []
        for pattern in self._patterns.values():
            if query_lower in pattern.name.lower() or query_lower in pattern.description.lower():
                results.append(pattern)
        return results

    @property
    def count(self) -> int:
        """Number of patterns in library."""
        return len(self._patterns)


class PatternExtractor:
    """
    Extracts patterns from test results and agent behavior.

    Analyzes successful and failed tests to identify reusable patterns.
    """

    def __init__(self, library: PatternLibrary | None = None):
        """
        Initialize pattern extractor.

        Args:
            library: Pattern library to store extracted patterns
        """
        self.library = library or PatternLibrary()
        self._pattern_counter = 0

    def _generate_id(self, prefix: str = "PAT") -> str:
        """Generate a unique pattern ID."""
        from uuid import uuid4

        return f"{prefix}-{uuid4().hex[:8]}"

    def extract_tool_sequence(
        self,
        tool_calls: list[str],
        success: bool,
        agent_id: str | None = None,
        test_id: str | None = None,
    ) -> Pattern | None:
        """
        Extract a tool sequence pattern from a test execution.

        Args:
            tool_calls: Ordered list of tool names called
            success: Whether the test passed
            agent_id: ID of the agent tested
            test_id: ID of the test case

        Returns:
            Extracted pattern or None if not significant
        """
        if len(tool_calls) < 2:
            return None

        pattern_type = (
            PatternType.SUCCESSFUL_TOOL_SEQUENCE if success else PatternType.FAILURE_MODE
        )

        pattern = Pattern(
            id=self._generate_id("TOOL"),
            name=f"Tool Sequence: {' -> '.join(tool_calls[:3])}{'...' if len(tool_calls) > 3 else ''}",
            type=pattern_type,
            category=PatternCategory.CORRECTNESS,
            description=f"{'Successful' if success else 'Failed'} tool sequence with {len(tool_calls)} calls",
            signature={
                "tool_sequence": tool_calls,
                "length": len(tool_calls),
                "success": success,
            },
            examples=[{"tools": tool_calls, "success": success}],
            source_agent=agent_id,
            source_test=test_id,
            confidence=0.6 if success else 0.5,
            tags=["tool-sequence", "auto-extracted"],
        )

        self.library.add(pattern)
        return pattern

    def extract_failure_pattern(
        self,
        error_type: str,
        error_message: str,
        context: dict[str, Any],
        agent_id: str | None = None,
        test_id: str | None = None,
    ) -> Pattern:
        """
        Extract a failure pattern from a test error.

        Args:
            error_type: Type of error (e.g., "ValueError", "timeout")
            error_message: Error message text
            context: Additional context about the failure
            agent_id: ID of the agent tested
            test_id: ID of the test case

        Returns:
            Extracted failure pattern
        """
        # Determine category based on error type
        category = PatternCategory.CORRECTNESS
        if "security" in error_type.lower() or "injection" in error_message.lower():
            category = PatternCategory.SECURITY
        elif "timeout" in error_type.lower() or "performance" in error_message.lower():
            category = PatternCategory.PERFORMANCE

        pattern = Pattern(
            id=self._generate_id("FAIL"),
            name=f"Failure: {error_type}",
            type=PatternType.FAILURE_MODE,
            category=category,
            description=f"Failure pattern: {error_message[:100]}",
            signature={
                "error_type": error_type,
                "error_keywords": self._extract_keywords(error_message),
                "context_keys": list(context.keys()),
            },
            examples=[{"error": error_message, "context": context}],
            source_agent=agent_id,
            source_test=test_id,
            confidence=0.7,
            tags=["failure", "auto-extracted", error_type.lower()],
        )

        self.library.add(pattern)
        return pattern

    def extract_vulnerability_pattern(
        self,
        vulnerability_type: str,
        severity: str,
        indicators: list[str],
        remediation: str | None = None,
    ) -> Pattern:
        """
        Extract a vulnerability signature pattern.

        Args:
            vulnerability_type: Type of vulnerability (e.g., "prompt_injection")
            severity: Severity level
            indicators: Indicators of the vulnerability
            remediation: Suggested remediation

        Returns:
            Extracted vulnerability pattern
        """
        pattern = Pattern(
            id=self._generate_id("VULN"),
            name=f"Vulnerability: {vulnerability_type.replace('_', ' ').title()}",
            type=PatternType.VULNERABILITY_SIGNATURE,
            category=PatternCategory.SECURITY,
            description=f"Security vulnerability pattern for {vulnerability_type}",
            signature={
                "vulnerability_type": vulnerability_type,
                "severity": severity,
                "indicators": indicators,
            },
            examples=[{"type": vulnerability_type, "indicators": indicators}],
            confidence=0.8,
            tags=["vulnerability", "security", severity.lower(), vulnerability_type],
        )

        if remediation:
            pattern.signature["remediation"] = remediation

        self.library.add(pattern)
        return pattern

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract key words from text for pattern matching."""
        # Simple keyword extraction (could be enhanced with NLP)
        words = text.lower().split()
        # Filter out common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in"}
        keywords = [w.strip(".,!?") for w in words if w.strip(".,!?") not in stop_words and len(w) > 2]
        return keywords[:max_keywords]


class PatternMatcher:
    """
    Matches patterns against new agents and test scenarios.

    Uses signature-based matching to identify relevant patterns.
    """

    def __init__(self, library: PatternLibrary):
        """
        Initialize pattern matcher.

        Args:
            library: Pattern library to match against
        """
        self.library = library

    def match_tool_sequence(
        self,
        tool_calls: list[str],
        min_similarity: float = 0.5,
    ) -> list[PatternMatch]:
        """
        Match a tool sequence against known patterns.

        Args:
            tool_calls: Tool calls to match
            min_similarity: Minimum similarity threshold

        Returns:
            List of pattern matches sorted by similarity
        """
        matches = []
        tool_patterns = self.library.filter_by_type(PatternType.SUCCESSFUL_TOOL_SEQUENCE)
        tool_patterns.extend(self.library.filter_by_type(PatternType.FAILURE_MODE))

        for pattern in tool_patterns:
            if "tool_sequence" not in pattern.signature:
                continue

            pattern_tools = pattern.signature["tool_sequence"]
            similarity = self._sequence_similarity(tool_calls, pattern_tools)

            if similarity >= min_similarity:
                matched = list(set(tool_calls) & set(pattern_tools))
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        similarity=similarity,
                        matched_elements=matched,
                        context={"input_length": len(tool_calls)},
                    )
                )

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def match_error(
        self,
        error_type: str,
        error_message: str,
        min_similarity: float = 0.5,
    ) -> list[PatternMatch]:
        """
        Match an error against known failure patterns.

        Args:
            error_type: Type of error
            error_message: Error message
            min_similarity: Minimum similarity threshold

        Returns:
            List of pattern matches sorted by similarity
        """
        matches = []
        failure_patterns = self.library.filter_by_type(PatternType.FAILURE_MODE)

        error_keywords = set(error_message.lower().split())

        for pattern in failure_patterns:
            sig = pattern.signature
            similarity = 0.0
            matched = []

            # Match error type
            if sig.get("error_type", "").lower() == error_type.lower():
                similarity += 0.5
                matched.append(f"error_type:{error_type}")

            # Match keywords
            pattern_keywords = set(sig.get("error_keywords", []))
            keyword_overlap = error_keywords & pattern_keywords
            if pattern_keywords:
                keyword_sim = len(keyword_overlap) / len(pattern_keywords)
                similarity += 0.5 * keyword_sim
                matched.extend([f"keyword:{k}" for k in keyword_overlap])

            if similarity >= min_similarity:
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        similarity=similarity,
                        matched_elements=matched,
                    )
                )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    def find_relevant_patterns(
        self,
        agent_tools: list[str],
        agent_states: list[str] | None = None,
        category: PatternCategory | None = None,
        min_confidence: float = 0.5,
    ) -> list[Pattern]:
        """
        Find patterns relevant to an agent's capabilities.

        Args:
            agent_tools: Tools available to the agent
            agent_states: States the agent can be in
            category: Filter by category
            min_confidence: Minimum confidence threshold

        Returns:
            List of relevant patterns
        """
        tool_set = set(agent_tools)
        relevant = []

        for pattern in self.library.list_all():
            if pattern.confidence < min_confidence:
                continue

            if category and pattern.category != category:
                continue

            # Check tool overlap
            pattern_tools = set(pattern.signature.get("tool_sequence", []))
            if pattern_tools and (tool_set & pattern_tools):
                relevant.append(pattern)
                continue

            # Check for any signature match
            sig_tools = []
            for key, value in pattern.signature.items():
                if isinstance(value, list):
                    sig_tools.extend([str(v) for v in value])

            if set(sig_tools) & tool_set:
                relevant.append(pattern)

        return relevant

    def _sequence_similarity(self, seq1: list[str], seq2: list[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0

        # Jaccard similarity for set overlap
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Bonus for order matching (LCS-based)
        lcs_len = self._lcs_length(seq1, seq2)
        order_bonus = lcs_len / max(len(seq1), len(seq2)) * 0.3

        return min(1.0, jaccard * 0.7 + order_bonus)

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]


# Built-in patterns library
BUILTIN_PATTERNS = [
    Pattern(
        id="BUILTIN-001",
        name="Tool Exhaustion Attack",
        type=PatternType.VULNERABILITY_SIGNATURE,
        category=PatternCategory.SECURITY,
        description="Pattern where attacker tries to exhaust all available tools",
        signature={
            "indicators": ["call all tools", "try every function", "enumerate capabilities"],
            "severity": "medium",
        },
        confidence=0.9,
        tags=["security", "built-in", "attack"],
    ),
    Pattern(
        id="BUILTIN-002",
        name="Prompt Injection Attempt",
        type=PatternType.VULNERABILITY_SIGNATURE,
        category=PatternCategory.SECURITY,
        description="Pattern indicating prompt injection attempts",
        signature={
            "indicators": [
                "ignore previous instructions",
                "disregard all prior",
                "new instructions",
                "system prompt",
            ],
            "severity": "high",
        },
        confidence=0.95,
        tags=["security", "built-in", "injection"],
    ),
    Pattern(
        id="BUILTIN-003",
        name="Information Extraction Attempt",
        type=PatternType.VULNERABILITY_SIGNATURE,
        category=PatternCategory.SECURITY,
        description="Pattern for attempts to extract sensitive information",
        signature={
            "indicators": [
                "reveal your instructions",
                "show system prompt",
                "what are your rules",
                "api key",
                "password",
            ],
            "severity": "high",
        },
        confidence=0.9,
        tags=["security", "built-in", "extraction"],
    ),
    Pattern(
        id="BUILTIN-004",
        name="State Loop Detection",
        type=PatternType.FAILURE_MODE,
        category=PatternCategory.RELIABILITY,
        description="Pattern for detecting infinite state loops",
        signature={
            "indicators": ["repeated state", "loop detected", "max iterations"],
            "max_repetitions": 3,
        },
        confidence=0.85,
        tags=["reliability", "built-in", "loop"],
    ),
]


def get_builtin_library() -> PatternLibrary:
    """Get a pattern library pre-populated with built-in patterns."""
    library = PatternLibrary()
    for pattern in BUILTIN_PATTERNS:
        if not library.get(pattern.id):
            library.add(pattern)
    return library
