"""Tests for the pattern learning system."""

import json
import tempfile
from pathlib import Path

import pytest

from quaestor.optimization.patterns import (
    BUILTIN_PATTERNS,
    Pattern,
    PatternCategory,
    PatternExtractor,
    PatternLibrary,
    PatternMatch,
    PatternMatcher,
    PatternType,
    get_builtin_library,
)


class TestPatternModel:
    """Tests for the Pattern model."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = Pattern(
            id="TEST-001",
            name="Test Pattern",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="A test pattern",
            signature={"tools": ["tool1", "tool2"]},
        )

        assert pattern.id == "TEST-001"
        assert pattern.name == "Test Pattern"
        assert pattern.type == PatternType.SUCCESSFUL_TOOL_SEQUENCE
        assert pattern.category == PatternCategory.CORRECTNESS
        assert pattern.confidence == 0.5  # Default

    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = Pattern(
            id="TEST-002",
            name="Serialization Test",
            type=PatternType.FAILURE_MODE,
            category=PatternCategory.SAFETY,
            description="Test serialization",
            signature={"error": "test_error"},
            tags=["test", "serialization"],
        )

        data = pattern.to_dict()

        assert data["id"] == "TEST-002"
        assert data["type"] == "failure_mode"
        assert data["category"] == "safety"
        assert data["tags"] == ["test", "serialization"]
        assert "created_at" in data
        assert "updated_at" in data

    def test_pattern_from_dict(self):
        """Test pattern deserialization."""
        data = {
            "id": "TEST-003",
            "name": "From Dict",
            "type": "vulnerability_signature",
            "category": "security",
            "description": "Created from dict",
            "signature": {"vuln": "test"},
            "confidence": 0.8,
            "match_count": 5,
            "tags": ["security"],
            "created_at": "2026-01-15T12:00:00+00:00",
            "updated_at": "2026-01-15T12:00:00+00:00",
        }

        pattern = Pattern.from_dict(data)

        assert pattern.id == "TEST-003"
        assert pattern.type == PatternType.VULNERABILITY_SIGNATURE
        assert pattern.category == PatternCategory.SECURITY
        assert pattern.confidence == 0.8
        assert pattern.match_count == 5

    def test_pattern_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = Pattern(
            id="TEST-004",
            name="Roundtrip Test",
            type=PatternType.EDGE_CASE,
            category=PatternCategory.RELIABILITY,
            description="Test roundtrip",
            signature={"key": "value"},
            examples=[{"input": "test"}],
            confidence=0.75,
            tags=["test"],
        )

        restored = Pattern.from_dict(original.to_dict())

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.type == original.type
        assert restored.signature == original.signature
        assert restored.confidence == original.confidence


class TestPatternMatch:
    """Tests for PatternMatch."""

    def test_pattern_match_creation(self):
        """Test creating a pattern match."""
        pattern = Pattern(
            id="PAT-001",
            name="Test",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={},
        )

        match = PatternMatch(
            pattern=pattern,
            similarity=0.85,
            matched_elements=["tool1", "tool2"],
        )

        assert match.similarity == 0.85
        assert match.is_strong_match is True
        assert len(match.matched_elements) == 2

    def test_weak_match(self):
        """Test weak match detection."""
        pattern = Pattern(
            id="PAT-002",
            name="Test",
            type=PatternType.FAILURE_MODE,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={},
        )

        match = PatternMatch(
            pattern=pattern,
            similarity=0.5,
            matched_elements=["tool1"],
        )

        assert match.is_strong_match is False

    def test_match_to_dict(self):
        """Test match serialization."""
        pattern = Pattern(
            id="PAT-003",
            name="Serialize Match",
            type=PatternType.AGENT_BEHAVIOR,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={},
        )

        match = PatternMatch(
            pattern=pattern,
            similarity=0.9,
            matched_elements=["elem1"],
            context={"test": True},
        )

        data = match.to_dict()

        assert data["pattern_id"] == "PAT-003"
        assert data["similarity"] == 0.9
        assert data["is_strong_match"] is True
        assert data["context"]["test"] is True


class TestPatternLibrary:
    """Tests for PatternLibrary."""

    @pytest.fixture
    def temp_library(self):
        """Create a temporary pattern library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield PatternLibrary(Path(tmpdir) / "patterns")

    def test_empty_library(self, temp_library):
        """Test empty library."""
        assert temp_library.count == 0
        assert temp_library.list_all() == []

    def test_add_and_get_pattern(self, temp_library):
        """Test adding and retrieving a pattern."""
        pattern = Pattern(
            id="LIB-001",
            name="Library Test",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={"tools": ["a", "b"]},
        )

        temp_library.add(pattern)

        assert temp_library.count == 1
        retrieved = temp_library.get("LIB-001")
        assert retrieved is not None
        assert retrieved.name == "Library Test"

    def test_remove_pattern(self, temp_library):
        """Test removing a pattern."""
        pattern = Pattern(
            id="LIB-002",
            name="To Remove",
            type=PatternType.FAILURE_MODE,
            category=PatternCategory.SAFETY,
            description="Test",
            signature={},
        )

        temp_library.add(pattern)
        assert temp_library.count == 1

        result = temp_library.remove("LIB-002")
        assert result is True
        assert temp_library.count == 0
        assert temp_library.get("LIB-002") is None

    def test_remove_nonexistent(self, temp_library):
        """Test removing non-existent pattern."""
        result = temp_library.remove("DOES-NOT-EXIST")
        assert result is False

    def test_filter_by_type(self, temp_library):
        """Test filtering by type."""
        pattern1 = Pattern(
            id="TYPE-001",
            name="Success",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={},
        )
        pattern2 = Pattern(
            id="TYPE-002",
            name="Failure",
            type=PatternType.FAILURE_MODE,
            category=PatternCategory.CORRECTNESS,
            description="Test",
            signature={},
        )

        temp_library.add(pattern1)
        temp_library.add(pattern2)

        success_patterns = temp_library.filter_by_type(PatternType.SUCCESSFUL_TOOL_SEQUENCE)
        assert len(success_patterns) == 1
        assert success_patterns[0].id == "TYPE-001"

    def test_filter_by_category(self, temp_library):
        """Test filtering by category."""
        pattern1 = Pattern(
            id="CAT-001",
            name="Safety",
            type=PatternType.FAILURE_MODE,
            category=PatternCategory.SAFETY,
            description="Test",
            signature={},
        )
        pattern2 = Pattern(
            id="CAT-002",
            name="Security",
            type=PatternType.VULNERABILITY_SIGNATURE,
            category=PatternCategory.SECURITY,
            description="Test",
            signature={},
        )

        temp_library.add(pattern1)
        temp_library.add(pattern2)

        security_patterns = temp_library.filter_by_category(PatternCategory.SECURITY)
        assert len(security_patterns) == 1
        assert security_patterns[0].id == "CAT-002"

    def test_filter_by_tags(self, temp_library):
        """Test filtering by tags."""
        pattern1 = Pattern(
            id="TAG-001",
            name="Tagged 1",
            type=PatternType.EDGE_CASE,
            category=PatternCategory.RELIABILITY,
            description="Test",
            signature={},
            tags=["test", "alpha"],
        )
        pattern2 = Pattern(
            id="TAG-002",
            name="Tagged 2",
            type=PatternType.EDGE_CASE,
            category=PatternCategory.RELIABILITY,
            description="Test",
            signature={},
            tags=["beta"],
        )

        temp_library.add(pattern1)
        temp_library.add(pattern2)

        alpha_patterns = temp_library.filter_by_tags(["alpha"])
        assert len(alpha_patterns) == 1
        assert alpha_patterns[0].id == "TAG-001"

    def test_search(self, temp_library):
        """Test searching patterns."""
        pattern1 = Pattern(
            id="SEARCH-001",
            name="Database Query Tool",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="Pattern for database queries",
            signature={},
        )
        pattern2 = Pattern(
            id="SEARCH-002",
            name="API Call Pattern",
            type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
            category=PatternCategory.CORRECTNESS,
            description="Pattern for REST API calls",
            signature={},
        )

        temp_library.add(pattern1)
        temp_library.add(pattern2)

        results = temp_library.search("database")
        assert len(results) == 1
        assert results[0].id == "SEARCH-001"

        results = temp_library.search("pattern")
        assert len(results) == 2

    def test_persistence(self):
        """Test pattern persistence across library instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "patterns"

            # Create library and add pattern
            lib1 = PatternLibrary(path)
            pattern = Pattern(
                id="PERSIST-001",
                name="Persistent Pattern",
                type=PatternType.AGENT_BEHAVIOR,
                category=PatternCategory.CORRECTNESS,
                description="Test persistence",
                signature={"key": "value"},
            )
            lib1.add(pattern)

            # Create new library instance
            lib2 = PatternLibrary(path)

            # Pattern should be loaded
            retrieved = lib2.get("PERSIST-001")
            assert retrieved is not None
            assert retrieved.name == "Persistent Pattern"


class TestPatternExtractor:
    """Tests for PatternExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create an extractor with temp library."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(Path(tmpdir) / "patterns")
            yield PatternExtractor(library)

    def test_extract_tool_sequence_success(self, extractor):
        """Test extracting successful tool sequence."""
        pattern = extractor.extract_tool_sequence(
            tool_calls=["search", "analyze", "summarize"],
            success=True,
            agent_id="agent-001",
            test_id="test-001",
        )

        assert pattern is not None
        assert pattern.type == PatternType.SUCCESSFUL_TOOL_SEQUENCE
        assert "successful_tool_sequence" in pattern.tags or "auto-extracted" in pattern.tags
        assert extractor.library.count == 1

    def test_extract_tool_sequence_failure(self, extractor):
        """Test extracting failed tool sequence."""
        pattern = extractor.extract_tool_sequence(
            tool_calls=["search", "crash"],
            success=False,
        )

        assert pattern is not None
        assert pattern.type == PatternType.FAILURE_MODE

    def test_extract_single_tool_returns_none(self, extractor):
        """Test that single tool call returns None."""
        pattern = extractor.extract_tool_sequence(
            tool_calls=["search"],
            success=True,
        )

        assert pattern is None

    def test_extract_failure_pattern(self, extractor):
        """Test extracting failure pattern."""
        pattern = extractor.extract_failure_pattern(
            error_type="ValueError",
            error_message="Invalid input: expected string but got integer",
            context={"input": 42, "expected": "string"},
            agent_id="agent-002",
        )

        assert pattern is not None
        assert pattern.type == PatternType.FAILURE_MODE
        assert pattern.category == PatternCategory.CORRECTNESS
        assert "ValueError" in pattern.name
        assert "failure" in pattern.tags

    def test_extract_security_failure(self, extractor):
        """Test extracting security-related failure."""
        pattern = extractor.extract_failure_pattern(
            error_type="SecurityError",
            error_message="Potential injection attack detected",
            context={},
        )

        assert pattern.category == PatternCategory.SECURITY

    def test_extract_vulnerability_pattern(self, extractor):
        """Test extracting vulnerability pattern."""
        pattern = extractor.extract_vulnerability_pattern(
            vulnerability_type="prompt_injection",
            severity="high",
            indicators=["ignore instructions", "new system prompt"],
            remediation="Add input sanitization",
        )

        assert pattern is not None
        assert pattern.type == PatternType.VULNERABILITY_SIGNATURE
        assert pattern.category == PatternCategory.SECURITY
        assert pattern.signature["remediation"] == "Add input sanitization"
        assert pattern.confidence == 0.8


class TestPatternMatcher:
    """Tests for PatternMatcher."""

    @pytest.fixture
    def matcher_with_patterns(self):
        """Create a matcher with pre-populated patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(Path(tmpdir) / "patterns")

            # Add test patterns
            library.add(
                Pattern(
                    id="MATCH-001",
                    name="Search-Analyze Pattern",
                    type=PatternType.SUCCESSFUL_TOOL_SEQUENCE,
                    category=PatternCategory.CORRECTNESS,
                    description="Search then analyze",
                    signature={"tool_sequence": ["search", "analyze", "report"]},
                    confidence=0.8,
                )
            )
            library.add(
                Pattern(
                    id="MATCH-002",
                    name="Timeout Failure",
                    type=PatternType.FAILURE_MODE,
                    category=PatternCategory.PERFORMANCE,
                    description="Timeout failure pattern",
                    signature={
                        "error_type": "TimeoutError",
                        "error_keywords": ["timeout", "exceeded", "limit"],
                    },
                    confidence=0.7,
                )
            )

            yield PatternMatcher(library)

    def test_match_tool_sequence_exact(self, matcher_with_patterns):
        """Test exact tool sequence matching."""
        matches = matcher_with_patterns.match_tool_sequence(
            tool_calls=["search", "analyze", "report"],
            min_similarity=0.5,
        )

        assert len(matches) >= 1
        assert matches[0].pattern.id == "MATCH-001"
        assert matches[0].similarity > 0.9

    def test_match_tool_sequence_partial(self, matcher_with_patterns):
        """Test partial tool sequence matching."""
        matches = matcher_with_patterns.match_tool_sequence(
            tool_calls=["search", "analyze"],
            min_similarity=0.3,
        )

        assert len(matches) >= 1
        # Should match with lower similarity
        assert matches[0].similarity < 1.0

    def test_match_tool_sequence_no_match(self, matcher_with_patterns):
        """Test no matching tool sequence."""
        matches = matcher_with_patterns.match_tool_sequence(
            tool_calls=["completely", "different", "tools"],
            min_similarity=0.8,
        )

        assert len(matches) == 0

    def test_match_error(self, matcher_with_patterns):
        """Test error pattern matching."""
        matches = matcher_with_patterns.match_error(
            error_type="TimeoutError",
            error_message="Request timeout exceeded limit",
            min_similarity=0.3,
        )

        assert len(matches) >= 1
        best_match = matches[0]
        assert best_match.pattern.id == "MATCH-002"

    def test_match_error_no_match(self, matcher_with_patterns):
        """Test no matching error pattern."""
        matches = matcher_with_patterns.match_error(
            error_type="ValueError",
            error_message="Something completely unrelated",
            min_similarity=0.8,
        )

        assert len(matches) == 0

    def test_find_relevant_patterns(self, matcher_with_patterns):
        """Test finding relevant patterns for an agent."""
        patterns = matcher_with_patterns.find_relevant_patterns(
            agent_tools=["search", "analyze"],
            min_confidence=0.5,
        )

        assert len(patterns) >= 1
        assert any(p.id == "MATCH-001" for p in patterns)

    def test_find_relevant_patterns_with_category(self, matcher_with_patterns):
        """Test finding relevant patterns filtered by category."""
        patterns = matcher_with_patterns.find_relevant_patterns(
            agent_tools=["search", "analyze"],
            category=PatternCategory.PERFORMANCE,
        )

        # Should not match correctness patterns
        assert not any(p.id == "MATCH-001" for p in patterns)


class TestBuiltinPatterns:
    """Tests for built-in patterns."""

    def test_builtin_patterns_exist(self):
        """Test that built-in patterns are defined."""
        assert len(BUILTIN_PATTERNS) > 0

    def test_builtin_patterns_valid(self):
        """Test that all built-in patterns are valid."""
        for pattern in BUILTIN_PATTERNS:
            assert pattern.id.startswith("BUILTIN-")
            assert pattern.type is not None
            assert pattern.category is not None
            assert len(pattern.signature) > 0

    def test_get_builtin_library(self):
        """Test getting library with built-in patterns."""
        library = get_builtin_library()

        assert library.count >= len(BUILTIN_PATTERNS)

        # Check specific patterns
        injection_pattern = library.get("BUILTIN-002")
        assert injection_pattern is not None
        assert "injection" in injection_pattern.name.lower()


class TestSequenceSimilarity:
    """Tests for sequence similarity algorithm."""

    @pytest.fixture
    def matcher(self):
        """Create a matcher for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            library = PatternLibrary(Path(tmpdir) / "patterns")
            yield PatternMatcher(library)

    def test_identical_sequences(self, matcher):
        """Test identical sequences have similarity 1.0."""
        sim = matcher._sequence_similarity(["a", "b", "c"], ["a", "b", "c"])
        assert sim == 1.0

    def test_empty_sequences(self, matcher):
        """Test empty sequences have similarity 0.0."""
        assert matcher._sequence_similarity([], []) == 0.0
        assert matcher._sequence_similarity(["a"], []) == 0.0
        assert matcher._sequence_similarity([], ["b"]) == 0.0

    def test_partial_overlap(self, matcher):
        """Test partial overlap has intermediate similarity."""
        sim = matcher._sequence_similarity(["a", "b", "c"], ["a", "b", "d"])
        assert 0.0 < sim < 1.0

    def test_no_overlap(self, matcher):
        """Test no overlap has low similarity."""
        sim = matcher._sequence_similarity(["a", "b"], ["c", "d"])
        assert sim < 0.5

    def test_lcs_length(self, matcher):
        """Test LCS length calculation."""
        assert matcher._lcs_length(["a", "b", "c"], ["a", "b", "c"]) == 3
        assert matcher._lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2
        assert matcher._lcs_length(["a"], ["b"]) == 0
