"""
Tests for the Static Linter module.

Tests no-LLM static analysis functionality.
"""

import pytest

from quaestor.analysis.linter import (
    Category,
    LinterConfig,
    LintResult,
    Severity,
    StaticLinter,
    lint_string,
)


class TestStaticLinter:
    """Tests for StaticLinter class."""

    @pytest.fixture
    def linter(self) -> StaticLinter:
        """Create a linter instance."""
        return StaticLinter()

    def test_lint_clean_code(self, linter: StaticLinter) -> None:
        """Test linting clean code with no issues."""
        source = '''
"""A well-documented module."""

def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


class Greeter:
    """A class that greets people."""

    def __init__(self, prefix: str = "Hello"):
        """Initialize with a greeting prefix."""
        self.prefix = prefix

    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"{self.prefix}, {name}!"
'''
        result = linter.lint_string(source)

        # Should have no serious issues
        assert result.error_count == 0

    def test_detect_missing_docstring(self, linter: StaticLinter) -> None:
        """Test detection of missing docstrings."""
        source = """
def undocumented_function(x):
    return x * 2


class UndocumentedClass:
    def also_undocumented(self):
        pass
"""
        result = linter.lint_string(source)

        # Should find missing docstrings
        docstring_issues = [i for i in result.issues if i.rule_id in ("Q001", "Q002")]
        assert len(docstring_issues) >= 2

        # Check categories
        for issue in docstring_issues:
            assert issue.category == Category.STYLE

    def test_detect_bare_except(self, linter: StaticLinter) -> None:
        """Test detection of bare except clauses."""
        source = '''
def risky_operation():
    """Do something risky."""
    try:
        dangerous()
    except:
        pass  # Bad!
'''
        result = linter.lint_string(source)

        # Should find bare except
        bare_except_issues = [i for i in result.issues if i.rule_id == "Q011"]
        assert len(bare_except_issues) == 1
        assert bare_except_issues[0].severity == Severity.ERROR

    def test_detect_async_without_error_handling(self, linter: StaticLinter) -> None:
        """Test detection of async functions without try/except."""
        source = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    response = await client.get(url)
    return await response.json()
'''
        result = linter.lint_string(source)

        # Should find missing error handling
        error_handling_issues = [i for i in result.issues if i.rule_id == "Q010"]
        assert len(error_handling_issues) == 1
        assert error_handling_issues[0].severity == Severity.WARNING

    def test_detect_hardcoded_secrets(self, linter: StaticLinter) -> None:
        """Test detection of hardcoded secrets."""
        source = '''
def configure():
    """Configure the API."""
    api_key = "sk-1234567890abcdef"
    secret = "super_secret_value"
    return api_key
'''
        result = linter.lint_string(source)

        # Should find hardcoded secrets
        secret_issues = [i for i in result.issues if i.rule_id == "Q030"]
        assert len(secret_issues) >= 1
        assert secret_issues[0].severity == Severity.ERROR
        assert secret_issues[0].category == Category.SECURITY

    def test_detect_eval_usage(self, linter: StaticLinter) -> None:
        """Test detection of eval/exec usage."""
        source = '''
def process_input(user_input: str) -> any:
    """Process user input."""
    result = eval(user_input)  # Dangerous!
    return result
'''
        result = linter.lint_string(source)

        # Should find eval usage
        eval_issues = [i for i in result.issues if i.rule_id == "Q031"]
        assert len(eval_issues) == 1
        assert "eval" in eval_issues[0].message

    def test_detect_exec_usage(self, linter: StaticLinter) -> None:
        """Test detection of exec usage."""
        source = '''
def run_code(code: str) -> None:
    """Run arbitrary code."""
    exec(code)  # Very dangerous!
'''
        result = linter.lint_string(source)

        # Should find exec usage
        exec_issues = [i for i in result.issues if i.rule_id == "Q031"]
        assert len(exec_issues) == 1
        assert "exec" in exec_issues[0].message

    def test_detect_sync_in_async(self, linter: StaticLinter) -> None:
        """Test detection of sync I/O in async functions."""
        source = '''
import time

async def slow_operation():
    """Do something slowly."""
    time.sleep(5)  # Blocks the event loop!
    return "done"
'''
        result = linter.lint_string(source)

        # Should find sync sleep in async
        sync_issues = [i for i in result.issues if i.rule_id == "Q040"]
        assert len(sync_issues) == 1
        assert sync_issues[0].category == Category.PERFORMANCE

    def test_detect_long_function(self, linter: StaticLinter) -> None:
        """Test detection of overly long functions."""
        # Generate a function with 60 lines
        lines = ["    x = 1"] * 60
        body = "\n".join(lines)
        source = f'''
def very_long_function():
    """This function is too long."""
{body}
    return x
'''
        result = linter.lint_string(source)

        # Should find long function
        long_issues = [i for i in result.issues if i.rule_id == "Q051"]
        assert len(long_issues) == 1

    def test_detect_potential_infinite_loop(self, linter: StaticLinter) -> None:
        """Test detection of potential infinite loops."""
        source = '''
def infinite_worker():
    """Run forever with no escape."""
    while True:
        do_work()
        # No break, no return!
'''
        result = linter.lint_string(source)

        # Should find infinite loop risk
        loop_issues = [i for i in result.issues if i.rule_id == "Q022"]
        assert len(loop_issues) == 1
        assert loop_issues[0].severity == Severity.ERROR

    def test_no_false_positive_while_true_with_break(self, linter: StaticLinter) -> None:
        """Test that while True with break is not flagged."""
        source = '''
def worker():
    """Run until done."""
    while True:
        result = do_work()
        if result.done:
            break
'''
        result = linter.lint_string(source)

        # Should NOT find infinite loop
        loop_issues = [i for i in result.issues if i.rule_id == "Q022"]
        assert len(loop_issues) == 0

    def test_no_false_positive_env_secrets(self, linter: StaticLinter) -> None:
        """Test that secrets from env vars are not flagged."""
        source = '''
import os

def configure():
    """Configure from environment."""
    api_key = os.environ.get("API_KEY")
    secret = os.getenv("SECRET")
    return api_key
'''
        result = linter.lint_string(source)

        # Should NOT find hardcoded secrets
        secret_issues = [i for i in result.issues if i.rule_id == "Q030"]
        assert len(secret_issues) == 0


class TestLinterConfig:
    """Tests for linter configuration."""

    def test_disable_rules(self) -> None:
        """Test disabling specific rules."""
        config = LinterConfig(disabled_rules={"Q001", "Q002"})
        linter = StaticLinter(config)

        source = """
def undocumented():
    pass

class AlsoUndocumented:
    pass
"""
        result = linter.lint_string(source)

        # Should not report docstring issues
        docstring_issues = [i for i in result.issues if i.rule_id in ("Q001", "Q002")]
        assert len(docstring_issues) == 0

    def test_enable_only_specific_rules(self) -> None:
        """Test enabling only specific rules."""
        config = LinterConfig(enabled_rules={"Q031"})  # Only unsafe eval
        linter = StaticLinter(config)

        source = """
def bad_code():
    eval("1+1")  # Should be caught

def missing_docs():  # Should NOT be caught
    pass
"""
        result = linter.lint_string(source)

        # Should only have Q031 issues
        assert all(i.rule_id == "Q031" for i in result.issues)

    def test_severity_override(self) -> None:
        """Test overriding rule severity."""
        config = LinterConfig(
            severity_overrides={"Q001": Severity.ERROR}  # Upgrade to error
        )
        linter = StaticLinter(config)

        source = """
def undocumented():
    pass
"""
        result = linter.lint_string(source)

        # Find the docstring issue
        doc_issues = [i for i in result.issues if i.rule_id == "Q001"]
        assert len(doc_issues) == 1
        assert doc_issues[0].severity == Severity.ERROR


class TestLintResult:
    """Tests for LintResult methods."""

    def test_result_counts(self) -> None:
        """Test issue count properties."""
        result = lint_string("""
def bad():
    eval("code")  # Error: Q031

def undocumented():  # Warning: Q001
    pass
""")

        assert result.error_count >= 1
        assert result.warning_count >= 0

    def test_has_errors(self) -> None:
        """Test has_errors property."""
        # Clean code
        clean_result = lint_string('''
def documented():
    """Has docstring."""
    pass
''')

        # Code with errors
        error_result = lint_string('''
def bad():
    """Has docstring but uses eval."""
    eval("bad")
''')

        assert clean_result.has_errors is False
        assert error_result.has_errors is True

    def test_get_issues_by_severity(self) -> None:
        """Test filtering issues by severity."""
        result = lint_string("""
def bad():
    eval("code")

def undocumented():
    pass
""")

        errors = result.get_issues_by_severity(Severity.ERROR)
        warnings = result.get_issues_by_severity(Severity.WARNING)

        assert all(i.severity == Severity.ERROR for i in errors)
        assert all(i.severity == Severity.WARNING for i in warnings)

    def test_get_issues_by_category(self) -> None:
        """Test filtering issues by category."""
        result = lint_string('''
def bad():
    """Documented but insecure."""
    eval("code")
    api_key = "secret123"
''')

        security_issues = result.get_issues_by_category(Category.SECURITY)

        assert all(i.category == Category.SECURITY for i in security_issues)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_lint_string(self) -> None:
        """Test lint_string convenience function."""
        result = lint_string("def foo(): pass")

        assert isinstance(result, LintResult)
