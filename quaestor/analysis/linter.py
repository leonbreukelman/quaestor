"""
Static Linter for Agent Code Analysis.

Provides fast, no-LLM static analysis for common issues in agent code.
Detects anti-patterns, security issues, and best practice violations.

Part of Phase 1: Core Analysis Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from quaestor.analysis.parser import ParsedCode, PythonParser, FunctionDef, ClassDef


class Severity(str, Enum):
    """Severity levels for lint issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Category(str, Enum):
    """Categories of lint rules."""
    CORRECTNESS = "correctness"
    SAFETY = "safety"
    SECURITY = "security"
    STYLE = "style"
    PERFORMANCE = "performance"
    BEST_PRACTICE = "best_practice"
    AGENT_PATTERN = "agent_pattern"


@dataclass
class LintRule:
    """Definition of a lint rule."""
    id: str
    name: str
    description: str
    severity: Severity
    category: Category
    enabled: bool = True


@dataclass
class LintIssue:
    """A single linting issue found in code."""
    rule_id: str
    severity: Severity
    category: Category
    message: str
    file_path: str
    line: int
    column: int
    suggestion: str | None = None
    code_snippet: str | None = None


@dataclass
class LintResult:
    """Complete linting results for a file or codebase."""
    file_path: str
    issues: list[LintIssue] = field(default_factory=list)
    rules_checked: list[str] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)
    
    @property
    def info_count(self) -> int:
        """Count of info-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.INFO)
    
    @property
    def has_errors(self) -> bool:
        """Check if any error-level issues exist."""
        return self.error_count > 0
    
    def get_issues_by_severity(self, severity: Severity) -> list[LintIssue]:
        """Get all issues with a specific severity."""
        return [i for i in self.issues if i.severity == severity]
    
    def get_issues_by_category(self, category: Category) -> list[LintIssue]:
        """Get all issues in a specific category."""
        return [i for i in self.issues if i.category == category]


@dataclass
class LinterConfig:
    """Configuration for the linter."""
    enabled_rules: set[str] = field(default_factory=set)
    disabled_rules: set[str] = field(default_factory=set)
    severity_overrides: dict[str, Severity] = field(default_factory=dict)
    
    # Safety settings (read-only by default per spec)
    read_only: bool = True
    
    def is_rule_enabled(self, rule_id: str) -> bool:
        """Check if a rule is enabled."""
        if self.disabled_rules and rule_id in self.disabled_rules:
            return False
        if self.enabled_rules:
            return rule_id in self.enabled_rules
        return True  # All rules enabled by default


# Built-in lint rules
BUILTIN_RULES: dict[str, LintRule] = {
    # Missing docstrings
    "Q001": LintRule(
        id="Q001",
        name="missing-function-docstring",
        description="Function is missing a docstring",
        severity=Severity.WARNING,
        category=Category.STYLE,
    ),
    "Q002": LintRule(
        id="Q002",
        name="missing-class-docstring",
        description="Class is missing a docstring",
        severity=Severity.WARNING,
        category=Category.STYLE,
    ),
    # Error handling
    "Q010": LintRule(
        id="Q010",
        name="missing-error-handler",
        description="Async function missing try/except for error handling",
        severity=Severity.WARNING,
        category=Category.SAFETY,
    ),
    "Q011": LintRule(
        id="Q011",
        name="bare-except",
        description="Bare except clause catches all exceptions",
        severity=Severity.ERROR,
        category=Category.CORRECTNESS,
    ),
    # Agent patterns
    "Q020": LintRule(
        id="Q020",
        name="tool-missing-decorator",
        description="Function appears to be a tool but lacks @tool decorator",
        severity=Severity.INFO,
        category=Category.AGENT_PATTERN,
    ),
    "Q021": LintRule(
        id="Q021",
        name="state-mutation-in-tool",
        description="Tool function mutates global state",
        severity=Severity.WARNING,
        category=Category.AGENT_PATTERN,
    ),
    "Q022": LintRule(
        id="Q022",
        name="infinite-loop-risk",
        description="Potential infinite loop detected in workflow",
        severity=Severity.ERROR,
        category=Category.CORRECTNESS,
    ),
    # Security
    "Q030": LintRule(
        id="Q030",
        name="hardcoded-secret",
        description="Potential hardcoded secret or API key",
        severity=Severity.ERROR,
        category=Category.SECURITY,
    ),
    "Q031": LintRule(
        id="Q031",
        name="unsafe-eval",
        description="Use of eval() or exec() is unsafe",
        severity=Severity.ERROR,
        category=Category.SECURITY,
    ),
    # Performance
    "Q040": LintRule(
        id="Q040",
        name="sync-in-async",
        description="Synchronous I/O in async function may block",
        severity=Severity.WARNING,
        category=Category.PERFORMANCE,
    ),
    # Best practices
    "Q050": LintRule(
        id="Q050",
        name="unused-import",
        description="Imported module is not used",
        severity=Severity.INFO,
        category=Category.BEST_PRACTICE,
    ),
    "Q051": LintRule(
        id="Q051",
        name="function-too-long",
        description="Function body exceeds recommended length",
        severity=Severity.INFO,
        category=Category.BEST_PRACTICE,
    ),
}


class StaticLinter:
    """
    Static linter for agent code.
    
    Performs fast, no-LLM analysis to detect:
    - Missing docstrings
    - Error handling issues
    - Agent-specific anti-patterns
    - Security vulnerabilities
    - Performance issues
    - Best practice violations
    
    SAFETY: Operates in read-only mode by default.
    Never modifies source files unless explicitly enabled.
    
    Usage:
        linter = StaticLinter()
        result = linter.lint_file("agent.py")
        
        # Or with parsed code
        result = linter.lint_parsed(parsed_code)
    """
    
    def __init__(self, config: LinterConfig | None = None):
        """Initialize the linter with optional configuration."""
        self.config = config or LinterConfig()
        self._parser = PythonParser()
        self._rules = BUILTIN_RULES.copy()
    
    def lint_file(self, file_path: str | Path) -> LintResult:
        """
        Lint a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            LintResult with all found issues
        """
        path = Path(file_path)
        parsed = self._parser.parse_file(path)
        return self.lint_parsed(parsed)
    
    def lint_string(self, source_code: str, file_name: str = "<string>") -> LintResult:
        """
        Lint Python source code string.
        
        Args:
            source_code: Python source code
            file_name: Name for reporting
            
        Returns:
            LintResult with all found issues
        """
        parsed = self._parser.parse_string(source_code, file_name)
        return self.lint_parsed(parsed)
    
    def lint_parsed(self, parsed: ParsedCode) -> LintResult:
        """
        Lint already-parsed code.
        
        Args:
            parsed: ParsedCode from PythonParser
            
        Returns:
            LintResult with all found issues
        """
        result = LintResult(file_path=parsed.source_file)
        
        # Track which rules we checked
        for rule_id in self._rules:
            if self.config.is_rule_enabled(rule_id):
                result.rules_checked.append(rule_id)
        
        # Run all enabled checks
        if self.config.is_rule_enabled("Q001"):
            self._check_function_docstrings(parsed, result)
        
        if self.config.is_rule_enabled("Q002"):
            self._check_class_docstrings(parsed, result)
        
        if self.config.is_rule_enabled("Q010"):
            self._check_async_error_handling(parsed, result)
        
        if self.config.is_rule_enabled("Q011"):
            self._check_bare_except(parsed, result)
        
        if self.config.is_rule_enabled("Q020"):
            self._check_tool_decorators(parsed, result)
        
        if self.config.is_rule_enabled("Q022"):
            self._check_infinite_loops(parsed, result)
        
        if self.config.is_rule_enabled("Q030"):
            self._check_hardcoded_secrets(parsed, result)
        
        if self.config.is_rule_enabled("Q031"):
            self._check_unsafe_eval(parsed, result)
        
        if self.config.is_rule_enabled("Q040"):
            self._check_sync_in_async(parsed, result)
        
        if self.config.is_rule_enabled("Q051"):
            self._check_function_length(parsed, result)
        
        return result
    
    def _add_issue(
        self,
        result: LintResult,
        rule_id: str,
        message: str,
        line: int,
        column: int = 0,
        suggestion: str | None = None,
        code_snippet: str | None = None,
    ) -> None:
        """Add an issue to the result."""
        rule = self._rules.get(rule_id)
        if not rule:
            return
        
        # Apply severity override if configured
        severity = self.config.severity_overrides.get(rule_id, rule.severity)
        
        result.issues.append(LintIssue(
            rule_id=rule_id,
            severity=severity,
            category=rule.category,
            message=message,
            file_path=result.file_path,
            line=line,
            column=column,
            suggestion=suggestion,
            code_snippet=code_snippet,
        ))
    
    def _check_function_docstrings(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for missing function docstrings."""
        for func in parsed.functions:
            if not func.docstring and not func.name.startswith("_"):
                loc = func.location
                line = loc.start_line if loc else 1
                self._add_issue(
                    result,
                    "Q001",
                    f"Function '{func.name}' is missing a docstring",
                    line,
                    suggestion=f'Add a docstring to describe what {func.name} does',
                )
        
        # Also check methods in classes
        for cls in parsed.classes:
            for method in cls.methods:
                if not method.docstring and not method.name.startswith("_"):
                    loc = method.location
                    line = loc.start_line if loc else 1
                    self._add_issue(
                        result,
                        "Q001",
                        f"Method '{cls.name}.{method.name}' is missing a docstring",
                        line,
                        suggestion=f'Add a docstring to describe what {method.name} does',
                    )
    
    def _check_class_docstrings(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for missing class docstrings."""
        for cls in parsed.classes:
            if not cls.docstring and not cls.name.startswith("_"):
                loc = cls.location
                line = loc.start_line if loc else 1
                self._add_issue(
                    result,
                    "Q002",
                    f"Class '{cls.name}' is missing a docstring",
                    line,
                    suggestion=f'Add a docstring to describe the purpose of {cls.name}',
                )
    
    def _check_async_error_handling(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check async functions for error handling."""
        for func in parsed.functions:
            if func.is_async and func.body_text:
                # Simple heuristic: async functions doing I/O should have try/except
                has_await = "await " in func.body_text
                has_try = "try:" in func.body_text or "try\n" in func.body_text
                
                if has_await and not has_try:
                    loc = func.location
                    line = loc.start_line if loc else 1
                    self._add_issue(
                        result,
                        "Q010",
                        f"Async function '{func.name}' has await but no try/except",
                        line,
                        suggestion="Add error handling for async operations",
                    )
    
    def _check_bare_except(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for bare except clauses."""
        # Check source code directly for "except:" pattern
        for i, line in enumerate(parsed.source_code.split("\n"), 1):
            stripped = line.strip()
            if stripped == "except:" or stripped.startswith("except: "):
                self._add_issue(
                    result,
                    "Q011",
                    "Bare 'except:' catches all exceptions including KeyboardInterrupt",
                    i,
                    suggestion="Use 'except Exception:' or catch specific exceptions",
                    code_snippet=line.strip(),
                )
    
    def _check_tool_decorators(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for functions that look like tools but lack decorator."""
        tool_indicators = ["execute", "run", "process", "handle", "fetch", "search", "create"]
        
        for func in parsed.functions:
            # Skip if already has tool decorator
            has_tool_decorator = any(
                d.name in ("tool", "function_tool", "agent_tool")
                for d in func.decorators
            )
            if has_tool_decorator:
                continue
            
            # Check if function name suggests it's a tool
            name_lower = func.name.lower()
            looks_like_tool = any(ind in name_lower for ind in tool_indicators)
            
            # Check if docstring mentions tool-like behavior
            doc_mentions_tool = False
            if func.docstring:
                doc_lower = func.docstring.lower()
                doc_mentions_tool = any(
                    word in doc_lower
                    for word in ["tool", "agent", "action", "capability"]
                )
            
            if looks_like_tool or doc_mentions_tool:
                loc = func.location
                line = loc.start_line if loc else 1
                self._add_issue(
                    result,
                    "Q020",
                    f"Function '{func.name}' appears to be a tool but lacks @tool decorator",
                    line,
                    suggestion="Add @tool decorator if this is an agent tool",
                )
    
    def _check_infinite_loops(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for potential infinite loops."""
        for func in parsed.functions:
            if not func.body_text:
                continue
            
            # Simple heuristic: while True without break
            if "while True:" in func.body_text or "while True\n" in func.body_text:
                has_break = "break" in func.body_text
                has_return = "return" in func.body_text
                
                if not has_break and not has_return:
                    loc = func.location
                    line = loc.start_line if loc else 1
                    self._add_issue(
                        result,
                        "Q022",
                        f"Function '{func.name}' has 'while True' without break/return",
                        line,
                        suggestion="Ensure loop has a termination condition",
                    )
    
    def _check_hardcoded_secrets(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for hardcoded secrets or API keys."""
        secret_patterns = [
            "api_key",
            "apikey",
            "secret",
            "password",
            "token",
            "auth_token",
            "private_key",
        ]
        
        for i, line in enumerate(parsed.source_code.split("\n"), 1):
            line_lower = line.lower()
            
            # Skip comments and imports
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("import "):
                continue
            
            # Check for assignments with secret-like names
            if "=" in line:
                for pattern in secret_patterns:
                    if pattern in line_lower:
                        # Check if it's assigning a string literal
                        if '= "' in line or "= '" in line:
                            # Skip if it's reading from env
                            if "os.environ" in line or "os.getenv" in line:
                                continue
                            self._add_issue(
                                result,
                                "Q030",
                                f"Potential hardcoded secret in assignment containing '{pattern}'",
                                i,
                                suggestion="Use environment variables or a secrets manager",
                                code_snippet=line.strip()[:60] + "...",
                            )
                            break
    
    def _check_unsafe_eval(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for use of eval() or exec()."""
        for i, line in enumerate(parsed.source_code.split("\n"), 1):
            stripped = line.strip()
            
            # Skip comments
            if stripped.startswith("#"):
                continue
            
            if "eval(" in line:
                self._add_issue(
                    result,
                    "Q031",
                    "Use of eval() is unsafe and can execute arbitrary code",
                    i,
                    suggestion="Use ast.literal_eval() for safe evaluation of literals",
                    code_snippet=stripped[:60],
                )
            
            if "exec(" in line:
                self._add_issue(
                    result,
                    "Q031",
                    "Use of exec() is unsafe and can execute arbitrary code",
                    i,
                    suggestion="Avoid exec() - use safer alternatives",
                    code_snippet=stripped[:60],
                )
    
    def _check_sync_in_async(self, parsed: ParsedCode, result: LintResult) -> None:
        """Check for synchronous I/O in async functions."""
        sync_io_patterns = [
            "open(",
            "requests.get",
            "requests.post",
            "urllib.request",
            "time.sleep(",
        ]
        
        for func in parsed.functions:
            if not func.is_async or not func.body_text:
                continue
            
            for pattern in sync_io_patterns:
                if pattern in func.body_text:
                    loc = func.location
                    line = loc.start_line if loc else 1
                    self._add_issue(
                        result,
                        "Q040",
                        f"Async function '{func.name}' uses sync I/O '{pattern.rstrip('(')}'",
                        line,
                        suggestion="Use async equivalents (aiofiles, httpx, asyncio.sleep)",
                    )
    
    def _check_function_length(
        self,
        parsed: ParsedCode,
        result: LintResult,
        max_lines: int = 50,
    ) -> None:
        """Check for functions that are too long."""
        for func in parsed.functions:
            if not func.body_text:
                continue
            
            line_count = func.body_text.count("\n") + 1
            if line_count > max_lines:
                loc = func.location
                line = loc.start_line if loc else 1
                self._add_issue(
                    result,
                    "Q051",
                    f"Function '{func.name}' is {line_count} lines (max recommended: {max_lines})",
                    line,
                    suggestion="Consider breaking into smaller functions",
                )


# Convenience functions
def lint_file(file_path: str | Path, config: LinterConfig | None = None) -> LintResult:
    """
    Lint a Python file.
    
    Args:
        file_path: Path to the Python file
        config: Optional linter configuration
        
    Returns:
        LintResult with all found issues
    """
    linter = StaticLinter(config)
    return linter.lint_file(file_path)


def lint_string(
    source_code: str,
    file_name: str = "<string>",
    config: LinterConfig | None = None,
) -> LintResult:
    """
    Lint Python source code string.
    
    Args:
        source_code: Python source code
        file_name: Name for reporting
        config: Optional linter configuration
        
    Returns:
        LintResult with all found issues
    """
    linter = StaticLinter(config)
    return linter.lint_string(source_code, file_name)
