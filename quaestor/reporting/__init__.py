"""
Quaestor Reporting.

Generate HTML, SARIF, and console reports.
"""

from quaestor.reporting.html import HTMLReportGenerator
from quaestor.reporting.sarif import (
    SARIFLocation,
    SARIFReport,
    SARIFResult,
    SARIFRule,
    create_sarif_from_issues,
    create_sarif_from_verdicts,
)

__all__ = [
    "HTMLReportGenerator",
    "SARIFLocation",
    "SARIFReport",
    "SARIFResult",
    "SARIFRule",
    "create_sarif_from_issues",
    "create_sarif_from_verdicts",
]
