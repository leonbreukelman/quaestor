"""
HTML Report Generator for Quaestor.

Generates human-readable HTML reports with coverage visualization,
verdict display, and test execution timelines.

Part of Phase 5: Coverage & Reporting.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, Template

from quaestor.coverage.tracker import CoverageDimension, CoverageReport
from quaestor.evaluation.models import Verdict
from quaestor.redteam.models import RedTeamReport

# Embedded HTML template with dark mode support
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quaestor Test Report - {{ report_title }}</title>
    <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f5f5;
            --text-primary: #333333;
            --text-secondary: #666666;
            --border-color: #ddd;
            --critical: #dc2626;
            --high: #ea580c;
            --medium: #f59e0b;
            --low: #10b981;
            --info: #3b82f6;
            --success: #22c55e;
            --shadow: rgba(0, 0, 0, 0.1);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --bg-primary: #1f2937;
                --bg-secondary: #111827;
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --border-color: #374151;
                --shadow: rgba(0, 0, 0, 0.3);
            }
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            background: var(--bg-primary);
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px var(--shadow);
        }

        h1 {
            color: var(--text-primary);
            margin-bottom: 10px;
        }

        .metadata {
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .section {
            background: var(--bg-primary);
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px var(--shadow);
        }

        h2 {
            color: var(--text-primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }

        .stat-card {
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid var(--border-color);
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: var(--text-primary);
        }

        .stat-label {
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 5px;
        }

        .coverage-bar {
            width: 100%;
            height: 30px;
            background: var(--bg-secondary);
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
            position: relative;
        }

        .coverage-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success), var(--info));
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.85em;
        }

        .dimension-coverage {
            margin: 15px 0;
        }

        .dimension-label {
            font-weight: 600;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
        }

        .verdict-list {
            margin-top: 20px;
        }

        .verdict-item {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 4px solid;
        }

        .verdict-critical {
            background: rgba(220, 38, 38, 0.1);
            border-color: var(--critical);
        }

        .verdict-high {
            background: rgba(234, 88, 12, 0.1);
            border-color: var(--high);
        }

        .verdict-medium {
            background: rgba(245, 158, 11, 0.1);
            border-color: var(--medium);
        }

        .verdict-low {
            background: rgba(16, 185, 129, 0.1);
            border-color: var(--low);
        }

        .verdict-info {
            background: rgba(59, 130, 246, 0.1);
            border-color: var(--info);
        }

        .verdict-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .verdict-title {
            font-weight: 600;
            font-size: 1.1em;
        }

        .verdict-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-critical {
            background: var(--critical);
            color: white;
        }

        .badge-high {
            background: var(--high);
            color: white;
        }

        .badge-medium {
            background: var(--medium);
            color: white;
        }

        .badge-low {
            background: var(--low);
            color: white;
        }

        .badge-info {
            background: var(--info);
            color: white;
        }

        .verdict-description {
            color: var(--text-secondary);
            margin-bottom: 10px;
        }

        .verdict-score {
            font-size: 0.9em;
            color: var(--text-secondary);
        }

        .timeline {
            margin-top: 20px;
        }

        .timeline-item {
            padding: 12px;
            border-left: 3px solid var(--info);
            margin-left: 10px;
            margin-bottom: 15px;
            background: var(--bg-secondary);
            border-radius: 0 6px 6px 0;
        }

        .timeline-time {
            font-size: 0.85em;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }

        .timeline-content {
            color: var(--text-primary);
        }

        .gap-list {
            margin-top: 15px;
        }

        .gap-item {
            padding: 10px 15px;
            background: rgba(245, 158, 11, 0.1);
            border-left: 3px solid var(--medium);
            margin-bottom: 10px;
            border-radius: 0 4px 4px 0;
        }

        .gap-dimension {
            font-weight: 600;
            color: var(--text-primary);
        }

        .gap-values {
            font-size: 0.9em;
            color: var(--text-secondary);
            margin-top: 5px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: var(--text-secondary);
            font-size: 0.9em;
        }

        .empty-state {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }

        /* Red Team Section Styles */
        .attack-list {
            margin-top: 20px;
        }

        .attack-item {
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 4px solid;
            background: var(--bg-secondary);
        }

        .attack-success {
            border-color: var(--critical);
            background: rgba(220, 38, 38, 0.1);
        }

        .attack-failed {
            border-color: var(--success);
        }

        .attack-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .attack-title {
            font-weight: 600;
            font-size: 1.1em;
        }

        .attack-badge {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
        }

        .attack-badge-success {
            background: var(--critical);
            color: white;
        }

        .attack-badge-failed {
            background: var(--success);
            color: white;
        }

        .attack-meta {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.9em;
            color: var(--text-secondary);
        }

        .attack-details {
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .vulnerability-tag {
            display: inline-block;
            padding: 2px 8px;
            margin-right: 5px;
            margin-bottom: 5px;
            border-radius: 4px;
            font-size: 0.8em;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
        }

        .severity-critical { color: var(--critical); border-color: var(--critical); }
        .severity-high { color: var(--high); border-color: var(--high); }
        .severity-medium { color: var(--medium); border-color: var(--medium); }
        .severity-low { color: var(--low); border-color: var(--low); }

        .progress-ring {
            width: 120px;
            height: 120px;
            margin: 0 auto;
        }

        .progress-ring circle {
            fill: none;
            stroke-width: 10;
        }

        .progress-ring .bg {
            stroke: var(--bg-secondary);
        }

        .progress-ring .progress {
            stroke: var(--success);
            stroke-linecap: round;
            transform: rotate(-90deg);
            transform-origin: 50% 50%;
            transition: stroke-dashoffset 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ report_title }}</h1>
            <div class="metadata">
                <div>Generated: {{ generation_time }}</div>
                {% if agent_name %}
                <div>Agent: {{ agent_name }}</div>
                {% endif %}
            </div>
        </header>

        {% if coverage_report %}
        <section class="section">
            <h2>📊 Coverage Summary</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ "%.1f"|format(coverage_report.overall_coverage) }}%</div>
                    <div class="stat-label">Overall Coverage</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ coverage_report.dimensions[CoverageDimension.TOOL].coverage_count if CoverageDimension.TOOL in coverage_report.dimensions else 0 }}</div>
                    <div class="stat-label">Tools Covered</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ coverage_report.dimensions[CoverageDimension.STATE].coverage_count if CoverageDimension.STATE in coverage_report.dimensions else 0 }}</div>
                    <div class="stat-label">States Covered</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ coverage_report.dimensions[CoverageDimension.TRANSITION].coverage_count if CoverageDimension.TRANSITION in coverage_report.dimensions else 0 }}</div>
                    <div class="stat-label">Transitions Covered</div>
                </div>
            </div>

            {% set tool_cov = coverage_report.dimensions.get(CoverageDimension.TOOL) %}
            {% if tool_cov %}
            <div class="dimension-coverage">
                <div class="dimension-label">
                    <span>Tool Coverage</span>
                    <span>{{ "%.1f"|format(tool_cov.coverage_percentage) }}%</span>
                </div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {{ tool_cov.coverage_percentage }}%">
                        {{ tool_cov.coverage_count }}/{{ tool_cov.total_items }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% set state_cov = coverage_report.dimensions.get(CoverageDimension.STATE) %}
            {% if state_cov %}
            <div class="dimension-coverage">
                <div class="dimension-label">
                    <span>State Coverage</span>
                    <span>{{ "%.1f"|format(state_cov.coverage_percentage) }}%</span>
                </div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {{ state_cov.coverage_percentage }}%">
                        {{ state_cov.coverage_count }}/{{ state_cov.total_items }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% set transition_cov = coverage_report.dimensions.get(CoverageDimension.TRANSITION) %}
            {% if transition_cov %}
            <div class="dimension-coverage">
                <div class="dimension-label">
                    <span>Transition Coverage</span>
                    <span>{{ "%.1f"|format(transition_cov.coverage_percentage) }}%</span>
                </div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {{ transition_cov.coverage_percentage }}%">
                        {{ transition_cov.coverage_count }}/{{ transition_cov.total_items }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% set invariant_cov = coverage_report.dimensions.get(CoverageDimension.INVARIANT) %}
            {% if invariant_cov %}
            <div class="dimension-coverage">
                <div class="dimension-label">
                    <span>Invariant Coverage</span>
                    <span>{{ "%.1f"|format(invariant_cov.coverage_percentage) }}%</span>
                </div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {{ invariant_cov.coverage_percentage }}%">
                        {{ invariant_cov.coverage_count }}/{{ invariant_cov.total_items }}
                    </div>
                </div>
            </div>
            {% endif %}

            {% if coverage_report.has_gaps %}
            <h3 style="margin-top: 30px; color: var(--text-primary);">Coverage Gaps</h3>
            <div class="gap-list">
                {% for dimension, dim_coverage in coverage_report.dimensions.items() %}
                {% if dim_coverage.uncovered_items %}
                <div class="gap-item">
                    <div class="gap-dimension">{{ dimension.value|capitalize }}</div>
                    <div class="gap-values">{{ dim_coverage.uncovered_items|list|sort|join(", ") }}</div>
                </div>
                {% endif %}
                {% endfor %}
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if verdicts %}
        <section class="section">
            <h2>⚖️ Evaluation Verdicts</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ verdicts|length }}</div>
                    <div class="stat-label">Total Verdicts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ verdict_counts.critical }}</div>
                    <div class="stat-label">Critical</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ verdict_counts.high }}</div>
                    <div class="stat-label">High</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ verdict_counts.medium }}</div>
                    <div class="stat-label">Medium</div>
                </div>
            </div>

            <div class="verdict-list">
                {% for verdict in verdicts %}
                <div class="verdict-item verdict-{{ verdict.severity.value }}">
                    <div class="verdict-header">
                        <div class="verdict-title">{{ verdict.title }}</div>
                        <span class="verdict-badge badge-{{ verdict.severity.value }}">{{ verdict.severity.value }}</span>
                    </div>
                    <div class="verdict-description">{{ verdict.description }}</div>
                    {% if verdict.score is not none %}
                    <div class="verdict-score">Score: {{ "%.2f"|format(verdict.score) }}</div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
        </section>
        {% else %}
        <section class="section">
            <h2>⚖️ Evaluation Verdicts</h2>
            <div class="empty-state">
                <p>No verdicts generated yet.</p>
            </div>
        </section>
        {% endif %}

        {% if redteam_report %}
        <section class="section">
            <h2>🔴 Red Team Assessment</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{{ redteam_report.total_attacks }}</div>
                    <div class="stat-label">Total Attacks</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ redteam_report.successful_attacks }}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" style="color: {% if redteam_report.success_rate > 30 %}var(--critical){% elif redteam_report.success_rate > 10 %}var(--medium){% else %}var(--success){% endif %}">{{ "%.1f"|format(redteam_report.success_rate) }}%</div>
                    <div class="stat-label">Attack Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{{ redteam_report.vulnerabilities_found|length }}</div>
                    <div class="stat-label">Vulnerability Types</div>
                </div>
            </div>

            {% if redteam_report.vulnerabilities_found %}
            <h3 style="margin-top: 20px; color: var(--text-primary);">Vulnerabilities Detected</h3>
            <div style="margin-top: 10px;">
                {% for vuln in redteam_report.vulnerabilities_found %}
                <span class="vulnerability-tag">{{ vuln.value }}</span>
                {% endfor %}
            </div>
            {% endif %}

            {% if redteam_report.results %}
            <h3 style="margin-top: 30px; color: var(--text-primary);">Attack Results</h3>
            <div class="attack-list">
                {% for result in redteam_report.results %}
                <div class="attack-item attack-{{ 'success' if result.success else 'failed' }}">
                    <div class="attack-header">
                        <div class="attack-title">{{ result.vulnerability_type }}</div>
                        <span class="attack-badge attack-badge-{{ 'success' if result.success else 'failed' }}">
                            {{ 'EXPLOITED' if result.success else 'BLOCKED' }}
                        </span>
                    </div>
                    <div class="attack-meta">
                        <span>Method: {{ result.attack_method }}</span>
                        <span class="severity-{{ result.severity }}">Severity: {{ result.severity|upper }}</span>
                        <span>Confidence: {{ "%.0f"|format(result.confidence * 100) }}%</span>
                    </div>
                    {% if result.input_prompt %}
                    <div class="attack-details">
                        <strong>Input:</strong>
{{ result.input_prompt[:500] }}{% if result.input_prompt|length > 500 %}...{% endif %}
                    </div>
                    {% endif %}
                    {% if result.agent_response and result.success %}
                    <div class="attack-details">
                        <strong>Response:</strong>
{{ result.agent_response[:500] }}{% if result.agent_response|length > 500 %}...{% endif %}
                    </div>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </section>
        {% endif %}

        {% if timeline %}
        <section class="section">
            <h2>⏱️ Test Execution Timeline</h2>
            <div class="timeline">
                {% for event in timeline %}
                <div class="timeline-item">
                    <div class="timeline-time">{{ event.timestamp }}</div>
                    <div class="timeline-content">{{ event.description }}</div>
                </div>
                {% endfor %}
            </div>
        </section>
        {% endif %}

        <footer>
            Generated by Quaestor - AI Agent Testing Framework
        </footer>
    </div>
</body>
</html>
"""


class HTMLReportGenerator:
    """
    Generate HTML reports from test results.

    Creates self-contained HTML reports with coverage visualization,
    verdict displays, and test execution timelines.
    """

    def __init__(self, template: str | None = None):
        """
        Initialize HTML report generator.

        Args:
            template: Optional custom Jinja2 template string
        """
        self.template_str = template or HTML_TEMPLATE
        self.env = Environment(autoescape=True)
        self.template: Template = self.env.from_string(self.template_str)

    def generate(
        self,
        output_path: str | Path,
        report_title: str = "Quaestor Test Report",
        agent_name: str | None = None,
        coverage_report: CoverageReport | None = None,
        verdicts: list[Verdict] | None = None,
        timeline: list[dict[str, Any]] | None = None,
        redteam_report: RedTeamReport | None = None,
    ) -> None:
        """
        Generate HTML report and write to file.

        Args:
            output_path: Path to write HTML report
            report_title: Title for the report
            agent_name: Optional agent name
            coverage_report: Optional coverage report data
            verdicts: Optional list of verdicts
            timeline: Optional timeline events
            redteam_report: Optional red team assessment report
        """
        # Count verdicts by severity
        verdict_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        if verdicts:
            for verdict in verdicts:
                verdict_counts[verdict.severity.value] += 1

        # Render template
        html_content = self.template.render(
            report_title=report_title,
            agent_name=agent_name,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            coverage_report=coverage_report,
            verdicts=verdicts or [],
            verdict_counts=verdict_counts,
            timeline=timeline or [],
            redteam_report=redteam_report,
            CoverageDimension=CoverageDimension,
        )

        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

    def generate_from_data(
        self,
        output_path: str | Path,
        data: dict[str, Any],
    ) -> None:
        """
        Generate HTML report from a data dictionary.

        Args:
            output_path: Path to write HTML report
            data: Dictionary with report data
        """
        self.generate(
            output_path=output_path,
            report_title=data.get("title", "Quaestor Test Report"),
            agent_name=data.get("agent_name"),
            coverage_report=data.get("coverage"),
            verdicts=data.get("verdicts"),
            timeline=data.get("timeline"),
            redteam_report=data.get("redteam"),
        )
