"""
Structured logging for optimization events.

Provides audit trail for all context changes.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextflow.core.schemas import OptimizationLog


class OptimizationLogger:
    """Logs all optimization attempts and results."""

    def __init__(self, log_file: Optional[Path] = None):
        """Initialize logger.

        Args:
            log_file: Optional file path for persistent logging.
                     If None, logs are kept in memory only.
        """
        self._log_file = log_file
        self._logs: list[OptimizationLog] = []

    def log_optimization(self, log_entry: OptimizationLog) -> None:
        """Log an optimization attempt.

        Args:
            log_entry: OptimizationLog entry
        """
        self._logs.append(log_entry)

        if self._log_file:
            self._write_to_file(log_entry)

    def _write_to_file(self, log_entry: OptimizationLog) -> None:
        """Write log entry to file in JSONL format."""
        with open(self._log_file, "a") as f:
            f.write(log_entry.model_dump_json() + "\n")

    def get_logs(
        self, agent_id: Optional[str] = None, limit: Optional[int] = None
    ) -> list[OptimizationLog]:
        """Retrieve logs.

        Args:
            agent_id: Filter by agent ID (optional)
            limit: Maximum number of logs to return (optional)

        Returns:
            List of OptimizationLog entries
        """
        logs = self._logs

        if agent_id:
            logs = [log for log in logs if log.agent_id == agent_id]

        # Most recent first
        logs = list(reversed(logs))

        if limit:
            logs = logs[:limit]

        return logs

    def get_success_rate(self, agent_id: str) -> float:
        """Calculate success rate for an agent's optimizations.

        Args:
            agent_id: Agent identifier

        Returns:
            Success rate (0.0 to 1.0)
        """
        agent_logs = [log for log in self._logs if log.agent_id == agent_id]

        if not agent_logs:
            return 0.0

        successful = sum(1 for log in agent_logs if log.applied)
        return successful / len(agent_logs)

    def print_summary(self, agent_id: Optional[str] = None) -> None:
        """Print a human-readable summary of logs.

        Args:
            agent_id: Filter by agent ID (optional)
        """
        logs = self.get_logs(agent_id=agent_id)

        if not logs:
            print("No optimization logs found.")
            return

        print(f"\n{'='*60}")
        print(f"Optimization Summary ({len(logs)} entries)")
        print(f"{'='*60}\n")

        for log in logs:
            status = "✓ APPLIED" if log.applied else "✗ REJECTED"
            print(f"[{log.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {log.agent_id}")
            print(f"  Status: {status}")
            print(f"  Version: {log.version_from} → {log.version_to or 'N/A'}")

            if log.patch:
                print(f"  Rationale: {log.patch.rationale}")

            if log.safety_result and not log.safety_result.passed:
                print(f"  Safety violations: {', '.join(log.safety_result.violations)}")

            if log.eval_report:
                print(f"  Eval: {log.eval_report.pass_fail.upper()}")

            if log.error:
                print(f"  Error: {log.error}")

            print()
