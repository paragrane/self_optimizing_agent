"""
Storage for agent execution traces.

Traces are used to diagnose issues and drive optimization.
"""

from typing import Protocol, Optional
from contextflow.core.schemas import AgentTrace


class TraceStore(Protocol):
    """Protocol for trace storage backends."""

    def add_trace(self, trace: AgentTrace) -> None:
        """Add an execution trace."""
        ...

    def get_traces(
        self, agent_id: str, limit: Optional[int] = None
    ) -> list[AgentTrace]:
        """Get traces for an agent, most recent first."""
        ...

    def get_error_traces(self, agent_id: str, limit: int = 10) -> list[AgentTrace]:
        """Get traces with errors for an agent."""
        ...

    def clear_traces(self, agent_id: str) -> None:
        """Clear all traces for an agent."""
        ...


class InMemoryTraceStore:
    """In-memory implementation of TraceStore.

    Suitable for development and testing.
    """

    def __init__(self, max_traces_per_agent: int = 1000):
        """Initialize with bounded trace storage."""
        self._traces: dict[str, list[AgentTrace]] = {}
        self._max_traces_per_agent = max_traces_per_agent

    def add_trace(self, trace: AgentTrace) -> None:
        """Add an execution trace."""
        if trace.agent_id not in self._traces:
            self._traces[trace.agent_id] = []

        self._traces[trace.agent_id].append(trace)

        # Bound size: keep most recent traces
        if len(self._traces[trace.agent_id]) > self._max_traces_per_agent:
            self._traces[trace.agent_id] = self._traces[trace.agent_id][
                -self._max_traces_per_agent :
            ]

    def get_traces(
        self, agent_id: str, limit: Optional[int] = None
    ) -> list[AgentTrace]:
        """Get traces for an agent, most recent first."""
        traces = self._traces.get(agent_id, [])
        # Reverse to get most recent first
        traces = list(reversed(traces))
        if limit:
            traces = traces[:limit]
        return traces

    def get_error_traces(self, agent_id: str, limit: int = 10) -> list[AgentTrace]:
        """Get traces with errors for an agent."""
        all_traces = self._traces.get(agent_id, [])
        error_traces = [t for t in all_traces if t.outcome == "error" or t.errors]
        # Most recent first
        error_traces = list(reversed(error_traces))
        return error_traces[:limit]

    def clear_traces(self, agent_id: str) -> None:
        """Clear all traces for an agent."""
        if agent_id in self._traces:
            del self._traces[agent_id]
