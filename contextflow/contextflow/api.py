"""
ContextFlow: Main facade for the library.

Provides a simple, high-level API for context optimization.
"""

from typing import Optional, Any
from contextflow.core.scoa import SCOA, AdapterProtocol
from contextflow.core.schemas import AgentProfile, AgentTrace, EvalReport
from contextflow.core.logging import OptimizationLogger
from contextflow.storage.prompt_store import InMemoryPromptStore
from contextflow.storage.memory_store import InMemoryMemoryStore
from contextflow.storage.trace_store import InMemoryTraceStore


class ContextFlow:
    """Main facade for ContextFlow library.

    Provides a simple API for:
    - Registering agents
    - Ingesting traces
    - Running optimization
    - Rolling back changes

    Example:
        cf = ContextFlow()
        cf.register_agent(profile, adapter, agent_handle)
        cf.ingest_traces(traces)
        reports = cf.run_once()
        cf.rollback("agent_1", "v2")
    """

    def __init__(
        self,
        prompt_store: Optional[InMemoryPromptStore] = None,
        memory_store: Optional[InMemoryMemoryStore] = None,
        trace_store: Optional[InMemoryTraceStore] = None,
        logger: Optional[OptimizationLogger] = None,
    ):
        """Initialize ContextFlow.

        Args:
            prompt_store: Optional custom prompt store
            memory_store: Optional custom memory store
            trace_store: Optional custom trace store
            logger: Optional custom logger
        """
        self._scoa = SCOA(
            prompt_store=prompt_store,
            memory_store=memory_store,
            trace_store=trace_store,
            logger=logger,
        )

    def register_agent(
        self,
        profile: AgentProfile,
        adapter: AdapterProtocol,
        agent_handle: Any = None,
    ) -> None:
        """Register an agent for optimization.

        Args:
            profile: Agent profile with metadata
            adapter: Framework adapter (LangGraph, CrewAI, etc.)
            agent_handle: Framework-specific agent object (optional)
        """
        self._scoa.register_agent(profile, adapter, agent_handle or profile)

    def ingest_traces(self, traces: list[AgentTrace]) -> None:
        """Ingest execution traces for analysis.

        Args:
            traces: List of agent execution traces
        """
        self._scoa.ingest_traces(traces)

    def run_once(self, agent_ids: Optional[list[str]] = None) -> list[EvalReport]:
        """Run optimization for specified agents sequentially.

        Args:
            agent_ids: List of agent IDs to optimize. If None, optimize all.

        Returns:
            List of EvalReport objects, one per agent
        """
        if agent_ids is None:
            # Optimize all registered agents
            agent_ids = list(self._scoa._agents.keys())

        reports = []
        for agent_id in agent_ids:
            try:
                report = self._scoa.optimize_agent(agent_id)
                reports.append(report)
            except Exception as e:
                # Log error and continue
                print(f"Error optimizing {agent_id}: {e}")

        return reports

    def rollback(self, agent_id: str, version: str) -> None:
        """Rollback an agent to a previous version.

        Args:
            agent_id: Agent to rollback
            version: Version tag to restore
        """
        self._scoa.rollback(agent_id, version)

    def get_logger(self) -> OptimizationLogger:
        """Get the optimization logger for inspection."""
        return self._scoa.get_logger()

    def get_trace_store(self) -> InMemoryTraceStore:
        """Get the trace store for inspection."""
        return self._scoa.get_trace_store()

    def get_prompt_store(self) -> InMemoryPromptStore:
        """Get the prompt store for inspection."""
        return self._scoa.get_prompt_store()
