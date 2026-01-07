"""
Self-Context Optimization Agent (SCOA) orchestrator.

Coordinates the sequential optimization loop for registered agents.
"""

from typing import Optional, Protocol, Any
from contextflow.core.schemas import (
    AgentProfile,
    AgentTrace,
    EvalReport,
    OptimizationLog,
)
from contextflow.core.optimizer import PromptOptimizer
from contextflow.core.eval import Evaluator
from contextflow.core.safety import SafetyValidator
from contextflow.core.logging import OptimizationLogger
from contextflow.core.patching import PromptPatcher
from contextflow.storage.prompt_store import InMemoryPromptStore
from contextflow.storage.memory_store import InMemoryMemoryStore
from contextflow.storage.trace_store import InMemoryTraceStore


class AdapterProtocol(Protocol):
    """Protocol that all framework adapters must implement."""

    def get_prompt(self, agent_handle: Any) -> str:
        """Get current system prompt from agent."""
        ...

    def set_prompt(self, agent_handle: Any, prompt: str) -> None:
        """Update agent's system prompt."""
        ...

    def get_memory(self, agent_handle: Any) -> Optional[str]:
        """Get agent's memory summary."""
        ...

    def set_memory(self, agent_handle: Any, memory_summary: str) -> None:
        """Update agent's memory summary."""
        ...

    def get_tool_allowlist(self, agent_handle: Any) -> list[str]:
        """Get list of allowed tools."""
        ...

    def get_policies(self, agent_handle: Any) -> dict[str, str]:
        """Get safety/privacy/compliance policies."""
        ...


class SCOA:
    """Self-Context Optimization Agent.

    Orchestrates the sequential optimization loop:
    1. Collect traces
    2. Diagnose issues
    3. Propose patch
    4. Validate safety
    5. Evaluate counterfactually
    6. Apply if passed
    7. Log results
    """

    def __init__(
        self,
        prompt_store: Optional[InMemoryPromptStore] = None,
        memory_store: Optional[InMemoryMemoryStore] = None,
        trace_store: Optional[InMemoryTraceStore] = None,
        logger: Optional[OptimizationLogger] = None,
    ):
        """Initialize SCOA.

        Args:
            prompt_store: Storage for versioned prompts
            memory_store: Storage for agent memories
            trace_store: Storage for execution traces
            logger: Optimization event logger
        """
        self._prompt_store = prompt_store or InMemoryPromptStore()
        self._memory_store = memory_store or InMemoryMemoryStore()
        self._trace_store = trace_store or InMemoryTraceStore()
        self._logger = logger or OptimizationLogger()

        self._optimizer = PromptOptimizer()
        self._evaluator = Evaluator()
        self._safety = SafetyValidator()
        self._patcher = PromptPatcher()

        # Registered agents: agent_id -> (profile, adapter, handle)
        self._agents: dict[str, tuple[AgentProfile, AdapterProtocol, Any]] = {}

    def register_agent(
        self,
        profile: AgentProfile,
        adapter: AdapterProtocol,
        agent_handle: Any,
    ) -> None:
        """Register an agent with SCOA.

        Args:
            profile: Agent profile with metadata
            adapter: Framework adapter for this agent
            agent_handle: Framework-specific agent object
        """
        # Store initial prompt
        self._prompt_store.save_prompt(
            profile.id, profile.current_system_prompt, profile.current_version
        )

        # Store initial memory if present
        if profile.memory_summary:
            self._memory_store.save_memory(profile.id, profile.memory_summary)

        # Register
        self._agents[profile.id] = (profile, adapter, agent_handle)

    def ingest_traces(self, traces: list[AgentTrace]) -> None:
        """Ingest execution traces for analysis.

        Args:
            traces: List of agent execution traces
        """
        for trace in traces:
            self._trace_store.add_trace(trace)

    def optimize_agent(self, agent_id: str) -> EvalReport:
        """Optimize a single agent.

        Runs the full optimization loop:
        collect → diagnose → propose → validate → evaluate → apply → log

        Args:
            agent_id: Agent to optimize

        Returns:
            EvalReport with results

        Raises:
            ValueError: If agent not registered
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")

        profile, adapter, agent_handle = self._agents[agent_id]

        log_entry = OptimizationLog(
            agent_id=agent_id,
            version_from=profile.current_version,
        )

        try:
            # Step 1: Collect traces
            traces = self._trace_store.get_traces(agent_id, limit=50)

            if not traces:
                # No traces yet; skip optimization
                log_entry.error = "No traces available for optimization"
                self._logger.log_optimization(log_entry)
                return EvalReport(
                    agent_id=agent_id,
                    metrics_before=self._evaluator._simulate_task_execution(
                        profile, profile.current_system_prompt, []
                    ),
                    metrics_after=self._evaluator._simulate_task_execution(
                        profile, profile.current_system_prompt, []
                    ),
                    pass_fail="fail",
                    notes="No traces available",
                )

            # Step 2 & 3: Diagnose and propose patch
            next_version = self._bump_version(profile.current_version)
            patch = self._optimizer.optimize_prompt(
                profile, traces, profile.current_version, next_version
            )
            log_entry.patch = patch
            log_entry.version_to = next_version

            # Step 4: Safety validation
            safety_result = self._safety.validate_patch(profile, patch)
            log_entry.safety_result = safety_result

            if not safety_result.passed:
                log_entry.applied = False
                self._logger.log_optimization(log_entry)
                return EvalReport(
                    agent_id=agent_id,
                    metrics_before=self._evaluator._simulate_task_execution(
                        profile, profile.current_system_prompt, []
                    ),
                    metrics_after=self._evaluator._simulate_task_execution(
                        profile, patch.new_prompt, []
                    ),
                    pass_fail="fail",
                    notes=f"Safety check failed: {', '.join(safety_result.violations)}",
                )

            # Step 5: Counterfactual evaluation
            eval_report = self._evaluator.evaluate_patch(
                profile,
                profile.current_system_prompt,
                patch.new_prompt,
                traces,
            )
            log_entry.eval_report = eval_report

            if eval_report.pass_fail == "fail":
                log_entry.applied = False
                self._logger.log_optimization(log_entry)
                return eval_report

            # Step 6: Apply patch
            self._apply_patch(profile, adapter, agent_handle, patch, next_version)
            log_entry.applied = True
            eval_report.applied_version = next_version

            # Step 7: Log
            self._logger.log_optimization(log_entry)

            return eval_report

        except Exception as e:
            log_entry.error = str(e)
            log_entry.applied = False
            self._logger.log_optimization(log_entry)
            raise

    def _apply_patch(
        self,
        profile: AgentProfile,
        adapter: AdapterProtocol,
        agent_handle: Any,
        patch,
        new_version: str,
    ) -> None:
        """Apply an approved patch to the agent.

        Updates:
        - Prompt store (versioned)
        - Agent via adapter
        - Profile in memory
        """
        # Save to prompt store
        self._prompt_store.save_prompt(profile.id, patch.new_prompt, new_version)
        self._prompt_store.save_patch(patch)

        # Update agent via adapter
        adapter.set_prompt(agent_handle, patch.new_prompt)

        # Update profile
        profile.current_system_prompt = patch.new_prompt
        profile.current_version = new_version

    def rollback(self, agent_id: str, version: str) -> None:
        """Rollback an agent to a previous prompt version.

        Args:
            agent_id: Agent to rollback
            version: Version tag to restore

        Raises:
            ValueError: If agent not registered or version not found
        """
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not registered")

        profile, adapter, agent_handle = self._agents[agent_id]

        # Retrieve prompt from store
        old_prompt = self._prompt_store.get_prompt(agent_id, version)
        if old_prompt is None:
            raise ValueError(f"Version {version} not found for agent {agent_id}")

        # Update agent
        adapter.set_prompt(agent_handle, old_prompt)

        # Update profile
        profile.current_system_prompt = old_prompt
        profile.current_version = version

        # Save as current version in store
        self._prompt_store.save_prompt(agent_id, old_prompt, version)

    def _bump_version(self, current_version: str) -> str:
        """Increment version tag.

        Args:
            current_version: Current version (e.g., 'v1')

        Returns:
            Next version (e.g., 'v2')
        """
        if current_version.startswith("v"):
            num = int(current_version[1:])
            return f"v{num + 1}"
        else:
            return "v2"

    def get_logger(self) -> OptimizationLogger:
        """Get the optimization logger."""
        return self._logger

    def get_trace_store(self) -> InMemoryTraceStore:
        """Get the trace store."""
        return self._trace_store

    def get_prompt_store(self) -> InMemoryPromptStore:
        """Get the prompt store."""
        return self._prompt_store
