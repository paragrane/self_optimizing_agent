"""
Storage for agent memory summaries.

Memory is bounded and managed to prevent unbounded growth.
"""

from typing import Protocol, Optional


class MemoryStore(Protocol):
    """Protocol for memory storage backends."""

    def save_memory(self, agent_id: str, memory_summary: str) -> None:
        """Save a memory summary for an agent."""
        ...

    def get_memory(self, agent_id: str) -> Optional[str]:
        """Get the current memory summary for an agent."""
        ...

    def clear_memory(self, agent_id: str) -> None:
        """Clear memory for an agent."""
        ...


class InMemoryMemoryStore:
    """In-memory implementation of MemoryStore.

    Suitable for development and testing.
    """

    def __init__(self, max_memory_chars: int = 10000):
        """Initialize with bounded memory size."""
        self._memories: dict[str, str] = {}
        self._max_memory_chars = max_memory_chars

    def save_memory(self, agent_id: str, memory_summary: str) -> None:
        """Save a memory summary for an agent.

        Truncates to max_memory_chars if needed.
        """
        if len(memory_summary) > self._max_memory_chars:
            memory_summary = memory_summary[-self._max_memory_chars:]
        self._memories[agent_id] = memory_summary

    def get_memory(self, agent_id: str) -> Optional[str]:
        """Get the current memory summary for an agent."""
        return self._memories.get(agent_id)

    def clear_memory(self, agent_id: str) -> None:
        """Clear memory for an agent."""
        if agent_id in self._memories:
            del self._memories[agent_id]
