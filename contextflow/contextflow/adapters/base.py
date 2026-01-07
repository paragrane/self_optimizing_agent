"""
Base adapter protocol for framework integration.

All framework adapters must implement this protocol.
"""

from typing import Protocol, Any, Optional


class BaseAdapter(Protocol):
    """Protocol that all framework adapters must implement.

    Provides a uniform interface for reading and updating agent context
    across different frameworks (LangGraph, CrewAI, custom).
    """

    def get_prompt(self, agent_handle: Any) -> str:
        """Get current system prompt from agent.

        Args:
            agent_handle: Framework-specific agent object

        Returns:
            Current system prompt string
        """
        ...

    def set_prompt(self, agent_handle: Any, prompt: str) -> None:
        """Update agent's system prompt.

        Args:
            agent_handle: Framework-specific agent object
            prompt: New system prompt
        """
        ...

    def get_memory(self, agent_handle: Any) -> Optional[str]:
        """Get agent's memory summary.

        Args:
            agent_handle: Framework-specific agent object

        Returns:
            Memory summary or None
        """
        ...

    def set_memory(self, agent_handle: Any, memory_summary: str) -> None:
        """Update agent's memory summary.

        Args:
            agent_handle: Framework-specific agent object
            memory_summary: New memory summary
        """
        ...

    def get_tool_allowlist(self, agent_handle: Any) -> list[str]:
        """Get list of allowed tools for the agent.

        Args:
            agent_handle: Framework-specific agent object

        Returns:
            List of tool names
        """
        ...

    def get_policies(self, agent_handle: Any) -> dict[str, str]:
        """Get safety/privacy/compliance policies.

        Args:
            agent_handle: Framework-specific agent object

        Returns:
            Dictionary of policy name -> policy text
        """
        ...


class SimpleAdapter:
    """Simple adapter for custom agents or testing.

    Assumes agent_handle is a dict-like object with:
    - 'system_prompt'
    - 'memory' (optional)
    - 'tools' (optional)
    - 'policies' (optional)
    """

    def get_prompt(self, agent_handle: Any) -> str:
        """Get current system prompt."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("system_prompt", "")
        return getattr(agent_handle, "system_prompt", "")

    def set_prompt(self, agent_handle: Any, prompt: str) -> None:
        """Update system prompt."""
        if isinstance(agent_handle, dict):
            agent_handle["system_prompt"] = prompt
        else:
            setattr(agent_handle, "system_prompt", prompt)

    def get_memory(self, agent_handle: Any) -> Optional[str]:
        """Get memory summary."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("memory")
        return getattr(agent_handle, "memory", None)

    def set_memory(self, agent_handle: Any, memory_summary: str) -> None:
        """Update memory summary."""
        if isinstance(agent_handle, dict):
            agent_handle["memory"] = memory_summary
        else:
            setattr(agent_handle, "memory", memory_summary)

    def get_tool_allowlist(self, agent_handle: Any) -> list[str]:
        """Get tool allowlist."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("tools", [])
        return getattr(agent_handle, "tools", [])

    def get_policies(self, agent_handle: Any) -> dict[str, str]:
        """Get policies."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("policies", {})
        return getattr(agent_handle, "policies", {})
