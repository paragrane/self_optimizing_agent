"""
LangGraph adapter for ContextFlow.

Integrates with LangGraph state graphs and provides utilities
for adding SCOA optimization nodes.
"""

from typing import Any, Optional


class LangGraphAdapter:
    """Adapter for LangGraph agents.

    Assumes LangGraph agent is a dict or object with:
    - 'config' or 'system_message' attribute
    - Optional 'memory' state
    - Optional 'tools' list
    - Optional 'policies' dict
    """

    def get_prompt(self, agent_handle: Any) -> str:
        """Get current system prompt from LangGraph agent.

        LangGraph agents typically store prompts in:
        - agent.config.get("system_message")
        - agent.system_message
        """
        if isinstance(agent_handle, dict):
            # Handle dict-based agent
            if "system_message" in agent_handle:
                return agent_handle["system_message"]
            if "config" in agent_handle and isinstance(agent_handle["config"], dict):
                return agent_handle["config"].get("system_message", "")
            return agent_handle.get("system_prompt", "")

        # Handle object-based agent
        if hasattr(agent_handle, "system_message"):
            return agent_handle.system_message
        if hasattr(agent_handle, "config") and hasattr(agent_handle.config, "get"):
            return agent_handle.config.get("system_message", "")
        return getattr(agent_handle, "system_prompt", "")

    def set_prompt(self, agent_handle: Any, prompt: str) -> None:
        """Update LangGraph agent's system prompt."""
        if isinstance(agent_handle, dict):
            agent_handle["system_message"] = prompt
            if "config" in agent_handle and isinstance(agent_handle["config"], dict):
                agent_handle["config"]["system_message"] = prompt
        else:
            if hasattr(agent_handle, "system_message"):
                agent_handle.system_message = prompt
            elif hasattr(agent_handle, "config"):
                if isinstance(agent_handle.config, dict):
                    agent_handle.config["system_message"] = prompt
                else:
                    setattr(agent_handle.config, "system_message", prompt)
            else:
                setattr(agent_handle, "system_message", prompt)

    def get_memory(self, agent_handle: Any) -> Optional[str]:
        """Get memory from LangGraph state."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("memory")
        return getattr(agent_handle, "memory", None)

    def set_memory(self, agent_handle: Any, memory_summary: str) -> None:
        """Update memory in LangGraph state."""
        if isinstance(agent_handle, dict):
            agent_handle["memory"] = memory_summary
        else:
            setattr(agent_handle, "memory", memory_summary)

    def get_tool_allowlist(self, agent_handle: Any) -> list[str]:
        """Get allowed tools."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("tools", [])
        return getattr(agent_handle, "tools", [])

    def get_policies(self, agent_handle: Any) -> dict[str, str]:
        """Get policies."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("policies", {})
        return getattr(agent_handle, "policies", {})


def create_scoa_nodes():
    """Create LangGraph nodes for SCOA optimization.

    Returns:
        Dictionary of node functions for integration into LangGraph
    """

    def scoa_collect(state: dict) -> dict:
        """Collect traces for optimization."""
        # This would fetch recent traces from the trace store
        return {"scoa_phase": "collect_done"}

    def scoa_diagnose(state: dict) -> dict:
        """Diagnose issues from traces."""
        return {"scoa_phase": "diagnose_done"}

    def scoa_patch(state: dict) -> dict:
        """Generate prompt patch."""
        return {"scoa_phase": "patch_done"}

    def scoa_validate(state: dict) -> dict:
        """Validate patch safety."""
        return {"scoa_phase": "validate_done"}

    def scoa_apply(state: dict) -> dict:
        """Apply approved patch."""
        return {"scoa_phase": "apply_done"}

    return {
        "scoa_collect": scoa_collect,
        "scoa_diagnose": scoa_diagnose,
        "scoa_patch": scoa_patch,
        "scoa_validate": scoa_validate,
        "scoa_apply": scoa_apply,
    }
