"""
CrewAI adapter for ContextFlow.

Integrates with CrewAI agents and crews.
"""

from typing import Any, Optional


class CrewAIAdapter:
    """Adapter for CrewAI agents.

    CrewAI agents typically have:
    - agent.role (description of role)
    - agent.goal (agent's goal)
    - agent.backstory (background context)
    - agent.tools (list of tools)

    We combine role + goal + backstory into the system prompt.
    """

    def get_prompt(self, agent_handle: Any) -> str:
        """Get current system prompt from CrewAI agent.

        Combines role, goal, and backstory into a unified prompt.
        """
        if isinstance(agent_handle, dict):
            # Dict-based agent
            role = agent_handle.get("role", "")
            goal = agent_handle.get("goal", "")
            backstory = agent_handle.get("backstory", "")
            system_prompt = agent_handle.get("system_prompt")

            if system_prompt:
                return system_prompt

            # Construct from CrewAI fields
            parts = []
            if role:
                parts.append(f"Role: {role}")
            if goal:
                parts.append(f"Goal: {goal}")
            if backstory:
                parts.append(f"Backstory: {backstory}")
            return "\n\n".join(parts)

        # Object-based agent
        if hasattr(agent_handle, "system_prompt"):
            return agent_handle.system_prompt

        # Construct from CrewAI attributes
        parts = []
        if hasattr(agent_handle, "role"):
            parts.append(f"Role: {agent_handle.role}")
        if hasattr(agent_handle, "goal"):
            parts.append(f"Goal: {agent_handle.goal}")
        if hasattr(agent_handle, "backstory"):
            parts.append(f"Backstory: {agent_handle.backstory}")

        return "\n\n".join(parts) if parts else ""

    def set_prompt(self, agent_handle: Any, prompt: str) -> None:
        """Update CrewAI agent's system prompt.

        Since CrewAI uses role/goal/backstory, we store the full prompt
        in a new 'system_prompt' attribute and optionally parse it back
        into components.
        """
        if isinstance(agent_handle, dict):
            agent_handle["system_prompt"] = prompt
            # Optionally update role/goal/backstory by parsing
            self._parse_and_update(agent_handle, prompt)
        else:
            setattr(agent_handle, "system_prompt", prompt)
            self._parse_and_update(agent_handle, prompt)

    def _parse_and_update(self, agent_handle: Any, prompt: str) -> None:
        """Parse prompt and update role/goal/backstory if possible."""
        # Simple parsing: look for "Role:", "Goal:", "Backstory:"
        lines = prompt.split("\n")
        current_section = None
        sections = {"role": [], "goal": [], "backstory": []}

        for line in lines:
            if line.startswith("Role:"):
                current_section = "role"
                sections["role"].append(line[5:].strip())
            elif line.startswith("Goal:"):
                current_section = "goal"
                sections["goal"].append(line[5:].strip())
            elif line.startswith("Backstory:"):
                current_section = "backstory"
                sections["backstory"].append(line[10:].strip())
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        # Update fields
        if sections["role"]:
            role_text = " ".join(sections["role"])
            if isinstance(agent_handle, dict):
                agent_handle["role"] = role_text
            else:
                setattr(agent_handle, "role", role_text)

        if sections["goal"]:
            goal_text = " ".join(sections["goal"])
            if isinstance(agent_handle, dict):
                agent_handle["goal"] = goal_text
            else:
                setattr(agent_handle, "goal", goal_text)

        if sections["backstory"]:
            backstory_text = " ".join(sections["backstory"])
            if isinstance(agent_handle, dict):
                agent_handle["backstory"] = backstory_text
            else:
                setattr(agent_handle, "backstory", backstory_text)

    def get_memory(self, agent_handle: Any) -> Optional[str]:
        """Get memory from CrewAI agent."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("memory")
        return getattr(agent_handle, "memory", None)

    def set_memory(self, agent_handle: Any, memory_summary: str) -> None:
        """Update memory in CrewAI agent."""
        if isinstance(agent_handle, dict):
            agent_handle["memory"] = memory_summary
        else:
            setattr(agent_handle, "memory", memory_summary)

    def get_tool_allowlist(self, agent_handle: Any) -> list[str]:
        """Get allowed tools from CrewAI agent."""
        if isinstance(agent_handle, dict):
            tools = agent_handle.get("tools", [])
        else:
            tools = getattr(agent_handle, "tools", [])

        # CrewAI tools might be objects; extract names
        tool_names = []
        for tool in tools:
            if isinstance(tool, str):
                tool_names.append(tool)
            elif hasattr(tool, "name"):
                tool_names.append(tool.name)
            elif hasattr(tool, "__name__"):
                tool_names.append(tool.__name__)

        return tool_names

    def get_policies(self, agent_handle: Any) -> dict[str, str]:
        """Get policies from CrewAI agent."""
        if isinstance(agent_handle, dict):
            return agent_handle.get("policies", {})
        return getattr(agent_handle, "policies", {})
