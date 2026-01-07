"""
Versioned storage for agent prompts.

Supports saving, retrieving, and rolling back to previous versions.
"""

from typing import Protocol, Optional
from datetime import datetime
from contextflow.core.schemas import AgentProfile, PromptPatch


class PromptStore(Protocol):
    """Protocol for prompt storage backends."""

    def save_prompt(self, agent_id: str, prompt: str, version: str) -> None:
        """Save a prompt version for an agent."""
        ...

    def get_prompt(self, agent_id: str, version: Optional[str] = None) -> Optional[str]:
        """Get a prompt version. If version is None, get latest."""
        ...

    def get_current_version(self, agent_id: str) -> Optional[str]:
        """Get the current version string for an agent."""
        ...

    def save_patch(self, patch: PromptPatch) -> None:
        """Save a prompt patch for audit trail."""
        ...

    def get_patch_history(self, agent_id: str) -> list[PromptPatch]:
        """Get all patches for an agent."""
        ...

    def list_versions(self, agent_id: str) -> list[str]:
        """List all version tags for an agent."""
        ...


class InMemoryPromptStore:
    """In-memory implementation of PromptStore.

    Suitable for development and testing. Use persistent store in production.
    """

    def __init__(self):
        # agent_id -> version -> prompt
        self._prompts: dict[str, dict[str, str]] = {}
        # agent_id -> current version
        self._current_versions: dict[str, str] = {}
        # agent_id -> list of patches
        self._patches: dict[str, list[PromptPatch]] = {}

    def save_prompt(self, agent_id: str, prompt: str, version: str) -> None:
        """Save a prompt version for an agent."""
        if agent_id not in self._prompts:
            self._prompts[agent_id] = {}
        self._prompts[agent_id][version] = prompt
        self._current_versions[agent_id] = version

    def get_prompt(self, agent_id: str, version: Optional[str] = None) -> Optional[str]:
        """Get a prompt version. If version is None, get latest."""
        if agent_id not in self._prompts:
            return None

        if version is None:
            version = self._current_versions.get(agent_id)
            if version is None:
                return None

        return self._prompts[agent_id].get(version)

    def get_current_version(self, agent_id: str) -> Optional[str]:
        """Get the current version string for an agent."""
        return self._current_versions.get(agent_id)

    def save_patch(self, patch: PromptPatch) -> None:
        """Save a prompt patch for audit trail."""
        if patch.agent_id not in self._patches:
            self._patches[patch.agent_id] = []
        self._patches[patch.agent_id].append(patch)

    def get_patch_history(self, agent_id: str) -> list[PromptPatch]:
        """Get all patches for an agent."""
        return self._patches.get(agent_id, [])

    def list_versions(self, agent_id: str) -> list[str]:
        """List all version tags for an agent."""
        if agent_id not in self._prompts:
            return []
        return sorted(self._prompts[agent_id].keys())
