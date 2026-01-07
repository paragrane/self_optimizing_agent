"""
Prompt patching logic.

Generates diffs and applies patches to prompts.
"""

import difflib
from contextflow.core.schemas import PromptPatch


class PromptPatcher:
    """Handles creation and application of prompt patches."""

    @staticmethod
    def create_patch(
        agent_id: str,
        old_prompt: str,
        new_prompt: str,
        version_from: str,
        version_to: str,
        rationale: str,
    ) -> PromptPatch:
        """Create a patch object with unified diff.

        Args:
            agent_id: Agent identifier
            old_prompt: Current prompt
            new_prompt: Proposed new prompt
            version_from: Current version tag
            version_to: New version tag
            rationale: Explanation for the change

        Returns:
            PromptPatch object
        """
        # Generate unified diff
        old_lines = old_prompt.splitlines(keepends=True)
        new_lines = new_prompt.splitlines(keepends=True)

        diff = "".join(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"{agent_id}_{version_from}",
                tofile=f"{agent_id}_{version_to}",
                lineterm="",
            )
        )

        return PromptPatch(
            agent_id=agent_id,
            version_from=version_from,
            version_to=version_to,
            diff=diff,
            rationale=rationale,
            new_prompt=new_prompt,
        )

    @staticmethod
    def apply_patch(old_prompt: str, patch: PromptPatch) -> str:
        """Apply a patch to get the new prompt.

        In practice, we just return the new_prompt from the patch.
        The diff is for human readability.

        Args:
            old_prompt: Current prompt (not used, kept for interface)
            patch: Patch to apply

        Returns:
            New prompt string
        """
        return patch.new_prompt

    @staticmethod
    def format_diff_for_display(diff: str) -> str:
        """Format a diff for terminal display.

        Adds color indicators (in text form for simplicity).
        """
        lines = diff.split("\n")
        formatted = []

        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                formatted.append(f"  + {line[1:]}")  # Addition
            elif line.startswith("-") and not line.startswith("---"):
                formatted.append(f"  - {line[1:]}")  # Deletion
            elif line.startswith("@@"):
                formatted.append(f"\n{line}\n")  # Hunk header
            else:
                formatted.append(f"    {line}")  # Context

        return "\n".join(formatted)
