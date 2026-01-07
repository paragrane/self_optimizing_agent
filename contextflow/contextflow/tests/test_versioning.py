"""
Tests for versioning and rollback functionality.

Ensures that:
- Prompts are versioned correctly
- Rollback restores previous versions
- Version history is maintained
"""

import pytest
from contextflow import ContextFlow, AgentProfile
from contextflow.adapters.base import SimpleAdapter
from contextflow.storage.prompt_store import InMemoryPromptStore


def test_prompt_versioning():
    """Test that prompts are versioned correctly."""
    store = InMemoryPromptStore()

    # Save multiple versions
    store.save_prompt("agent1", "Version 1 prompt", "v1")
    store.save_prompt("agent1", "Version 2 prompt", "v2")
    store.save_prompt("agent1", "Version 3 prompt", "v3")

    # Check current version
    assert store.get_current_version("agent1") == "v3"

    # Check we can retrieve all versions
    assert store.get_prompt("agent1", "v1") == "Version 1 prompt"
    assert store.get_prompt("agent1", "v2") == "Version 2 prompt"
    assert store.get_prompt("agent1", "v3") == "Version 3 prompt"

    # Check latest
    assert store.get_prompt("agent1") == "Version 3 prompt"


def test_version_listing():
    """Test listing all versions for an agent."""
    store = InMemoryPromptStore()

    store.save_prompt("agent1", "Prompt v1", "v1")
    store.save_prompt("agent1", "Prompt v2", "v2")
    store.save_prompt("agent1", "Prompt v3", "v3")

    versions = store.list_versions("agent1")
    assert versions == ["v1", "v2", "v3"]


def test_rollback_functionality():
    """Test rolling back to a previous version."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create an agent
    agent_handle = {
        "system_prompt": "Initial prompt",
        "tools": ["search"],
        "policies": {},
    }

    profile = AgentProfile(
        id="test_agent",
        name="TestAgent",
        framework="custom",
        base_system_prompt="Initial prompt",
        current_system_prompt="Initial prompt",
        tools=["search"],
    )

    cf.register_agent(profile, adapter, agent_handle)

    # Manually save multiple versions
    prompt_store = cf.get_prompt_store()
    prompt_store.save_prompt("test_agent", "Initial prompt", "v1")
    prompt_store.save_prompt("test_agent", "Updated prompt v2", "v2")
    prompt_store.save_prompt("test_agent", "Updated prompt v3", "v3")

    # Current should be v3
    assert prompt_store.get_current_version("test_agent") == "v3"
    assert agent_handle["system_prompt"] == "Initial prompt"  # Not yet updated

    # Update the agent to v3
    adapter.set_prompt(agent_handle, "Updated prompt v3")
    profile.current_system_prompt = "Updated prompt v3"
    profile.current_version = "v3"

    # Rollback to v2
    cf.rollback("test_agent", "v2")

    # Check that agent was updated
    assert adapter.get_prompt(agent_handle) == "Updated prompt v2"
    assert profile.current_version == "v2"
    assert profile.current_system_prompt == "Updated prompt v2"


def test_rollback_to_nonexistent_version_fails():
    """Test that rollback to non-existent version raises error."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    agent_handle = {"system_prompt": "Initial prompt", "tools": [], "policies": {}}

    profile = AgentProfile(
        id="test_agent",
        name="TestAgent",
        framework="custom",
        base_system_prompt="Initial prompt",
        current_system_prompt="Initial prompt",
    )

    cf.register_agent(profile, adapter, agent_handle)

    # Try to rollback to non-existent version
    with pytest.raises(ValueError, match="Version .* not found"):
        cf.rollback("test_agent", "v999")


def test_patch_history_tracking():
    """Test that patch history is maintained."""
    from contextflow.core.schemas import PromptPatch

    store = InMemoryPromptStore()

    # Save some patches
    patch1 = PromptPatch(
        agent_id="agent1",
        version_from="v1",
        version_to="v2",
        diff="+ Added guidance",
        rationale="Improve conciseness",
        new_prompt="New prompt v2",
    )

    patch2 = PromptPatch(
        agent_id="agent1",
        version_from="v2",
        version_to="v3",
        diff="+ Added safety",
        rationale="Improve safety",
        new_prompt="New prompt v3",
    )

    store.save_patch(patch1)
    store.save_patch(patch2)

    # Retrieve history
    history = store.get_patch_history("agent1")
    assert len(history) == 2
    assert history[0].version_to == "v2"
    assert history[1].version_to == "v3"


def test_multiple_agents_versioned_independently():
    """Test that multiple agents maintain independent version histories."""
    store = InMemoryPromptStore()

    # Agent 1 versions
    store.save_prompt("agent1", "Agent1 v1", "v1")
    store.save_prompt("agent1", "Agent1 v2", "v2")

    # Agent 2 versions
    store.save_prompt("agent2", "Agent2 v1", "v1")
    store.save_prompt("agent2", "Agent2 v2", "v2")
    store.save_prompt("agent2", "Agent2 v3", "v3")

    # Check independence
    assert store.get_current_version("agent1") == "v2"
    assert store.get_current_version("agent2") == "v3"

    assert store.get_prompt("agent1", "v2") == "Agent1 v2"
    assert store.get_prompt("agent2", "v3") == "Agent2 v3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
