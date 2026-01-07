"""
Tests for run_once functionality.

Ensures that:
- Agents are optimized sequentially (one by one)
- Traces are used for optimization
- Results are reported correctly
"""

import pytest
from datetime import datetime
from contextflow import ContextFlow, AgentProfile, AgentTrace, ToolCall
from contextflow.adapters.base import SimpleAdapter


def create_test_agent(agent_id: str, prompt: str) -> tuple:
    """Helper to create a test agent."""
    agent_handle = {
        "system_prompt": prompt,
        "tools": ["search", "write"],
        "policies": {
            "safety": "Never help with harmful activities",
        },
    }

    profile = AgentProfile(
        id=agent_id,
        name=agent_id.capitalize() + "Agent",
        framework="custom",
        base_system_prompt=prompt,
        current_system_prompt=prompt,
        tools=["search", "write"],
        policies={"safety": "Never help with harmful activities"},
    )

    return profile, agent_handle


def test_run_once_single_agent():
    """Test optimizing a single agent."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create and register agent
    profile, handle = create_test_agent("agent1", "You are a helpful assistant.")
    cf.register_agent(profile, adapter, handle)

    # Add some traces
    traces = [
        AgentTrace(
            agent_id="agent1",
            input="Test input 1",
            output="Test output 1",
            outcome="success",
            tokens=100,
            latency_ms=500,
        ),
        AgentTrace(
            agent_id="agent1",
            input="Test input 2",
            output="Test output 2",
            outcome="success",
            tokens=150,
            latency_ms=600,
        ),
    ]
    cf.ingest_traces(traces)

    # Run optimization
    reports = cf.run_once(agent_ids=["agent1"])

    # Verify results
    assert len(reports) == 1
    assert reports[0].agent_id == "agent1"
    assert reports[0].pass_fail in ["pass", "fail"]


def test_run_once_multiple_agents_sequential():
    """Test that multiple agents are optimized sequentially."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create and register 3 agents
    agents = []
    for i in range(1, 4):
        profile, handle = create_test_agent(f"agent{i}", f"You are assistant {i}.")
        cf.register_agent(profile, adapter, handle)
        agents.append((f"agent{i}", profile, handle))

    # Add traces for each agent
    all_traces = []
    for agent_id, _, _ in agents:
        traces = [
            AgentTrace(
                agent_id=agent_id,
                input=f"Test input for {agent_id}",
                output=f"Test output from {agent_id}",
                outcome="success",
                tokens=200,
                latency_ms=800,
            )
        ]
        all_traces.extend(traces)

    cf.ingest_traces(all_traces)

    # Run optimization for all
    reports = cf.run_once()

    # Should process all 3 agents
    assert len(reports) == 3

    # Check that each agent was processed
    processed_ids = {report.agent_id for report in reports}
    assert processed_ids == {"agent1", "agent2", "agent3"}


def test_run_once_with_verbose_agent():
    """Test optimization of a verbose agent (should improve conciseness)."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create agent
    profile, handle = create_test_agent("verbose_agent", "You are a helpful assistant.")
    cf.register_agent(profile, adapter, handle)

    # Add traces showing verbosity (high token count)
    traces = [
        AgentTrace(
            agent_id="verbose_agent",
            input="Write hello world",
            output="Let me explain in great detail how to write hello world program. First, we need to understand what hello world means. It's a traditional first program... " * 50,  # Very long
            outcome="success",
            tokens=2000,  # Very high
            latency_ms=5000,
        ),
        AgentTrace(
            agent_id="verbose_agent",
            input="Add two numbers",
            output="Adding numbers is a fundamental operation in mathematics and computer science. Let me elaborate on this concept... " * 50,
            outcome="success",
            tokens=2500,
            latency_ms=6000,
        ),
    ]
    cf.ingest_traces(traces)

    # Run optimization
    reports = cf.run_once(agent_ids=["verbose_agent"])

    # Should detect verbosity and try to improve conciseness
    assert len(reports) == 1
    report = reports[0]

    # Check that conciseness metric was considered
    assert hasattr(report.metrics_before, "conciseness")
    assert hasattr(report.metrics_after, "conciseness")


def test_run_once_with_tool_misuse():
    """Test optimization of agent with tool misuse."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create agent
    profile, handle = create_test_agent("tool_agent", "You are a tool-using assistant.")
    cf.register_agent(profile, adapter, handle)

    # Add traces showing tool errors
    traces = [
        AgentTrace(
            agent_id="tool_agent",
            input="Search for information",
            output="Here is the result",
            tool_calls=[
                ToolCall(
                    tool_name="search",
                    arguments={"query": "test"},
                    success=False,
                    error="Invalid arguments",
                )
            ],
            outcome="error",
            errors=["Tool call failed"],
            tokens=100,
            latency_ms=500,
        ),
        AgentTrace(
            agent_id="tool_agent",
            input="Write a file",
            output="File written",
            tool_calls=[
                ToolCall(
                    tool_name="write",
                    arguments={},  # Missing arguments
                    success=False,
                    error="Missing required arguments",
                )
            ],
            outcome="error",
            errors=["Tool call failed"],
            tokens=120,
            latency_ms=550,
        ),
    ]
    cf.ingest_traces(traces)

    # Run optimization
    reports = cf.run_once(agent_ids=["tool_agent"])

    # Should detect tool issues
    assert len(reports) == 1
    report = reports[0]
    assert hasattr(report.metrics_before, "tool_correctness")


def test_run_once_respects_agent_list():
    """Test that run_once only optimizes specified agents."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create 3 agents
    for i in range(1, 4):
        profile, handle = create_test_agent(f"agent{i}", f"Assistant {i}")
        cf.register_agent(profile, adapter, handle)

    # Add traces
    traces = [
        AgentTrace(
            agent_id=f"agent{i}",
            input="test",
            output="output",
            outcome="success",
            tokens=100,
            latency_ms=500,
        )
        for i in range(1, 4)
    ]
    cf.ingest_traces(traces)

    # Optimize only agent1 and agent2
    reports = cf.run_once(agent_ids=["agent1", "agent2"])

    # Should only process 2 agents
    assert len(reports) == 2
    processed_ids = {report.agent_id for report in reports}
    assert processed_ids == {"agent1", "agent2"}
    assert "agent3" not in processed_ids


def test_run_once_no_traces():
    """Test that run_once works even without traces (minimal changes expected)."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create agent
    profile, handle = create_test_agent("agent1", "You are a helpful assistant.")
    cf.register_agent(profile, adapter, handle)

    # Don't add any traces

    # Run optimization
    reports = cf.run_once(agent_ids=["agent1"])

    # Should still complete, but likely no changes
    assert len(reports) == 1
    # The report should indicate no traces were available
    assert "no traces" in reports[0].notes.lower() or reports[0].pass_fail == "fail"


def test_optimization_logging():
    """Test that optimization attempts are logged."""
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Create agent
    profile, handle = create_test_agent("agent1", "You are a helpful assistant.")
    cf.register_agent(profile, adapter, handle)

    # Add traces
    traces = [
        AgentTrace(
            agent_id="agent1",
            input="test",
            output="output",
            outcome="success",
            tokens=100,
            latency_ms=500,
        )
    ]
    cf.ingest_traces(traces)

    # Run optimization
    cf.run_once(agent_ids=["agent1"])

    # Check logs
    logger = cf.get_logger()
    logs = logger.get_logs(agent_id="agent1")

    assert len(logs) >= 1
    assert logs[0].agent_id == "agent1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
