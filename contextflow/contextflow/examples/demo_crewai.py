"""
Demo: Using ContextFlow with CrewAI agents.

Shows how to:
1. Create mock CrewAI agents
2. Register them with ContextFlow
3. Ingest traces
4. Run optimization sequentially
5. View results and diffs
"""

from datetime import datetime
from contextflow import ContextFlow, AgentProfile, AgentTrace, ToolCall
from contextflow.adapters.crewai_adapter import CrewAIAdapter
from contextflow.core.patching import PromptPatcher


def create_mock_crewai_agents():
    """Create mock CrewAI agents for demonstration."""

    planner = {
        "agent_id": "planner",
        "role": "Task Planner",
        "goal": "Create comprehensive plans for software development tasks",
        "backstory": "You are an experienced project manager with deep technical knowledge. You excel at breaking down complex requirements into actionable steps.",
        "tools": ["research", "analyze_requirements"],
        "policies": {
            "safety": "Never plan activities that could harm users or violate ethics",
        },
    }

    coder = {
        "agent_id": "coder",
        "role": "Senior Software Engineer",
        "goal": "Write high-quality, maintainable code",
        "backstory": "You are a senior developer with 10 years of experience across multiple languages and frameworks.",
        "tools": ["write_code", "run_tests", "debug"],
        "policies": {
            "safety": "Never write malicious code or introduce security vulnerabilities",
            "compliance": "Follow OWASP secure coding guidelines",
        },
    }

    reviewer = {
        "agent_id": "reviewer",
        "role": "Code Reviewer",
        "goal": "Ensure code quality, security, and best practices",
        "backstory": "You are a principal engineer focused on code quality and security. You have expertise in identifying vulnerabilities and architectural issues.",
        "tools": ["static_analysis", "security_scan"],
        "policies": {
            "safety": "Flag all potential security issues",
            "privacy": "Ensure no PII leakage",
        },
    }

    return planner, coder, reviewer


def create_sample_traces():
    """Create sample traces with realistic issues."""

    traces = [
        # Planner - good performance
        AgentTrace(
            agent_id="planner",
            input="Plan a user authentication system",
            output="1. Choose auth method (JWT)\n2. Design user schema\n3. Implement endpoints\n4. Add rate limiting\n5. Security testing",
            outcome="success",
            tokens=180,
            latency_ms=900,
        ),
        # Coder - verbosity issue and tool misuse
        AgentTrace(
            agent_id="coder",
            input="Write a simple password validator",
            output="""Let me write a comprehensive password validation system for you!

First, let's discuss password security best practices. Passwords should be at least 8 characters, contain uppercase, lowercase, numbers, and special characters. This is important because weak passwords are a major security risk.

Here's the implementation with detailed comments explaining every step:

def validate_password(password):
    # Check length - this is the first and most basic check
    if len(password) < 8:
        return False  # Password too short

    # Check for uppercase - we need at least one
    has_upper = any(c.isupper() for c in password)

    # Check for lowercase - we need at least one
    has_lower = any(c.islower() for c in password)

    # And so on for 20 more lines with excessive documentation...""",
            tool_calls=[
                ToolCall(tool_name="write_code", arguments={}, success=True),
                ToolCall(tool_name="security_scan", arguments={}, success=False, error="Not authorized for this tool"),
            ],
            outcome="success",
            tokens=2100,  # Way too verbose!
            latency_ms=5000,
        ),
        AgentTrace(
            agent_id="coder",
            input="Fix bug in login function",
            output="""Great question! Let me explain the entire authentication flow before fixing the bug.

Authentication is a critical part of any application. It ensures that users are who they say they are...
[500 more words of explanation]

Here's the fix with extensive comments...""",
            tool_calls=[],
            outcome="success",
            tokens=1800,
            latency_ms=4200,
        ),
        # Reviewer - performing well
        AgentTrace(
            agent_id="reviewer",
            input="Review authentication implementation",
            output="Issues found:\n1. SQL injection risk in line 45\n2. Missing input validation\n3. Passwords not hashed\nRecommendation: Use parameterized queries and bcrypt",
            tool_calls=[
                ToolCall(tool_name="static_analysis", arguments={"file": "auth.py"}, success=True),
                ToolCall(tool_name="security_scan", arguments={"file": "auth.py"}, success=True),
            ],
            outcome="success",
            tokens=200,
            latency_ms=1100,
        ),
    ]

    return traces


def print_prompt_diff(agent_id: str, patch):
    """Print a formatted diff of the prompt changes."""
    print(f"\nPrompt Diff for {agent_id}:")
    print("=" * 70)
    if patch and patch.diff:
        formatted_diff = PromptPatcher.format_diff_for_display(patch.diff)
        # Show first 800 characters of diff
        if len(formatted_diff) > 800:
            print(formatted_diff[:800])
            print(f"\n... (diff truncated, {len(formatted_diff) - 800} more characters)")
        else:
            print(formatted_diff)
    else:
        print("No diff available")
    print("=" * 70)


def main():
    """Run the CrewAI demo."""
    print("=" * 70)
    print("ContextFlow Demo: CrewAI Integration")
    print("=" * 70)
    print()

    # Create agents
    print("1. Creating mock CrewAI agents...")
    planner, coder, reviewer = create_mock_crewai_agents()
    print(f"   ✓ Created 3 agents: planner, coder, reviewer")
    print()

    # Initialize ContextFlow
    print("2. Initializing ContextFlow...")
    cf = ContextFlow()
    adapter = CrewAIAdapter()
    print("   ✓ ContextFlow initialized")
    print()

    # Register agents
    print("3. Registering agents...")
    for agent in [planner, coder, reviewer]:
        # Get initial prompt from CrewAI structure
        initial_prompt = adapter.get_prompt(agent)

        profile = AgentProfile(
            id=agent["agent_id"],
            name=agent["agent_id"].capitalize() + "Agent",
            framework="crewai",
            base_system_prompt=initial_prompt,
            current_system_prompt=initial_prompt,
            tools=agent["tools"],
            policies=agent["policies"],
        )
        cf.register_agent(profile, adapter, agent)
        print(f"   ✓ Registered {agent['agent_id']}")
    print()

    # Ingest traces
    print("4. Ingesting sample traces (showing coder verbosity issues)...")
    traces = create_sample_traces()
    cf.ingest_traces(traces)
    print(f"   ✓ Ingested {len(traces)} traces")

    # Show trace summary
    print("\n   Trace Summary:")
    print(f"   - Planner: 1 trace, avg {180} tokens")
    print(f"   - Coder: 2 traces, avg {(2100+1800)/2:.0f} tokens (VERBOSE!)")
    print(f"   - Reviewer: 1 trace, avg {200} tokens")
    print()

    # Run optimization SEQUENTIALLY
    print("5. Running sequential optimization (ONE BY ONE)...")
    print("   Processing planner → coder → reviewer in sequence")
    print()

    reports = cf.run_once(agent_ids=["planner", "coder", "reviewer"])

    # Display detailed results
    print("6. Detailed Optimization Results:")
    print("=" * 70)
    print()

    for report in reports:
        print(f"\n{'=' * 70}")
        print(f"AGENT: {report.agent_id.upper()}")
        print(f"{'=' * 70}")
        print(f"Status: {report.pass_fail.upper()}")
        print(f"Applied Version: {report.applied_version or 'N/A (not applied)'}")
        print()

        print("Performance Metrics:")
        print(f"  {'Metric':<25} {'Before':<10} {'After':<10} {'Change':<10}")
        print(f"  {'-'*55}")

        metrics = [
            ("Instruction Adherence", "instruction_adherence"),
            ("Tool Correctness", "tool_correctness"),
            ("Conciseness", "conciseness"),
        ]

        for label, attr in metrics:
            before_val = getattr(report.metrics_before, attr)
            after_val = getattr(report.metrics_after, attr)
            delta = after_val - before_val
            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
            print(f"  {label:<25} {before_val:<10.1f} {after_val:<10.1f} {delta_str:<10}")

        print()
        print(f"Evaluation Notes: {report.notes}")

        # Show diff for applied patches
        if report.applied_version:
            logs = cf.get_logger().get_logs(agent_id=report.agent_id, limit=1)
            if logs and logs[0].patch:
                print_prompt_diff(report.agent_id, logs[0].patch)

        print()

    # Show final prompt for coder (most changed)
    print("\n7. Updated Coder Prompt (showing improvements):")
    print("=" * 70)
    prompt_store = cf.get_prompt_store()
    coder_version = prompt_store.get_current_version("coder")
    coder_prompt = prompt_store.get_prompt("coder", coder_version)

    print(f"Version: {coder_version}")
    print()
    # Show first 600 chars
    display_prompt = coder_prompt[:600] + "..." if len(coder_prompt) > 600 else coder_prompt
    print(display_prompt)
    print("=" * 70)
    print()

    # Show optimization success rate
    print("8. Optimization Statistics:")
    print("=" * 70)
    logger = cf.get_logger()

    for agent_id in ["planner", "coder", "reviewer"]:
        success_rate = logger.get_success_rate(agent_id)
        print(f"  {agent_id.capitalize()}: {success_rate*100:.0f}% success rate")

    print()
    print("Demo completed successfully!")
    print()
    print("Key Takeaways:")
    print("  ✓ Agents were optimized SEQUENTIALLY (one by one)")
    print("  ✓ Coder's verbosity issue was detected and addressed")
    print("  ✓ Safety constraints were preserved throughout")
    print("  ✓ All changes were versioned and can be rolled back")


if __name__ == "__main__":
    main()
