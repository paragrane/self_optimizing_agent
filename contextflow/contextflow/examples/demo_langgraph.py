"""
Demo: Using ContextFlow with LangGraph agents.

Shows how to:
1. Create mock LangGraph agents
2. Register them with ContextFlow
3. Ingest traces
4. Run optimization sequentially
5. View results
"""

from datetime import datetime
from contextflow import ContextFlow, AgentProfile, AgentTrace, ToolCall
from contextflow.adapters.langgraph_adapter import LangGraphAdapter


def create_mock_langgraph_agents():
    """Create mock LangGraph agents for demonstration."""

    planner_agent = {
        "agent_id": "planner",
        "system_message": "You are a planning agent. Create detailed plans for software tasks.",
        "tools": ["search_docs", "read_file", "list_tasks"],
        "policies": {
            "safety": "Never suggest harmful or malicious plans",
            "privacy": "Never include sensitive user data in plans",
        },
    }

    coder_agent = {
        "agent_id": "coder",
        "system_message": "You are a coding agent. Write clean, efficient code.",
        "tools": ["write_file", "read_file", "run_tests"],
        "policies": {
            "safety": "Never write code that could harm users or systems",
            "compliance": "Follow secure coding best practices",
        },
    }

    reviewer_agent = {
        "agent_id": "reviewer",
        "system_message": "You are a code review agent. Review code for quality and security.",
        "tools": ["read_file", "analyze_code", "suggest_improvements"],
        "policies": {
            "safety": "Flag all security vulnerabilities",
            "privacy": "Check for data leak risks",
        },
    }

    return planner_agent, coder_agent, reviewer_agent


def create_sample_traces():
    """Create sample traces demonstrating various issues."""

    # Planner traces - generally good
    planner_traces = [
        AgentTrace(
            agent_id="planner",
            timestamp=datetime.now(),
            input="Create a plan for building a REST API",
            output="Plan:\n1. Design database schema\n2. Create API endpoints\n3. Add authentication\n4. Write tests\n5. Deploy",
            tool_calls=[
                ToolCall(tool_name="search_docs", arguments={"query": "REST API best practices"}, success=True)
            ],
            outcome="success",
            tokens=250,
            latency_ms=1200,
        ),
        AgentTrace(
            agent_id="planner",
            timestamp=datetime.now(),
            input="Plan microservices migration",
            output="Migration plan:\n1. Identify service boundaries\n2. Extract domain logic\n3. Set up API gateways\n4. Implement service discovery\n5. Gradual rollout",
            tool_calls=[],
            outcome="success",
            tokens=300,
            latency_ms=1500,
        ),
    ]

    # Coder traces - showing verbosity and tool misuse issues
    coder_traces = [
        AgentTrace(
            agent_id="coder",
            timestamp=datetime.now(),
            input="Write a function to validate email",
            output="""Here's a comprehensive email validation function with extensive documentation and examples:

def validate_email(email):
    '''
    This function validates email addresses according to RFC 5322 standards.
    It performs multiple checks including format validation, domain verification,
    and length constraints. The function is designed to be robust and handle
    edge cases gracefully.

    Parameters:
    - email (str): The email address to validate

    Returns:
    - bool: True if valid, False otherwise

    Examples:
    >>> validate_email('test@example.com')
    True
    >>> validate_email('invalid.email')
    False
    '''
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

This function is production-ready and handles all common email formats...""",
            tool_calls=[
                ToolCall(tool_name="search_docs", arguments={}, success=False, error="Wrong tool used"),
            ],
            outcome="success",
            tokens=1200,  # Very verbose!
            latency_ms=3000,
        ),
        AgentTrace(
            agent_id="coder",
            timestamp=datetime.now(),
            input="Add error handling to database connection",
            output="""Certainly! I'll provide a complete implementation with extensive error handling...[400 more words of explanation]

def connect_db():
    try:
        conn = create_connection()
        return conn
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

Let me explain every line in detail...""",
            tool_calls=[
                ToolCall(tool_name="read_file", arguments={"path": "db.py"}, success=True),
                ToolCall(tool_name="analyze_code", arguments={}, success=False, error="Not in allowlist"),
            ],
            outcome="success",
            tokens=1500,
            latency_ms=3500,
        ),
    ]

    # Reviewer traces - good quality
    reviewer_traces = [
        AgentTrace(
            agent_id="reviewer",
            timestamp=datetime.now(),
            input="Review this code: def login(user, pwd): return db.query(f'SELECT * FROM users WHERE name={user}')",
            output="CRITICAL: SQL injection vulnerability detected. Use parameterized queries instead.",
            tool_calls=[
                ToolCall(tool_name="analyze_code", arguments={"code": "..."}, success=True)
            ],
            outcome="success",
            tokens=150,
            latency_ms=800,
        ),
    ]

    return planner_traces + coder_traces + reviewer_traces


def main():
    """Run the demo."""
    print("=" * 70)
    print("ContextFlow Demo: LangGraph Integration")
    print("=" * 70)
    print()

    # Create agents
    print("1. Creating mock LangGraph agents...")
    planner, coder, reviewer = create_mock_langgraph_agents()
    print(f"   ✓ Created 3 agents: planner, coder, reviewer")
    print()

    # Initialize ContextFlow
    print("2. Initializing ContextFlow...")
    cf = ContextFlow()
    adapter = LangGraphAdapter()
    print("   ✓ ContextFlow initialized")
    print()

    # Register agents
    print("3. Registering agents...")
    for agent in [planner, coder, reviewer]:
        profile = AgentProfile(
            id=agent["agent_id"],
            name=agent["agent_id"].capitalize() + "Agent",
            framework="langgraph",
            base_system_prompt=agent["system_message"],
            current_system_prompt=agent["system_message"],
            tools=agent["tools"],
            policies=agent["policies"],
        )
        cf.register_agent(profile, adapter, agent)
        print(f"   ✓ Registered {agent['agent_id']}")
    print()

    # Ingest traces
    print("4. Ingesting sample traces...")
    traces = create_sample_traces()
    cf.ingest_traces(traces)
    print(f"   ✓ Ingested {len(traces)} traces")
    print()

    # Run optimization
    print("5. Running sequential optimization (ONE BY ONE)...")
    print()
    reports = cf.run_once()

    # Display results
    print("6. Optimization Results:")
    print("=" * 70)
    print()

    for report in reports:
        print(f"Agent: {report.agent_id}")
        print(f"Result: {report.pass_fail.upper()}")
        print(f"Applied Version: {report.applied_version or 'N/A'}")
        print()
        print(f"Metrics Before:")
        print(f"  - Instruction Adherence: {report.metrics_before.instruction_adherence:.1f}/10")
        print(f"  - Tool Correctness: {report.metrics_before.tool_correctness:.1f}/10")
        print(f"  - Conciseness: {report.metrics_before.conciseness:.1f}/10")
        print()
        print(f"Metrics After:")
        print(f"  - Instruction Adherence: {report.metrics_after.instruction_adherence:.1f}/10")
        print(f"  - Tool Correctness: {report.metrics_after.tool_correctness:.1f}/10")
        print(f"  - Conciseness: {report.metrics_after.conciseness:.1f}/10")
        print()
        print(f"Notes: {report.notes}")
        print()
        print("-" * 70)
        print()

    # Show updated prompts
    print("7. Updated Prompts:")
    print("=" * 70)
    print()

    for agent_id in ["planner", "coder", "reviewer"]:
        prompt_store = cf.get_prompt_store()
        current_version = prompt_store.get_current_version(agent_id)
        prompt = prompt_store.get_prompt(agent_id, current_version)

        print(f"Agent: {agent_id} (version: {current_version})")
        print(f"Prompt (first 500 chars):")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        print()
        print("-" * 70)
        print()

    # Show optimization logs
    print("8. Optimization Log Summary:")
    print("=" * 70)
    cf.get_logger().print_summary()

    print()
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
