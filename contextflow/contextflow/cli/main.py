"""
CLI interface for ContextFlow.

Commands:
- contextflow run-once --agents all
- contextflow run-once --agents PlannerAgent,CoderAgent
- contextflow rollback --agent CoderAgent --version v3
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
from contextflow import ContextFlow, AgentProfile, AgentTrace
from contextflow.adapters.base import SimpleAdapter


def load_traces_from_file(filepath: Path) -> list[AgentTrace]:
    """Load traces from a JSONL file."""
    traces = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                traces.append(AgentTrace(**data))
    return traces


def load_agent_profiles_from_file(filepath: Path) -> list[AgentProfile]:
    """Load agent profiles from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        return [AgentProfile(**profile) for profile in data]


def cmd_run_once(args):
    """Run optimization once for specified agents."""
    print("ContextFlow: Running optimization")
    print("=" * 70)
    print()

    # Load configuration
    config_path = Path(args.config) if args.config else Path("contextflow_config.json")

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Please create a config file with agent profiles.")
        print("Example: contextflow_config.json")
        return 1

    # Load agent profiles
    print(f"Loading agent profiles from {config_path}...")
    try:
        profiles = load_agent_profiles_from_file(config_path)
        print(f"✓ Loaded {len(profiles)} agent profiles")
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return 1

    # Initialize ContextFlow
    cf = ContextFlow()
    adapter = SimpleAdapter()

    # Register agents
    print("\nRegistering agents...")
    for profile in profiles:
        # Create a simple agent handle (dict-based)
        agent_handle = {
            "system_prompt": profile.current_system_prompt,
            "tools": profile.tools,
            "policies": profile.policies,
        }
        cf.register_agent(profile, adapter, agent_handle)
        print(f"  ✓ {profile.id}")

    # Load traces if provided
    if args.traces:
        traces_path = Path(args.traces)
        if not traces_path.exists():
            print(f"\nWarning: Traces file not found: {traces_path}")
            print("Proceeding without traces (may result in minimal changes)")
            traces = []
        else:
            print(f"\nLoading traces from {traces_path}...")
            try:
                traces = load_traces_from_file(traces_path)
                print(f"✓ Loaded {len(traces)} traces")
            except Exception as e:
                print(f"Error loading traces: {e}")
                return 1

        if traces:
            cf.ingest_traces(traces)

    # Determine which agents to optimize
    if args.agents == "all":
        agent_ids = [p.id for p in profiles]
    else:
        agent_ids = [a.strip() for a in args.agents.split(",")]

    print(f"\nOptimizing agents: {', '.join(agent_ids)}")
    print("=" * 70)
    print()

    # Run optimization
    reports = cf.run_once(agent_ids=agent_ids)

    # Display results
    print("\nOptimization Results:")
    print("=" * 70)

    for report in reports:
        status_icon = "✓" if report.pass_fail == "pass" else "✗"
        print(f"\n{status_icon} {report.agent_id}")
        print(f"  Status: {report.pass_fail.upper()}")
        print(f"  Version: {report.applied_version or 'N/A'}")

        # Show metric changes
        delta_adherence = report.metrics_after.instruction_adherence - report.metrics_before.instruction_adherence
        delta_tools = report.metrics_after.tool_correctness - report.metrics_before.tool_correctness
        delta_concise = report.metrics_after.conciseness - report.metrics_before.conciseness

        if abs(delta_adherence) > 0.5 or abs(delta_tools) > 0.5 or abs(delta_concise) > 0.5:
            print(f"  Changes:")
            if abs(delta_adherence) > 0.5:
                print(f"    - Instruction adherence: {delta_adherence:+.1f}")
            if abs(delta_tools) > 0.5:
                print(f"    - Tool correctness: {delta_tools:+.1f}")
            if abs(delta_concise) > 0.5:
                print(f"    - Conciseness: {delta_concise:+.1f}")

        if report.notes:
            print(f"  Notes: {report.notes}")

    print()
    print("=" * 70)
    print("Optimization complete!")
    return 0


def cmd_rollback(args):
    """Rollback an agent to a previous version."""
    print("ContextFlow: Rolling back agent")
    print("=" * 70)
    print()

    # Load configuration
    config_path = Path(args.config) if args.config else Path("contextflow_config.json")

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1

    # Load agent profiles and initialize
    try:
        profiles = load_agent_profiles_from_file(config_path)
    except Exception as e:
        print(f"Error loading profiles: {e}")
        return 1

    cf = ContextFlow()
    adapter = SimpleAdapter()

    for profile in profiles:
        agent_handle = {
            "system_prompt": profile.current_system_prompt,
            "tools": profile.tools,
            "policies": profile.policies,
        }
        cf.register_agent(profile, adapter, agent_handle)

    # Perform rollback
    try:
        print(f"Rolling back {args.agent} to version {args.version}...")
        cf.rollback(args.agent, args.version)
        print(f"✓ Successfully rolled back {args.agent} to {args.version}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ContextFlow: Self-optimizing context for multi-agent systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run-once command
    run_parser = subparsers.add_parser(
        "run-once",
        help="Run optimization once for specified agents"
    )
    run_parser.add_argument(
        "--agents",
        type=str,
        default="all",
        help="Comma-separated agent IDs or 'all' (default: all)"
    )
    run_parser.add_argument(
        "--traces",
        type=str,
        help="Path to traces JSONL file (optional)"
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (default: contextflow_config.json)"
    )

    # rollback command
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Rollback an agent to a previous version"
    )
    rollback_parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Agent ID to rollback"
    )
    rollback_parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version to restore (e.g., v2)"
    )
    rollback_parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file (default: contextflow_config.json)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run-once":
        return cmd_run_once(args)
    elif args.command == "rollback":
        return cmd_rollback(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
