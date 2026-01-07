"""
Evaluation logic for comparing prompt versions.

Runs counterfactual evaluation: before vs after.
"""

from typing import Callable, Optional
import random
from contextflow.core.schemas import EvalMetrics, EvalReport, AgentTrace, AgentProfile


class SyntheticTask:
    """A synthetic task for evaluation."""

    def __init__(self, task_id: str, prompt: str, expected_tools: list[str] = None):
        self.task_id = task_id
        self.prompt = prompt
        self.expected_tools = expected_tools or []


class Evaluator:
    """Evaluates agent performance with before/after prompts.

    Uses synthetic tasks and simple rubrics.
    """

    def __init__(self, llm_client: Optional[Callable] = None):
        """Initialize evaluator.

        Args:
            llm_client: Optional LLM client for running evaluations.
                       If None, uses mock evaluation.
        """
        self._llm_client = llm_client

    def evaluate_patch(
        self,
        profile: AgentProfile,
        old_prompt: str,
        new_prompt: str,
        recent_traces: list[AgentTrace],
    ) -> EvalReport:
        """Evaluate a prompt patch using counterfactual evaluation.

        Generates synthetic tasks, runs them with old and new prompts,
        and compares performance.

        Args:
            profile: Agent profile
            old_prompt: Current prompt
            new_prompt: Proposed new prompt
            recent_traces: Recent execution traces (for context)

        Returns:
            EvalReport with metrics and pass/fail decision
        """
        # Generate synthetic tasks based on agent role
        tasks = self._generate_synthetic_tasks(profile, recent_traces)

        # Run tasks with old prompt
        metrics_before = self._run_tasks(profile, old_prompt, tasks)

        # Run tasks with new prompt
        metrics_after = self._run_tasks(profile, new_prompt, tasks)

        # Determine pass/fail
        pass_fail = self._determine_pass_fail(metrics_before, metrics_after)

        notes = self._generate_notes(metrics_before, metrics_after)

        return EvalReport(
            agent_id=profile.id,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            pass_fail=pass_fail,
            notes=notes,
            applied_version=None if pass_fail == "fail" else profile.current_version,
        )

    def _generate_synthetic_tasks(
        self, profile: AgentProfile, traces: list[AgentTrace]
    ) -> list[SyntheticTask]:
        """Generate synthetic tasks based on agent role and past behavior."""
        tasks = []

        # Role-based task generation
        if "plan" in profile.name.lower():
            tasks.append(
                SyntheticTask(
                    "plan_1",
                    "Create a plan for building a REST API with user authentication",
                    ["search", "read_docs"],
                )
            )
            tasks.append(
                SyntheticTask(
                    "plan_2",
                    "Break down the task of refactoring a monolithic app into microservices",
                    ["analyze", "list_components"],
                )
            )
        elif "code" in profile.name.lower():
            tasks.append(
                SyntheticTask(
                    "code_1",
                    "Write a Python function to validate email addresses",
                    ["write_file"],
                )
            )
            tasks.append(
                SyntheticTask(
                    "code_2",
                    "Implement error handling for a database connection function",
                    ["read_file", "write_file"],
                )
            )
        elif "review" in profile.name.lower():
            tasks.append(
                SyntheticTask(
                    "review_1",
                    "Review this code for security vulnerabilities: def login(user, pwd): return db.query(f'SELECT * FROM users WHERE name={user}')",
                    ["analyze_code"],
                )
            )
            tasks.append(
                SyntheticTask(
                    "review_2",
                    "Check if this implementation follows best practices for error handling",
                    ["read_file", "check_style"],
                )
            )
        else:
            # Generic tasks
            tasks.append(
                SyntheticTask("generic_1", "Analyze the given input and provide recommendations")
            )
            tasks.append(
                SyntheticTask("generic_2", "Summarize the key points from this context")
            )

        return tasks[:3]  # Limit to 3 tasks

    def _run_tasks(
        self, profile: AgentProfile, prompt: str, tasks: list[SyntheticTask]
    ) -> EvalMetrics:
        """Run tasks and compute metrics.

        In production, this would call the LLM. For demo, we simulate.
        """
        if self._llm_client:
            return self._run_tasks_with_llm(profile, prompt, tasks)
        else:
            return self._simulate_task_execution(profile, prompt, tasks)

    def _run_tasks_with_llm(
        self, profile: AgentProfile, prompt: str, tasks: list[SyntheticTask]
    ) -> EvalMetrics:
        """Run tasks using actual LLM (not implemented in demo)."""
        # This would invoke the LLM client for each task
        # For now, fall back to simulation
        return self._simulate_task_execution(profile, prompt, tasks)

    def _simulate_task_execution(
        self, profile: AgentProfile, prompt: str, tasks: list[SyntheticTask]
    ) -> EvalMetrics:
        """Simulate task execution for demo purposes.

        Uses heuristics based on prompt characteristics.
        """
        # Analyze prompt characteristics
        prompt_length = len(prompt)
        has_conciseness_guidance = any(
            word in prompt.lower()
            for word in ["concise", "brief", "short", "succinct"]
        )
        has_tool_guidance = any(
            word in prompt.lower()
            for word in ["tool", "function", "call", "use"]
        )
        has_safety_guidance = any(
            word in prompt.lower()
            for word in ["safe", "secure", "policy", "never", "don't"]
        )
        has_format_guidance = any(
            word in prompt.lower()
            for word in ["json", "format", "structure", "schema"]
        )

        # Compute metrics based on prompt quality indicators
        instruction_adherence = 5.0
        if has_format_guidance:
            instruction_adherence += 2.0
        if prompt_length > 500:
            instruction_adherence += 1.0

        tool_correctness = 5.0
        if has_tool_guidance:
            tool_correctness += 3.0
        if any(tool in prompt.lower() for tool in profile.tools):
            tool_correctness += 1.0

        conciseness = 5.0
        if has_conciseness_guidance:
            conciseness += 3.0
        else:
            # Penalize if no conciseness guidance
            conciseness -= 1.0

        safety_compliance = has_safety_guidance

        # Add some randomness for realism
        random.seed(hash(prompt) % 10000)
        instruction_adherence = min(10.0, instruction_adherence + random.uniform(-0.5, 0.5))
        tool_correctness = min(10.0, tool_correctness + random.uniform(-0.5, 0.5))
        conciseness = min(10.0, conciseness + random.uniform(-0.5, 0.5))

        return EvalMetrics(
            instruction_adherence=instruction_adherence,
            tool_correctness=tool_correctness,
            conciseness=conciseness,
            safety_compliance=safety_compliance,
            task_success_rate=0.7 + random.uniform(0, 0.2),
            avg_tokens=prompt_length * 0.3,  # Rough proxy
            hallucination_risk=max(0, 5.0 - prompt_length / 200),
        )

    def _determine_pass_fail(
        self, before: EvalMetrics, after: EvalMetrics
    ) -> str:
        """Determine if the patch should be applied.

        Pass if:
        - All metrics improve or stay roughly the same
        - Safety compliance is maintained
        """
        if not after.safety_compliance:
            return "fail"

        # Compute overall score
        def overall_score(m: EvalMetrics) -> float:
            return (
                m.instruction_adherence
                + m.tool_correctness
                + m.conciseness
            ) / 3.0

        before_score = overall_score(before)
        after_score = overall_score(after)

        # Allow small regression (within 0.5 points)
        if after_score >= before_score - 0.5:
            return "pass"
        else:
            return "fail"

    def _generate_notes(
        self, before: EvalMetrics, after: EvalMetrics
    ) -> str:
        """Generate human-readable notes about the evaluation."""
        notes = []

        delta_adherence = after.instruction_adherence - before.instruction_adherence
        delta_tools = after.tool_correctness - before.tool_correctness
        delta_concise = after.conciseness - before.conciseness

        if delta_adherence > 0.5:
            notes.append(f"Instruction adherence improved (+{delta_adherence:.1f})")
        elif delta_adherence < -0.5:
            notes.append(f"Instruction adherence decreased ({delta_adherence:.1f})")

        if delta_tools > 0.5:
            notes.append(f"Tool usage improved (+{delta_tools:.1f})")
        elif delta_tools < -0.5:
            notes.append(f"Tool usage decreased ({delta_tools:.1f})")

        if delta_concise > 0.5:
            notes.append(f"Conciseness improved (+{delta_concise:.1f})")
        elif delta_concise < -0.5:
            notes.append(f"Conciseness decreased ({delta_concise:.1f})")

        if not after.safety_compliance:
            notes.append("CRITICAL: Safety compliance violated")

        return "; ".join(notes) if notes else "No significant changes"
