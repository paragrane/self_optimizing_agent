"""
Core data models for ContextFlow.

All data structures use Pydantic for validation and serialization.
"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation by an agent."""

    tool_name: str
    arguments: dict
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class AgentTrace(BaseModel):
    """A single execution trace of an agent.

    Captures input, output, tool calls, and performance metrics.
    """

    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    input: str
    output: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    outcome: Literal["success", "error", "timeout"] = "success"
    errors: list[str] = Field(default_factory=list)
    tokens: int = 0
    latency_ms: float = 0.0

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentProfile(BaseModel):
    """Profile of an agent registered with ContextFlow.

    Contains identity, current prompt, policies, and configuration.
    """

    id: str
    name: str
    framework: Literal["langgraph", "crewai", "custom"] = "custom"
    base_system_prompt: str
    current_system_prompt: str
    tools: list[str] = Field(default_factory=list)
    policies: dict[str, str] = Field(default_factory=dict)
    owner: str = "default"
    current_version: str = "v1"
    memory_summary: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PromptPatch(BaseModel):
    """A proposed or applied change to an agent's context.

    Contains diff, rationale, and safety validation results.
    """

    agent_id: str
    version_from: str
    version_to: str
    diff: str  # Unified diff format
    rationale: str
    risk_flags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    new_prompt: str = ""  # Full new prompt

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EvalMetrics(BaseModel):
    """Evaluation metrics for an agent's performance."""

    instruction_adherence: float = Field(ge=0, le=10, default=5.0)
    tool_correctness: float = Field(ge=0, le=10, default=5.0)
    conciseness: float = Field(ge=0, le=10, default=5.0)
    safety_compliance: bool = True
    task_success_rate: float = Field(ge=0, le=1, default=0.5)
    avg_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    hallucination_risk: float = Field(ge=0, le=10, default=0.0)


class EvalReport(BaseModel):
    """Report from evaluating a prompt patch.

    Contains before/after metrics and pass/fail decision.
    """

    agent_id: str
    metrics_before: EvalMetrics
    metrics_after: EvalMetrics
    pass_fail: Literal["pass", "fail"] = "pass"
    notes: str = ""
    applied_version: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetyCheckResult(BaseModel):
    """Result from running safety validation on a patch."""

    passed: bool
    policy_lock_check: bool = True
    secret_scan_check: bool = True
    tool_scope_check: bool = True
    injection_hardening_check: bool = True
    violations: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OptimizationLog(BaseModel):
    """Audit log entry for an optimization attempt."""

    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    version_from: str
    version_to: Optional[str] = None
    patch: Optional[PromptPatch] = None
    safety_result: Optional[SafetyCheckResult] = None
    eval_report: Optional[EvalReport] = None
    applied: bool = False
    error: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
