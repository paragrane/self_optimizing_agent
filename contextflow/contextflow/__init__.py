"""
ContextFlow: Self-optimizing context for multi-agent systems.

Main exports:
- ContextFlow: Main facade
- AgentProfile, AgentTrace, EvalReport: Core data models
- SCOA: Self-Context Optimization Agent orchestrator
"""

from contextflow.api import ContextFlow
from contextflow.core.scoa import SCOA
from contextflow.core.schemas import (
    AgentProfile,
    AgentTrace,
    ToolCall,
    EvalReport,
    EvalMetrics,
    PromptPatch,
    SafetyCheckResult,
    OptimizationLog,
)
from contextflow.adapters.base import BaseAdapter
from contextflow.storage.prompt_store import InMemoryPromptStore
from contextflow.storage.memory_store import InMemoryMemoryStore
from contextflow.storage.trace_store import InMemoryTraceStore

__version__ = "0.1.0"

__all__ = [
    "ContextFlow",
    "SCOA",
    "AgentProfile",
    "AgentTrace",
    "ToolCall",
    "EvalReport",
    "EvalMetrics",
    "PromptPatch",
    "SafetyCheckResult",
    "OptimizationLog",
    "BaseAdapter",
    "InMemoryPromptStore",
    "InMemoryMemoryStore",
    "InMemoryTraceStore",
]
