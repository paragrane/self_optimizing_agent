# ContextFlow

**Self-optimizing context for multi-agent systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

ContextFlow is a framework-agnostic library that enables continuous, autonomous improvement of agent context (system prompts, memory, tool hints) in multi-agent systems. It implements a Self-Context Optimization Agent (SCOA) that observes behavior, diagnoses issues, proposes updates, validates safety, and applies improvements—all without modifying business logic.

## Why ContextFlow?

Traditional agentic systems suffer from:
- **Prompt drift**: Static prompts degrade as requirements evolve
- **Verbosity creep**: Agents become increasingly verbose over time
- **Tool misuse**: Agents call wrong tools or use incorrect parameters
- **Context pollution**: Agent memory grows unbounded

ContextFlow solves these by treating **agent context as a living artifact** that evolves based on observed behavior.

## Key Features

- ✅ **Sequential optimization**: Optimizes agents one-by-one for stable convergence
- ✅ **Framework-agnostic**: Works with LangGraph, CrewAI, and custom agents
- ✅ **Safety-first**: Hard constraints prevent removing safety policies, introducing secrets, or adding unauthorized tools
- ✅ **Versioned & auditable**: Full version history with rollback support
- ✅ **Counterfactual evaluation**: Tests patches before applying
- ✅ **Production-ready**: Clean APIs, type hints, comprehensive tests

## Installation

```bash
pip install contextflow
```

For development:
```bash
git clone https://github.com/contextflow/contextflow.git
cd contextflow
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
from contextflow import ContextFlow, AgentProfile, AgentTrace
from contextflow.adapters.base import SimpleAdapter

# Initialize
cf = ContextFlow()
adapter = SimpleAdapter()

# Register an agent
profile = AgentProfile(
    id="coder",
    name="CoderAgent",
    framework="custom",
    base_system_prompt="You are a coding assistant.",
    current_system_prompt="You are a coding assistant.",
    tools=["write_file", "read_file"],
    policies={"safety": "Never write malicious code"},
)

agent_handle = {"system_prompt": profile.current_system_prompt}
cf.register_agent(profile, adapter, agent_handle)

# Ingest execution traces
traces = [
    AgentTrace(
        agent_id="coder",
        input="Write hello world",
        output="Let me write a comprehensive hello world program with extensive documentation...",  # Too verbose!
        outcome="success",
        tokens=1500,  # High token count
        latency_ms=3000,
    )
]
cf.ingest_traces(traces)

# Run optimization
reports = cf.run_once(agent_ids=["coder"])

# View results
for report in reports:
    print(f"Agent: {report.agent_id}")
    print(f"Status: {report.pass_fail}")
    print(f"Conciseness: {report.metrics_before.conciseness:.1f} → {report.metrics_after.conciseness:.1f}")
```

### With LangGraph

```python
from contextflow import ContextFlow, AgentProfile
from contextflow.adapters.langgraph_adapter import LangGraphAdapter

cf = ContextFlow()
adapter = LangGraphAdapter()

# Your LangGraph agent
langgraph_agent = {
    "system_message": "You are a planning agent.",
    "tools": ["search", "analyze"],
}

profile = AgentProfile(
    id="planner",
    name="PlannerAgent",
    framework="langgraph",
    base_system_prompt=langgraph_agent["system_message"],
    current_system_prompt=langgraph_agent["system_message"],
    tools=langgraph_agent["tools"],
)

cf.register_agent(profile, adapter, langgraph_agent)
# ... ingest traces and optimize
```

### With CrewAI

```python
from contextflow import ContextFlow, AgentProfile
from contextflow.adapters.crewai_adapter import CrewAIAdapter

cf = ContextFlow()
adapter = CrewAIAdapter()

# Your CrewAI agent
crewai_agent = {
    "role": "Senior Engineer",
    "goal": "Write high-quality code",
    "backstory": "You are an expert developer...",
    "tools": ["write_code", "run_tests"],
}

profile = AgentProfile(
    id="coder",
    name="CoderAgent",
    framework="crewai",
    base_system_prompt=adapter.get_prompt(crewai_agent),
    current_system_prompt=adapter.get_prompt(crewai_agent),
    tools=["write_code", "run_tests"],
)

cf.register_agent(profile, adapter, crewai_agent)
# ... ingest traces and optimize
```

## CLI Usage

ContextFlow provides a CLI for quick optimization runs:

```bash
# Optimize all agents
contextflow run-once --agents all --config agents.json --traces traces.jsonl

# Optimize specific agents
contextflow run-once --agents PlannerAgent,CoderAgent --config agents.json

# Rollback to previous version
contextflow rollback --agent CoderAgent --version v3 --config agents.json
```

**Configuration format** (`agents.json`):
```json
[
  {
    "id": "coder",
    "name": "CoderAgent",
    "framework": "custom",
    "base_system_prompt": "You are a coding assistant.",
    "current_system_prompt": "You are a coding assistant.",
    "tools": ["write_file", "read_file"],
    "policies": {"safety": "Never write malicious code"},
    "current_version": "v1"
  }
]
```

**Traces format** (`traces.jsonl`):
```jsonl
{"agent_id": "coder", "input": "test", "output": "result", "outcome": "success", "tokens": 100, "latency_ms": 500, "timestamp": "2024-01-01T12:00:00"}
```

## How It Works

### Sequential Optimization Loop

For each agent (in sequence):

1. **Collect**: Fetch recent execution traces
2. **Diagnose**: Identify issues (verbosity, tool misuse, errors, etc.)
3. **Propose**: Generate a prompt patch with improvements
4. **Validate**: Run safety checks:
   - Policy lock (can't remove safety constraints)
   - Secret scan (no API keys/tokens in prompts)
   - Tool scope (can't add unauthorized tools)
   - Injection hardening (must resist prompt injection)
5. **Evaluate**: Run counterfactual tests (before vs after)
6. **Apply**: If passed, update the prompt
7. **Log**: Record all changes for audit trail

### Safety Guarantees

ContextFlow enforces hard constraints:

- ✅ **Policy Lock**: Never removes or weakens safety/privacy/compliance text
- ✅ **Secret Scan**: Rejects patches containing API keys, tokens, passwords
- ✅ **Tool Scope**: Rejects patches referencing unauthorized tools
- ✅ **Injection Hardening**: Ensures prompts resist user override attempts

Any patch that fails safety checks is rejected and logged.

### Versioning & Rollback

Every prompt update creates a new version (`v1`, `v2`, `v3`, ...). Full history is maintained:

```python
# Rollback to previous version
cf.rollback("coder", "v2")

# View version history
prompt_store = cf.get_prompt_store()
versions = prompt_store.list_versions("coder")
print(versions)  # ['v1', 'v2', 'v3']
```

## Examples

See `contextflow/examples/` for complete demos:

- **`demo_langgraph.py`**: Full LangGraph integration with 3 agents
- **`demo_crewai.py`**: CrewAI integration showing verbosity fixes

Run demos:
```bash
python -m contextflow.examples.demo_langgraph
python -m contextflow.examples.demo_crewai
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ContextFlow Facade                      │
│  - register_agent()   - run_once()   - rollback()           │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌─────▼──────┐   ┌─────▼──────┐
    │ Storage │      │   SCOA     │   │  Adapters  │
    │ Layer   │      │ Optimizer  │   │ (LangGraph,│
    └─────────┘      └────────────┘   │  CrewAI)   │
                                      └────────────┘
```

**Core Components**:
- `SCOA`: Orchestrates optimization loop
- `Optimizer`: Generates prompt improvements
- `Evaluator`: Runs counterfactual evaluation
- `SafetyValidator`: Enforces safety constraints
- `PromptStore`, `MemoryStore`, `TraceStore`: Versioned storage
- `Adapters`: Framework integration (LangGraph, CrewAI, custom)

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest contextflow/tests/test_safety.py -v

# Run with coverage
pytest --cov=contextflow --cov-report=html
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black contextflow/

# Type checking
mypy contextflow/

# Linting
ruff contextflow/
```

## Roadmap

- [ ] Multi-agent co-optimization (consider inter-agent dependencies)
- [ ] Real-time streaming optimization
- [ ] RL-based optimization (beyond LLM reasoning)
- [ ] Federated learning (share improvements across deployments)
- [ ] Integration with more frameworks (AutoGen, etc.)
- [ ] Cloud-native storage backends (PostgreSQL, Redis)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use ContextFlow in your research, please cite:

```bibtex
@software{contextflow2024,
  title = {ContextFlow: Self-optimizing context for multi-agent systems},
  author = {ContextFlow Contributors},
  year = {2024},
  url = {https://github.com/contextflow/contextflow}
}
```

## Acknowledgments

Built with:
- [Pydantic](https://docs.pydantic.dev/) for data validation
- Inspired by meta-learning, prompt optimization, and self-improving systems research

---

**Questions?** Open an [issue](https://github.com/contextflow/contextflow/issues) or start a [discussion](https://github.com/contextflow/contextflow/discussions).
