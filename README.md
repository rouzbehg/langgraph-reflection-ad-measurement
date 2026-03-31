# LangGraph Reflection Agent for RCT Diagnosis

This repository contains a minimal but well-structured educational project for diagnosing problems in randomized controlled experiments with an LLM-based agent. The focus is on agentic patterns, not production ML systems.

## What is included

- Synthetic experiment generator with controllable failure modes
- A single-agent pipeline with `Analyzer -> Critic -> Reviser`
- Light tool use for SRM and pre-period balance checks
- Structured JSON outputs at every stage
- LangSmith tracing hooks for prompts, tool calls, and stage outputs
- Evaluation utilities that compare pre-reflection and post-reflection predictions
- Three notebooks with step-by-step explanations

## Quick start

1. Create a Python 3.10+ virtual environment.
2. Install the project:

```bash
pip install -e .
```

3. Ensure the copied `.env` file contains the keys you want to use. Typical variables:

```bash
export OPENAI_API_KEY=...
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=langgraph-reflection-ad-measurement
```

4. Generate and diagnose one synthetic experiment:

```bash
python -m rct_diagnosis_agent.runner single --seed 7 --failure-modes srm imbalance
```

5. Run a small evaluation:

```bash
python -m rct_diagnosis_agent.runner evaluate --count 12 --seed 7
```

## Project layout

- `src/rct_diagnosis_agent/data.py`: synthetic experiment generator
- `src/rct_diagnosis_agent/tools.py`: SRM and pre-period balance tools
- `src/rct_diagnosis_agent/agent.py`: Analyzer, Critic, Reviser, and LangGraph orchestration
- `src/rct_diagnosis_agent/evaluation.py`: evaluation loop and metrics
- `src/rct_diagnosis_agent/runner.py`: CLI entrypoints
- `src/rct_diagnosis_agent/tracing.py`: LangSmith helpers
- `prompts/`: versioned prompts for each stage
- `notebooks/`: walkthrough notebooks

## Notes

- The project uses experiment-level summaries rather than raw user-level logs.
- Reflection is implemented as an explicit critique-and-revision pass.
- Tool use is intentionally light and easy to inspect.
- If LangSmith is configured, each stage and tool invocation will appear in the trace.
