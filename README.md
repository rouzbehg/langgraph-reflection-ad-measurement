# LangGraph Reflection Agent for RCT Diagnosis

This repository contains a minimal but well-structured educational project for diagnosing problems in randomized controlled experiments with an LLM-based agent. The focus is on agentic patterns, not production ML systems.

## What is included

- Synthetic experiment generator with controllable failure modes
- Disk-backed synthetic dataset workflow with strict load-or-generate behavior
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

4. Materialize the reusable synthetic dataset once:

```bash
rct-materialize --count 25 --seed 7
```

5. Generate and diagnose one stored experiment:

```bash
rct-diagnose single --dataset-path data/synthetic/dataset_v1/campaigns.parquet --experiment-index 0
```

6. Run a small evaluation against the stored dataset:

```bash
rct-diagnose evaluate --dataset-path data/synthetic/dataset_v1/campaigns.parquet --count 12
```

## CLI

After `pip install -e .`, the project exposes two console commands:

- `rct-materialize`
  - create or reuse the persisted synthetic dataset
- `rct-diagnose`
  - run the agent pipeline with the existing `single` and `evaluate` subcommands

Examples:

```bash
rct-materialize --count 25 --seed 7
rct-diagnose single --experiment-index 0
rct-diagnose evaluate --count 12
```

## Project layout

- `src/rct_diagnosis_agent/data.py`: synthetic experiment generator
- `src/rct_diagnosis_agent/materialize.py`: create or reuse on-disk synthetic datasets
- `src/rct_diagnosis_agent/tools.py`: SRM and pre-period balance tools
- `src/rct_diagnosis_agent/agent.py`: Analyzer, Critic, Reviser, and LangGraph orchestration
- `src/rct_diagnosis_agent/evaluation.py`: evaluation loop and metrics
- `src/rct_diagnosis_agent/runner.py`: CLI entrypoints
- `src/rct_diagnosis_agent/tracing.py`: LangSmith helpers
- `prompts/`: versioned prompts for each stage
- `notebooks/`: walkthrough notebooks

## Notes

- The project uses experiment-level summaries rather than raw user-level logs.
- Synthetic data is stored on disk as `data/synthetic/dataset_v1/campaigns.parquet`.
- The standard persisted dataset entrypoint is `load_or_generate_data(config)`.
- All standard entrypoints use a generate-if-missing pattern: they load the existing versioned dataset first and only generate one if the file does not exist.
- Each stored experiment includes campaign-level metrics, treatment/control arm metrics for impressions, clicks, spend, revenue, experiment aggregates, latent factors, and hidden causal truth for evaluation only.
- Reflection is implemented as an explicit critique-and-revision pass.
- Tool use is intentionally light and easy to inspect.
- If LangSmith is configured, each stage and tool invocation will appear in the trace.
