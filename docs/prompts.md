# Prompts

This file is part of the system design. Prompts are not throwaway notes: they capture product intent, architecture constraints, and patch history.

Rules for future updates:

- append new prompt context instead of overwriting older prompt history
- preserve earlier instructions unless explicitly superseded
- read this file before making architectural or workflow changes

## Foundation Prompt

Build a minimal but well-structured Python project that implements an LLM-based agent for diagnosing issues in randomized controlled experiments (RCTs), with a focus on reflection and light tool use.

Core expectations:

- keep a single-agent pipeline with `Analyzer -> Critic -> Reviser`
- include light tool use for SRM and pre-period balance checks
- keep structured JSON outputs
- provide synthetic experiment generation
- provide evaluation before and after reflection
- integrate LangSmith tracing
- explain the system with notebooks

## Patch Prompt: Data Realism and Persistence

Extend the existing project by improving the synthetic data generator realism and enforcing a strict data persistence policy.

Patch constraints:

- this is a patch to the existing system
- do not modify agent logic, workflow, or tools
- do not rewrite architecture
- only improve data generation and persistence behavior
- keep changes backward compatible

Patch requirements:

- separate campaign-level and experiment-level synthetic fields
- add explicit hidden causal truth such as baseline rate, treatment effect, incremental conversions, and iROAS
- generate conversions with binomial processes
- link pre-period and post-period outcomes
- replace free-form latent assumptions with structured latent factors while keeping old fields for compatibility
- enforce generate-once, reuse-always behavior
- avoid in-memory-only datasets
- check for dataset existence before generation
- use deterministic dataset identity through `dataset_id` and `random_seed`
- store datasets under `data/synthetic/dataset_v1/campaigns.parquet`
- use a single `load_or_generate_data(config)` entrypoint

Why this matters:

- prompts encode both functionality and guardrails
- later contributors should understand why the architecture stayed stable while the data layer evolved
- keeping prompt history makes incremental patching safer than silent rewrites
