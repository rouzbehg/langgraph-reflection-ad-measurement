# AGENTS

## Project Purpose

This repository is an educational LangGraph-based project for diagnosing issues in randomized controlled experiments with an LLM agent. The core goals are clarity, observability, reflection, and reproducible synthetic data.

## Key Components

- agent: analyzer, critic, reviser pipeline
- tools: SRM test and pre-period balance test
- data generator: synthetic campaign and experiment summary generation
- persistence layer: versioned synthetic dataset storage and reuse
- evaluation: before-vs-after reflection metrics
- notebooks: explanatory walkthroughs and analysis

## Working Rules

- do not rewrite architecture without explicit instruction
- prefer incremental patches over large refactors
- keep agent logic, workflow, and tool behavior stable unless directly asked to change them
- treat prompts and persistence rules as part of the system design
- prioritize reproducibility and backward compatibility when changing the data layer
