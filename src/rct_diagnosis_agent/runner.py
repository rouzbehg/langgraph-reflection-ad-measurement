from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

from .agent import build_default_agent
from .data import DatasetConfig, DEFAULT_DATASET_PATH, load_experiments, load_or_generate_data
from .evaluation import evaluate_agent


def _parse_failure_modes(raw: str | None) -> Sequence[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the RCT diagnosis reflection agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="Generate and analyze one synthetic experiment.")
    single_parser.add_argument("--seed", type=int, default=7)
    single_parser.add_argument("--failure-modes", type=str, default=None)
    single_parser.add_argument("--model", type=str, default="gpt-4o-mini")
    single_parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    single_parser.add_argument("--dataset-count", type=int, default=25)
    single_parser.add_argument("--experiment-index", type=int, default=0)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on multiple synthetic experiments.")
    eval_parser.add_argument("--count", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=7)
    eval_parser.add_argument("--failure-modes", type=str, default=None)
    eval_parser.add_argument("--model", type=str, default="gpt-4o-mini")
    eval_parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    eval_parser.add_argument("--dataset-count", type=int, default=25)

    args = parser.parse_args()
    agent = build_default_agent(model=args.model)
    failure_modes = _parse_failure_modes(args.failure_modes)
    dataset_path = Path(args.dataset_path)

    experiments = load_or_generate_data(
        DatasetConfig(
            count=args.dataset_count,
            seed=args.seed,
            failure_modes=failure_modes,
            dataset_path=dataset_path,
        )
    )

    if args.command == "single":
        experiment = experiments[args.experiment_index]
        result = agent.run(experiment)
        print(json.dumps(result.model_dump(), indent=2))
        return

    rows, summary = evaluate_agent(agent, experiments[: args.count])
    print(json.dumps({"summary": summary.model_dump(), "rows": [row.model_dump() for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
