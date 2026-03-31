from __future__ import annotations

import argparse

from .data import DEFAULT_DATASET_PATH, ensure_dataset, load_experiments, save_experiments, SyntheticExperimentGenerator


def _parse_failure_modes(raw: str | None):
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize synthetic RCT experiment data to disk.")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--failure-modes", type=str, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    failure_modes = _parse_failure_modes(args.failure_modes)
    if args.force:
        generator = SyntheticExperimentGenerator(seed=args.seed)
        path = save_experiments(
            generator.generate_many(count=args.count, failure_modes=failure_modes),
            output_path=args.output,
        )
    else:
        path = ensure_dataset(
            dataset_path=args.output,
            count=args.count,
            seed=args.seed,
            failure_modes=failure_modes,
        )

    print(f"{path} ({len(load_experiments(path))} experiments)")


if __name__ == "__main__":
    main()
