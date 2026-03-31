from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .schemas import ExperimentSummary, IssueName, LatentFactors, SegmentSummary


DATASET_VERSION = "dataset_v1"
DATASET_ROOT = Path(__file__).resolve().parents[2] / "data" / "synthetic"
DEFAULT_DATASET_PATH = DATASET_ROOT / DATASET_VERSION / "campaigns.parquet"
LEGACY_DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "synthetic_experiments.jsonl"


@dataclass(frozen=True)
class DatasetConfig:
    count: int = 25
    seed: int = 7
    failure_modes: Sequence[str] | None = None
    dataset_version: str = DATASET_VERSION
    dataset_id: str | None = None
    dataset_path: str | Path | None = None

    @property
    def resolved_dataset_id(self) -> str:
        if self.dataset_id:
            return self.dataset_id
        return f"synthetic_rct_{self.dataset_version}_seed_{self.seed}_count_{self.count}"

    @property
    def path(self) -> Path:
        if self.dataset_path is not None:
            return Path(self.dataset_path)
        return DATASET_ROOT / self.dataset_version / "campaigns.parquet"


@dataclass
class FailureConfig:
    srm: bool = False
    imbalance: bool = False
    low_power: bool = False
    outliers: bool = False
    simpsons_paradox: bool = False

    def labels(self) -> List[IssueName]:
        labels: List[IssueName] = []
        if self.srm:
            labels.append("srm")
        if self.imbalance:
            labels.append("pre_period_imbalance")
        if self.low_power:
            labels.append("low_statistical_power")
        if self.outliers:
            labels.append("outliers")
        if self.simpsons_paradox:
            labels.append("simpsons_paradox")
        return labels


class SyntheticExperimentGenerator:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.np_random = np.random.default_rng(seed)

    def sample_failure_config(self) -> FailureConfig:
        return FailureConfig(
            srm=bool(self.np_random.random() < 0.25),
            imbalance=bool(self.np_random.random() < 0.25),
            low_power=bool(self.np_random.random() < 0.35),
            outliers=bool(self.np_random.random() < 0.20),
            simpsons_paradox=bool(self.np_random.random() < 0.15),
        )

    def generate(
        self,
        experiment_id: str,
        failure_modes: Sequence[str] | None = None,
        *,
        dataset_id: str | None = None,
        random_seed: int | None = None,
    ) -> ExperimentSummary:
        config = self._config_from_names(failure_modes) if failure_modes else self.sample_failure_config()

        impressions = int(self.np_random.integers(50_000, 5_000_001))
        ctr = float(self.np_random.uniform(0.008, 0.045))
        clicks = int(self.np_random.binomial(impressions, ctr))
        avg_conversion_value = round(float(self.np_random.uniform(35.0, 220.0)), 2)
        avg_cpc = float(self.np_random.uniform(0.35, 3.2))
        spend = round(clicks * avg_cpc * float(self.np_random.uniform(0.92, 1.08)), 2)

        total_users = int(min(max(clicks, 800), self.np_random.integers(2_000, 30_001)))
        if config.low_power:
            total_users = int(max(300, total_users * 0.18))

        treatment_share = 0.5
        if config.srm:
            treatment_share = float(self.np_random.choice([0.36, 0.41, 0.62, 0.67]))
        treatment_users = int(round(total_users * treatment_share))
        control_users = max(1, total_users - treatment_users)
        treatment_users = max(1, treatment_users)

        true_baseline_conversion_rate = float(self.np_random.uniform(0.03, 0.16))
        treatment_effect = float(self.np_random.uniform(-0.003, 0.018))
        if config.low_power:
            treatment_effect *= 0.2

        noise_level = self._noise_level(config)
        noise_scale = {"low": 0.002, "medium": 0.006, "high": 0.012}[noise_level]
        common_trend = float(self.np_random.normal(0.002, noise_scale / 2.0))
        imbalance_shift = float(self.np_random.normal(0.0, noise_scale / 2.0))
        if config.imbalance:
            imbalance_shift += float(self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(0.01, 0.03))

        control_pre_mean = self._clip_rate(true_baseline_conversion_rate + float(self.np_random.normal(0.0, noise_scale / 2.0)))
        treatment_pre_mean = self._clip_rate(control_pre_mean + imbalance_shift)
        control_post_mean = self._clip_rate(control_pre_mean + common_trend + float(self.np_random.normal(0.0, noise_scale)))
        treatment_post_mean = self._clip_rate(
            treatment_pre_mean + common_trend + treatment_effect + float(self.np_random.normal(0.0, noise_scale))
        )

        segment_summaries: List[SegmentSummary] = []
        treatment_heterogeneity = bool(config.simpsons_paradox or self.np_random.random() < 0.2)
        if config.simpsons_paradox:
            segment_summaries, control_post_mean, treatment_post_mean = self._simpsons_segments()
            treatment_heterogeneity = True

        control_conversions = int(self.np_random.binomial(control_users, control_post_mean))
        treatment_conversions = int(self.np_random.binomial(treatment_users, treatment_post_mean))
        total_conversions = control_conversions + treatment_conversions
        revenue = round(total_conversions * avg_conversion_value, 2)

        control_conversion_rate = round(control_conversions / control_users, 4)
        treatment_conversion_rate = round(treatment_conversions / treatment_users, 4)
        control_pre_std = self._rate_std(control_pre_mean, control_users)
        treatment_pre_std = self._rate_std(treatment_pre_mean, treatment_users)
        control_post_std = self._rate_std(control_post_mean, control_users)
        treatment_post_std = self._rate_std(treatment_post_mean, treatment_users)
        if config.outliers:
            control_post_std = round(control_post_std * float(self.np_random.uniform(1.8, 2.5)), 4)
            treatment_post_std = round(treatment_post_std * float(self.np_random.uniform(1.9, 2.8)), 4)

        true_incremental_conversions = round(treatment_users * treatment_effect, 4)
        true_incremental_revenue = true_incremental_conversions * avg_conversion_value
        true_iroas = round(true_incremental_revenue / spend, 4) if spend else 0.0

        notes = self._notes_for_compatibility(config, noise_level, treatment_heterogeneity)
        latent_factors = LatentFactors(
            pre_period_imbalance=config.imbalance,
            outliers=config.outliers,
            treatment_heterogeneity=treatment_heterogeneity,
            noise_level=noise_level,
        )

        return ExperimentSummary(
            experiment_id=experiment_id,
            hypothesis="Treatment should improve conversion rate.",
            dataset_id=dataset_id,
            random_seed=random_seed,
            expected_treatment_share=0.5,
            impressions=impressions,
            clicks=clicks,
            spend=spend,
            conversions=total_conversions,
            avg_conversion_value=avg_conversion_value,
            revenue=revenue,
            control_size=control_users,
            treatment_size=treatment_users,
            control_users=control_users,
            treatment_users=treatment_users,
            control_conversions=control_conversions,
            treatment_conversions=treatment_conversions,
            control_pre_mean=round(control_pre_mean, 4),
            treatment_pre_mean=round(treatment_pre_mean, 4),
            control_pre_std=control_pre_std,
            treatment_pre_std=treatment_pre_std,
            control_post_mean=round(control_post_mean, 4),
            treatment_post_mean=round(treatment_post_mean, 4),
            control_post_std=control_post_std,
            treatment_post_std=treatment_post_std,
            control_conversion_rate=control_conversion_rate,
            treatment_conversion_rate=treatment_conversion_rate,
            notes=notes,
            latent_factors=latent_factors,
            segment_summaries=segment_summaries,
            hidden_truth=config.labels(),
            true_baseline_conversion_rate=round(true_baseline_conversion_rate, 4),
            true_treatment_effect=round(treatment_effect, 4),
            true_incremental_conversions=true_incremental_conversions,
            true_iROAS=true_iroas,
        )

    def generate_many(
        self,
        count: int,
        failure_modes: Sequence[str] | None = None,
        *,
        dataset_id: str | None = None,
        random_seed: int | None = None,
    ) -> List[ExperimentSummary]:
        return [
            self.generate(
                f"exp_{idx:04d}",
                failure_modes=failure_modes,
                dataset_id=dataset_id,
                random_seed=random_seed,
            )
            for idx in range(count)
        ]

    def _config_from_names(self, names: Sequence[str]) -> FailureConfig:
        normalized = {name.strip().lower() for name in names}
        return FailureConfig(
            srm="srm" in normalized,
            imbalance="imbalance" in normalized or "pre_period_imbalance" in normalized,
            low_power="low_power" in normalized or "low_statistical_power" in normalized,
            outliers="outliers" in normalized,
            simpsons_paradox="simpsons" in normalized or "simpsons_paradox" in normalized,
        )

    def _noise_level(self, config: FailureConfig) -> str:
        if config.outliers:
            return "high"
        if config.low_power:
            return "medium"
        return str(self.np_random.choice(["low", "medium", "medium", "high"]))

    def _simpsons_segments(self) -> tuple[List[SegmentSummary], float, float]:
        returning = SegmentSummary(
            name="returning_users",
            weight_control=0.30,
            weight_treatment=0.70,
            control_conversion_rate=0.27,
            treatment_conversion_rate=0.25,
        )
        new_users = SegmentSummary(
            name="new_users",
            weight_control=0.70,
            weight_treatment=0.30,
            control_conversion_rate=0.06,
            treatment_conversion_rate=0.05,
        )
        control_rate = (returning.weight_control * returning.control_conversion_rate) + (
            new_users.weight_control * new_users.control_conversion_rate
        )
        treatment_rate = (returning.weight_treatment * returning.treatment_conversion_rate) + (
            new_users.weight_treatment * new_users.treatment_conversion_rate
        )
        return [returning, new_users], round(control_rate, 4), round(treatment_rate, 4)

    def _clip_rate(self, value: float) -> float:
        return float(np.clip(value, 0.001, 0.999))

    def _rate_std(self, rate: float, sample_size: int) -> float:
        return round(float(np.sqrt(max(rate * (1.0 - rate), 1e-9) / max(sample_size, 1))), 4)

    def _notes_for_compatibility(
        self,
        config: FailureConfig,
        noise_level: str,
        treatment_heterogeneity: bool,
    ) -> List[str]:
        notes: List[str] = []
        if config.outliers:
            notes.append("A small number of extreme spenders may be stretching the post-period variance.")
        if config.low_power:
            notes.append("Observed lift is small relative to noise and sample size.")
        if config.simpsons_paradox:
            notes.append("Segment mix changed noticeably between control and treatment.")
        if treatment_heterogeneity:
            notes.append("Treatment response differs across audience segments.")
        notes.append(f"Latent noise level is {noise_level}.")
        return notes


def experiments_to_rows(experiments: Iterable[ExperimentSummary]) -> List[dict]:
    return [experiment.model_dump() for experiment in experiments]


def _row_for_storage(experiment: ExperimentSummary) -> dict:
    row = experiment.model_dump()
    for field_name in ("notes", "hidden_truth", "segment_summaries", "latent_factors"):
        row[field_name] = json.dumps(row[field_name])
    return row


def _row_from_storage(row: dict) -> ExperimentSummary:
    parsed = dict(row)
    for field_name in ("notes", "hidden_truth", "segment_summaries", "latent_factors"):
        value = parsed.get(field_name)
        if isinstance(value, str):
            parsed[field_name] = json.loads(value)
    return ExperimentSummary.model_validate(parsed)


def save_experiments(experiments: Iterable[ExperimentSummary], output_path: str | Path = DEFAULT_DATASET_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for experiment in experiments:
                handle.write(json.dumps(experiment.model_dump()))
                handle.write("\n")
        return path

    rows = [_row_for_storage(experiment) for experiment in experiments]
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def load_experiments(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> List[ExperimentSummary]:
    path = Path(dataset_path)
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [ExperimentSummary.model_validate_json(line) for line in handle if line.strip()]
    frame = pd.read_parquet(path)
    return [_row_from_storage(record) for record in frame.to_dict(orient="records")]


def load_or_generate_data(config: DatasetConfig) -> List[ExperimentSummary]:
    path = config.path
    if path.exists():
        return load_experiments(path)

    generator = SyntheticExperimentGenerator(seed=config.seed)
    experiments = generator.generate_many(
        count=config.count,
        failure_modes=config.failure_modes,
        dataset_id=config.resolved_dataset_id,
        random_seed=config.seed,
    )
    save_experiments(experiments, output_path=path)
    return load_experiments(path)


def ensure_dataset(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    count: int = 25,
    seed: int = 7,
    failure_modes: Sequence[str] | None = None,
) -> Path:
    path = Path(dataset_path)
    if path.exists():
        return path

    if path == DEFAULT_DATASET_PATH:
        load_or_generate_data(DatasetConfig(count=count, seed=seed, failure_modes=failure_modes))
        return path

    generator = SyntheticExperimentGenerator(seed=seed)
    experiments = generator.generate_many(
        count=count,
        failure_modes=failure_modes,
        dataset_id=f"custom_dataset_seed_{seed}_count_{count}",
        random_seed=seed,
    )
    save_experiments(experiments, output_path=path)
    return path
