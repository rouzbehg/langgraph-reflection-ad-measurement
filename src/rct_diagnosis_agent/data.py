from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .schemas import ExperimentSummary, IssueName, SegmentSummary


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
        self.random = random.Random(seed)

    def sample_failure_config(self) -> FailureConfig:
        return FailureConfig(
            srm=self.random.random() < 0.25,
            imbalance=self.random.random() < 0.25,
            low_power=self.random.random() < 0.35,
            outliers=self.random.random() < 0.20,
            simpsons_paradox=self.random.random() < 0.15,
        )

    def generate(
        self,
        experiment_id: str,
        failure_modes: Sequence[str] | None = None,
    ) -> ExperimentSummary:
        config = self._config_from_names(failure_modes) if failure_modes else self.sample_failure_config()
        base_size = self.random.randint(1800, 9000)
        treatment_share = 0.5
        treatment_size = int(base_size * treatment_share)
        control_size = base_size - treatment_size

        if config.srm:
            treatment_share = self.random.choice([0.35, 0.4, 0.62, 0.68])
            treatment_size = int(base_size * treatment_share)
            control_size = base_size - treatment_size

        pre_mean = self.random.uniform(0.08, 0.18)
        control_pre_mean = pre_mean
        treatment_pre_mean = pre_mean
        control_pre_std = self.random.uniform(0.04, 0.09)
        treatment_pre_std = control_pre_std * self.random.uniform(0.95, 1.05)

        if config.imbalance:
            treatment_pre_mean += self.random.choice([-1, 1]) * self.random.uniform(0.015, 0.04)

        true_lift = self.random.uniform(-0.004, 0.02)
        if config.low_power:
            true_lift *= 0.25
            control_size = max(250, control_size // 6)
            treatment_size = max(250, treatment_size // 6)

        control_post_mean = control_pre_mean + self.random.uniform(-0.004, 0.008)
        treatment_post_mean = treatment_pre_mean + true_lift + self.random.uniform(-0.004, 0.008)
        control_post_std = control_pre_std * self.random.uniform(0.95, 1.2)
        treatment_post_std = treatment_pre_std * self.random.uniform(0.95, 1.2)

        notes: List[str] = []
        if config.outliers:
            control_post_std *= self.random.uniform(1.7, 2.4)
            treatment_post_std *= self.random.uniform(1.9, 2.8)
            notes.append("A small number of extreme spenders may be stretching the post-period variance.")

        control_conversion_rate = max(0.001, min(0.999, control_post_mean))
        treatment_conversion_rate = max(0.001, min(0.999, treatment_post_mean))
        segments: List[SegmentSummary] = []

        if config.simpsons_paradox:
            segments = self._simpsons_segments()
            control_conversion_rate = round(
                sum(seg.weight_control * seg.control_conversion_rate for seg in segments), 4
            )
            treatment_conversion_rate = round(
                sum(seg.weight_treatment * seg.treatment_conversion_rate for seg in segments), 4
            )
            notes.append("Segment mix changed noticeably between control and treatment.")

        if config.low_power:
            notes.append("Observed lift is small relative to noise and sample size.")

        return ExperimentSummary(
            experiment_id=experiment_id,
            hypothesis="Treatment should improve conversion rate.",
            expected_treatment_share=0.5,
            control_size=control_size,
            treatment_size=treatment_size,
            control_pre_mean=round(control_pre_mean, 4),
            treatment_pre_mean=round(treatment_pre_mean, 4),
            control_pre_std=round(control_pre_std, 4),
            treatment_pre_std=round(treatment_pre_std, 4),
            control_post_mean=round(control_post_mean, 4),
            treatment_post_mean=round(treatment_post_mean, 4),
            control_post_std=round(control_post_std, 4),
            treatment_post_std=round(treatment_post_std, 4),
            control_conversion_rate=round(control_conversion_rate, 4),
            treatment_conversion_rate=round(treatment_conversion_rate, 4),
            notes=notes,
            segment_summaries=segments,
            hidden_truth=config.labels(),
        )

    def generate_many(
        self,
        count: int,
        failure_modes: Sequence[str] | None = None,
    ) -> List[ExperimentSummary]:
        return [self.generate(f"exp_{idx:04d}", failure_modes=failure_modes) for idx in range(count)]

    def _config_from_names(self, names: Sequence[str]) -> FailureConfig:
        normalized = {name.strip().lower() for name in names}
        return FailureConfig(
            srm="srm" in normalized,
            imbalance="imbalance" in normalized or "pre_period_imbalance" in normalized,
            low_power="low_power" in normalized or "low_statistical_power" in normalized,
            outliers="outliers" in normalized,
            simpsons_paradox="simpsons" in normalized or "simpsons_paradox" in normalized,
        )

    def _simpsons_segments(self) -> List[SegmentSummary]:
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
        return [returning, new_users]


def experiments_to_rows(experiments: Iterable[ExperimentSummary]) -> List[dict]:
    return [experiment.model_dump() for experiment in experiments]
