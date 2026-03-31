from __future__ import annotations

import math
from typing import Any, Dict

from .schemas import ExperimentSummary, ToolObservation
from .tracing import traced


def _normal_survival(z_score: float) -> float:
    return 0.5 * math.erfc(z_score / math.sqrt(2.0))


def _chi_square_df1_survival(statistic: float) -> float:
    return math.erfc(math.sqrt(max(statistic, 0.0) / 2.0))


@traced(name="tool::srm_test", run_type="tool")
def srm_test(experiment: ExperimentSummary) -> ToolObservation:
    total = experiment.control_size + experiment.treatment_size
    expected_treatment = total * experiment.expected_treatment_share
    expected_control = total - expected_treatment
    statistic = ((experiment.control_size - expected_control) ** 2) / max(expected_control, 1e-9)
    statistic += ((experiment.treatment_size - expected_treatment) ** 2) / max(expected_treatment, 1e-9)
    p_value = _chi_square_df1_survival(statistic)
    flagged = p_value < 0.01
    return ToolObservation(
        tool_name="srm_test",
        summary="Chi-square sample ratio mismatch check against the expected allocation.",
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        flagged=flagged,
        details={
            "expected_control_size": round(expected_control, 2),
            "expected_treatment_size": round(expected_treatment, 2),
            "observed_control_size": experiment.control_size,
            "observed_treatment_size": experiment.treatment_size,
        },
    )


@traced(name="tool::pre_period_balance_test", run_type="tool")
def pre_period_balance_test(experiment: ExperimentSummary) -> ToolObservation:
    variance = (experiment.control_pre_std ** 2) / max(experiment.control_size, 1)
    variance += (experiment.treatment_pre_std ** 2) / max(experiment.treatment_size, 1)
    standard_error = math.sqrt(max(variance, 1e-12))
    z_score = (experiment.treatment_pre_mean - experiment.control_pre_mean) / standard_error
    p_value = 2.0 * _normal_survival(abs(z_score))
    flagged = p_value < 0.05
    return ToolObservation(
        tool_name="pre_period_balance_test",
        summary="Two-sided normal approximation for the difference in pre-period means.",
        statistic=round(z_score, 4),
        p_value=round(p_value, 6),
        flagged=flagged,
        details={
            "difference_in_means": round(experiment.treatment_pre_mean - experiment.control_pre_mean, 6),
            "standard_error": round(standard_error, 6),
        },
    )


TOOL_REGISTRY = {
    "srm_test": srm_test,
    "pre_period_balance_test": pre_period_balance_test,
}


def tool_descriptions() -> Dict[str, Dict[str, Any]]:
    return {
        "srm_test": {
            "purpose": "Detect sample ratio mismatch between expected and observed group sizes.",
            "returns": ["chi_square_statistic", "p_value", "flagged"],
        },
        "pre_period_balance_test": {
            "purpose": "Check whether control and treatment appear balanced before the experiment starts.",
            "returns": ["z_score", "p_value", "flagged"],
        },
    }
