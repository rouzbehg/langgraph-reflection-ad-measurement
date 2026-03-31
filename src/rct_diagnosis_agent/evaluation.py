from __future__ import annotations

from typing import Iterable, List, Sequence

from .agent import RCTDiagnosisAgent
from .schemas import EvaluationRow, EvaluationSummary, ExperimentSummary


def _extract_issue_names(result_issues: Sequence) -> List[str]:
    return [issue.issue for issue in result_issues]


def _exact_match(left: Sequence[str], right: Sequence[str]) -> float:
    return float(set(left) == set(right))


def _recall(predicted: Sequence[str], truth: Sequence[str]) -> float:
    truth_set = set(truth)
    if not truth_set:
        return 1.0 if not predicted else 0.0
    return len(set(predicted) & truth_set) / len(truth_set)


def evaluate_agent(agent: RCTDiagnosisAgent, experiments: Iterable[ExperimentSummary]) -> tuple[List[EvaluationRow], EvaluationSummary]:
    rows: List[EvaluationRow] = []
    exact_before_total = 0.0
    exact_after_total = 0.0
    recall_before_total = 0.0
    recall_after_total = 0.0

    for experiment in experiments:
        result = agent.run(experiment)
        before = _extract_issue_names(result.analyzer.issues)
        after = _extract_issue_names(result.final.issues)
        truth = list(experiment.hidden_truth)
        exact_before_total += _exact_match(before, truth)
        exact_after_total += _exact_match(after, truth)
        recall_before_total += _recall(before, truth)
        recall_after_total += _recall(after, truth)
        rows.append(
            EvaluationRow(
                experiment_id=experiment.experiment_id,
                ground_truth=truth,
                before_reflection=before,
                after_reflection=after,
                analyzer_confidence=result.analyzer.confidence,
                final_confidence=result.final.confidence,
            )
        )

    count = len(rows)
    summary = EvaluationSummary(
        count=count,
        exact_match_before=exact_before_total / count if count else 0.0,
        exact_match_after=exact_after_total / count if count else 0.0,
        recall_before=recall_before_total / count if count else 0.0,
        recall_after=recall_after_total / count if count else 0.0,
        improvement_exact_match=(exact_after_total - exact_before_total) / count if count else 0.0,
        improvement_recall=(recall_after_total - recall_before_total) / count if count else 0.0,
    )
    return rows, summary
