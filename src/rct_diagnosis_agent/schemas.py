from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IssueName = Literal["srm", "pre_period_imbalance", "low_statistical_power", "outliers", "simpsons_paradox"]


class SegmentSummary(BaseModel):
    name: str
    weight_control: float
    weight_treatment: float
    control_conversion_rate: float
    treatment_conversion_rate: float


class ExperimentSummary(BaseModel):
    experiment_id: str
    hypothesis: str
    expected_treatment_share: float = 0.5
    control_size: int
    treatment_size: int
    control_pre_mean: float
    treatment_pre_mean: float
    control_pre_std: float
    treatment_pre_std: float
    control_post_mean: float
    treatment_post_mean: float
    control_post_std: float
    treatment_post_std: float
    control_conversion_rate: float
    treatment_conversion_rate: float
    primary_metric: str = "conversion_rate"
    notes: List[str] = Field(default_factory=list)
    segment_summaries: List[SegmentSummary] = Field(default_factory=list)
    hidden_truth: List[IssueName] = Field(default_factory=list)

    def to_prompt_dict(self) -> Dict[str, Any]:
        payload = self.model_dump()
        payload.pop("hidden_truth", None)
        return payload


class IssueDiagnosis(BaseModel):
    issue: IssueName
    severity: Literal["low", "medium", "high"]
    evidence: str


class ToolObservation(BaseModel):
    tool_name: str
    summary: str
    statistic: float
    p_value: float
    flagged: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    tool_name: Literal["srm_test", "pre_period_balance_test"]
    rationale: str


class AnalyzerPlan(BaseModel):
    tool_calls: List[ToolCall] = Field(default_factory=list)
    notes: str


class AnalyzerOutput(BaseModel):
    issues: List[IssueDiagnosis] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)


class CriticOutput(BaseModel):
    missed_checks: List[str] = Field(default_factory=list)
    logical_errors: List[str] = Field(default_factory=list)
    confidence_adjustment: Optional[str] = None
    notes: str


class FinalOutput(BaseModel):
    issues: List[IssueDiagnosis] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AgentRunResult(BaseModel):
    experiment: ExperimentSummary
    analyzer: AnalyzerOutput
    critic: CriticOutput
    final: FinalOutput


class EvaluationRow(BaseModel):
    experiment_id: str
    ground_truth: List[IssueName]
    before_reflection: List[IssueName]
    after_reflection: List[IssueName]
    analyzer_confidence: float
    final_confidence: float


class EvaluationSummary(BaseModel):
    count: int
    exact_match_before: float
    exact_match_after: float
    recall_before: float
    recall_after: float
    improvement_exact_match: float
    improvement_recall: float
