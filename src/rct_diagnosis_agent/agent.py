from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from .llm import OpenAIStructuredLLM, StructuredLLM, build_prompt
from .schemas import (
    AgentRunResult,
    AnalyzerOutput,
    AnalyzerPlan,
    CriticOutput,
    ExperimentSummary,
    FinalOutput,
    ToolObservation,
)
from .tools import TOOL_REGISTRY
from .tracing import attach_run_metadata, traced

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:
    StateGraph = None  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]
    END = "__end__"  # type: ignore[assignment]
    _LANGGRAPH_IMPORT_ERROR = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


class AgentState(TypedDict, total=False):
    experiment: ExperimentSummary
    analyzer_plan: AnalyzerPlan
    tool_observations: List[ToolObservation]
    analyzer_output: AnalyzerOutput
    critic_output: CriticOutput
    final_output: FinalOutput


def _load_prompt(name: str) -> str:
    return Path(__file__).resolve().parents[2].joinpath("prompts", name).read_text(encoding="utf-8")


class RCTDiagnosisAgent:
    def __init__(self, llm: StructuredLLM) -> None:
        if StateGraph is None:
            raise ImportError(f"langgraph is required to use RCTDiagnosisAgent: {_LANGGRAPH_IMPORT_ERROR}")
        self.llm = llm
        self.analyzer_system = _load_prompt("analyzer_system.md")
        self.critic_system = _load_prompt("critic_system.md")
        self.reviser_system = _load_prompt("reviser_system.md")
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("analyzer_plan", self._analyzer_plan)
        graph.add_node("run_tools", self._run_tools)
        graph.add_node("analyzer", self._analyzer)
        graph.add_node("critic", self._critic)
        graph.add_node("reviser", self._reviser)
        graph.add_edge(START, "analyzer_plan")
        graph.add_edge("analyzer_plan", "run_tools")
        graph.add_edge("run_tools", "analyzer")
        graph.add_edge("analyzer", "critic")
        graph.add_edge("critic", "reviser")
        graph.add_edge("reviser", END)
        return graph.compile()

    @traced(name="analyzer_plan", run_type="llm")
    def _analyzer_plan(self, state: AgentState) -> AgentState:
        experiment = state["experiment"]
        prompt = build_prompt(
            self.analyzer_system,
            task="Choose which statistical tools should be called before the analyzer writes its diagnosis.",
            available_tools=json.dumps(list(TOOL_REGISTRY.keys()), indent=2),
            experiment_summary=json.dumps(experiment.to_prompt_dict(), indent=2),
        )
        plan = self.llm.invoke_structured(prompt, AnalyzerPlan)
        return {**state, "analyzer_plan": plan}

    @traced(name="run_tools", run_type="chain")
    def _run_tools(self, state: AgentState) -> AgentState:
        experiment = state["experiment"]
        observations: List[ToolObservation] = []
        for call in state["analyzer_plan"].tool_calls[:2]:
            tool_fn = TOOL_REGISTRY[call.tool_name]
            observations.append(tool_fn(experiment))
        return {**state, "tool_observations": observations}

    @traced(name="analyzer", run_type="llm")
    def _analyzer(self, state: AgentState) -> AgentState:
        experiment = state["experiment"]
        observations = [obs.model_dump() for obs in state.get("tool_observations", [])]
        heuristic_hints = _heuristic_hints(experiment)
        prompt = build_prompt(
            self.analyzer_system,
            task="Produce an initial diagnosis. Use tool observations and experiment evidence. Return only schema fields.",
            experiment_summary=json.dumps(experiment.to_prompt_dict(), indent=2),
            tool_observations=json.dumps(observations, indent=2),
            heuristic_hints=json.dumps(heuristic_hints, indent=2),
        )
        output = self.llm.invoke_structured(prompt, AnalyzerOutput)
        output.tool_observations = state.get("tool_observations", [])
        return {**state, "analyzer_output": output}

    @traced(name="critic", run_type="llm")
    def _critic(self, state: AgentState) -> AgentState:
        experiment = state["experiment"]
        analyzer_output = state["analyzer_output"]
        prompt = build_prompt(
            self.critic_system,
            task="Critique the analyzer diagnosis.",
            experiment_summary=json.dumps(experiment.to_prompt_dict(), indent=2),
            analyzer_output=json.dumps(analyzer_output.model_dump(), indent=2),
        )
        critique = self.llm.invoke_structured(prompt, CriticOutput)
        return {**state, "critic_output": critique}

    @traced(name="reviser", run_type="llm")
    def _reviser(self, state: AgentState) -> AgentState:
        experiment = state["experiment"]
        attach_run_metadata(
            {
                "experiment_id": experiment.experiment_id,
                "ground_truth": experiment.hidden_truth,
            },
            ["rct-diagnosis", "reflection"],
        )
        prompt = build_prompt(
            self.reviser_system,
            task="Write the final diagnosis after reflection.",
            experiment_summary=json.dumps(experiment.to_prompt_dict(), indent=2),
            analyzer_output=json.dumps(state["analyzer_output"].model_dump(), indent=2),
            critic_output=json.dumps(state["critic_output"].model_dump(), indent=2),
            heuristic_hints=json.dumps(_heuristic_hints(experiment), indent=2),
        )
        final_output = self.llm.invoke_structured(prompt, FinalOutput)
        return {**state, "final_output": final_output}

    def run(self, experiment: ExperimentSummary) -> AgentRunResult:
        result = self.graph.invoke({"experiment": experiment})
        return AgentRunResult(
            experiment=experiment,
            analyzer=result["analyzer_output"],
            critic=result["critic_output"],
            final=result["final_output"],
        )


def _heuristic_hints(experiment: ExperimentSummary) -> Dict[str, Any]:
    total_n = experiment.control_size + experiment.treatment_size
    conversion_diff = experiment.treatment_conversion_rate - experiment.control_conversion_rate
    pooled_rate = (experiment.control_conversion_rate + experiment.treatment_conversion_rate) / 2.0
    approx_se = (2.0 * pooled_rate * max(1.0 - pooled_rate, 1e-6) / max(total_n / 2.0, 1.0)) ** 0.5
    low_power = total_n < 1500 or abs(conversion_diff) < 1.5 * approx_se
    outlier_risk = max(experiment.control_post_std, experiment.treatment_post_std) > 2.2 * max(
        experiment.control_pre_std,
        experiment.treatment_pre_std,
        1e-6,
    )

    simpsons_risk = False
    if experiment.segment_summaries:
        pooled_delta = experiment.treatment_conversion_rate - experiment.control_conversion_rate
        segment_deltas = [
            segment.treatment_conversion_rate - segment.control_conversion_rate for segment in experiment.segment_summaries
        ]
        simpsons_risk = all(delta < 0 for delta in segment_deltas) and pooled_delta > 0

    return {
        "total_sample_size": total_n,
        "observed_conversion_diff": round(conversion_diff, 6),
        "approx_standard_error": round(approx_se, 6),
        "low_power_risk": low_power,
        "outlier_risk": outlier_risk,
        "simpsons_paradox_risk": simpsons_risk,
    }


def build_default_agent(model: str = "gpt-4o-mini") -> RCTDiagnosisAgent:
    return RCTDiagnosisAgent(OpenAIStructuredLLM(model=model))
