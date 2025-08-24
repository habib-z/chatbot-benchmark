from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from chatbot_evaluation.metrics.m_answer_relevancy import AnswerRelevancyFileBacked
# >>> Replace imports below with your actual module paths
# Faithfulness file-backed (saves details on sample._ragas_details)
from chatbot_evaluation.metrics.m_faithfulness import FaithfulnessFileBacked
# FactualCorrectness file-backed (your earlier version)
from chatbot_evaluation.metrics.m_factual_correctness import FactualCorrectnessFileBacked


@dataclass
class MetricSpec:
    name: str
    impl_id: str
    params: Dict[str, Any]
    datasets: list[str]

def build_metric(spec: MetricSpec, manifest: dict, llm_judge, embeddings):
    name = spec.name.lower()
    print("building metric name:", name)
    metric_prompts = manifest["prompts"].get(name)
    print("metric prompts:", metric_prompts)
    print("spec:", spec)
    if not metric_prompts:
        raise ValueError(f"No prompts configured for metric '{name}'")


    base_dir = metric_prompts["base_dir"]
    spec_param={}

    # extract versions from ids if present
    def _ver(idstr: str) -> str | None:
        return idstr.split("@")[-1] if "@" in idstr else None


    if name == "faithfulness":
        print("metric_prompts")
        print(metric_prompts)
        # derive a common base_dir (parent of role dirs)
        spec_param = {
            "statement_generator_version": _ver(metric_prompts["statement_generator"]["id"]),
            "nli_judge_version": _ver(metric_prompts["nli_judge"]["id"]),
        }
        print(f"spec_param: {spec_param}")
        params = dict(spec.params or {})
        params["spec_param"] = spec_param
        return FaithfulnessFileBacked(llm=llm_judge, base_dir=base_dir, **params)

    if name == "factual_correctness":
        spec_param = {
            "claim_decomposition_version": _ver(metric_prompts["claim_decomposition"]["id"]),
            "nli_judge_version": _ver(metric_prompts["nli_judge"]["id"]),
        }
        print(f"spec_param: {spec_param}")
        params = dict(spec.params or {})
        params["spec_param"] = spec_param

        return FactualCorrectnessFileBacked(llm=llm_judge, base_dir=base_dir, **params)
        # raise NotImplementedError("Hook your factual correctness builder here")

    if name == "answer_relevancy":
        spec_param = {
            "response_relevance_version": _ver(metric_prompts["response_relevance"]["id"]),
        }
        print(f"spec_param: {spec_param}")
        params = dict(spec.params or {})
        params["spec_param"] = spec_param

        return AnswerRelevancyFileBacked(llm=llm_judge, base_dir=base_dir,embeddings=embeddings, **params)
    raise ValueError(f"Unknown metric: {spec.name}")

def run_metric(metric_obj, samples: Iterable[Any]) -> List[dict]:
    out = []
    has_eval = hasattr(metric_obj, "single_turn_eval")
    for s in samples:
        if has_eval:
            res = metric_obj.single_turn_eval(s)
            score = res["score"]
            details = res["details"]
        else:
            score = metric_obj.single_turn_score(s)
            details = getattr(s, "_ragas_details", {}).get(metric_obj.name) or {}
        row = {
            "qid": getattr(s, "id", None),
            "query": s.user_input,
            "response": s.response,
            "reference": getattr(s, "reference", None),
            "retrieved_contexts": getattr(s, "retrieved_contexts", None),
            "metric": metric_obj.name,
            "score": score,
            "details": details,
        }
        out.append(row)
    return out