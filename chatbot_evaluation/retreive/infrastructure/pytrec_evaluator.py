# chatbot_evaluation/retreive/infrastructure/pytrec_evaluator.py
from __future__ import annotations
from typing import Dict, List, Any, Optional
import numpy as np
import pytrec_eval

# If you have an Evaluator interface, keep the class name the same.
class PytrecEvaluator:
    """
    Normalizes human-friendly metric specs to pytrec_eval names and evaluates.
    """

    @staticmethod
    def _normalize_measures(requested: List[str], ks: Optional[List[int]]) -> List[str]:
        """
        requested examples:
          ["precision@k", "recall@k", "ndcg@k", "mrr"] with ks=[5,10]
        OR already-pytrec names:
          ["P_10", "ndcg_cut_5", "recall.10", "recip_rank"]
        OR legacy names:
          ["recall_10"]  -> convert to "recall.10"
        """
        out = set()

        def add_for_k(template: str):
            assert ks, "ks must be provided when using @k specifications"
            for k in ks:
                out.add(template.format(k=k))

        for m in requested:
            m0 = str(m).strip().lower()

            # friendly specs
            if m0 in ("precision@k", "p@k"):
                add_for_k("P_{k}")
                continue
            if m0 in ("recall@k",):
                add_for_k("recall.{k}")
                continue
            if m0 in ("ndcg@k", "ndcg_cut@k"):
                add_for_k("ndcg_cut_{k}")
                continue
            if m0 in ("mrr", "recip_rank"):
                out.add("recip_rank")
                continue
            if m0 == "map":
                out.add("map")
                continue

            # already pytrec?
            if m0.startswith("p_"):
                out.add("P_" + m0.split("_", 1)[1])
                continue
            if m0.startswith("ndcg_cut_"):
                out.add("ndcg_cut_" + m0.split("_", 2)[2] if m0.count("_") > 1 else m0)
                continue
            if m0.startswith("recall."):
                out.add("recall." + m0.split(".", 1)[1])
                continue

            # legacy "recall_10"
            if m0.startswith("recall_"):
                out.add("recall." + m0.split("_", 1)[1])
                continue

            raise ValueError(f"Unsupported measure spec: {m}")

        return sorted(out)

    def evaluate(
        self,
        run: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        requested_measures: List[str],
        ks: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            'per_query': {qid: {'P_5':..., 'recall.5':..., 'ndcg_cut_5':..., 'recip_rank':...}, ...},
            'per_k': {5: {'precision@k':..., 'recall@k':..., 'ndcg@k':...}, 10: {...}},
            'overall': {'recip_rank': ..., 'map': ...}
          }
        """
        measures = self._normalize_measures(requested_measures, ks)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
        per_query = evaluator.evaluate(run)  # qid -> {metric_name: value}

        # Aggregate per-k
        per_k = {}
        if ks:
            for k in ks:
                row = {}
                # precision
                vals = [d.get(f"P_{k}") for d in per_query.values() if f"P_{k}" in d]
                if vals:
                    row["precision@k"] = float(np.mean(vals))
                # recall
                vals = [d.get(f"recall.{k}") for d in per_query.values() if f"recall.{k}" in d]
                if vals:
                    row["recall@k"] = float(np.mean(vals))
                # ndcg
                vals = [d.get(f"ndcg_cut_{k}") for d in per_query.values() if f"ndcg_cut_{k}" in d]
                if vals:
                    row["ndcg@k"] = float(np.mean(vals))

                if row:
                    per_k[k] = row

        # Overall non-k metrics
        overall = {}
        for name in ("recip_rank", "map"):
            vals = [d.get(name) for d in per_query.values() if name in d]
            if vals:
                overall[name] = float(np.mean(vals))

        return {"per_query": per_query, "per_k": per_k, "overall": overall}