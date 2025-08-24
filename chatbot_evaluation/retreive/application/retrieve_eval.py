# chatbot_evaluation/retreive/application/retrieve_eval.py
from __future__ import annotations
from typing import List, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm
import time

from chatbot_evaluation.retreive.domain.entities import Query
from chatbot_evaluation.retreive.domain.interfaces import Embedder, VectorDB
from chatbot_evaluation.retreive.infrastructure.logger import get_logger

logger = get_logger(__name__)


def _precision_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    return len(set(top) & gold_ids) / float(k)


def _recall_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    top = ranked_ids[:k]
    return len(set(top) & gold_ids) / float(len(gold_ids))


def _mrr_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold_ids:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    import math
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if did in gold_ids else 0.0
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    R = min(len(gold_ids), k)
    if R == 0:
        return 0.0
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, R + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _to_sorted_rows(hits: List[Dict[str, Any]], score_mode: str) -> List[Tuple[str, float]]:
    """
    hits: [{'chunk_id' or 'doc_id': str|int, 'score': float}, ...]
    score_mode:
      - 'similarity' : higher is better (pass-through)
      - 'distance'   : smaller is better (negate to make higher better)
    """
    rows: List[Tuple[str, float]] = []
    for h in hits:
        did = h.get("chunk_id", h.get("doc_id"))
        if did is None:
            continue
        s = float(h["score"])
        if score_mode == "distance":
            s = -s
        rows.append((str(did), s))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def retrieve_and_evaluate(
    queries: List[Query],
    qrels: Dict[str, Dict[str, int]],   # qrels[qid][docid] = relevance (>0 = relevant)
    embedder: Embedder,
    vectordb: VectorDB,
    collection_name: str,
    cutoffs: List[int] = (5, 10),
    score_mode: str = "similarity",
) -> Dict[str, Any]:
    """
    Pure-Python IR metrics (no pytrec_eval). Returns:
      {
        "per_query": {
          "<qid>": {
            "metrics": {"P@5":..., "recall@5":..., "ndcg@5":..., "RR@5":..., "P@10":..., ...},
            "docs": [("<doc_id>", <score>), ...]   # sorted desc, top = max(cutoffs)
          },
          ...
        },
        "per_k": { 5: {"precision@k":..., "recall@k":..., "ndcg@k":..., "mrr":...}, 10: {...} }
      }
    """
    start = time.time()
    ks = sorted({int(k) for k in cutoffs})
    k_max = max(ks)

    # normalize qrels to string ids + set-of-relevant
    gold: Dict[str, set[str]] = {
        str(qid): {str(did) for did, rel in dids.items() if rel and rel > 0}
        for qid, dids in qrels.items()
    }

    per_query_docs: Dict[str, List[Tuple[str, float]]] = {}
    per_query_metrics: Dict[str, Dict[str, float]] = {}

    for q in tqdm(queries, desc="Retrieving"):
        qid = str(q.query_id)
        qvec = embedder.encode([q.text])[0]
        hits = vectordb.search(collection_name, qvec, k_max)  # [{'doc_id'|'chunk_id','score'}, ...]
        rows = _to_sorted_rows(hits, score_mode=score_mode)
        per_query_docs[qid] = rows[:k_max]

        ranked_ids = [did for (did, _s) in rows]
        gold_ids = gold.get(qid, set())

        out: Dict[str, float] = {}
        for k in ks:
            out[f"P@{k}"]      = _precision_at_k(ranked_ids, gold_ids, k)
            out[f"recall@{k}"] = _recall_at_k(ranked_ids, gold_ids, k)
            out[f"ndcg@{k}"]   = _ndcg_at_k(ranked_ids, gold_ids, k)
            out[f"RR@{k}"]     = _mrr_at_k(ranked_ids, gold_ids, k)
        per_query_metrics[qid] = out

    # Macro per-k
    per_k: Dict[int, Dict[str, float]] = {}
    for k in ks:
        P_vals  = [m.get(f"P@{k}", 0.0) for m in per_query_metrics.values()]
        R_vals  = [m.get(f"recall@{k}", 0.0) for m in per_query_metrics.values()]
        N_vals  = [m.get(f"ndcg@{k}", 0.0) for m in per_query_metrics.values()]
        RR_vals = [m.get(f"RR@{k}", 0.0) for m in per_query_metrics.values()]
        per_k[k] = {
            "precision@k": float(np.mean(P_vals)) if P_vals else 0.0,
            "recall@k":    float(np.mean(R_vals)) if R_vals else 0.0,
            "ndcg@k":      float(np.mean(N_vals)) if N_vals else 0.0,
            "mrr":         float(np.mean(RR_vals)) if RR_vals else 0.0,
        }

    elapsed = time.time() - start
    logger.info(f"Retrieved {len(queries)} queries in {elapsed:.2f}s (avg {elapsed/len(queries):.3f}s/q)")

    return {
        "per_query": {
            qid: {"metrics": per_query_metrics.get(qid, {}), "docs": per_query_docs.get(qid, [])}
            for qid in per_query_docs.keys()
        },
        "per_k": per_k,
    }