import json, math, numpy as np
from collections import defaultdict
from typing import Any, Dict, List, Union

from pathlib import Path



# ---------- utils ----------
def _safe_div(n, d):
    return float(n) / float(d) if (d is not None and d != 0) else float("nan")

def _nanmean(xs: List[float]) -> float:
    arr = np.array(xs, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def _is_merged(rows: List[Dict[str, Any]]) -> bool:
    """Heuristic: a 'merged' row has a 'metrics' list inside."""
    return bool(rows) and isinstance(rows[0].get("metrics"), list)


# --- retrieval aggregation from TREC-style results.json ---
# Maps our friendly names -> TREC keys template
_TREC_KEY = {
    "precision": "P_{k}",
    "recall":    "recall_{k}",
    "ndcg":      "ndcg_cut_{k}",
    # MRR is not per-k in trec_eval; key is just 'recip_rank'
}

def _mean_of_key(per_query: Dict[str, Dict], key: str) -> float | None:
    vals = [v[key] for v in per_query.values() if key in v]
    return float(np.mean(vals)) if vals else None


def flatten_trec_results_to_per_k(results_path: str | Path, ks: List[int] | None = None) -> List[dict]:
    """
    Read retrieval/results.json and return a JSONL-ready list of per-k rows:
      [{"k": 5, "precision@k": ..., "recall@k": ..., "ndcg@k": ..., "mrr": ...}, ...]
    Robust to string keys ("5") and missing metrics (skips None).
    """
    p = Path(results_path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    per_k = data.get("per_k") or {}
    # Coerce keys to int (results.json may store "5" instead of 5)
    parsed: Dict[int, Dict[str, Any]] = {}
    for k_str, vals in per_k.items():
        try:
            k_int = int(k_str)
        except Exception:
            continue
        parsed[k_int] = dict(vals or {})

    # If ks was provided (e.g., from suite), filter to those; otherwise use whatever is present
    k_list = sorted(set(int(k) for k in (ks or parsed.keys())))
    out: List[dict] = []
    for k in k_list:
        vals = parsed.get(k, {})
        row = {"k": int(k)}
        # Keep only known keys if present; cast to float for JSONL cleanliness
        for key in ("precision@k", "recall@k", "ndcg@k", "mrr"):
            v = vals.get(key, None)
            if v is not None:
                row[key] = float(v)
        out.append(row)
    return out


def aggregate_retrieval_from_per_k(per_k_rows: List[dict]) -> Dict[str, Any]:
    """
    Turn per-k rows into a flat metrics dict for aggregates.json and gates.
    Names:
      retrieval_precision@5, retrieval_recall@10, retrieval_ndcg@5, retrieval_mrr
    For MRR, we take the value from the largest k present (they’re usually identical).
    """
    metrics: Dict[str, float] = {}
    if not per_k_rows:
        return {"metrics": metrics}

    # Index by k
    by_k: Dict[int, dict] = {int(r["k"]): r for r in per_k_rows if "k" in r}
    if not by_k:
        return {"metrics": metrics}

    # Add precision/recall/ndcg for each k present
    for k, row in by_k.items():
        if "precision@k" in row:
            metrics[f"retrieval_precision@{k}"] = float(row["precision@k"])
        if "recall@k" in row:
            metrics[f"retrieval_recall@{k}"] = float(row["recall@k"])
        if "ndcg@k" in row:
            metrics[f"retrieval_ndcg@{k}"] = float(row["ndcg@k"])

    # MRR once (take from the largest k that has it)
    k_max = max(by_k.keys())
    if "mrr" in by_k[k_max]:
        metrics["retrieval_mrr"] = float(by_k[k_max]["mrr"])

    return {"metrics": metrics}

def aggregate_retrieval(rows: List[Dict]):
    """
    rows: [{'k': 5, 'recall@k': 0.87, 'mrr@k': 0.52, 'ndcg@k': 0.61}, ...]
    Returns a flat metrics dict with names like 'retrieval_recall@5'.
    """
    metrics = {}
    for r in rows:
        k = r.get("k")
        if k is None:
            continue
        for key in ("recall@k", "mrr@k", "ndcg@k"):
            if key in r and r[key] is not None:
                metrics[f"retrieval_{key.replace('@k', f'@{k}') }"] = float(r[key])
    return {"metrics": metrics}


def aggregate_retrieval_from_trec_results(results_path: Union[str, "Path"], ks: List[int]) -> Dict:
    """
    results.json format (per-query):
      {
        "1": {"P_10": 0.1, "recall_10": 1.0, "ndcg_cut_10": 0.2890, "recip_rank": 1.0, ...},
        "2": {...},
        ...
      }
    Returns:
      {"metrics": {"retrieval_precision@10": ..., "retrieval_recall@10": ..., "retrieval_ndcg@10": ..., "retrieval_mrr": ...}}
    """
    with open(results_path, "r", encoding="utf-8") as f:
        per_query = json.load(f)  # dict[qid] -> dict

    def mean_of(key: str):
        vals = [v[key] for v in per_query.values() if key in v]
        return float(np.mean(vals)) if vals else None

    out: Dict[str, float] = {}

    for k in ks:
        p = mean_of(f"P_{k}")           # TREC: precision at k
        r = mean_of(f"recall_{k}")      # TREC: recall at k
        nd = mean_of(f"ndcg_cut_{k}")   # TREC: nDCG@k

        if p is not None:  out[f"retrieval_precision@{k}"] = p
        if r is not None:  out[f"retrieval_recall@{k}"]    = r
        if nd is not None: out[f"retrieval_ndcg@{k}"]      = nd

    mrr = mean_of("recip_rank")         # cutoff-agnostic MRR (if present)
    if mrr is not None:
        out["retrieval_mrr"] = mrr

    return {"metrics": out}
# ---------- merging (leave as-is, but use only if needed) ----------
def merge_metric_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group by qid, merge metrics per sample. (Use only if input is per-metric rows.)"""
    by_qid: Dict[str, Dict[str, Any]] = {}
    metrics_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in rows:
        qid = r.get("qid") or f"noid:{hash((r.get('query'), r.get('response')))}"
        if qid not in by_qid:
            by_qid[qid] = {
                "qid": qid,
                "query": r.get("query"),
                "response": r.get("response"),
                "reference": r.get("reference"),
                "retrieved_contexts": r.get("retrieved_contexts"),
                "metrics": []
            }
        metrics_map[qid].append({
            "name": r["metric"],
            "score": r["score"],
            "raw": r.get("details") or r.get("raw")  # normalize
        })

    merged = []
    for qid, base in by_qid.items():
        base["metrics"] = metrics_map[qid]
        merged.append(base)
    return merged

# ---------- aggregation with special handling ----------
def aggregate(merged_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes:
      - generic means per metric (e.g., 'faithfulness_mean')
      - faithfulness micro/macro precision
      - factual_correctness micro/macro precision/recall/f1
    """
    n = len(merged_rows)
    generic: Dict[str, List[float]] = defaultdict(list)

    # Faithfulness accumulators
    faith_per_sample_prec: List[float] = []
    faith_supported_sum = 0
    faith_total_sum = 0

    # Factual accumulators
    factual_per_sample_prec: List[float] = []
    factual_per_sample_rec: List[float] = []
    factual_per_sample_f1:  List[float] = []
    factual_tp_sum = 0
    factual_fp_sum = 0
    factual_fn_sum = 0

    # Optional rich breakdown
    by_metric = {
        "faithfulness": {"per_sample": [], "micro": {}},
        "factual_correctness": {"per_sample": [], "micro": {}},
    }

    for row in merged_rows:
        qid = row.get("qid")
        for m in row.get("metrics", []):
            name = m.get("name")
            score = m.get("score")
            raw = m.get("raw") or m.get("details") or {}

            # always keep generic mean
            generic[name].append(float(score) if score is not None else float("nan"))

            if name == "faithfulness":
                # micro: supported/total
                # try artifacts.verdicts first, else nli.output.statements
                verdicts = ((raw.get("artifacts") or {}).get("verdicts")) \
                           or (((raw.get("nli") or {}).get("output") or {}).get("statements")) \
                           or []
                total = len(verdicts)
                supported = sum(int(v.get("verdict", 0)) for v in verdicts)
                if total > 0:
                    faith_supported_sum += supported
                    faith_total_sum += total
                    # macro per-sample (precision == score)
                    ps = _safe_div(supported, total)
                    faith_per_sample_prec.append(ps)
                    by_metric["faithfulness"]["per_sample"].append({
                        "qid": qid, "supported": supported, "total": total, "precision": ps
                    })
                else:
                    # still record the sample with NaN precision
                    faith_per_sample_prec.append(float("nan"))
                    by_metric["faithfulness"]["per_sample"].append({
                        "qid": qid, "supported": 0, "total": 0, "precision": float("nan")
                    })

            elif name == "factual_correctness":
                # Prefer counts from artifacts.counts; derive if missing
                counts = (raw.get("artifacts") or {}).get("counts") or {}
                tp = counts.get("tp"); fp = counts.get("fp"); fn = counts.get("fn")

                if tp is None or fp is None or fn is None:
                    # derive from verdict lists when counts missing
                    a2r_list = ((raw.get("artifacts") or {}).get("a2r_verdicts")) \
                               or (((raw.get("nli") or {}).get("a2r") or {}).get("output") or {}).get("statements") \
                               or []
                    r2a_list = ((raw.get("artifacts") or {}).get("r2a_verdicts")) \
                               or (((raw.get("nli") or {}).get("r2a") or {}).get("output") or {}).get("statements") \
                               or []
                    a2r_arr = [int(v.get("verdict", 0)) for v in a2r_list]
                    r2a_arr = [int(v.get("verdict", 0)) for v in r2a_list]
                    tp = int(sum(a2r_arr))
                    fp = int(len(a2r_arr) - tp)
                    # if recall path missing, fn=0
                    fn = int(len(r2a_arr) - sum(r2a_arr)) if r2a_list else 0

                tp = int(tp or 0); fp = int(fp or 0); fn = int(fn or 0)

                factual_tp_sum += tp
                factual_fp_sum += fp
                factual_fn_sum += fn

                prec = _safe_div(tp, tp + fp)
                rec  = _safe_div(tp, tp + fn)
                f1   = _safe_div(2 * prec * rec, (prec + rec)) if (not math.isnan(prec) and not math.isnan(rec) and (prec + rec) > 0) else (0.0 if (prec == 0.0 and rec == 0.0) else float("nan"))

                factual_per_sample_prec.append(prec)
                factual_per_sample_rec.append(rec)
                factual_per_sample_f1.append(f1)

                by_metric["factual_correctness"]["per_sample"].append({
                    "qid": qid, "tp": tp, "fp": fp, "fn": fn,
                    "precision": prec, "recall": rec, "f1": f1
                })

    # compute micros
    faithfulness_precision_micro = _safe_div(faith_supported_sum, faith_total_sum) if faith_total_sum > 0 else float("nan")

    fp_denom_p = (factual_tp_sum + factual_fp_sum)
    fp_denom_r = (factual_tp_sum + factual_fn_sum)
    factual_precision_micro = _safe_div(factual_tp_sum, fp_denom_p) if (fp_denom_p > 0) else float("nan")
    factual_recall_micro    = _safe_div(factual_tp_sum, fp_denom_r) if (fp_denom_r > 0) else float("nan")
    if (not math.isnan(factual_precision_micro) and not math.isnan(factual_recall_micro) and (factual_precision_micro + factual_recall_micro) > 0):
        factual_f1_micro = 2 * factual_precision_micro * factual_recall_micro / (factual_precision_micro + factual_recall_micro)
    else:
        factual_f1_micro = 0.0 if (factual_precision_micro == 0.0 and factual_recall_micro == 0.0) else float("nan")

    # macro
    metrics = {}
    # generic means for any metric name
    for k, v in generic.items():
        metrics[f"{k}_mean"] = _nanmean(v)

    # faithfulness macro/micro
    if faith_per_sample_prec:
        metrics["faithfulness_precision_macro"] = _nanmean(faith_per_sample_prec)
    metrics["faithfulness_precision_micro"] = faithfulness_precision_micro

    by_metric["faithfulness"]["micro"] = {
        "supported": faith_supported_sum,
        "total": faith_total_sum,
        "precision": faithfulness_precision_micro,
    }

    # factual macro/micro
    if factual_per_sample_prec:
        metrics["factual_precision_macro"] = _nanmean(factual_per_sample_prec)
        metrics["factual_recall_macro"]    = _nanmean(factual_per_sample_rec)
        metrics["factual_f1_macro"]        = _nanmean(factual_per_sample_f1)

    metrics["factual_precision_micro"] = factual_precision_micro
    metrics["factual_recall_micro"]    = factual_recall_micro
    metrics["factual_f1_micro"]        = factual_f1_micro

    by_metric["factual_correctness"]["micro"] = {
        "tp": factual_tp_sum, "fp": factual_fp_sum, "fn": factual_fn_sum,
        "precision": factual_precision_micro,
        "recall": factual_recall_micro,
        "f1": factual_f1_micro,
    }

    return {"n": n, "metrics": metrics, "by_metric": by_metric}