# evalkit/retrieval_phase.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json, time

from chatbot_evaluation.evalkit.retrieval_bundle import load_bundle, make_normalizer, build_embedder, build_qdrant_repo, \
    score_mode_from_bundle
from chatbot_evaluation.evalkit.files import write_jsonl, write_json  # your fixed function

# Simple IR metrics (no dependency on pytrec; keep it self-contained)
def recall_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids: return 0.0
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)

def mrr_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked_ids: List[str], gold_ids: set[str], k: int) -> float:
    # binary relevance
    import math
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if did in gold_ids else 0.0
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
    # ideal DCG (all relevant at top)
    R = min(len(gold_ids), k)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, R + 1))
    return dcg / idcg if idcg > 0 else 0.0

def compute_per_query_metrics(ranked_ids: List[str], gold_ids: set[str], k_list: List[int]) -> Dict[str, float]:
    out = {}
    for k in k_list:
        out[f"recall@{k}"] = recall_at_k(ranked_ids, gold_ids, k)
        out[f"mrr@{k}"]    = mrr_at_k(ranked_ids, gold_ids, k)
        out[f"ndcg@{k}"]   = ndcg_at_k(ranked_ids, gold_ids, k)
    return out

def read_corpus_jsonl(path: str, id_field: str, text_field: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            mp[str(obj[id_field])] = obj[text_field]
    return mp

# def run_retrieval_phase_(
#     suite: dict,
#     run_dir: Path,
#     SentenceEmbedderCls,
#     QdrantRepositoryCls,
# ) -> Dict[str, List[str]]:
#     """
#     Returns contexts_map: qid -> [topk passage texts]
#     Also writes judgments.retrieval.jsonl.
#     """
#     if "retrieval" not in suite:
#         return {}
#
#     rconf = suite["retrieval"]
#     bundle_path = rconf["bundle"]
#     eval_k_list = rconf.get("eval", {}).get("top_k", [5])
#     couple = bool(rconf.get("couple_into_generation", False))
#     dataset_path = rconf["dataset"]["path"]
#
#     # 1) Load bundle and build adapters
#     rb = load_bundle(bundle_path)
#     normalizer = make_normalizer(rb)
#     embedder = build_embedder(rb, SentenceEmbedderCls)
#     vectordb = build_qdrant_repo(rb, QdrantRepositoryCls)
#
#     # 2) Build collection name (stable but unique per run)
#     coll = f"{rb.qdrant.collection_prefix}_{int(time.time())}"
#
#     # 3) Ingest if needed (if you already have an indexed collection, skip this)
#     #    Here we read corpus jsonl (as declared in the bundle) and upsert.
#     corpus_map = read_corpus_jsonl(rb.corpus.path, rb.corpus.id_field, rb.corpus.text_field)
#     docs = [{"id": did, "text": normalizer(txt)} for did, txt in corpus_map.items()]
#     vectordb.create_or_replace_collection(coll, vector_size=embedder.dim(), metric=rb.qdrant.metric)
#     vectordb.upsert_texts(coll, docs, embedder)  # your repo likely has methods like this
#
#     # 4) Load retrieval dataset (your DS already exists on disk)
#     #    Expected structure (you showed earlier):
#     #      queries: list of {query_id, query}, qrels: dict[query_id][doc_id] = relevance
#     from datasets import load_from_disk
#     ds = load_from_disk(dataset_path)
#     queries = [{"id": str(q["query_id"]), "text": normalizer(q["query"])} for q in ds["queries"]]
#     qrels = {}
#     for row in ds["qrels"]:
#         if row["relevance"] > 0:
#             qrels.setdefault(str(row["query_id"]), set()).add(str(row["doc_id"]))
#
#     # 5) Search & metrics
#     judgments = []
#     contexts_map: Dict[str, List[str]] = {}
#     k_max = max(eval_k_list) if eval_k_list else rb.defaults_top_k
#
#     for q in queries:
#         qid, qtext = q["id"], q["text"]
#         ranked = vectordb.search(coll, qtext, top_k=k_max, embedder=embedder)  # -> list[{"doc_id":..., "score":...}]
#         ranked_ids = [r["doc_id"] for r in ranked]
#         perq = compute_per_query_metrics(ranked_ids, qrels.get(qid, set()), eval_k_list)
#
#         metrics = [{"name": f"retrieval_{k}", "score": float(v), "raw": None} for k, v in perq.items()]
#
#         judgments.append({
#             "qid": qid,
#             "query": qtext,
#             "response": None,
#             "reference": None,
#             "retrieved_contexts": None,
#             "metrics": metrics,
#             "retrieval": {
#                 "ranked": ranked,                 # retain IDs + scores
#                 "gold": list(qrels.get(qid, [])), # ground truth doc ids
#             }
#         })
#
#         if couple:
#             # build the context list (texts) for generation phase
#             contexts_map[qid] = [corpus_map[d] for d in ranked_ids[:rb.defaults_top_k] if d in corpus_map]
#
#     write_jsonl(run_dir / "judgments.retrieval.jsonl", judgments)
#     return contexts_map if couple else {}

from typing import Tuple, List, Dict
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
import time

from chatbot_evaluation.evalkit.retrieval_bundle import load_bundle, make_normalizer, build_embedder, build_qdrant_repo
from chatbot_evaluation.retreive.application.ingest_docs import ingest_documents
from chatbot_evaluation.retreive.application.retrieve_eval import retrieve_and_evaluate
from chatbot_evaluation.retreive.domain.entities import Document, Query
from chatbot_evaluation.retreive.infrastructure.logger import get_logger
from chatbot_evaluation.retreive.infrastructure.pytrec_evaluator import PytrecEvaluator

log = get_logger(__name__)

def run_retrieval_phase(retrieval_manifest: dict, run_dir: Path) -> Tuple[dict, List[dict]]:
    """
    Ingests corpus (if needed), runs retrieval for multiple cutoffs, persists artifacts.

    Returns:
      contexts_map  : {qid: [top passages (strings)]} for the largest k
      retrieval_rows: [{"k": 5, "precision@k":..., "recall@k":..., "ndcg@k":..., "mrr":...}, ...]
    """
    # 1) Load bundle and adapters
    bundle_file = retrieval_manifest["bundle"]["file"] \
        if isinstance(retrieval_manifest.get("bundle"), dict) else retrieval_manifest["bundle"]
    rb = load_bundle(bundle_file)

    normalizer = make_normalizer(rb)
    device = "cpu"  # flip to 'cuda' if you like
    embedder = build_embedder(rb, device=device)
    vectordb = build_qdrant_repo(rb)
    score_mode = score_mode_from_bundle(rb)

    # 2) Load HF retrieval dataset (expects 'corpus','queries','qrels')
    ds_cfg = retrieval_manifest.get("dataset", {})
    ds_path = ds_cfg.get("path")
    if not ds_path:
        raise ValueError("retrieval.dataset.path is required in the suite/manifest.")
    dsd = load_from_disk(ds_path)

    corpus  = [Document(str(d["doc_id"]), normalizer(d["text"])) for d in dsd["corpus"]]
    queries = [Query(str(q["query_id"]), normalizer(q.get("query", q.get("text", "")))) for q in dsd["queries"]]

    # qrels to str keys
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for row in dsd["qrels"]:
        qid = str(row["query_id"]); did = str(row["doc_id"])
        rel = int(row.get("relevance", row.get("rel", 1)))
        if rel > 0:
            qrels[qid][did] = rel

    # 3) Ingest once (idempotent)
    coll = f"coll_{ds_cfg.get('name','retr')}"
    if not vectordb.has_collection(coll):
        ingest_documents(corpus, embedder, vectordb, coll)

    # 4) Evaluate @ multiple cutoffs
    eval_cfg: Dict = retrieval_manifest.get("eval", {}) or {}
    top_ks: List[int] = eval_cfg.get("top_k", [5])

    results = retrieve_and_evaluate(
        queries=queries,
        qrels=qrels,
        embedder=embedder,
        vectordb=vectordb,
        collection_name=coll,
        cutoffs=top_ks,
        score_mode=score_mode,
    )
    print("results = retrieve_and_evaluate")
    print(results)
    # 5) Build contexts_map from the largest k
    k_big = max(top_ks)
    doc_text = {d.doc_id: d.text for d in corpus}
    contexts_map: Dict[str, List[str]] = {}
    for q in queries:
        docs = results.get("per_query", {}).get(str(q.query_id), {}).get("docs", [])[:k_big]
        contexts_map[str(q.query_id)] = [doc_text.get(doc_id, "") for (doc_id, _s) in docs]

    # 6) Flatten per-k rows for per_k.jsonl
    per_k = results.get("per_k", {})
    retrieval_rows = []
    for k, vals in sorted(per_k.items(), key=lambda kv: int(kv[0])):
        row = {"k": int(k)}
        # vals looks like {"precision@k": ..., "recall@k": ..., "ndcg@k": ..., "mrr": ...}
        for kk, vv in vals.items():
            if vv is not None:
                row[kk] = float(vv)  # force native float
        retrieval_rows.append(row)
    # 7) Persist artifacts
    rdir = run_dir / "retrieval"
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "results.json", results)
    write_jsonl(rdir / "per_k.jsonl", retrieval_rows)

    ctx_rows = []
    for q in queries:
        ctx_rows.append({
            "qid": str(q.query_id),
            "query": q.text,
            "top_k": k_big,
            "contexts": contexts_map.get(str(q.query_id), []),
        })
    write_jsonl(rdir / "contexts.jsonl", ctx_rows)



    return contexts_map, retrieval_rows