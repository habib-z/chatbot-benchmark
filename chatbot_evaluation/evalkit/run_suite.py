# chatbot_evaluation/evalkit/run_suite.py
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import logging
import pandas as pd

from ragas.dataset_schema import SingleTurnSample

from chatbot_evaluation.evalkit.resolve import write_manifest_and_lock
from chatbot_evaluation.evalkit.files import ensure_dir, write_jsonl, write_json, write_yaml, read_yaml
from chatbot_evaluation.evalkit.metrics_registry import MetricSpec, build_metric, run_metric
from chatbot_evaluation.evalkit.merge import merge_metric_rows, aggregate
from chatbot_evaluation.evalkit.reporting import NoopReporter
from chatbot_evaluation.evalkit.gates import evaluate_gates
# ... your imports ...
from chatbot_evaluation.evalkit.retrieval_phase import run_retrieval_phase
from chatbot_evaluation.evalkit.gen_autofill import ChatClientHTTP, autofill_answers_for_dataset
# ...
# NEW:
from chatbot_evaluation.evalkit.retrieval_phase import run_retrieval_phase
from chatbot_evaluation.evalkit.gen_autofill import ChatClientHTTP, autofill_answers_for_dataset

log = logging.getLogger("run_suite")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

def _to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return [x]
    return []

def load_samples(jsonl_path: str):
    df = pd.read_json(jsonl_path, lines=True)
    samples = []
    for _, row in df.iterrows():
        samples.append(
            SingleTurnSample(
                user_input=row.get("query") or row.get("user_input"),
                response=row.get("response"),
                reference=row.get("reference"),
                retrieved_contexts=_to_list(row.get("retrieved_contexts")),
                id=row.get("qid"),
            )
        )
    return samples

def init_judge(manifest: dict):
    from chatbot_evaluation.temp.judge import get_local_judge
    return get_local_judge()

def init_embeddings(manifest: dict):
    # Try using the retrieval bundle embedder
    try:
        from chatbot_evaluation.evalkit.retrieval_bundle import load_bundle, build_embedder
        import torch
        rb_entry = manifest.get("retrieval", {}).get("bundle")
        bundle_file = rb_entry["file"] if isinstance(rb_entry, dict) else rb_entry
        if not bundle_file:
            return None
        rb = load_bundle(bundle_file)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        se = build_embedder(rb, device=device)  # your SentenceEmbedder

        class _EmbAdapter:
            def __init__(self, se): self.se = se
            def embed_query(self, text: str):
                return self.se.encode([text])[0]
            def embed_documents(self, texts: list[str]):
                return self.se.encode(list(texts))

        return _EmbAdapter(se)
    except Exception as e:
        log.warning(f"init_embeddings: falling back to None ({e})")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="../config/suite.yaml")
    args = ap.parse_args()

    # 1) resolve suite -> manifest + lock
    temp_out = ".run_temp"
    ensure_dir(temp_out)
    manifest, lock = write_manifest_and_lock(temp_out, args.suite)

    # 2) run directory
    out_base = manifest["output"]["base_dir"]
    run_id = manifest["run"]["id"]
    run_dir = Path(out_base) / run_id
    metrics_dir = run_dir / "metrics"
    ensure_dir(str(metrics_dir))

    # persist manifest+lock inside run dir
    write_yaml(run_dir / "manifest.yaml", manifest)
    write_yaml(run_dir / "lock.yaml", lock)

    # ---------- NEW: retrieval phase ----------
    # ---------- retrieval phase ----------
    contexts_map = {}
    retrieval_agg = {}  # dict with {"metrics": {...}}
    per_k_rows = []  # rows we will write to retrieval/per_k.jsonl

    if manifest.get("retrieval"):
        log.info("Running retrieval phase…")
        # Your existing function should create retrieval/results.json (TREC-style per-query)
        contexts_map, _raw_rows = run_retrieval_phase(manifest["retrieval"], run_dir)

        # Build per_k from results.json regardless of what run_retrieval_phase returned
        from chatbot_evaluation.evalkit.merge import (
            flatten_trec_results_to_per_k,
            aggregate_retrieval_from_per_k,
        )
        retrieval_dir = run_dir / "retrieval"
        ks = manifest.get("retrieval", {}).get("eval", {}).get("top_k", [10])

        per_k_rows = flatten_trec_results_to_per_k(retrieval_dir / "results.json", ks=ks)
        write_jsonl(retrieval_dir / "per_k.jsonl", per_k_rows)

        retrieval_agg = aggregate_retrieval_from_per_k(per_k_rows)
        write_json(retrieval_dir / "aggregates.json", retrieval_agg)
    else:
        log.info("No retrieval block; skipping retrieval phase.")

    # ---------- NEW: optional E2E generation ----------
    gen_cfg = manifest.get("generation", {})
    if gen_cfg.get("autofill_answers"):
        base_url = gen_cfg.get("chat_api", {}).get("base_url")
        if not base_url:
            raise ValueError("generation.autofill_answers=true but chat_api.base_url missing")
        client = ChatClientHTTP(base_url=base_url, timeout_s=int(gen_cfg.get("chat_api", {}).get("timeout_s", 30)))
        ds_out_dir = run_dir / "datasets";
        ds_out_dir.mkdir(parents=True, exist_ok=True)
        for ds in manifest["datasets"]:
            in_path = ds["path"]
            out_path = ds_out_dir / f"{ds['id'].replace('/', '_').replace('@', '-')}.autofilled.jsonl"
            log.info(f"Autofilling answers for dataset={ds['id']} -> {out_path}")
            autofill_answers_for_dataset(in_jsonl=in_path, out_jsonl=out_path,
                                         chat_client=client, contexts_map=contexts_map or {})
            ds["path"] = str(out_path)  # use autofilled copy for this run
    # 3) init judge + embeddings
    llm_judge = init_judge(manifest)
    embeddings = init_embeddings(manifest)

    # 4) build metric objects
    metric_specs = [
        MetricSpec(name=m["name"], impl_id=m["impl_id"], params=m["params"], datasets=m["datasets"])
        for m in manifest["metrics"]
    ]
    metric_objs = {ms.name: build_metric(ms, manifest, llm_judge, embeddings) for ms in metric_specs}

    # 5) run metrics per dataset -> write per-metric files
    # all_rows = list(retrieval_rows)  # include retrieval rows in overall aggregation
    all_rows = []
    for ds in manifest["datasets"]:
        ds_id = ds["id"]
        ds_samples = load_samples(ds["path"])
        for mspec in metric_specs:
            if ds_id not in mspec.datasets:
                continue
            mobj = metric_objs[mspec.name]
            log.info(f"Running metric={mspec.name} on dataset={ds_id} ({len(ds_samples)} samples)")
            rows = run_metric(mobj, ds_samples)
            for r in rows:
                r["dataset_id"] = ds_id
            per_file = metrics_dir / f"{mspec.name}__{ds_id.replace('/','_').replace('@','-')}.jsonl"
            write_jsonl(per_file, rows)
            all_rows.extend(rows)

    # 6) merge judgments
    merged = merge_metric_rows(all_rows)
    write_jsonl(run_dir / "judgments.jsonl", merged)

    # 7) aggregates
    agg_overall = aggregate(merged)
    per_ds = {}
    for ds in manifest["datasets"]:
        ds_rows = [r for r in all_rows if r.get("dataset_id") == ds["id"]]
        ds_merged = merge_metric_rows(ds_rows)
        per_ds[ds["id"]] = aggregate(ds_merged)
    agg = {
        "overall": agg_overall,  # QA metrics (faithfulness/factual/etc.)
        "per_dataset": per_ds,  # QA per dataset
        "retrieval": retrieval_agg,  # Retrieval metrics (flat dict under 'metrics')
    }
    write_json(run_dir / "aggregates.json", agg)


    # 8) gates
    gate_space = dict(agg_overall.get("metrics", {}))
    gate_space.update(retrieval_agg.get("metrics", {}))  # now you can gate on 'retrieval_recall@5', etc.

    # evaluate_gates expects {'metrics': {...}}
    gates = evaluate_gates({"metrics": gate_space}, manifest.get("reporting", {}).get("gates", {}))
    write_json(run_dir / "gates.json", gates)

    # 9) reporter
    rep = NoopReporter()
    rep.start_run(manifest)
    rep.log_metrics(agg_overall["metrics"])
    rep.log_artifact(str(run_dir / "manifest.yaml"))
    rep.log_artifact(str(run_dir / "judgments.jsonl"))
    rep.log_artifact(str(run_dir / "aggregates.json"))
    rep.log_artifact(str(run_dir / "gates.json"))
    rep.end_run()

    print(f"[OK] {run_dir}")

if __name__ == "__main__":
    main()