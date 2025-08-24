from __future__ import annotations
import os, json, argparse
from pathlib import Path
import pandas as pd

from ragas.dataset_schema import SingleTurnSample

from .files import ensure_dir, read_yaml, write_yaml, write_jsonl
from .manifest import build_manifest
from .metrics_registry import MetricSpec, build_metric, run_metric
from .merge import merge_metric_rows, aggregate
from .reporting import NoopReporter, MLflowReporter

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
                response=row["response"],
                reference=row.get("reference"),
                retrieved_contexts=_to_list(row.get("retrieved_contexts")),
                id=row.get("qid"),
            )
        )
    return samples

def init_judge(judges_cfg: dict):
    kind = (judges_cfg.get("llm", {}).get("kind") or "http").lower()
    if kind == "http":
        # minimal wrapper expected by your metrics — you’ll pass this into your file-backed classes
        # Replace with your get_local_judge()
        from chatbot_evaluation.temp.judge import get_local_judge
        return get_local_judge()
    elif kind == "openai":
        # provide your own OpenAI LangChain LLM wrapper to match your metrics
        raise NotImplementedError("Wire OpenAI judge wrapper")
    else:
        raise NotImplementedError(f"Judge kind not supported yet: {kind}")

def init_embeddings(judges_cfg: dict):
    # Provide your embeddings object if a metric requires it (answer_relevancy)
    # Example: SentenceTransformer or LangChain HF wrapper.
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="config/seed.yaml")
    ap.add_argument("--out_base", default=None, help="Override output base dir from seed.output.base_dir")
    ap.add_argument("--reporter", choices=["none","mlflow"], default="none")
    args = ap.parse_args()

    seed = read_yaml(args.seed)
    out_base = args.out_base or seed["output"]["base_dir"]

    # 1) Manifest
    manifest = build_manifest(seed)
    run_id = manifest["run"]["id"]
    run_dir = Path(out_base) / run_id
    metrics_dir = run_dir / "metrics"
    ensure_dir(str(metrics_dir))

    # 2) Samples
    samples = load_samples(seed["dataset"]["path"])

    # 3) Judges / embeddings
    llm_judge = init_judge(seed.get("judges", {}))
    embeddings = init_embeddings(seed.get("judges", {}))

    # 4) Instantiate metrics
    prompt_base_dirs = {k: str(Path(v).parent) for k, v in seed.get("prompts", {}).items()}  # adjust as needed
    metric_specs = [MetricSpec(**m) for m in seed.get("metrics", [])]
    metric_objs = []
    for ms in metric_specs:
        metric = build_metric(ms, llm_judge=llm_judge, embeddings=embeddings, prompt_base_dirs=prompt_base_dirs)
        metric_objs.append(metric)

    # 5) Run metrics -> write per-metric JSONL
    all_rows = []
    for m in metric_objs:
        rows = run_metric(m, samples)
        write_jsonl(metrics_dir / f"{m.name}.jsonl", rows)
        all_rows.extend(rows)

    # 6) Merge per-metric rows into single judgments.jsonl
    merged = merge_metric_rows(all_rows)
    write_jsonl(run_dir / "judgments.jsonl", merged)

    # 7) Aggregates
    agg = aggregate(merged)
    with open(run_dir / "aggregates.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    # 8) Persist manifest last (after we know outputs exist)
    write_yaml(run_dir / "manifest.yaml", manifest)

    # 9) Reporting (optional MLflow)
    if args.reporter == "mlflow":
        rep = MLflowReporter()
    else:
        rep = NoopReporter()
    rep.start_run(manifest)
    rep.log_metrics(agg["metrics"])
    rep.log_artifact(str(run_dir / "manifest.yaml"))
    rep.log_artifact(str(run_dir / "judgments.jsonl"))
    rep.log_artifact(str(run_dir / "aggregates.json"))
    rep.end_run()

    print(f"[OK] Run artifacts: {run_dir}")

if __name__ == "__main__":
    main()