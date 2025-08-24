import json
from pathlib import Path
import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from chatbot_evaluation.temp.judge import get_local_judge

from chatbot_evaluation.metrics.m_factual_correctness import FactualCorrectnessFileBacked

# defaults (edit in-file; no argparse required)
DEFAULT_BENCH_PATH ="../benchmarks/asiyeh/gemma-12b-ref-context.jsonl"      # JSONL with response, reference[, qid]
DEFAULT_BASE_DIR   = "../prompts/factual_correctness/claim_decomposition/v1"
DEFAULT_OUT_DIR    = "runs/factual_correctness"
DEFAULT_LIMIT      = 2

def main(
    bench_path: str = DEFAULT_BENCH_PATH,
    base_dir: str   = DEFAULT_BASE_DIR,
    out_dir: str    = DEFAULT_OUT_DIR,
    limit: int | None = DEFAULT_LIMIT,
    mode: str = "f1",
    beta: float = 1.0,
    atomicity: str = "low",
    coverage: str = "high",
):
    llm = get_local_judge()
    df = pd.read_json(bench_path, lines=True)

    samples = []
    for _, row in df.iterrows():
        samples.append(
            SingleTurnSample(
                response=row["response"],
                reference=row["reference"],
                id=row.get("qid"),
            )
        )
    if limit is not None:
        samples = samples[:limit]

    metric = FactualCorrectnessFileBacked(
        llm=llm,
        base_dir=base_dir,
        mode=mode,
        beta=float(beta),
        atomicity=atomicity,   # "low" or "high"
        coverage=coverage,     # "low" or "high"
    )

    scores = []
    judgments = []
    for s in samples:
        score = metric.single_turn_score(s)
        scores.append(score)

        details = getattr(s, "_ragas_details", {}).get("factual_correctness") or {}
        if not details:
            key = getattr(s, "id", None) or (hash(s.response), hash(s.reference))
            details = getattr(metric, "_details_cache", {}).get(key, {})

        judgments.append({
            "qid": getattr(s, "id", None),
            "response": s.response,
            "reference": s.reference,
            "factual_correctness_score": score,
            "factual_correctness_details": details,
        })
        print(score)

    print(scores)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    with (outp / "judgments.jsonl").open("w", encoding="utf-8") as f:
        for row in judgments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
