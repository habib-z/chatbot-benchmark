import json
from pathlib import Path
import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from chatbot_evaluation.temp.judge import get_local_judge
from chatbot_evaluation.metrics.m_faithfulness import FaithfulnessFileBacked

# NOTE: this import was unused and can cause confusion; remove it.
# from faithfulness_from_files import Faithfulness

# -------- defaults (edit here; no CLI required) --------
DEFAULT_BENCH_PATH = "../benchmarks/asiyeh/gemma-12b-ref-context.jsonl"
DEFAULT_BASE_DIR = "../prompts/faithfulness/"
DEFAULT_OUTPUT_DIR = "runs/faithfulness"

def _to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return []

def main(
    bench_path: str = DEFAULT_BENCH_PATH,
    base_dir: str = DEFAULT_BASE_DIR,
    out_dir: str = DEFAULT_OUTPUT_DIR,
    limit: int | None = 2,
):
    evaluator_llm = get_local_judge()

    df = pd.read_json(bench_path, lines=True)
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
    if limit is not None:
        samples = samples[:limit]


    metric_faith = FaithfulnessFileBacked(llm=evaluator_llm, base_dir=base_dir)

    scores = []
    judgments = []
    for s in samples:
        print(s)
        score = metric_faith.single_turn_score(s)
        scores.append(score)

        # 1) try sample-attached details
        details = getattr(s, "_ragas_details", {}).get("faithfulness") or {}

        # 2) fallback to metric-level cache (handles internal copies)
        if not details:
            key = getattr(s, "id", None)
            print(f"key is: {key}")
            if key is None:
                key = (hash(s.user_input), hash(s.response))

            print(f"key is: {key}")
            details = getattr(metric_faith, "_details_cache", {}).get(key, {})

        judgments.append({
            "qid": getattr(s, "id", None),
            "query": s.user_input,
            "response": s.response,
            "retrieved_contexts": s.retrieved_contexts,
            "faithfulness_score": score,
            "faithfulness_details": details,
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