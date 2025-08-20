import argparse, json
from pathlib import Path
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from langchain_openai import ChatOpenAI
from ragas.llms.base import LangchainLLMWrapper

from file_metric_factual import FactualFromFiles, load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="benchmarks/factual_correctness/v0",
                    help="folder containing instruction.en.md, output_schema.json, examples.jsonl, rubric.yaml")
    ap.add_argument("--samples", default="benchmarks/factual_correctness/v0/datasets/samples.clientA.v0.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini", help="judge model name (OpenAI or your vLLM-compatible)")
    args = ap.parse_args()

    # Judge LLM (uses OPENAI_API_KEY; respects OPENAI_BASE_URL if you run vLLM)
    judge = LangchainLLMWrapper(ChatOpenAI(model_name=args.model, temperature=0.0))

    # Metric + dataset
    metric = FactualFromFiles(llm=judge, base_dir=args.base_dir)
    ds: EvaluationDataset = load_dataset(args.samples)

    # Run
    results = evaluate(dataset=ds, metrics=[metric], llm=judge)
    print("AGGREGATES:", results)  # {'factual_fbeta': 0.xx}

    # Save per-sample raw verdict + numeric components
    run_dir = Path("runs") / f"factual_{Path(args.samples).stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "judgments.jsonl", "w", encoding="utf-8") as f:
        for s in ds.samples:
            det = getattr(s, "_ragas_details", {})
            row = {
                "qid": getattr(s, "id", None),
                "user_input": s.user_input,
                "reference": s.reference,
                "assistant_text": s.response,
                "factual_verdict": det.get("factual_verdict"),     # RAW LLM RESULT (lists of claims etc.)
                "precision": det.get("factual_precision"),
                "recall": det.get("factual_recall"),
                "f_beta": det.get("factual_fbeta"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"aggregates": results,
                   "base_dir": args.base_dir,
                   "samples": args.samples,
                   "model": args.model}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
