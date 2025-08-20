# run_faithfulness_from_files.py
import argparse, json
from pathlib import Path
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from langchain_openai import ChatOpenAI
from ragas.llms.base import LangchainLLMWrapper

from file_faithfulness_metric import FaithfulnessFromFiles, load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="benchmarks/faithfulness/v0",
                    help="folder containing statement_generator.* and nli_judge.* files")
    ap.add_argument("--samples", default="benchmarks/faithfulness/v0/datasets/samples.clientA.v0.jsonl")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI/vLLM-compatible model name")
    args = ap.parse_args()

    llm = LangchainLLMWrapper(ChatOpenAI(model_name=args.model, temperature=0.0))
    metric = FaithfulnessFromFiles(llm=llm, base_dir=args.base_dir)
    ds: EvaluationDataset = load_dataset(args.samples)

    # run evaluation (RAGAS aggregator returns overall score dict)
    results = evaluate(dataset=ds, metrics=[metric], llm=llm)
    print("AGGREGATES:", results)

    out_dir = Path("runs") / f"faithfulness_{Path(args.samples).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save per-item score + RAW judge verdict JSON
    with (out_dir / "judgments.jsonl").open("w", encoding="utf-8") as f:
        for s in ds.samples:
            det = getattr(s, "_ragas_details", {}).get("faithfulness_details", {})
            row = {
                "qid": getattr(s, "id", None),
                "user_input": s.user_input,
                "response": s.response,
                "retrieved_contexts": s.retrieved_contexts,
                "faithfulness_score": det.get("score"),
                "statements": det.get("statements"),
                "verdicts": det.get("verdicts"),  # list of {"statement","reason","verdict"}
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"aggregates": results,
                   "base_dir": args.base_dir,
                   "samples": args.samples,
                   "model": args.model}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()