import json
from pathlib import Path
import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from chatbot_evaluation.temp.judge import get_local_judge

# pick ONE embeddings provider you actually use in your stack.
# Example (HuggingFace local or auto-downloaded):
from langchain_community.embeddings import HuggingFaceEmbeddings
# If you use OpenAI, swap to: from langchain_openai import OpenAIEmbeddings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from chatbot_evaluation.metrics.m_answer_relevancy import AnswerRelevancy

# -------- defaults (edit here; no CLI required) --------
DEFAULT_BENCH_PATH ="../benchmarks/asiyeh/gemma-12b-ref-context.jsonl" # JSONL with user_input, response[, qid]
DEFAULT_BASE_DIR   = "../prompts/answer_relevancy/response_relevance/v1"
DEFAULT_OUT_DIR    = "runs/answer_relevancy"
DEFAULT_LIMIT      = 2
DEFAULT_STRICTNESS = 3

def main(
    bench_path: str = DEFAULT_BENCH_PATH,
    base_dir: str   = DEFAULT_BASE_DIR,
    out_dir: str    = DEFAULT_OUT_DIR,
    limit: int | None = DEFAULT_LIMIT,
    strictness: int = DEFAULT_STRICTNESS,
):
    llm = get_local_judge()
    # Multilingual embedding recommended for Persian
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    embeddings = HuggingFaceEmbeddings(
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        model_kwargs={"trust_remote_code": True},  # Arctic uses custom code
        encode_kwargs={"prompt_name": "query", "normalize_embeddings": True}
    )

    df = pd.read_json(bench_path, lines=True)

    samples = []
    for _, row in df.iterrows():
        samples.append(
            SingleTurnSample(
                user_input=row["query"],
                response=row["response"],
                id=row.get("qid"),
            )
        )
    if limit is not None:
        samples = samples[:limit]

    metric = AnswerRelevancy(
        llm=llm,
        embeddings=embeddings,
        base_dir=base_dir,
        strictness=strictness,
    )

    scores = []
    judgments = []
    for s in samples:
        print("sample __________________________________ ")
        print("sample", s)
        score = metric.single_turn_score(s)
        scores.append(score)

        # prefer sample-attached details; fallback to metric cache
        details = getattr(s, "_ragas_details", {}).get("answer_relevancy") or {}
        if not details:
            key = getattr(s, "id", None) or (hash(s.user_input), hash(s.response))
            details = getattr(metric, "_details_cache", {}).get(key, {})

        judgments.append({
            "qid": getattr(s, "id", None),
            "user_input": s.user_input,
            "response": s.response,
            "answer_relevancy_score": score,
            "answer_relevancy_details": details,
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
