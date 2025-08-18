import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict, Counter
import random

def extract_dataset_stats(dataset_path: Path):
    corpus = load_dataset("json", data_files=str(dataset_path / "corpus.jsonl"), split="train")
    queries = load_dataset("json", data_files=str(dataset_path / "queries.jsonl"), split="train")
    qrels_df = pd.read_csv(dataset_path / "qrels.tsv", sep="\t", names=["query_id", "_", "doc_id", "score"])

    # Build qrel dictionary
    qrels = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels[row["query_id"]][row["doc_id"]] = int(row["score"])

    rel_dist = Counter(qrels_df["score"])
    queries_with_2plus_rels = sum(1 for v in qrels.values() if len(v) >= 2)

    stats = {
        "dataset": dataset_path.name,
        "num_docs": len(corpus),
        "num_queries": len(queries),
        "num_qrels": len(qrels_df),
        "avg_rels_per_query": round(sum(len(v) for v in qrels.values()) / len(qrels), 2),
        "queries_with_2+_rels (%)": round(100 * queries_with_2plus_rels / len(qrels), 2),
        "rel_score_distribution": dict(rel_dist)
    }

    print(f"\n📊 [{dataset_path.name}] Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats

def write_sample_file(dataset_path: Path, output_dir: Path, num_samples=3, num_neg=2):
    corpus = load_dataset("json", data_files=str(dataset_path / "corpus.jsonl"), split="train")
    queries = load_dataset("json", data_files=str(dataset_path / "queries.jsonl"), split="train")
    qrels_df = pd.read_csv(dataset_path / "qrels.tsv", sep="\t", names=["query_id", "_", "doc_id", "score"])

    corpus_dict = {doc["doc_id"]: doc["text"] for doc in corpus}
    query_dict = {q["query_id"]: q["text"] for q in queries}
    rels = defaultdict(set)
    for _, row in qrels_df.iterrows():
        if int(row["score"]) > 0:
            rels[row["query_id"]].add(row["doc_id"])

    qids = list(rels.keys())
    sample_qids = random.sample(qids, min(num_samples, len(qids)))

    output_path = output_dir / f"{dataset_path.name}_samples.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in sample_qids:
            f.write("=" * 100 + "\n")
            f.write(f"Query: {query_dict.get(qid, '[Missing]')}\n\n")

            f.write("→ Positive Docs:\n")
            for did in rels[qid]:
                doc_text = corpus_dict.get(did, '[Missing]')
                f.write(f"  [+] {doc_text[:200].replace('\n', ' ')}...\n")

            all_doc_ids = set(corpus_dict.keys())
            negatives = list(all_doc_ids - rels[qid])
            sample_neg = random.sample(negatives, min(num_neg, len(negatives)))
            f.write("\n→ Negative Docs:\n")
            for did in sample_neg:
                doc_text = corpus_dict.get(did, '[Missing]')
                f.write(f"  [-] {doc_text[:200].replace('\n', ' ')}...\n")
            f.write("\n")

    print(f"✅ Saved samples to: {output_path.name}")

def analyze_all_beir_datasets(base_dir: str, sample_out_dir: str):
    base_path = Path(base_dir)
    out_path = Path(sample_out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    stats = []
    print("🚀 Starting BEIR dataset summary...\n")
    for subdir in sorted(os.listdir(base_path)):
        dataset_path = base_path / subdir
        if not dataset_path.is_dir():
            continue
        try:
            print(f"\n📂 Processing dataset: {subdir}")
            s = extract_dataset_stats(dataset_path)
            stats.append(s)
            write_sample_file(dataset_path, out_path)
        except Exception as e:
            print(f"[ERROR] Failed on {subdir}: {e}")

    df = pd.DataFrame(stats)
    summary_path = out_path / "dataset_statistics.csv"
    df.to_csv(summary_path, index=False)
    print(f"\n📊 All dataset stats saved to: {summary_path}")

# 🔧 SET YOUR PATH HERE:
if __name__ == "__main__":
    analyze_all_beir_datasets(
        base_dir="beir/",
        sample_out_dir="output_beir_summary"
    )
