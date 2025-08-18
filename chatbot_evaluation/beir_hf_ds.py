import json
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

def beir_to_hf_dataset(corpus_path, queries_path, qrels_path, output_dir=None):
    # Load corpus
    corpus = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus.append({
                "doc_id": obj["doc_id"],
                "text": obj["text"],
                "title": obj.get("title", "")
            })

    # Load queries
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            queries.append({
                "query_id": obj["query_id"],
                "query": obj["text"],
                "correct_answer": obj.get("correct_answer", "")
            })

    # Load qrels
    qrels = []
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            qid, _, docid, rel = line.strip().split('\t')
            qrels.append({
                "query_id": qid,
                "doc_id": docid,
                "relevance": int(rel)
            })

    # Convert to Hugging Face datasets
    corpus_ds = Dataset.from_pandas(pd.DataFrame(corpus))
    queries_ds = Dataset.from_pandas(pd.DataFrame(queries))
    qrels_ds = Dataset.from_pandas(pd.DataFrame(qrels))

    dataset_dict = DatasetDict({
        "corpus": corpus_ds,
        "queries": queries_ds,
        "qrels": qrels_ds
    })

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_dict.save_to_disk(str(output_path))
        print(f"Hugging Face dataset saved to {output_path}")

    return dataset_dict



if __name__ == '__main__':

    ds_name="abank"
    ds_name="touche2020_fa"
    ds_name="miracl_fa"
    ds_name="scifact_fa"
    ds_name="persian_web_document_retrieval"
    ds_name="cqad_upstack_mathematica_retrieval_fa"

    beir_dir= f"./beir/{ds_name}_beir/"
    ds_name =f"{ds_name}_ds"
    # Usage Example
    dataset_dict = beir_to_hf_dataset(
        f"{beir_dir}corpus.jsonl",
        f"{beir_dir}queries.jsonl",
        f"{beir_dir}qrels.tsv",
        output_dir=f"dataset/{ds_name}"
    )
    # print(dataset_dict)
    # output_dir = "dataset/abank_ds"
    # dataset_dict.save_to_disk(output_dir)