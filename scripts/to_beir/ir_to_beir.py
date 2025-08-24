from datasets import load_dataset
from pathlib import Path
import json

def convert_hf_ir_benchmark_to_beir_format(
    dataset_name: str,
    output_dir: str,
    corpus_split: str = "test",
    queries_split: str = "test",
    qrels_split: str = "test",
    config_name: str = None,
    include_empty_answer: bool = True
):
    """
    Convert a HuggingFace-style IR benchmark (e.g., Fa-MTEB) into BEIR format.

    Args:
        dataset_name (str): The name of the dataset on HuggingFace (e.g. 'mteb/Touche2020-Fa').
        output_dir (str): Path to output directory.
        corpus_split (str): Split name for corpus (default: 'test').
        queries_split (str): Split name for queries (default: 'test').
        qrels_split (str): Split name for qrels (default: 'test').
        config_name (str): Builder config name if applicable (e.g. 'corpus').
        include_empty_answer (bool): If True, adds a dummy `correct_answer` to each query.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load each part of the dataset
    corpus  = load_dataset(dataset_name, name="corpus",  split=corpus_split)
    queries = load_dataset(dataset_name, name="queries", split=queries_split)
    qrels   = load_dataset(dataset_name, name="qrels",   split=qrels_split)

    # Write corpus.jsonl
    with open(output_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            entry = {
                "doc_id": doc["_id"],
                "text": doc["text"],
                "title": ""
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write queries.jsonl
    with open(output_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            entry = {
                "query_id": q["_id"],
                "text": q["text"]
            }
            if include_empty_answer:
                entry["correct_answer"] = ""
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write qrels.tsv
    with open(output_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\t0\tdoc-id\t1\n")  # Optional header
        for r in qrels:
            f.write(f"{r['query-id']}\t0\t{r['corpus-id']}\t{r['score']}\n")

    print(f"✔ Converted: {dataset_name}")
    print(f"→ Files saved in {output_dir}")
    print("• corpus.jsonl\n• queries.jsonl\n• qrels.tsv")

from pathlib import Path
import json

def convert_ir_iter_datasets_to_beir_format(dataset_id: str, output_dir: str, binary_only: bool = True):
    """
    Convert an ir_datasets IR collection (e.g., miracl/fa/dev) into BEIR format.

    Args:
        dataset_id (str): Dataset ID from ir_datasets (e.g., 'miracl/fa/dev').
        output_dir (str): Path to output directory.
        binary_only (bool): If True, keep only qrels with relevance > 0.
    """
    import ir_datasets, collections

    ds = ir_datasets.load(dataset_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Corpus
    with open(output_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in ds.docs_iter():
            json.dump({"doc_id": doc.doc_id, "text": doc.text, "title": ""}, f, ensure_ascii=False)
            f.write("\n")

    # 2. Queries
    with open(output_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in ds.queries_iter():
            json.dump({
                "query_id": q.query_id,
                "text": q.text,
                "correct_answer": ""  # Placeholder
            }, f, ensure_ascii=False)
            f.write("\n")

    # 3. Qrels
    with open(output_dir / "qrels.tsv", "w", encoding="utf-8") as f:
        f.write("query-id\t0\tdoc-id\t1\n")  # Optional header
        for rel in ds.qrels_iter():
            if not binary_only or rel.relevance > 0:
                f.write(f"{rel.query_id}\t0\t{rel.doc_id}\t{int(rel.relevance)}\n")

    print(f"✔ Converted {dataset_id} to BEIR format in {output_dir}")

if __name__ == '__main__':
    ...
    # convert_hf_ir_benchmark_to_beir_format(
    #     dataset_name="mteb/Touche2020-Fa",
    #     output_dir="../beir/touche2020_fa_beir"
    # )

    # convert_hf_ir_benchmark_to_beir_format(
    #     dataset_name="mteb/SciFact-Fa",
    #     output_dir="./beir/scifact_fa_beir"
    # )
    # convert_hf_ir_benchmark_to_beir_format(
    #     dataset_name="mteb/PersianWebDocumentRetrieval",
    #     output_dir="./beir/persian_web_document_retrieval_beir",
    # )
    # convert_hf_ir_benchmark_to_beir_format(
    #     dataset_name="mteb/CQADupstackMathematicaRetrieval-Fa",
    #     output_dir="./beir/cqad_upstack_mathematica_retrieval_fa_beir",
    # )

    # output_dir="./beir/miracl_fa_beir"
    # convert_ir_iter_datasets_to_beir_format(
    #     dataset_id="miracl/fa/dev",
    #     output_dir=output_dir
    # )

    # convert_ir_iter_datasets_to_beir_format(
    #     dataset_id="hc4/fa/dev",
    #     output_dir="./beir/hc4_fa_beir",
    #     binary_only=False
    # )

