import pandas as pd
import json
from pathlib import Path


def convert_excel_to_beir_format(chunks_path: str, queries_path: str, output_dir: str):
    """
    Convert two Excel files (chunks and queries) to BEIR/MTEB standard format:
    - corpus.jsonl
    - queries.jsonl
    - qrels.tsv

    Each query will also store 'correct_answer' for additional context.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read Excel files
    chunks_df = pd.read_excel(chunks_path)
    queries_df = pd.read_excel(queries_path)

    # Validate required columns
    required_chunks_cols = {'chunk_id', 'text'}
    required_queries_cols = {'query_id', 'query', 'correct_answer', 'correct_chunk_ids'}
    if not required_chunks_cols.issubset(chunks_df.columns):
        raise ValueError(f"Chunks file missing columns: {required_chunks_cols - set(chunks_df.columns)}")
    if not required_queries_cols.issubset(queries_df.columns):
        raise ValueError(f"Queries file missing columns: {required_queries_cols - set(queries_df.columns)}")

    # 1. Create corpus.jsonl
    corpus_path = output_dir / "corpus.jsonl"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        for _, row in chunks_df.iterrows():
            doc_id = str(row['chunk_id'])
            text = str(row['text'])
            entry = {"doc_id": doc_id, "text": text, "title": ""}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 2. Create queries.jsonl (with correct_answer included)
    queries_path_out = output_dir / "queries.jsonl"
    with open(queries_path_out, 'w', encoding='utf-8') as f:
        for _, row in queries_df.iterrows():
            q_id = str(row['query_id'])
            query_text = str(row['query'])
            correct_answer = str(row['correct_answer'])
            entry = {"query_id": q_id, "text": query_text, "correct_answer": correct_answer}
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 3. Create qrels.tsv
    qrels_path = output_dir / "qrels.tsv"
    with open(qrels_path, 'w', encoding='utf-8') as f:
        f.write("query-id\t0\tdoc-id\t1\n")  # header (optional for clarity)
        for _, row in queries_df.iterrows():
            q_id = str(row['query_id'])
            correct_chunk_ids = str(row['correct_chunk_ids'])
            chunk_ids = [cid.strip() for cid in correct_chunk_ids.split(',')]
            for doc_id in chunk_ids:
                f.write(f"{q_id}\t0\t{doc_id}\t1\n")

    print(f"Files created at: {output_dir}")
    print(f"- {corpus_path.name}")
    print(f"- {queries_path_out.name}")
    print(f"- {qrels_path.name}")


if __name__ == "__main__":
    chunks_file = "../../data/abank/v1/chuck_jul_30.xlsx"
    queries_file = "../../data/abank/v1//q_chuck_id.xlsx"
    output_directory = "../../data/jsonl/abank/v1"

    convert_excel_to_beir_format(chunks_file, queries_file, output_directory)
