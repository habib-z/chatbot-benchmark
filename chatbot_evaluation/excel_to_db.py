import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any
import ast

class IRDataset(Dataset):
    """
    Custom PyTorch Dataset for Information Retrieval
    containing documents, queries, and qrels.
    """
    def __init__(self, docs: Dict[str, str], queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]):
        self.docs = docs
        self.queries = queries
        self.qrels = qrels
        self.query_ids = list(queries.keys())

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        query_text = self.queries[query_id]
        relevant_docs = self.qrels.get(query_id, {})
        return {
            "query_id": query_id,
            "query": query_text,
            "relevant_docs": relevant_docs
        }


def load_excel_to_ir_dataset(chunks_path: str, queries_path: str) -> IRDataset:
    """
    Load Excel files and convert them to an IRDataset with standard structure:
    docs: {chunk_id: text}
    queries: {query_id: query_text}
    qrels: {query_id: {chunk_id: relevance_score}}
    """
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

    # Create docs dictionary
    docs = {str(row['chunk_id']): str(row['text']) for _, row in chunks_df.iterrows()}

    # Create queries dictionary
    queries = {str(row['query_id']): str(row['query']) for _, row in queries_df.iterrows()}

    # Create qrels dictionary
    qrels = {}
    for _, row in queries_df.iterrows():
        query_id = str(row['query_id'])
        correct_chunk_ids = str(row['correct_chunk_ids'])
        # Convert "1,2,3" to list
        chunk_ids = [chunk.strip() for chunk in correct_chunk_ids.split(',')]
        qrels[query_id] = {cid: 1 for cid in chunk_ids}  # relevance score = 1

    return IRDataset(docs=docs, queries=queries, qrels=qrels)


if __name__ == "__main__":
    # Example usage
    chunks_file = "jul30/chuck_jul_30.xlsx"
    queries_file = "jul30/q_chuck_id.xlsx"

    dataset = load_excel_to_ir_dataset(chunks_file, queries_file)

    # Example: Inspect first item
    print("Number of queries:", len(dataset))
    print("Sample item:", dataset[0])
