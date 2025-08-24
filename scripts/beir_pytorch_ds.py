import json
from pathlib import Path
from torch.utils.data import Dataset

class BEIRDataset(Dataset):
    def __init__(self, corpus_path: str, queries_path: str, qrels_path: str):
        self.docs = self._load_jsonl(corpus_path)
        self.queries = self._load_jsonl(queries_path)
        self.qrels = self._load_qrels(qrels_path)
        self.query_ids = list(self.queries.keys())

    def _load_jsonl(self, path):
        data = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                data[obj["_id"]] = obj
        return data

    def _load_qrels(self, path):
        qrels = {}
        with open(path, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                qid, _, docid, rel = line.strip().split('\t')
                qrels.setdefault(qid, {})[docid] = int(rel)
        return qrels

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, idx):
        qid = self.query_ids[idx]
        query_text = self.queries[qid]["text"]
        correct_answer = self.queries[qid].get("correct_answer", "")
        relevant_docs = self.qrels.get(qid, {})
        return {
            "query_id": qid,
            "query": query_text,
            "correct_answer": correct_answer,
            "relevant_docs": {docid: self.docs[docid]["text"] for docid in relevant_docs}
        }

# Usage
dataset = BEIRDataset("datasets/retrieval/beir/abank_beir/corpus.jsonl",
                      "datasets/retrieval/beir/abank_beir/queries.jsonl",
                      "datasets/retrieval/beir/abank_beir/qrels.tsv")
print(dataset[0])
