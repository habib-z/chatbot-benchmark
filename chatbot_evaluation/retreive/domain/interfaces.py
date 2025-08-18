# domain/interfaces.py
from typing import List, Dict

class Embedder:
    def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class VectorDB:
    def create_collection(self, name: str, dim: int):
        raise NotImplementedError

    def upsert(self, name: str, ids: List[int], vectors: List[List[float]], payloads: List[dict]):
        raise NotImplementedError

    def search(self, name: str, query_vec: List[float], top_k: int) -> List[dict]:
        raise NotImplementedError

class Evaluator:
    def evaluate(self, run: Dict, qrels: Dict, metrics: List[str]) -> Dict:
        raise NotImplementedError
