
from qdrant_client import QdrantClient, models
from chatbot_evaluation.retreive.domain.interfaces import VectorDB

class QdrantRepository(VectorDB):
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port, timeout=60)

    def create_collection(self, name, dim):
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
        )

    def upsert(self, name, ids, vectors, payloads):
        self.client.upsert(name, points=models.Batch(ids=ids, vectors=vectors, payloads=payloads))

    def search(self, name, query_vec, top_k):
        res = self.client.search(name, query_vector=query_vec, limit=top_k, with_vectors=False)
        return [{"chunk_id": p.payload["chunk_id"], "score": p.score} for p in res]

    def has_collection(self,collection_name):
        return self.client.collection_exists(collection_name)
