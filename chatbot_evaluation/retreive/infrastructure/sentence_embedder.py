# infrastructure/sentence_embedder.py
from sentence_transformers import SentenceTransformer
from retreive.domain.interfaces import Embedder

class SentenceEmbedder(Embedder):
    def __init__(self, model_name: str, device: str):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
