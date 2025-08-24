# application/ingest_docs.py
from chatbot_evaluation.retreive.domain.entities import Document
from chatbot_evaluation.retreive.domain.interfaces import Embedder, VectorDB
from chatbot_evaluation.retreive.infrastructure.logger import get_logger
from tqdm import tqdm
import time

logger = get_logger(__name__)

def ingest_documents(
    docs: list[Document],
    embedder: Embedder,
    vectordb: VectorDB,
    collection_name: str,
    batch_size: int = 256
):
    start = time.time()
    dim = len(embedder.encode([docs[0].text])[0])
    vectordb.create_collection(collection_name, dim)

    ids, vecs, payloads = [], [], []
    for i, doc in enumerate(tqdm(docs, desc=f"Ingesting into {collection_name}")):
        vec = embedder.encode([doc.text])[0]
        ids.append(i)
        vecs.append(vec)
        payloads.append({"chunk_id": doc.doc_id, "base_doc_id": doc.doc_id.split("#")[0]})

        if len(ids) == batch_size:
            vectordb.upsert(collection_name, ids, vecs, payloads)
            ids, vecs, payloads = [], [], []
    if ids:
        vectordb.upsert(collection_name, ids, vecs, payloads)

    logger.info(f"Ingested {len(docs)} docs into {collection_name} in {time.time()-start:.2f}s")
