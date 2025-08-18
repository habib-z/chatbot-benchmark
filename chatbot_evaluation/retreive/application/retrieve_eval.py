# application/retrieve_eval.py
from retreive.domain.entities import Query, Qrels
from retreive.domain.interfaces import Embedder, VectorDB, Evaluator
from retreive.infrastructure.logger import get_logger
from tqdm import tqdm
import time

logger = get_logger(__name__)

def retrieve_and_evaluate(
    queries: list[Query],
    qrels: Qrels,
    embedder: Embedder,
    vectordb: VectorDB,
    evaluator: Evaluator,
    collection_name: str,
    top_k: int = 5
):
    start = time.time()
    run = {}
    for query in tqdm(queries, desc="Retrieving"):
        qvec = embedder.encode([query.text])[0]
        results = vectordb.search(collection_name, qvec, top_k)
        run[query.query_id] = {r["chunk_id"]: 1 - r["score"] for r in results}

    metrics = [f"recall_{top_k}", f"P_{top_k}", f"ndcg_cut_{top_k}"]
    scores = evaluator.evaluate(run, qrels, metrics)
    elapsed = time.time() - start
    logger.info(f"Retrieved {len(queries)} queries in {elapsed:.2f}s (avg {elapsed/len(queries):.3f}s/q)")
    return scores
