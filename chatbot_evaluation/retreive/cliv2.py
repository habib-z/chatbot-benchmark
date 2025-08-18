# presentation/cli.py
import yaml
import torch
from infrastructure.logger import get_logger, log_manifest

import torch
from collections import defaultdict
from infrastructure.sentence_embedder import SentenceEmbedder
from infrastructure.qdrant_repo import QdrantRepository
from infrastructure.pytrec_evaluator import PytrecEvaluator
from application.ingest_docs import ingest_documents
from application.retrieve_eval import retrieve_and_evaluate
from datasets import load_from_disk
from domain.entities import Document, Query
import time
logger = get_logger(__name__)

def main(config_path="configs/default.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # prepare retrievers from config (dense, bm25, hybrid, reranker)
    # ... (plug into same interfaces)

    ds = load_from_disk("../dataset/retreive/scifact_fa_ds")
    corpus = [Document(d["doc_id"], d["text"]) for d in ds["corpus"]]
    queries = [Query(q["query_id"], q["query"]) for q in ds["queries"]]
    qrels = defaultdict(dict)
    for row in ds["qrels"]:
        if row["relevance"] > 0:  # keep only positives
            qrels[row["query_id"]][row["doc_id"]] = row["relevance"]

    embedder = SentenceEmbedder("Snowflake/snowflake-arctic-embed-l-v2.0",
                                device="cuda" if torch.cuda.is_available() else "cpu")
    vectordb = QdrantRepository(host="185.255.91.144", port=30081)
    evaluator = PytrecEvaluator()


    # collection naming
    coll = f"{cfg['qdrant']['collection_prefix']}_{cfg['experiment']['name']}_{int(time.time())}"

    # run ingestion + retrieval + eval
    ingest_documents(corpus, embedder, vectordb, coll)
    results = retrieve_and_evaluate(queries, qrels, embedder, vectordb, evaluator, coll,
                                    top_k=cfg["retrieval"]["top_k"])

    logger.info(f"Final Results: {results}")
    log_manifest(cfg, results, "runs")
