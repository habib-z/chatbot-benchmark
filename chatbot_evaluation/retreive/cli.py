
import torch
from collections import defaultdict
from infrastructure.sentence_embedder import SentenceEmbedder
from infrastructure.qdrant_repo import QdrantRepository
from infrastructure.pytrec_evaluator import PytrecEvaluator
from application.ingest_docs import ingest_documents
from application.retrieve_eval import retrieve_and_evaluate
from datasets import load_from_disk
from domain.entities import Document, Query

def main():
    ds = load_from_disk("../dataset/retreive/scifact_fa_ds")
    corpus = [Document(d["doc_id"], d["text"]) for d in ds["corpus"]]
    queries = [Query(q["query_id"], q["query"]) for q in ds["queries"]]
    qrels = defaultdict(dict)
    for row in ds["qrels"]:
        if row["relevance"] > 0:  # keep only positives
            qrels[row["query_id"]][row["doc_id"]] = row["relevance"]



    embedder = SentenceEmbedder("Snowflake/snowflake-arctic-embed-l-v2.0", device="cuda" if torch.cuda.is_available() else "cpu")
    vectordb = QdrantRepository(host="185.255.91.144", port=30081)
    evaluator = PytrecEvaluator()

    coll = "scifact_fa_snow"
    ingest_documents(corpus, embedder, vectordb, coll)
    scores = retrieve_and_evaluate(queries, qrels, embedder, vectordb, evaluator, coll)

    print(scores)

if __name__ == "__main__":
    main()
