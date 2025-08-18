# -*- coding: utf-8 -*-
"""hf_dataset_pipeline.ipynb"""

# !pip install -U datasets sentence-transformers qdrant-client tqdm pytrec_eval

from datasets import load_from_disk
from collections import defaultdict
import random
import textwrap
import statistics
import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import pytrec_eval
import torch

wrap = lambda s: textwrap.fill(s, 88)

# ✅ Load local Hugging Face DatasetDict
ds = load_from_disk("dataset/retreive/scifact_fa_ds")

# Extract splits
corpus = ds["corpus"]
queries_ds = ds["queries"]
qrels_ds = ds["qrels"]

# Convert corpus to dict for fast access
docs = {corpus[i]["doc_id"]: corpus[i]["text"] for i in range(len(corpus))}
queries = {queries_ds[i]["query_id"]: queries_ds[i]["query"] for i in range(len(queries_ds))}

# Convert qrels to dict {qid: {docid: score}}
qrels = defaultdict(dict)
for i in range(len(qrels_ds)):
    qid = qrels_ds[i]["query_id"]
    did = qrels_ds[i]["doc_id"]
    score = qrels_ds[i]["relevance"]
    if score > 0:  # keep only positives
        qrels[qid][did] = score

print(f"docs:{len(docs):,}  queries:{len(queries):,}  positives avg:"
      f"{statistics.mean(len(r) for r in qrels.values()):.2f}")

# ✅ Eyeball helper
def show_random_query(pos_k=-1, neg_k=2):
    qid = random.choice(list(qrels.keys()))
    print("\n" + "="*100)
    print("Query:", wrap(queries[qid]), "\n")
    if pos_k < 0:
        pos_k = len(qrels[qid])
        print(f"pos num = {pos_k}")
    for did in random.sample(list(qrels[qid].keys()), k=min(pos_k, len(qrels[qid]))):
        print("[POS]", wrap(docs[did][:700]), "\n")
    neg_pool = [d for d in docs if d not in qrels[qid]]
    for did in random.sample(neg_pool, k=neg_k):
        print("[NEG]", wrap(docs[did][:700]), "\n")

# Test the helper
show_random_query()

# ✅ Qdrant setup
model_name = 'Snowflake/snowflake-arctic-embed-l-v2.0'

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)

embedder = SentenceTransformer(model_name, device=device)

client = QdrantClient(host="185.255.91.144", port=30081, timeout=60)
COLL = "scifact_fa_snow"
dim = embedder.get_sentence_embedding_dimension()

client.recreate_collection(
    collection_name=COLL,
    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
)

# ✅ Embed & upload docs in batches
BATCH = 256
ids, vecs, payloads = [], [], []
for i, (doc_id, text) in enumerate(tqdm.tqdm(docs.items(), desc="Embedding")):
    vec = embedder.encode(text, normalize_embeddings=True)
    payload = {
        "chunk_id": doc_id,
        "base_doc_id": doc_id.split('#')[0]
    }
    ids.append(i)
    vecs.append(vec)
    payloads.append(payload)

    if len(ids) == BATCH:
        client.upsert(collection_name=COLL, points=models.Batch(ids=ids, vectors=vecs, payloads=payloads))
        ids, vecs, payloads = [], [], []
# final flush
if ids:
    client.upsert(collection_name=COLL, points=models.Batch(ids=ids, vectors=vecs, payloads=payloads))

# ✅ Retrieval & Evaluation
TOP_K = 5
run = {}
for qid, qtext in tqdm.tqdm(queries.items(), desc="Retrieving"):
    qvec = embedder.encode(qtext, normalize_embeddings=True,prompt_name="query")
    res = client.search(collection_name=COLL, query_vector=qvec, limit=TOP_K, with_vectors=False)
    run[qid] = {point.payload["chunk_id"]: 1 - point.score for point in res}  # higher better

# Prepare ground truth
gt = {qid: rels for qid, rels in qrels.items()}
metrics = {f'recall_{TOP_K}', f'P_{TOP_K}', f'ndcg_cut_{TOP_K}'}
evaluator = pytrec_eval.RelevanceEvaluator(gt, metrics)
scores = evaluator.evaluate(run)

R = np.mean([s[f"recall_{TOP_K}"] for s in scores.values()])
P = np.mean([s[f"P_{TOP_K}"] for s in scores.values()])
nDCG = np.mean([s[f"ndcg_cut_{TOP_K}"] for s in scores.values()])

print(f"Recall@{TOP_K}:{R:.3f}  Precision@{TOP_K}:{P:.3f}  nDCG@{TOP_K}:{nDCG:.3f}")
