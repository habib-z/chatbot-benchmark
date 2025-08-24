# pip install datasets
from datasets import load_dataset
import random, textwrap
wrap = lambda s: textwrap.fill(s, 88)

# load the three builder‑configs (split = "test")
corpus  = load_dataset("mteb/Touche2020-Fa", "corpus",  split="test")
queries = load_dataset("mteb/Touche2020-Fa", "queries", split="test")
qrels   = load_dataset("mteb/Touche2020-Fa", "qrels",   split="test")

# build helpers
docs = {d["_id"]: d["text"] for d in corpus}
rels = {}
for r in qrels:
    rels.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]

# pick one query that has ≥2 different scores (0/1/2/3)
qid = next(q for q,r in rels.items() if len(set(r.values()))>1)

print("Query:", wrap(next(q["text"] for q in queries if q["_id"]==qid)), "\n")
for score in sorted(set(rels[qid].values()), reverse=True):
    print(f"--- passages with score{score} ---")
    for did in [d for d,s in rels[qid].items() if s==score][:2]:
        print(wrap(docs[did][:500]), "\n")



