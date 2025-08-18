# First time: pip install -U datasets tqdm
from datasets import load_dataset
import random, textwrap, collections, statistics

wrap = lambda s: textwrap.fill(s, 88)

# ── 1. load the three builder‑configs — only "test" split exists
task = "mteb/PersianWebDocumentRetrieval"
corpus  = load_dataset(task, "corpus",  split="test")    # 167 k passages
queries = load_dataset(task, "queries", split="test")    #   951 queries
qrels   = load_dataset(task, "qrels",   split="test")    # 18 308 judgements

print(f"◎ docs:{len(corpus):,}  queries:{len(queries):,}  qrels:{len(qrels):,}")

# ── 2. maps for quick access
docs = {d["_id"]: d["text"] for d in corpus}
queries_map = {q["_id"]: q["text"] for q in queries}

q2rels = collections.defaultdict(list)      # query → [doc_id …]
for row in qrels:
    if row["score"] > 0:                    # keep positives only
        q2rels[row["query-id"]].append(row["corpus-id"])

pos_counts = [len(v) for v in q2rels.values()]
print("◎ avg positives/q:", f"{statistics.mean(pos_counts):.2f}",
      "| min/max:", min(pos_counts), max(pos_counts))

# ── 3. pretty‑print two queries (show 3 positives + 2 negatives)
def show(qid, neg_k=2):
    print("\n" + "="*100)
    print("Query:", wrap(queries_map[qid]), "\n")

    for did in q2rels[qid][:3]:
        print(f"[POS] doc {did}\n" + wrap(docs[did][:700]) +
              (" …" if len(docs[did]) > 700 else ""), "\n")

    neg_pool = [d for d in docs if d not in q2rels[qid]]
    for did in random.sample(neg_pool, k=neg_k):
        print(f"[NEG] doc {did}\n" + wrap(docs[did][:700]) +
              (" …" if len(docs[did]) > 700 else ""), "\n")

# pick two queries with ≥3 positives so ordering matters
rich = [q for q, pos in q2rels.items() if len(pos) >= 3]
for q in random.sample(rich, k=2):
    show(q)
