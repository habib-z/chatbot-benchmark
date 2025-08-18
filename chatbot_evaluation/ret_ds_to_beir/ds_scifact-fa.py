# pip install datasets tqdm --upgrade
from datasets import load_dataset
import random, textwrap, statistics, collections

WRAP = lambda s: textwrap.fill(s, width=88)

# ─────────────────────────────────────────────────────────────── 1. LOAD
task = "mteb/SciFact-Fa"

corpus  = load_dataset(task, "corpus",  split="test")   # ≈ 5 000 docs
queries = load_dataset(task, "queries", split="test")   #   1 409 queries
qrels   = load_dataset(task, "qrels",   split="test")   #   2 680 judgments

print(f"\n◎ Sizes — corpus:{len(corpus):,}  queries:{len(queries):,}  qrels:{len(qrels):,}")

# ─────────────────────────────────────────────────────────────── 2. MAPS
corpus_map = {r["_id"]: r["text"] for r in corpus}
qrels_map  = {}
for r in qrels:
    qrels_map.setdefault(r["query-id"], {})[r["corpus-id"]] = r["score"]

# stats on positives
pos_counts = [len(ps) for ps in qrels_map.values()]
print(f"◎ Avg positives/query: {statistics.mean(pos_counts):.2f} "
      f"(min={min(pos_counts)}, max={max(pos_counts)})")
print("◎ Positives‑per‑query histogram:", collections.Counter(pos_counts))

# ─────────────────────────────────────────────────────────────── 3. PRINTER
def show_query(qid, neg_k=2):
    qtext = next(q["text"] for q in queries if q["_id"] == qid)
    pos   = qrels_map[qid]

    print("\n" + "="*100)
    print(f"Query {qid}:\n" + WRAP(qtext) + "\n")
    print(f"► {len(pos)} relevant passage(s)")

    for did, score in pos.items():
        snippet = corpus_map[did][:700]
        print(f"\n  [REL score={score}] doc {did}")
        print("  " + WRAP(snippet) + (" …" if len(snippet) == 700 else ""))

    neg_ids = random.sample([d for d in corpus_map if d not in pos], k=neg_k)
    print(f"\n► {neg_k} irrelevant passages (contrast)")
    for did in neg_ids:
        snippet = corpus_map[did][:700]
        print(f"\n  [IRREL] doc {did}")
        print("  " + WRAP(snippet) + (" …" if len(snippet) == 700 else ""))

# ─────────────────────────────────────────────────────────────── 4. DISPLAY
multi = [qid for qid, ps in qrels_map.items() if len(ps) > 1]
if not multi:
    print("\nNo queries with multiple positives in this set!")
else:
    for qid in random.sample(multi, k=min(2, len(multi))):
        show_query(qid)
