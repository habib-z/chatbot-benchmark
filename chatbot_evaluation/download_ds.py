import ir_datasets, collections, textwrap, statistics, random
wrap = lambda s: textwrap.fill(s, 88)

ds = ir_datasets.load("miracl/fa/dev")              # 632 queries / 210 k docs
docs = {d.doc_id: d.text for d in ds.docs_iter()}

queries = {q.query_id: q.text for q in ds.queries_iter()}
qrels   = collections.defaultdict(dict)             # qid → {doc: rel}
for qr in ds.qrels_iter():
    if qr.relevance > 0:                            # keep only positives
        qrels[qr.query_id][qr.doc_id] = 1           # binary

print(f"docs:{len(docs):,}  queries:{len(queries):,}  positives avg:"
      f"{statistics.mean(len(r) for r in qrels.values()):.2f}")




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