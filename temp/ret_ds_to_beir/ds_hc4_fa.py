# pip install ir_datasets
import ir_datasets, random, textwrap
wrap = lambda s: textwrap.fill(s, 88)

ds   = ir_datasets.load("hc4/fa/dev")     # dev split = 60 Persian news queries
qid  = random.choice(list(ds.queries_iter())).query_id
query = ds.queries[qid]

print("Query‑EN:", query.title)
print("Query‑FA:", query.mt_title, "\n")

# positive docs: score 2 > 1 > 0
for doc_id, rel in sorted(ds.qrels_iter(), key=lambda x: -x[2]):
    if doc_id.startswith(qid):  # qrels_iter yields all topics; filter current
        doc = ds.docs[doc_id]
        print(f"score {rel} →", wrap(doc.text[:500]), "\n")
        if rel == 0: break       # show one negative
