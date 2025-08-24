from datasets import load_dataset
from collections import Counter
import numpy as np

def positives_stats(dataset_slug):
    qrels = load_dataset(dataset_slug, "qrels", split="test")
    bag   = {}
    for r in qrels:
        bag.setdefault(r["query-id"], set()).add(r["corpus-id"])
    counts = [len(s) for s in bag.values()]
    return dict(Counter(counts)), np.mean(counts)

print(positives_stats("mteb/CQADupstackMathematicaRetrieval-Fa"))
# e.g. {1: 483, 2: 271, 3: 38, 4: 12}  mean ≈ 1.63
