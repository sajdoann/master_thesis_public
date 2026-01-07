import json
import random

DOCS_FILE = "data/docs_all.jsonl"
QUERIES_FILE = "data/queries_all.jsonl"


NUM_DOCS = 100000

OUT_DOCS = f"data/n{NUM_DOCS}_docs.jsonl"
OUT_QUERIES = f"data/n{NUM_DOCS}_queries.jsonl"

# ---- Load docs ----
docs = []
with open(DOCS_FILE, encoding="utf-8") as f:
    for line in f:
        docs.append(json.loads(line))

print(f"Loaded {len(docs)} docs")

# ---- Sample N docs ----
sampled_docs = random.sample(docs, NUM_DOCS)

# Get selected docIDs
small_doc_ids = set(d["docID"] for d in sampled_docs)

print(f"Selected {len(small_doc_ids)} docs")

# ---- Load queries + filter ----
small_queries = []
with open(QUERIES_FILE, encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)

        # Check if query touches any of the sampled docs
        keep_indices = [i for i, did in enumerate(q["docID"]) if did in small_doc_ids]
        if not keep_indices:
            continue

        # Filter to only selected docs
        new_q = {
            "queryID": q["queryID"],
            "query": q["query"],
            "docID": [q["docID"][i] for i in keep_indices],
            "source": [q["source"][i] for i in keep_indices],
            "label": [q["label"][i] for i in keep_indices],
        }

        small_queries.append(new_q)

print(f"Kept {len(small_queries)} queries that reference the sampled docs")

# ---- Save filtered docs ----
with open(OUT_DOCS, "w", encoding="utf-8") as f:
    for d in sampled_docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

# ---- Save filtered queries ----
with open(OUT_QUERIES, "w", encoding="utf-8") as f:
    for q in small_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

print(f"Wrote {OUT_DOCS} and {OUT_QUERIES}")