import csv
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


mode = "all"


# --- INPUT FILES ---
files = [
        #"data/train_big.tsv",
        "data/train_small.tsv",
        "data/dev.tsv",
        "data/test.tsv"]

if mode == 'all':
    files = [
        "data/dev.tsv",
        "data/test.tsv",
        #"data/train_big.tsv",
        "data/train_small.tsv"
    ]
elif mode == 'test':
    files = [
        "data/test.tsv"
    ]
elif mode == 'devtest':
    files = [
        "data/dev.tsv",
        "data/test.tsv"
    ]


# --- OUTPUT FILES ---
DOCS_OUT = f"data/docs_{mode}.jsonl"
QUERIES_OUT = f"data/queries_{mode}.jsonl"

# Internal mappings
doc2id = {}
query2id = {}

next_docID = 0
next_queryID = 0

# queryID -> { 'query': str, 'docID':[], 'source':[], 'label':[] }
query_data = defaultdict(lambda: {"query": None, "docID": [], "source": [], "label": []})

docs_output = []

def clean(x):
    return x.strip() if isinstance(x, str) else ""

def count_lines(filepath):
    with open(filepath, "rb") as f:
        return sum(1 for _ in f)

for f in files:
    if not Path(f).exists():
        print(f"File not found: {f}")
        continue

    total = count_lines(f) - 1  # skip header
    print(f"Processing {f} ({total:,} lines)")

    with open(f, newline='', encoding='utf-8') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')

        for row in tqdm(reader, total=total, desc=f"Reading {f}", unit="rows"):
            query = clean(row.get("query"))
            url = clean(row.get("url"))
            title = clean(row.get("title"))
            doc = clean(row.get("doc"))
            label = float(row.get("label"))



            # ---- Build full text field for doc ----
            full_text = f"title: {title}\nurl: {url}\ntext: {doc}"

            # ---- Assign doc ID ----
            doc_key = url + "::" + title
            if doc_key not in doc2id:
                doc2id[doc_key] = next_docID
                docs_output.append({
                    "docID": next_docID,
                    "source": url,
                    "text": full_text
                })
                next_docID += 1

            did = doc2id[doc_key]

            # skip query making for non relevant
            if label <= 0:
                continue

            # ---- Assign query ID ----
            if query not in query2id:
                query2id[query] = next_queryID
                next_queryID += 1
            qid = query2id[query]

            # ---- Store query info ----
            query_data[qid]["query"] = query
            query_data[qid]["docID"].append(did)
            query_data[qid]["source"].append(url)
            query_data[qid]["label"].append(label)
    print(f"processed file {f}")

print("Writing output files...")

# --- WRITE DOCS ---
with open(DOCS_OUT, "w", encoding="utf-8") as f:
    for doc in tqdm(docs_output, desc="Writing docs", unit="docs"):
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

# --- WRITE QUERIES (only queries with relevant docs) ---
with open(QUERIES_OUT, "w", encoding="utf-8") as f:
    for qid, info in tqdm(query_data.items(), desc="Writing queries", unit="queries"):
        if len(info["docID"]) == 0:
            continue
        out = {
            "queryID": qid,
            "query": info["query"],
            "docID": info["docID"],
            "source": info["source"],
            "label": info["label"]
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Saved {len(docs_output):,} docs → {DOCS_OUT}")
print(f"Saved {len(query_data):,} queries (before filtering) → {QUERIES_OUT}")
