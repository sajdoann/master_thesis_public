import json
import random
from pathlib import Path
from tqdm import tqdm

# Reproducibility
random.seed(42)

DOCS_ALL = "data/docs_all.jsonl"
QUERIES_TEST = "data/queries_test.jsonl"
DOCS_TEST = "data/docs_test.jsonl"
OUTPUT = "data/docs_test_100000.jsonl"
TARGET_SIZE = 100_000


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    print("Loading docs_all…")
    docs_all = load_jsonl(DOCS_ALL)
    print(f"   Loaded {len(docs_all):,} docs from docs_all")

    print("Loading docs_test…")
    docs_test = load_jsonl(DOCS_TEST)
    print(f"   Loaded {len(docs_test):,} docs in the existing test split")

    print("Loading queries_test…")
    queries_test = load_jsonl(QUERIES_TEST)

    # Collect all relevant docIDs
    print("Collecting docIDs that are relevant for queries_test…")
    relevant_docIDs = set()
    for q in queries_test:
        for did in q["docID"]:
            relevant_docIDs.add(did)
    print(f"   Found {len(relevant_docIDs):,} relevant docIDs (to exclude)")

    # Index docs_all by docID
    docs_all_by_id = {doc["docID"]: doc for doc in docs_all}

    # docIDs already in test set
    test_docIDs = {d["docID"] for d in docs_test}

    # Select all non-relevant, non-test docs
    print("Filtering candidate docs for filling…")
    candidate_docs = [
        d for d in docs_all
        if d["docID"] not in test_docIDs
        and d["docID"] not in relevant_docIDs
    ]
    print(f"   Candidates available: {len(candidate_docs):,}")

    needed = TARGET_SIZE - len(docs_test)
    if needed <= 0:
        print("docs_test already has ≥ 100,000 docs. Nothing to add.")
        return

    print(f"Need to add {needed:,} docs")

    if needed > len(candidate_docs):
        raise ValueError(
            f" Not enough non-relevant docs: need {needed}, but only {len(candidate_docs)} available."
        )

    #  Sample filler docs
    print("Sampling filler docs (seeded)…")
    filler = random.sample(candidate_docs, needed)

    # Combine
    print("Combining final dataset…")
    final_docs = docs_test + filler
    assert len(final_docs) == TARGET_SIZE

    # Save
    print(f"Saving {TARGET_SIZE:,} docs → {OUTPUT}")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for d in tqdm(final_docs, desc="Writing output"):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Done!")
    print(f"Final file: {OUTPUT}")


if __name__ == "__main__":
    main()
