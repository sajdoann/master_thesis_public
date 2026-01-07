#!/bin/bash
# ================================================================
# Run embedders_experimetns_try.py for all defined embedders
# ================================================================
# run from: (thesis_venv) a@onyx:~/codes/master-thesis$ bash embedders_experiments/run_retrieval.sh
#
# COST CONTROL FOR OPENAI:
#   - Cost estimation runs automatically before OpenAI API calls
#   - Default budget limit: $10 USD (set OPENAI_BUDGET_LIMIT env var to override)
#   - Example: export OPENAI_BUDGET_LIMIT=5.0  # Set $5 limit
#   - Script will abort if estimated cost exceeds budget limit

# Activate your virtual environment (adjust path if needed)
source /home/a/.virtualenvs/thesis_venv/bin/activate

# Base paths
OUTPUT_DIR="embedders_experiments/results"
DATASET="wikipedia_nlp_small" # or choose other dataset
QUERIES="queries"

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# List of all embedders to test
EMBEDDERS=(
  "minilm"
 #"e5"
 #"fernet"
 #"mpnet"
 #"paraphrase"
  #"qwen3"
)

# Loop through each embedder
for EMBEDDER in "${EMBEDDERS[@]}"; do
  echo "===================================================================="
  echo " Running experiment for embedder: $EMBEDDER"
  echo "===================================================================="

  # Output filename for this embedder
  OUTFILE="$OUTPUT_DIR/results_$DATASET.csv" # optionally split by embedder "$OUTPUT_DIR/results_${EMBEDDER}.csv"

  # Run experiment (local source for qwen3, auto for others)
  if [ "$EMBEDDER" == "qwen3" ]; then
    SOURCE="local"
  else
    SOURCE="auto"
  fi

  mkdir -p "$OUTPUT_DIR/logs"

  python -m embedders_experiments.cli run \
      --embedder "$EMBEDDER" \
      --source "$SOURCE" \
      --dataset "embedders_experiments/datasets/$DATASET" \
      --queries "embedders_experiments/datasets/$DATASET/$QUERIES.jsonl" \
      --database "./chroma_data" \
      --chunk-size 1024 \
      --overlap 0 \
      --first_chunk_only 1 \
      --db-type "qdrant"
      --out "$OUTFILE" 2>&1 | tee "$OUTPUT_DIR/logs/log_$DATASET_$EMBEDDER.log" \


  # Check exit code
  if [ $? -eq 0 ]; then
    echo "✅ Finished: $EMBEDDER"
  else
    echo "❌ Failed: $EMBEDDER"
  fi
  echo
done

echo "🏁 All experiments completed. Results are in $OUTFILE"
