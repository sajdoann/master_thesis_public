#!/bin/bash
#SBATCH --job-name=openai_embed
#SBATCH --time=30:00:00
#SBATCH --output=openai_%j.out
#SBATCH --mem=102G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu-ms

echo "Starting OpenAI embedder batch job"
echo "SLURM job ID: $SLURM_JOB_ID"

# ========== WORKDIR & ENV ==========
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR

source master-thesis-env/bin/activate
pip install openai

# HuggingFace cache local only
export HUGGINGFACE_HUB_CACHE="$WORKDIR/models"
export TRANSFORMERS_CACHE="$WORKDIR/models"
export HF_HOME="$WORKDIR/models"

# ========== PATHS ==========
EXP_DIR=~/personal_work_troja/master-thesis/
cd $EXP_DIR

OUTPUT_DIR="embedders_experiments/results"
DATASET="test_dareczech"
QUERIES="queries"


mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# ================================================================
# Cost control for OpenAI API
# ================================================================
export OPENAI_BUDGET_LIMIT=2.0   # default = 2 USD (override if needed)

# ================================================================
# Embedders to test
# ================================================================
EMBEDDERS=(
  "text-embedding-3-small"
  # add more if needed
)

# ================================================================
# Run loop
# ================================================================
for EMBEDDER in "${EMBEDDERS[@]}"; do
  echo "===================================================================="
  echo " Running experiment for embedder: $EMBEDDER"
  echo "===================================================================="

  # If using qwen locally
  if [ "$EMBEDDER" == "qwen3" ]; then
    SOURCE="local"
  else
    SOURCE="auto"
  fi

  OUTFILE="$OUTPUT_DIR/results_${DATASET}.csv"
  LOGFILE="$OUTPUT_DIR/logs/log_${DATASET}_${EMBEDDER}.log"

  python -m embedders_experiments.cli run \
      --embedder "$EMBEDDER" \
      --source "$SOURCE" \
      --dataset "embedders_experiments/datasets/$DATASET" \
      --queries "embedders_experiments/datasets/$DATASET/$QUERIES.jsonl" \
      --database "./chroma_data" \
      --chunk-size 1024 \
      --overlap 0 \
      --first_chunk_only 1 \
      --out "$OUTFILE" 2>&1 | tee "$LOGFILE"

  if [ $? -eq 0 ]; then
    echo "✅ Finished: $EMBEDDER"
  else
    echo "❌ Failed: $EMBEDDER"
  fi

  echo
done

echo "🏁 All OpenAI experiments completed. Results saved to $OUTPUT_DIR"
