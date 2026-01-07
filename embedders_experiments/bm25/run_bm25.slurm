#!/bin/bash
#SBATCH --job-name=bm25_retrieval
#SBATCH --time=52:00:00
#SBATCH --output=bm25_%j.out
#SBATCH --mem=60G
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu-ms

echo " Starting BM25 retrieval job"

# ================== ARGUMENT CHECK ==================
# Allow overriding defaults via environment variables or command line args
MODE=${1:-udpipe}  # default to udpipe if not provided
K1=${2:-2}       # default BM25 k1 parameter
B=${3:-1}       # default BM25 b parameter
DATASET=${4:-test_dareczech}  # default dataset

echo "Configuration:"
echo "   Mode: $MODE"
echo "   BM25 k1: $K1"
echo "   BM25 b: $B"
echo "   Dataset: $DATASET"

# ================== WORKDIR & ENV ==================
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR || exit 1

# Activate environment
source master-thesis-env/bin/activate
pip install PyStemmer

# Cache setup (local only)
export HUGGINGFACE_HUB_CACHE="$WORKDIR/models"
export TRANSFORMERS_CACHE="$WORKDIR/models"
export HF_HOME="$WORKDIR/models"

# ================== PATHS ==================
EXP_DIR=~/personal_work_troja/master-thesis/
cd $EXP_DIR || exit 1

OUTPUT_DIR="embedders_experiments/results"
QUERIES="queries"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Output + log files
OUTFILE="$OUTPUT_DIR/results_${DATASET}_bm25_${MODE}_k1${K1}_b${B}.csv"
LOGFILE="$OUTPUT_DIR/logs/log_${DATASET}_bm25_${MODE}_k1${K1}_b${B}.log"

# ================== RUN BM25 ==================
echo "  Running BM25 retrieval..."
echo "   Dataset: embedders_experiments/datasets/$DATASET"
echo "   Queries: embedders_experiments/datasets/$DATASET/$QUERIES.jsonl"
echo "   Output: $OUTFILE"
echo "   Log: $LOGFILE"

python -m embedders_experiments.bm25.run_bm25 \
    --dataset "embedders_experiments/datasets/$DATASET" \
    --queries "embedders_experiments/datasets/$DATASET/$QUERIES.jsonl" \
    --out "$OUTFILE" \
    --mode "$MODE" \
    --k1 "$K1" \
    --b "$B" 2>&1 | tee "$LOGFILE"

EXIT_CODE=$?

# ================== CHECK STATUS ==================
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ BM25 finished successfully"
  echo "📊 Results saved to: $OUTFILE"
else
  echo "❌ BM25 run FAILED"
  echo "📋 Check log file: $LOGFILE"
fi

echo "Done. Results saved to $OUTFILE"

