#!/bin/bash
#SBATCH --job-name=embed_cpu_batch
#SBATCH --time=60:00:00
#SBATCH --output=cpu_%j.out
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu-ms

echo " Starting CPU embedder batch job"

# ========== WORKDIR & ENV ==========
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR

source master-thesis-env/bin/activate

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

# ========== EMBEDDERS ==========
EMBEDDERS=(
  "minilm"
  "e5"
  "fernet"
  "mpnet"
  "paraphrase"
)

# ========== RUN LOOP ==========
for EMBEDDER in "${EMBEDDERS[@]}"; do
  echo "===================================================================="
  echo " Running $EMBEDDER on CPU dataset $DATASET"
  echo "===================================================================="

  if [ "$EMBEDDER" == "qwen3" ]; then
    SOURCE="local"
  else
    SOURCE="auto"
  fi

  OUTFILE="$OUTPUT_DIR/results_$DATASET.csv"
  LOGFILE="$OUTPUT_DIR/logs/log_${DATASET}_${EMBEDDER}.log"


  python -m embedders_experiments.cli run \
      --embedder "$EMBEDDER" \
      --source "$SOURCE" \
      --device "cpu" \
      --dataset "embedders_experiments/datasets/$DATASET" \
      --queries "embedders_experiments/datasets/$DATASET/$QUERIES.jsonl" \
      --database "$WORKDIR/chroma_cpu_lrc" \
      --chunk-size 1024 \
      --first_chunk_only 1 \
      --overlap 0 \
      --out "$OUTFILE" 2>&1 | tee "$LOGFILE"

  if [ $? -eq 0 ]; then
    echo "✅ Finished: $EMBEDDER"
  else
    echo "❌ Failed: $EMBEDDER"
  fi
done

echo "🏁 All CPU experiments done. Results stored in $OUTPUT_DIR"
