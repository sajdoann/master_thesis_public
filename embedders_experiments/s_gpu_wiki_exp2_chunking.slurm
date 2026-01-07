#!/bin/bash
#SBATCH --job-name=embedder_run
#SBATCH --partition=gpu-ms
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8

###############################
# 🔧 PARAMS
###############################
DATASET=${1:-wikipedia_nlp}
MODEL_NAME=${2:-qwen3-4b}

echo "📘 Dataset: $DATASET"
echo "🧠 Model:   $MODEL_NAME"

###############################
# 📂 OUTPUT
###############################
JOB_ID="${SLURM_JOB_ID:-$$}"
OUTNAME="gpu_${DATASET}_${MODEL_NAME}_${JOB_ID}.out"
exec > >(tee "$OUTNAME") 2>&1

###############################
# 📁 ENV
###############################
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR || exit 1
source master-thesis-env/bin/activate

export HUGGINGFACE_HUB_CACHE="$WORKDIR/models"
export TRANSFORMERS_CACHE="$WORKDIR/models"
export HF_HOME="$WORKDIR/models"

if [ -f ~/.hf_token ]; then
    export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
    huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN
fi

EXP_DIR=~/personal_work_troja/master-thesis
cd $EXP_DIR || exit 1

OUTPUT_DIR="embedders_experiments/results"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

OUTFILE="$OUTPUT_DIR/results_chunking_${DATASET}.csv"

###############################
# 🧪 GRID DEFINITION
###############################
CHUNK_SIZES=(256 512 1024)
OVERLAPS_256=(0 25 50)
OVERLAPS_512=(0 50 100)
OVERLAPS_1024=(0 100 200)

###############################
# 🚀 GRID RUNS
###############################
for CHUNK in "${CHUNK_SIZES[@]}"; do
    case $CHUNK in
        256) OVERLAPS=("${OVERLAPS_256[@]}") ;;
        512) OVERLAPS=("${OVERLAPS_512[@]}") ;;
        1024) OVERLAPS=("${OVERLAPS_1024[@]}") ;;
    esac

    for OVERLAP in "${OVERLAPS[@]}"; do
        RUN_ID="${JOB_ID}_${CHUNK}_${OVERLAP}"
        CHROMA_DB="/tmp/chroma_${RUN_ID}_${MODEL_NAME}"
        LOGFILE="$LOG_DIR/log_${DATASET}_${MODEL_NAME}_${CHUNK}_${OVERLAP}.log"

        echo "▶ chunk=$CHUNK overlap=$OVERLAP first_chunk_only=0"

        python -m embedders_experiments.cli run \
            --embedder "$MODEL_NAME" \
            --source "local" \
            --device "cuda" \
            --dataset "embedders_experiments/datasets/$DATASET" \
            --queries "embedders_experiments/datasets/$DATASET/queries.jsonl" \
            --database "$CHROMA_DB" \
            --chunk-size "$CHUNK" \
            --overlap "$OVERLAP" \
            --first_chunk_only 0 \
            --prompt-mode "custom" \
            --batch-size 100 \
            --out "$OUTFILE" 2>&1 | tee "$LOGFILE"

        rm -rf "$CHROMA_DB"
    done
done

###############################
# 🧪 BASELINE RUN
###############################
echo "▶ BASELINE: chunk=1024 overlap=0 first_chunk_only=1"

BASE_CHROMA="/tmp/chroma_${JOB_ID}_baseline_${MODEL_NAME}"
BASE_LOG="$LOG_DIR/log_${DATASET}_${MODEL_NAME}_baseline.log"

python -m embedders_experiments.cli run \
    --embedder "$MODEL_NAME" \
    --source "local" \
    --device "cuda" \
    --dataset "embedders_experiments/datasets/$DATASET" \
    --queries "embedders_experiments/datasets/$DATASET/queries.jsonl" \
    --database "$BASE_CHROMA" \
    --chunk-size 1024 \
    --overlap 0 \
    --first_chunk_only 1 \
    --prompt-mode "custom" \
    --batch-size 100 \
    --out "$OUTFILE" 2>&1 | tee "$BASE_LOG"

rm -rf "$BASE_CHROMA"

echo "🏁 All runs finished"
echo "📄 Results: $OUTFILE"
