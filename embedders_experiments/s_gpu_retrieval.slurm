#!/bin/bash
#SBATCH --job-name=embedder_run
#SBATCH --partition=gpu-ms
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a40:1

###############################
# 🔧 PARAMS
###############################
DATASET=${1:-test_dareczech}            # default dataset
MODEL_NAME=${2:-qwen3-4b}                # default model

echo "📘 Dataset: $DATASET"
echo "🧠 Model:   $MODEL_NAME"

###############################
# 📂 DYNAMIC OUTPUT NAME
###############################
JOB_ID="${SLURM_JOB_ID:-$$}"   # use slurm ID if available, else unix PID
OUTNAME="gpu_${DATASET}_${MODEL_NAME}_${JOB_ID}.out"


# redirect all stdout/stderr into the dynamic file
exec > >(tee "$OUTNAME") 2>&1

echo "📝 Output will be saved to: $OUTNAME"

###############################
# 📁 WORKDIR & ENV
###############################
WORKDIR=/lnet/work/people/sajdokova
cd $WORKDIR || exit 1

source master-thesis-env/bin/activate


pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

export HUGGINGFACE_HUB_CACHE="$WORKDIR/models"
export TRANSFORMERS_CACHE="$WORKDIR/models"
export HF_HOME="$WORKDIR/models"

   # Load Hugging Face token if available
if [ -f ~/.hf_token ]; then
	export HUGGINGFACE_HUB_TOKEN=$(cat ~/.hf_token)
elif [ -f ~/.cache/huggingface/hub/.token ]; then
	export HUGGINGFACE_HUB_TOKEN=$(cat ~/.cache/huggingface/hub/.token)
else
	echo "⚠️ Warning: Hugging Face token not found. Gated models may fail."
fi
huggingface-cli login --token $HUGGINGFACE_HUB_TOKEN


EXP_DIR=~/personal_work_troja/master-thesis/
cd $EXP_DIR || exit 1

OUTPUT_DIR="embedders_experiments/results"
QUERIES="queries"

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"

OUTFILE="$OUTPUT_DIR/results_${DATASET}.csv"
LOGFILE="$OUTPUT_DIR/logs/log_${DATASET}_${MODEL_NAME}.log"

###############################
# 💾 CHROMADB TEMP DIR
###############################
JOB_ID="${SLURM_JOB_ID:-$$}"

if [ -d "/tmp" ] && [ -w "/tmp" ]; then
    CHROMA_DB="/tmp/chroma_${JOB_ID}_${MODEL_NAME}"
else
    CHROMA_DB="$WORKDIR/chroma_${JOB_ID}_${MODEL_NAME}"
fi

mkdir -p "$CHROMA_DB"
echo "💽 ChromaDB path: $CHROMA_DB"

###############################
# 🚀 RUNNING EXPERIMENT
###############################
echo "⚙️ Running model $MODEL_NAME on dataset $DATASET ..."

python -m embedders_experiments.cli run \
    --embedder "$MODEL_NAME" \
    --source "local" \
    --device "cuda" \
    --dataset "embedders_experiments/datasets/$DATASET" \
    --queries "embedders_experiments/datasets/$DATASET/$QUERIES.jsonl" \
    --database "$CHROMA_DB" \
    --chunk-size 1024 \
    --prompt-mode "custom" \
    --first_chunk_only 1 \
    --overlap 0 \
    --batch-size 100 \
    --out "$OUTFILE" 2>&1 | tee "$LOGFILE"

EXIT_CODE=$?

###############################
# 🧹 CLEANUP
###############################
if [ $EXIT_CODE -eq 0 ]; then
    echo "🧹 Cleaning up ChromaDB..."
    rm -rf "$CHROMA_DB" || true
else
    echo "⚠️ Run failed — keeping ChromaDB at: $CHROMA_DB"
fi

echo "🏁 Done."
echo "📄 Results: $OUTFILE"
echo "📝 Log:     $LOGFILE"

