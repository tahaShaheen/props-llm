#!/bin/bash
#SBATCH --job-name=vllm_array
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=1-50%20
#SBATCH --time=0-03:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -o /dev/null
#SBATCH -e /dev/null
#SBATCH --exclude=sg025
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=tashahee@asu.edu

set -euo pipefail

JOB_LOG_DIR="slurm_logs/${SLURM_ARRAY_JOB_ID:-manual_$(date +%s)}"
mkdir -p "$JOB_LOG_DIR"
OUT_LOG_FILE="$JOB_LOG_DIR/llm_output_array_${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}.out"
ERR_LOG_FILE="$JOB_LOG_DIR/llm_error_array_${SLURM_ARRAY_JOB_ID:-na}_${SLURM_ARRAY_TASK_ID:-na}.err"
exec >"$OUT_LOG_FILE" 2>"$ERR_LOG_FILE"

module purge
module load mamba
source activate irl

REPETITION_ID=""
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    REPETITION_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "Resolved repetition id: ${REPETITION_ID} (folder suffix for repetition_*)"
fi

# 1. Generate a collision-proof port
# Pick a random port between 10000 and 65000 to avoid TOCTOU collisions 
# if multiple array tasks land on the exact same physical node.
PORT=$(shuf -i 10000-65000 -n 1)

# Ensure the randomly selected port isn't currently in use on this specific node
while ss -tuln | grep -q ":$PORT " ; do
    PORT=$(shuf -i 10000-65000 -n 1)
done

echo "Assigned randomized free port: $PORT"

# 2. Cleanup function to kill vLLM server on exit
cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
    fi
}
trap cleanup EXIT

# 3. Start vLLM server with a Retry Loop for transient OS errors
MODEL_PATH="/data/datasets/community/huggingface/models--openai--gpt-oss-120b/snapshots/8b193b0ef83bd41b40eb71fee8f1432315e02a3e"
VLLM_PID=""
MAX_START_RETRIES=5
START_ATTEMPT=1

while [ $START_ATTEMPT -le $MAX_START_RETRIES ]; do
    echo "Starting vLLM via Apptainer (Attempt $START_ATTEMPT of $MAX_START_RETRIES)..."

    # --nv passes the A100 GPUs to the container
    # --bind /data:/data allows the container to read the model weights
    apptainer run --nv --bind /data:/data /home/$USER/vllm-latest.sif \
        --model "$MODEL_PATH" \
        --served-model-name "gpt-oss:120b" \
        --tensor-parallel-size 2 \
        --port $PORT &

    # python -m vllm.entrypoints.openai.api_server \
        # --model "$MODEL_PATH" \
        # --served-model-name "gpt-oss:120b" \
        # --tensor-parallel-size 2 \
        # --max-model-len 8192 \
        # --port $PORT &

    VLLM_PID=$!

    # Wait 5 seconds to see if Apptainer crashes instantly (e.g., LDAP/UID errors)
    sleep 5
    
    if kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Apptainer successfully booted and stabilized. PID: $VLLM_PID"
        break # Exit the retry loop, it's alive!
    else
        echo "Warning: Apptainer crashed immediately. Waiting 15 seconds before retrying..."
        sleep 15
        START_ATTEMPT=$((START_ATTEMPT + 1))
    fi
done

# If we exhausted all retries and it's still dead, kill the SLURM job
if [ $START_ATTEMPT -gt $MAX_START_RETRIES ]; then
    echo "CRITICAL ERROR: Apptainer failed to boot after $MAX_START_RETRIES attempts. Hardware or OS issue suspected." >&2
    exit 1
fi

# 4. Wait for vLLM to load the 120B model into VRAM
echo "Loading 120B model into VRAM. Waiting for vLLM server to be ready..."
MAX_URL_RETRIES=60
COUNTER=0

while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)" != "200" ]; do
    # Fail-fast check during the long weight-loading process (e.g., if it runs Out of Memory)
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "CRITICAL ERROR: The vLLM process crashed while loading model weights! Exiting immediately." >&2
        exit 1
    fi

    sleep 10
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -ge $MAX_URL_RETRIES ]; then
        echo "Error: vLLM server failed to serve model after 10 minutes." >&2
        exit 1
    fi
done
echo "vLLM server is up and running on port $PORT!"

# ==========================================
# 5. EXECUTE THE ACTUAL TRAINING LOOP
# ==========================================
echo "Navigating to project directory..."
cd /home/$USER/taha/props-llm

echo "Launching main.py..."
python main.py \
    --config configs/walker2d/walker2d_propsp.yaml \
    --repetition_id "${REPETITION_ID:-$SLURM_ARRAY_TASK_ID}" \
    --port "$PORT"