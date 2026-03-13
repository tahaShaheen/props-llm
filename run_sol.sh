#!/bin/bash
#SBATCH --job-name=vllm_array
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=1-2%2
#SBATCH --time=0-08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH -o llm_output_array_%A_%a.out
#SBATCH -e llm_error_array_%A_%a.err

set -euo pipefail

module purge
module load mamba
source activate irl

# 1. Generate a collision-proof port
# Pick a random port between 10000 and 65000 to avoid TOCTOU collisions 
# if multiple array tasks land on the exact same physical node.
PORT=$(shuf -i 10000-65000 -n 1)

# Ensure the randomly selected port isn't currently in use on this specific node
while ss -tuln | grep -q ":$PORT " ; do
    PORT=$(shuf -i 10000-65000 -n 1)
done

echo "Assigned randomized free port: $PORT"

# 2. Start vLLM server in the background
# WARNING: Ensure this MODEL_PATH points to Hugging Face weights!
MODEL_PATH="/data/datasets/community/models/gpt-oss-120b"
VLLM_PID=""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "gpt-oss:120b" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --port $PORT &

VLLM_PID=$!
echo "vLLM server started on port $PORT with PID: $VLLM_PID"

# 3. Cleanup function to kill vLLM server on exit
cleanup() {
    echo "Cleaning up vLLM server (PID: $VLLM_PID)..."
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
    fi
}
# Register cleanup immediately
trap cleanup EXIT

# 4. Wait for vLLM to load the 120B model into VRAM
echo "Loading 120B model into VRAM. Waiting for vLLM server to boot..."
MAX_RETRIES=60
COUNTER=0

# Use a strict HTTP 200 OK check. This prevents the script from advancing 
# if vLLM is awake but returning 503/404 errors while still warming up.
while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${PORT}/v1/models)" != "200" ]; do
    sleep 10
    COUNTER=$((COUNTER + 1))
    if [ $COUNTER -ge $MAX_RETRIES ]; then
        echo "Error: vLLM server failed to start after 10 minutes." >&2
        exit 1
    fi
done
echo "vLLM server is up and running on port $PORT!"

# 5. Navigate to project directory
cd /home/$USER/taha/props-llm

# 6. Run training, passing both the run_id and the dynamic port
python main.py --config configs/cartpole/cartpole_propsp.yaml --run_id "$SLURM_ARRAY_TASK_ID" --port "$PORT"