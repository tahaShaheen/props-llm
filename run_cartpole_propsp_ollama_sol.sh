#!/bin/bash
#SBATCH --job-name=vllm_array
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --array=1-50%3
#SBATCH --time=0-08:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:2
#SBATCH --output=logs/array_%A_%a.out

set -euo pipefail

module load mamba
source activate props_env

MODEL_PATH="/data/datasets/community/models/gpt-oss-120b"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --port 8000 &

echo "Loading 120B model into VRAM. Waiting for vLLM server to boot..."
while ! curl -s http://localhost:8000/v1/models > /dev/null; do
    sleep 10
done
echo "vLLM server is up and running!"

cd /scratch/$USER/your_project_folder

python main.py --config configs/cartpole/cartpole_propsp.yaml --run_id "$SLURM_ARRAY_TASK_ID"
