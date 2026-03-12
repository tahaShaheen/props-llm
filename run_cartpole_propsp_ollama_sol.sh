#!/bin/bash
#SBATCH -p public
#SBATCH -q public
#SBATCH --gres=gpu:a100:2
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 0-08:00
#SBATCH -J props_cartpole_ollama
#SBATCH -o llm_output_%j.out
#SBATCH -e llm_error_%j.err

set -euo pipefail

module purge
module load mamba
module load ollama/0.15.6

# Activate your environment
source activate irl

# POINT OLLAMA TO YOUR PERSONAL SCRATCH SPACE SO IT CAN DOWNLOAD
export OLLAMA_MODELS=/scratch/$USER/taha/ollama_cache
export OLLAMA_HOST=127.0.0.1:11434

cleanup() {
  if [[ -n "${OLLAMA_PID:-}" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" || true
  fi
}
trap cleanup EXIT

# Start Ollama server
ollama serve > ollama_server_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!

# Wait for server to boot up
for _ in {1..60}; do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null; then
  echo "Ollama server failed to start on ${OLLAMA_HOST}" >&2
  exit 1
fi

# DOWNLOAD THE 120B MODEL BEFORE RUNNING PYTHON
echo "Pulling gpt-oss:120b... This may take several minutes."
ollama pull gpt-oss:120b
echo "Download complete!"

cd /home/$USER/taha/props-llm

python main.py --config configs/cartpole/cartpole_propsp.yaml
