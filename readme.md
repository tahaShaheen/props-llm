# Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs


This repo serves as the code base for Prompted Policy Search (ProPS and ProPS<sup>+</sup>). The project website is [here](https://props-llm.github.io/).

<p align="center">
<img src = "static/banner.gif" width ="800" />
</p>

## Key Takeaways
In this paper, we demonstrate that:
1. LLMs can perform numerical optimization for Reinforcement Learning (RL) tasks.
2. LLMs can incorporate semantics signals, (e.g., goals, domain knowledge, ...), leading to more informed exploraton and sample-efficient learning.
3. Our proposed ProPS outperforms all baselines on 8 out of 15 Gymnasium tasks.


# Getting Started

## Install RL Tasks

- The RL tasks are based on gymnasium. Please install according to `https://github.com/Farama-Foundation/Gymnasium`
- There are 2 customized environments in the folders `./envs/gym-maze-master` and `./envs/gym-navigation-main`. If you want to train the maze or navigation agent, please pip install the packages.

## Install the LLM APIs

We utilized the standard Google Gemini, Openai, and Anthropic APIs. Please install the packages accordingly.

- `https://ai.google.dev/gemini-api/docs`
- `https://platform.openai.com/docs/overview`
- `https://docs.anthropic.com/en/release-notes/api`

## Start Training
In order to run an experiment, please run `python main.py --config <configuration_file>`.

## Recent Changes (February 2026 - Taha)

### Parameter Parsing is a bit more Robust
Removed explanation requirement in the j2 files. Changed the `num_optim_semantic` file to not ask for an explanation at the end. I'm using reasoning models and using the reasoning traces as explanations. I increased focus on the actual output format. 

### Local Ollama LLM Integration
- Added support for local Ollama models
- Usage: Set `llm_model_name: ollama:model_name` in your YAML config (e.g., `ollama:deepseek-r1:8b`)
- No additional code changes needed
- Important: the same YAML field is also used for local OpenAI-compatible servers such as vLLM when `main.py` is launched with `--port`. In that case, `OPENAI_BASE_URL` is set automatically and `gpt-oss` models are routed through the OpenAI-compatible client instead of the native Ollama API.

### Configurable Ollama Context Window
- Added `ollama_num_ctx` parameter (default: 4096) to configure token context limits
- Set in YAML: `ollama_num_ctx: 50000`
- Token counting via Ollama's tokenize API
- Displays token counts on terminal: `[TOKENS] ollama:model: X tokens` to diagnose Context Window Saturation/Forgetting thing-a-majig
- Note: `ollama_num_ctx` applies only to the native Ollama code path. It does not control context length when the model is served through vLLM/OpenAI-compatible mode.

### Context Guard System
- The guard logic is currently disabled in code.
- Token counting is still available for diagnostics, but prompts are no longer blocked for exceeding a computed context limit.

### Parameter Parsing is a bit more Robust
- Handles multiple outputs:
  - `params[0]: value`
  - `params[0] = value`
  - space/comma-separated numbers
- Removes `<think>` tags from reasoning models (DeepSeek-R1, Qwen)
- Strips code block markers (```), preserves full response in logs

### Visualization (Live)
- Real-time plots update after each episode
- Saved as `training_progress.png` in logdir

### Parallel Environment Visualizer (`eval_parallel_policy.py`)
Tool to visualize trained policies across multiple environment instances simultaneously in a grid.

- Loads trained parameters from `overall_log.csv` or other sepcified config file. 
- Runs 10 parallel environment instances (configurable)
- Shows per-environment rewards and mean reward across all instances
- Spacebar restart to re-run the same policy on new instances

**Usage:**
```bash
# Basic usage: 10 envs with config's max_traj_length
python eval_parallel_policy.py --episode 5 --config configs/cartpole/cartpole_propsp.yaml --render

# Custom number of environments
python eval_parallel_policy.py --episode 10 --config configs/mountaincar/mountaincar_props.yaml --render --num_envs 20

# Override max steps
python eval_parallel_policy.py --episode 5 --config configs/cartpole/cartpole_propsp.yaml --render --max_steps 1000

# Stats only (no visualization)
python eval_parallel_policy.py --episode 5 --config configs/cartpole/cartpole_propsp.yaml
```

- Uses Gymnasium's `AsyncVectorEnv` for parallelization
- Each environment runs independently, capturing RGB frames via a custom `RenderInInfoWrapper`
- Frames are collected asynchronously and synchronized for grid animation
- Reward tracking shows cumulative rewards updating in real-time
- Color-coded labels: GREEN for successful completion (≥max_steps-2 frames), RED for premature termination
- Matplotlib animation with configurable FPS (default: 10)
- Automatically pulls `max_traj_length` from YAML config if `--max_steps` not specified

- **SPACEBAR**: Restart simulation with the same parameters
- **Close window**: Exit the visualizer

### Human-in-the-Loop Feedback System (`propspf` task)
Interactive training mode where human observers provide qualitative feedback to guide LLM policy optimization.

1. LLM generates policy parameters and the policy executes
2. At configurable intervals, `eval_parallel_policy.py` launches automatically to visualize the best-performing policy
3. Human observer watches the policy execute and provides qualitative feedback
4. Feedback is incorporated into the LLM prompt for subsequent iterations
5. LLM uses both reward numbers AND human observations to improve the policy

**YAML Configuration:**
```yaml
task: cont_state_llm_num_optim_semantics_with_feedback  # or dist_state_llm_num_optim_semantics_with_feedback
feedback_interval: 5                    # Ask for feedback every N episodes
include_feedback_with_params: true      # Show feedback with params in LLM prompts
param_min: -6.0                        # Minimum valid parameter value (continuous tasks only)
param_max: 6.0                         # Maximum valid parameter value (continuous tasks only)
```

**Example Configs:**
- `configs/walker2d/walker2d_propspf.yaml` - Walker2D with human feedback (continuous)
- `configs/frozenlake/frozenlake_propspf.yaml` - FrozenLake with human feedback (discrete)

### Reduced Params Buffer
- Buffer is curated to show only the top-K and most recent-J parameter sets
- Prevents Repetitive Degeneration ("Context Loop") caused by large, repetitive buffer dumps, which is a problem for smaller LLMs (32b models)
- This means that the user prompt size is smaller and almost a fixed size
- Allows smaller context windows, making it easier to fit models + context + OS into memory

### Failure Recovery
- Increased retry attempts from 5 to 10 for robust training
- Warning messages in prompts on failed retries:
  - Shows attempt count (e.g., "attempt 3 out of 10")
  - Lists number of previous failures
  - Remaining attempts
  - Emphasizes format correctness and value ranges
- Logs full error details to terminal (doesn't save it to logs)

### Version Constraints
- `numpy<2.0` (gymnasium requires this; `np.bool8` was removed in 2.0+)
- `pandas<3.0` (compatibility with numpy 1.x)
- Added `requests` for Ollama API calls to do the context token counting

### Example Configuration
```yaml
task: cont_state_llm_num_optim_semantics
llm_model_name: ollama:deepseek-r1:8b
ollama_num_ctx: 50000  # Adjust based on your prompts and model
num_episodes: 100
num_evaluation_episodes: 20
bias: true
optimum: 500
search_step_size: 1.0
```

### Runs on Sol (Apptainer + vLLM)
Experiments can now be launched on Sol by starting a vLLM OpenAI-compatible server inside Apptainer and then pointing `main.py` to that local port.

**Current launch pattern (`run_sol.sh`):**
```bash
apptainer run --nv --bind /data:/data /home/$USER/vllm-latest.sif \
  --model "$MODEL_PATH" \
  --served-model-name "gpt-oss:120b" \
  --tensor-parallel-size 2 \
  --port $PORT &

python main.py \
  --config <your_config.yaml> \
  --repetition_id <job_or_array_id> \
  --port "$PORT"
```

The current `run_sol.sh` checked into the repo still points to `configs/cartpole/cartpole_propsp.yaml`; change the `--config` argument there to run a different experiment on Sol.

**What matters here:**
- `--model "$MODEL_PATH"` selects the actual checkpoint/weights loaded by vLLM.
- `--served-model-name "gpt-oss:120b"` is the API-visible model name that the training code must request.
- `--port "$PORT"` causes `main.py` to set `OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1`, which switches `gpt-oss` requests to the OpenAI-compatible client.

### Model Name Precedence: YAML vs vLLM Server
When using Sol/vLLM, the YAML and the server launch script have different jobs:

- YAML `llm_model_name` decides what model name the Python client asks for.
- vLLM `--served-model-name` decides which API name the server exposes.
- vLLM `--model` decides which checkpoint is actually loaded.

For example, with:
```yaml
llm_model_name: ollama:gpt-oss:120b
```
and:
```bash
--served-model-name "gpt-oss:120b"
--model "$MODEL_PATH"
```
the `ollama:` prefix is stripped internally, the client requests `gpt-oss:120b`, and the real model used is the checkpoint at `MODEL_PATH`.

**In practice:**
- If `main.py` is launched with `--port`, the vLLM/OpenAI-compatible server takes precedence for `gpt-oss` models.
- If `--port` is not provided and `OPENAI_BASE_URL` is not set, then `llm_model_name: ollama:...` uses the native Ollama backend instead.

### Optimizer Template Selection
Optimizer Jinja files are not auto-selected from the task name. The runner loads exactly what is written in the YAML:

- `llm_si_template_name`
- `llm_output_conversion_template_name`

These usually point to the same file for compatibility. The main prompt is the important one; the output-conversion template is largely legacy in the current numeric optimization paths.

**Current template families:**
- `num_optim.j2`: continuous numeric optimization (`cont_space_llm_num_optim`)
- `num_optim_semantic.j2`: continuous optimization with environment semantics (`cont_state_llm_num_optim_semantics`)
- `num_optim_candidates.j2`: discrete-state candidate policies (`dist_state_llm_num_optim`)
- `num_optim_candidates_semantics.j2`: discrete-state candidate policies with semantics (`dist_state_llm_num_optim_semantics`)
- `*_feedback.j2`: human/semantic feedback variants (`propspf` configs)

The file `configs/optim_template_map.md` is documentation only. It was generated to show which YAML currently points to which optimizer template; it is not read by the training code.

### Original Prompt Variants
The original optimizer prompts can now be used without rewriting the current YAML structure.

**Available original templates:**
- `num_optim_original.j2`
- `num_optim_semantic_original.j2`
- `num_optim_candidates_original.j2`
- `num_optim_candidates_semantics_original.j2`

**Two ways to use them:**

1. Point the YAML directly at the original files:
```yaml
llm_si_template_name: num_optim_original.j2
llm_output_conversion_template_name: num_optim_original.j2
```

2. Or keep the current template names and use the variant switch:
```yaml
optim_template_variant: original
```

With `optim_template_variant: original`, `main.py` automatically rewrites template names such as `num_optim.j2` to `num_optim_original.j2` if that file exists.

**Important:**
- This affects only optimizer prompt templates, not environment-description templates.
- If a requested `_original` file does not exist, the code falls back to the current template and prints a warning.
