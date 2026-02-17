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

### Configurable Ollama Context Window
- Added `ollama_num_ctx` parameter (default: 4096) to configure token context limits
- Set in YAML: `ollama_num_ctx: 50000`
- Token counting via Ollama's tokenize API
- Displays token counts on terminal: `[TOKENS] ollama:model: X tokens` to diagnose Context Window Saturation/Forgetting thing-a-majig

### Context Guard System
- Automatic check before sending prompts to Ollama: `[GUARD] Input prompt: X tokens / Y context limit`
- Prevents context overflow by throwing error with suggestions
- Calculates recommended `ollama_num_ctx` based on remaining iterations and tokens per parameter line

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
