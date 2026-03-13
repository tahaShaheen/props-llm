import yaml
import argparse
import os
import time
from runner import (
    llm_num_optim_runner,
)
from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner
from runner import llm_num_optim_semantics_feedback_runner
# import gym_maze
# import gym_navigation
from envs import nim, pong


def apply_template_variant(config):
    variant = config.get("optim_template_variant")
    if not variant:
        return

    template_dir = config.get("template_dir")
    if not template_dir:
        return

    keys = ["llm_si_template_name", "llm_output_conversion_template_name"]
    for key in keys:
        template_name = config.get(key)
        if not template_name:
            continue

        base_name, ext = os.path.splitext(template_name)
        candidate_name = f"{base_name}_{variant}{ext}" if ext else f"{template_name}_{variant}"
        candidate_path = os.path.join(template_dir, candidate_name)

        if os.path.exists(candidate_path):
            config[key] = candidate_name
        else:
            print(
                f"[WARNING] Requested template variant '{variant}' for {key}, but '{candidate_name}' was not found in '{template_dir}'. Using '{template_name}'."
            )


def resolve_repetition_id(cli_repetition_id=None, cli_run_id=None):
    if cli_repetition_id:
        return str(cli_repetition_id)
    if cli_run_id:
        return str(cli_run_id)

    slurm_array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    slurm_array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    if slurm_array_job_id and slurm_array_task_id:
        return f"{slurm_array_job_id}_{slurm_array_task_id}"

    env_candidates = [
        slurm_array_task_id,
        slurm_array_job_id,
        os.getenv("SLURM_JOB_ID"),
        os.getenv("JOB_ID"),
        os.getenv("PBS_JOBID"),
    ]
    for candidate in env_candidates:
        if candidate:
            return str(candidate)

    return str(int(time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--repetition_id",
        type=str,
        default=None,
        help="Repetition/job id used for log folder naming",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Deprecated alias for --repetition_id",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for local OpenAI-compatible server (e.g., vLLM)",
    )
    
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config.pop("num_repeats", None)
    config.pop("num_runs", None)
    base_logdir = config.get("logdir")
    repetition_id = resolve_repetition_id(args.repetition_id, args.run_id)

    run_config = dict(config)

    # Route OpenAI-compatible local servers (like vLLM) via env var.
    # Existing agent code reads OPENAI_BASE_URL to switch from Ollama mode.
    if args.port:
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{args.port}/v1"

    if base_logdir:
        run_config["logdir"] = os.path.join(base_logdir, f"repetition_{repetition_id}")

    apply_template_variant(run_config)

    if run_config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
        llm_num_optim_runner.run_training_loop(**run_config)
    elif run_config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
        llm_num_optim_semantics_runner.run_training_loop(**run_config)
    elif run_config["task"] in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]:
        run_config["config_path"] = args.config
        llm_num_optim_semantics_feedback_runner.run_training_loop(**run_config)
    else:
        raise ValueError(f"Task {run_config['task']} not recognized.")


if __name__ == "__main__":
    main()
