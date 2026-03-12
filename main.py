import yaml
import argparse
import os
from runner import (
    llm_num_optim_runner,
)
from runner import llm_num_optim_runner
from runner import llm_num_optim_semantics_runner
from runner import llm_num_optim_semantics_feedback_runner
# import gym_maze
# import gym_navigation
from envs import nim, pong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    has_repeat_key = "num_repeats" in config or "num_runs" in config
    num_repeats = int(config.pop("num_repeats", config.pop("num_runs", 1)))
    base_logdir = config.get("logdir")

    for repeat_idx in range(num_repeats):
        run_config = dict(config)
        if has_repeat_key and base_logdir:
            run_config["logdir"] = os.path.join(base_logdir, f"run_{repeat_idx}")

        if run_config["task"] in ["cont_space_llm_num_optim", "cont_space_llm_num_optim_rndm_proj", "dist_state_llm_num_optim"]:
            llm_num_optim_runner.run_training_loop(**run_config)
        elif run_config["task"] in ["dist_state_llm_num_optim_semantics", "cont_state_llm_num_optim_semantics"]:
            llm_num_optim_semantics_runner.run_training_loop(**run_config)
        elif run_config["task"] in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]:
            # Pass config path for eval_parallel_policy visualization
            run_config["config_path"] = args.config
            llm_num_optim_semantics_feedback_runner.run_training_loop(**run_config)
        else:
            raise ValueError(f"Task {run_config['task']} not recognized.")


if __name__ == "__main__":
    main()
