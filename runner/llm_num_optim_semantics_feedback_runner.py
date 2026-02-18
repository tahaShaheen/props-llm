"""
Runner for LLM-based RL policy optimization with human-in-the-loop feedback.

This runner:
1. Gets policy parameters from the LLM
2. Evaluates the policy
3. Shows the policy execution to a human via eval_parallel_policy
4. Collects human feedback from the terminal
5. Passes feedback to the LLM for the next iteration
6. Logs feedback in overall_log.csv
"""

from world.continuous_space_general_world import ContinualSpaceGeneralWorld
from world.discrete_state_general_world import DiscreteStateGeneralWorld
from agent.llm_num_optim_q_table_semantics import LLMNumOptimQTableSemanticsAgent
from agent.llm_num_optim_linear_policy_semantics import LLMNumOptimSemanticAgent
from utils.plotting import update_training_plot
from utils.console import red, green, yellow, blue
from jinja2 import Environment, FileSystemLoader
import os
from prompt_toolkit import PromptSession
import sys
import traceback
import numpy as np
import ast
import subprocess
import sys
import csv


def build_feedback_history(overall_log_path: str, max_rows: int = 20) -> str:
    """Build compact textual history from overall_log.csv for feedback prompting."""
    if not os.path.exists(overall_log_path):
        return ""

    rows = []
    try:
        with open(overall_log_path, "r", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                rows.append(row)
    except Exception:
        return ""

    if not rows:
        return ""

    selected = rows[-max_rows:]
    lines = []
    for row in selected:
        try:
            it = row.get("Iteration", "")
            actual_reward = row.get("Total Reward", "")
            guessed_reward = row.get("Guessed Reward", "")
            actual_feedback = row.get("Human Feedback", "")
            guessed_feedback = row.get("Guessed Feedback", "")
            lines.append(
                f"Iteration {it}: actual_reward={actual_reward}, guessed_reward={guessed_reward}, "
                f"actual_feedback={actual_feedback}, guessed_feedback={guessed_feedback}"
            )
        except Exception:
            continue
    return "\n".join(lines)


def get_human_feedback(
    episode: int,
    reward: float,
    config_path: str,
    logdir: str,
    use_gemini_feedback: bool = False,
    gemini_feedback_model: str = "gemini-2.0-flash",
    gemini_feedback_template=None,
    env_description_text: str = "",
    overall_log_path: str = "",
    gemini_feedback_history_rows: int = 20,
    gemini_api_key_env: str = "GEMINI_API_KEY",
) -> str:


    """Show episode visualization and collect feedback (human or Gemini)."""
    print("\n" + "=" * 80)
    print(blue(f"EPISODE {episode} COMPLETE - Average Reward: {reward:.2f}"))
    print("=" * 80)
    
    # Run the visualization
    print(yellow("\nLaunching policy visualization..."))
    print(yellow("Watch the policy execute, then close the window to provide feedback.\n"))
    
    try:
        # Run eval_parallel_policy.py with the current episode
        log_file = os.path.join(logdir, "overall_log.csv")
        feedback_gif_path = os.path.join(logdir, f"episode_{episode}", "feedback_preview.gif")
        os.makedirs(os.path.dirname(feedback_gif_path), exist_ok=True)
        cmd = [
            sys.executable,
            "eval_parallel_policy.py",
            "--config", config_path,
            "--episode", str(episode),
            "--render",
            "--num_envs", "5",
            "--save_video_path", feedback_gif_path,
        ]
        if use_gemini_feedback:
            cmd.append("--no_window")
        
        # Run and wait for the visualization to complete
        result = subprocess.run(cmd, capture_output=False)
        
    except Exception as e:
        print(red(f"Warning: Could not launch visualization: {e}"))
        print(yellow("Continuing without visualization..."))

    if use_gemini_feedback:
        print("\n" + "=" * 80)
        print(green("GEMINI FEEDBACK TIME"))
        print("=" * 80)
        try:
            from google import genai

            api_key = os.getenv(gemini_api_key_env, "")
            client = genai.Client(api_key=api_key) if api_key else genai.Client()

            history_text = build_feedback_history(
                overall_log_path or os.path.join(logdir, "overall_log.csv"),
                max_rows=gemini_feedback_history_rows,
            )

            if gemini_feedback_template is not None:
                prompt_text = gemini_feedback_template.render(
                    {
                        "episode": episode,
                        "reward": f"{reward:.2f}",
                        "environment_description": env_description_text,
                        "history": history_text,
                    }
                )
            else:
                prompt_text = (
                    "You are evaluating an RL agent rollout GIF. "
                    "Provide concise feedback with good, bad, and how to improve.\n"
                    f"Episode: {episode}\n"
                    f"Reward: {reward:.2f}\n"
                    f"Environment:\n{env_description_text}\n"
                    f"History:\n{history_text}\n"
                )

            uploaded_file = client.files.upload(file=feedback_gif_path)
            response = client.models.generate_content(
                model=gemini_feedback_model,
                contents=[uploaded_file, prompt_text],
            )
            feedback = (response.text or "").strip()

            if feedback:
                print(green(f"\n✓ Gemini feedback recorded: \"{feedback}\""))
            else:
                print(yellow("\n(Gemini returned empty feedback, continuing...)"))
            return feedback
        except Exception as e:
            print(red(f"Gemini feedback failed: {e}"))
            print(yellow("Falling back to manual terminal feedback..."))
    
    # Prompt for human feedback using prompt_toolkit
    print("\n" + "=" * 80)
    print(green("FEEDBACK TIME"))
    print("=" * 80)
    print("You just observed the policy in action.")
    print("Provide feedback to help the LLM improve the policy.")
    print("Examples:")
    print("  - 'The agent falls over too quickly'")
    print("  - 'Movement is too jerky, needs smoother actions'")
    print("  - 'Good forward motion but unstable'")
    print("  - 'Agent barely moves, try larger action values'")
    print("\nPress ENTER with no input to skip feedback for this iteration.")
    print("=" * 80)
    
    try:
        # Use prompt_toolkit for professional input experience with arrow keys, history, etc.
        session = PromptSession()
        feedback = session.prompt("\nYour feedback: ").strip()
    except EOFError:
        feedback = ""
    except KeyboardInterrupt:
        print(yellow("\n(Feedback cancelled)"))
        feedback = ""
    
    if feedback:
        print(green(f"\n✓ Feedback recorded: \"{feedback}\""))
    else:
        print(yellow("\n(No feedback provided, continuing...)"))
    
    return feedback


def update_overall_log_feedback_for_episode(overall_log_path: str, target_episode: int, feedback: str):
    """Update Human Feedback column for a specific episode row in overall_log.csv."""
    if not os.path.exists(overall_log_path):
        return

    with open(overall_log_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames or "Human Feedback" not in fieldnames:
        return

    updated = False
    for row in rows:
        try:
            episode_value = int(str(row.get("Iteration", "")).strip())
        except Exception:
            continue
        if episode_value == target_episode:
            row["Human Feedback"] = feedback
            updated = True
            break

    if not updated:
        return

    with open(overall_log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        writer.writeheader()
        writer.writerows(rows)


def restore_agent_state_from_overall_log(agent, overall_log_path: str) -> int:
    """Restore replay buffer and feedback/prediction state from overall_log.csv.

    Returns number of restored rows.
    """
    if not os.path.exists(overall_log_path):
        return 0

    try:
        with open(overall_log_path, "r", newline="") as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            rows = list(reader)
    except Exception as e:
        print(red(f"Failed reading {overall_log_path} for replay restore: {e}"))
        return 0

    def parse_float(value):
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    def parse_params_vector(value):
        import re

        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None

        try:
            parsed = ast.literal_eval(text)
            arr = np.array(parsed, dtype=float).reshape(-1)
            if arr.size > 0:
                return arr
        except Exception:
            pass

        nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
        if not nums:
            return None
        return np.array([float(x) for x in nums], dtype=float).reshape(-1)

    records = []
    for row in rows:
        try:
            episode = int(str(row.get("Iteration", "")).strip())
        except Exception:
            continue

        reward = parse_float(row.get("Total Reward", ""))
        params = parse_params_vector(row.get("Parameters", ""))
        if reward is None or params is None:
            continue

        guessed_reward = parse_float(row.get("Guessed Reward", ""))
        guessed_feedback = str(row.get("Guessed Feedback", "") or "").strip()
        human_feedback = str(row.get("Human Feedback", "") or "").strip()

        records.append((episode, params, reward, human_feedback, guessed_reward, guessed_feedback))

    records.sort(key=lambda item: item[0])

    if hasattr(agent, "replay_buffer") and hasattr(agent.replay_buffer, "buffer"):
        agent.replay_buffer.buffer.clear()

    if hasattr(agent, "feedback_buffer"):
        agent.feedback_buffer = {}
    if hasattr(agent, "predicted_reward_buffer"):
        agent.predicted_reward_buffer = {}
    if hasattr(agent, "predicted_feedback_buffer"):
        agent.predicted_feedback_buffer = {}
    if hasattr(agent, "episode_numbers"):
        agent.episode_numbers = []

    restored = 0
    for episode, params, reward, human_feedback, guessed_reward, guessed_feedback in records:
        agent.replay_buffer.add(params, reward)
        buffer_idx = len(agent.replay_buffer.buffer) - 1

        if hasattr(agent, "store_episode_number"):
            agent.store_episode_number(buffer_idx, episode)
        if human_feedback and hasattr(agent, "store_feedback"):
            agent.store_feedback(episode, human_feedback)
        if hasattr(agent, "store_predicted_outcomes"):
            agent.store_predicted_outcomes(episode, guessed_reward, guessed_feedback)

        restored += 1

    return restored


def run_training_loop(
    task,
    num_episodes,
    gym_env_name,
    render_mode,
    logdir,
    dim_actions,
    dim_states,
    max_traj_count,
    max_traj_length,
    template_dir,
    llm_si_template_name,
    llm_output_conversion_template_name,
    llm_model_name,
    num_evaluation_episodes,
    warmup_episodes,
    warmup_dir,
    config_path=None,  # Added to pass config path for eval_parallel_policy
    bias=None,
    rank=None,
    optimum=None,
    search_step_size=0.1,
    env_kwargs=None,
    env_desc_file=None,
    ollama_num_ctx=4096,
    ollama_num_predict=None,
    include_trajectories=True,
    include_feedback_with_params=False,  # NEW: Toggle for feedback display
    feedback_interval=5,  # NEW: Ask for feedback every N episodes (only for best params)
    param_min=None,  # NEW: Minimum parameter value for validation (continuous tasks)
    param_max=None,  # NEW: Maximum parameter value for validation (continuous tasks)
    use_gemini_feedback=False,
    gemini_feedback_model="gemini-2.0-flash",
    gemini_feedback_template_name=None,
    gemini_feedback_history_rows=20,
    gemini_api_key_env="GEMINI_API_KEY",
):
    assert task in ["dist_state_llm_num_optim_semantics_with_feedback", "cont_state_llm_num_optim_semantics_with_feedback"]
    
    # Validate and warn about optimum parameter
    if optimum is None:
        print("[WARNING] optimum is None! Ensure 'optimum' is defined in your YAML config file.")
        optimum = 1.0  # Fallback default
    else:
        print(f"[INFO] Using optimum value from config: {optimum}")

    def format_parameters_for_csv(parameters):
        try:
            if isinstance(parameters, str):
                parsed = ast.literal_eval(parameters)
            else:
                parsed = parameters
            arr = np.array(parsed, dtype=float).flatten()
            # Convert to space-separated string manually to avoid numpy line breaks
            arr_str = '[' + ' '.join(f'{x:.5g}' for x in arr) + ']'
            return arr_str
        except Exception:
            return str(parameters).replace('\n', ' ').replace('\r', '')

    def format_parameters_for_prompt(parameters):
        import re

        try:
            if isinstance(parameters, str):
                text = parameters.strip()
                parsed = None
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
                    if nums:
                        parsed = [float(x) for x in nums]
                    else:
                        parsed = text
            else:
                parsed = parameters
            arr = np.array(parsed, dtype=float).flatten()
            return ", ".join([f"params[{i}]: {x:.5g}" for i, x in enumerate(arr)])
        except Exception:
            return str(parameters).replace('\n', ' ').replace('\r', '')

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    gemini_feedback_template = (
        jinja2_env.get_template(gemini_feedback_template_name)
        if gemini_feedback_template_name
        else None
    )
    env_description_text = ""
    if env_desc_file:
        try:
            env_description_text = jinja2_env.get_template(env_desc_file).render()
        except Exception:
            env_description_text = env_desc_file
    
    if task == "dist_state_llm_num_optim_semantics_with_feedback":
        world = DiscreteStateGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
            env_kwargs=env_kwargs,
        )

        agent = LLMNumOptimQTableSemanticsAgentWithFeedback(
            logdir,
            dim_actions,
            dim_states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            optimum,
            env_kwargs=env_kwargs,
            env_desc_file=env_desc_file,
            num_episodes=num_episodes,
            ollama_num_ctx=ollama_num_ctx,
            ollama_num_predict=ollama_num_predict,
            include_trajectories=include_trajectories,
            include_feedback_with_params=include_feedback_with_params,
        )
    else:
        world = ContinualSpaceGeneralWorld(
            gym_env_name,
            render_mode,
            max_traj_length,
        )

        agent = LLMNumOptimSemanticAgentWithFeedback(
            logdir,
            dim_actions,
            dim_states,
            max_traj_count,
            max_traj_length,
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            num_evaluation_episodes,
            bias,
            optimum,
            search_step_size,
            env_desc_file=env_desc_file,
            num_episodes=num_episodes,
            ollama_num_ctx=ollama_num_ctx,
            ollama_num_predict=ollama_num_predict,
            include_trajectories=include_trajectories,
            include_feedback_with_params=include_feedback_with_params,
            param_min=param_min,
            param_max=param_max,
        )

    print('init done')

    overall_log_path = f"{logdir}/overall_log.csv"
    feedback_log_path = f"{logdir}/feedback_log.csv"
    start_episode = 0
    logged_episodes = set()
    has_existing_log = False
    
    if os.path.exists(overall_log_path):
        try:
            with open(overall_log_path, "r") as log_file:
                lines = [line.strip() for line in log_file.readlines() if line.strip()]
            if len(lines) > 1:
                has_existing_log = True
                for row in lines[1:]:
                    try:
                        logged_episodes.add(int(row.split(",", 1)[0]))
                    except ValueError:
                        continue
                if logged_episodes:
                    start_episode = max(logged_episodes) + 1
                print(green(f"Resuming overall_log.csv from episode {start_episode}"))
        except Exception as e:
            print(red(f"Failed to parse existing overall_log.csv, starting from 0: {e}"))
            start_episode = 0

    if start_episode > 0:
        restored_count = restore_agent_state_from_overall_log(agent, overall_log_path)
        agent.training_episodes = start_episode
        print(green(f"Restored replay state from overall_log.csv ({restored_count} entries)"))
    else:
        if not warmup_dir:
            warmup_dir = f"{logdir}/warmup"
            os.makedirs(warmup_dir, exist_ok=True)
            agent.random_warmup(world, warmup_dir, warmup_episodes)
        else:
            agent.replay_buffer.load(warmup_dir)

    overall_fieldnames = [
        "Iteration",
        "CPU Time",
        "API Time",
        "Total Episodes",
        "Total Steps",
        "Total Reward",
        "Guessed Reward",
        "Context Size",
        "Num Attempts",
        "Human Feedback",
        "Guessed Feedback",
        "Explanation",
        "Parameters",
    ]
    overall_log_file = open(overall_log_path, "a" if start_episode > 0 else "w", newline="")
    overall_log_writer = csv.DictWriter(
        overall_log_file,
        fieldnames=overall_fieldnames,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    if start_episode == 0:
        overall_log_writer.writeheader()
        overall_log_file.flush()

    # Initialize feedback log
    feedback_fieldnames = [
        "Iteration",
        "Actual Reward",
        "Guessed Reward",
        "Actual Feedback",
        "Guessed Feedback",
    ]
    feedback_log_file = open(feedback_log_path, "a" if start_episode > 0 else "w", newline="")
    feedback_log_writer = csv.DictWriter(
        feedback_log_file,
        fieldnames=feedback_fieldnames,
        quoting=csv.QUOTE_NONNUMERIC,
    )
    if start_episode == 0:
        feedback_log_writer.writeheader()
        feedback_log_file.flush()

    # Track feedback for intervals
    interval_best_episode = None
    interval_best_reward = float('-inf')
    
    # Initialize feedback tracking across iterations
    current_feedback = ""
    last_policy_params = ""
    episode_params_map = {}
    
    for episode in range(start_episode, num_episodes):
        if episode in logged_episodes:
            print(green(f"Skipping already logged episode {episode}"))
            continue
        print(f"\n{'='*80}")
        print(f"EPISODE: {episode}")
        print(f"{'='*80}")
        
        # Create log dir
        curr_episode_dir = f"{logdir}/episode_{episode}"
        if os.path.exists(curr_episode_dir):
            print(green(f"Reusing existing episode directory: {curr_episode_dir}"))
        else:
            print(f"Creating log directory: {curr_episode_dir}")
            os.makedirs(curr_episode_dir, exist_ok=True)
        
        training_succeeded = False
        for trial_idx in range(10):
            try:
                # Train with feedback from previous iteration
                request_feedback_prediction = ((episode + 1) % feedback_interval == 0)
                cpu_time, api_time, total_episodes, total_steps, total_reward, parameters, context_size, num_attempts = agent.train_policy_with_feedback(
                    world, 
                    curr_episode_dir, 
                    attempt_idx=trial_idx,
                    human_feedback=current_feedback,
                    last_policy_params=last_policy_params,
                    request_feedback_prediction=request_feedback_prediction,
                )

                guessed_reward = getattr(agent, "last_predicted_reward", None)
                guessed_feedback = (
                    (getattr(agent, "last_predicted_feedback", "") or "")
                    if request_feedback_prediction
                    else ""
                )
                guessed_explanation = getattr(agent, "last_explanation", "") or ""
                if hasattr(agent, "store_predicted_outcomes"):
                    agent.store_predicted_outcomes(episode, guessed_reward, guessed_feedback)
                
                # Format parameters for CSV
                formatted_parameters = format_parameters_for_csv(parameters)
                
                # Human feedback in this row should correspond to this episode's params
                # (may be empty until interval feedback is collected and row is updated later).
                feedback_for_this_episode = (
                    agent.get_feedback_for_episode(episode)
                    if hasattr(agent, "get_feedback_for_episode")
                    else ""
                )

                guessed_reward_value = "" if guessed_reward is None else guessed_reward

                overall_log_writer.writerow(
                    {
                        "Iteration": episode,
                        "CPU Time": cpu_time,
                        "API Time": api_time,
                        "Total Episodes": total_episodes,
                        "Total Steps": total_steps,
                        "Total Reward": total_reward,
                        "Guessed Reward": guessed_reward_value,
                        "Context Size": context_size,
                        "Num Attempts": num_attempts,
                        "Human Feedback": feedback_for_this_episode,
                        "Guessed Feedback": guessed_feedback,
                        "Explanation": guessed_explanation,
                        "Parameters": formatted_parameters,
                    }
                )
                overall_log_file.flush()
                
                # Log guessed-vs-actual reward every episode
                feedback_log_writer.writerow(
                    {
                        "Iteration": episode,
                        "Actual Reward": total_reward,
                        "Guessed Reward": guessed_reward_value,
                        "Actual Feedback": "",
                        "Guessed Feedback": guessed_feedback,
                    }
                )
                feedback_log_file.flush()
                
                guessed_reward_file_path = f"{curr_episode_dir}/guessed_reward.txt"
                with open(guessed_reward_file_path, "w") as f:
                    f.write("" if guessed_reward is None else str(guessed_reward))

                if request_feedback_prediction:
                    guessed_feedback_file_path = f"{curr_episode_dir}/guessed_feedback.txt"
                    with open(guessed_feedback_file_path, "w") as f:
                        f.write(guessed_feedback)

                guessed_explanation_file_path = f"{curr_episode_dir}/guessed_explanation.txt"
                with open(guessed_explanation_file_path, "w") as f:
                    f.write(guessed_explanation)
                
                # Update training progress plot
                update_training_plot(logdir)
                print(green(f"{trial_idx + 1}th trial attempt succeeded in training"))
                
                # Store current params as indexed params[i] list for next iteration prompt
                last_policy_params = format_parameters_for_prompt(parameters)
                episode_params_map[episode] = last_policy_params
                
                # Track best in current interval
                if total_reward > interval_best_reward:
                    interval_best_reward = total_reward
                    interval_best_episode = episode
                
                # Track which replay buffer index corresponds to this episode.
                # Must use the actual replay buffer position (warmup entries may exist).
                agent.store_episode_number(len(agent.replay_buffer.buffer) - 1, episode)
                
                training_succeeded = True
                break
            except Exception as e:
                if isinstance(e, KeyError):
                    print(red(f"{trial_idx + 1}th trial attempt failed: INVALID ACTION"))
                else:
                    print(
                        red(
                            f"{trial_idx + 1}th trial attempt failed with error in training: {e}"
                        )
                    )
                    print(red(f"Error type: {type(e).__name__}"))
                    traceback.print_exc()
                continue
        
        if not training_succeeded:
            print(f"Episode {episode} failed to train after 10 attempts")
            break
        
        # === FEEDBACK INTERVAL CHECK ===
        # Ask for feedback every feedback_interval episodes, on the best params from that batch
        if (episode + 1) % feedback_interval == 0:
            print(f"\n{'='*80}")
            print(blue(f"FEEDBACK INTERVAL COMPLETE (Episodes {episode + 1 - feedback_interval} to {episode})"))
            print(blue(f"Best params from this interval: Episode {interval_best_episode} with reward {interval_best_reward:.2f}"))
            print(f"{'='*80}")
            
            current_feedback = get_human_feedback(
                episode=interval_best_episode,
                reward=interval_best_reward,
                config_path=config_path,
                logdir=logdir,
                use_gemini_feedback=use_gemini_feedback,
                gemini_feedback_model=gemini_feedback_model,
                gemini_feedback_template=gemini_feedback_template,
                env_description_text=env_description_text,
                overall_log_path=overall_log_path,
                gemini_feedback_history_rows=gemini_feedback_history_rows,
                gemini_api_key_env=gemini_api_key_env,
            )

            # Ensure the prompt shows params that actually received this feedback.
            if interval_best_episode in episode_params_map:
                last_policy_params = episode_params_map[interval_best_episode]
            
            # Store feedback in agent for inclusion with params in future iterations
            if current_feedback:
                print(f"[INFO] Storing feedback for episode {interval_best_episode}")
                feedback_episode_dir = f"{logdir}/episode_{interval_best_episode}"
                os.makedirs(feedback_episode_dir, exist_ok=True)
                feedback_file_path = f"{feedback_episode_dir}/human_feedback.txt"
                with open(feedback_file_path, "w") as f:
                    f.write(current_feedback)
            agent.store_feedback(interval_best_episode, current_feedback)

            # IMPORTANT: close append handle before in-place rewrite to avoid file corruption.
            overall_log_file.flush()
            overall_log_file.close()
            update_overall_log_feedback_for_episode(overall_log_path, interval_best_episode, current_feedback)
            overall_log_file = open(overall_log_path, "a", newline="")
            overall_log_writer = csv.DictWriter(
                overall_log_file,
                fieldnames=overall_fieldnames,
                quoting=csv.QUOTE_NONNUMERIC,
            )

            guessed_reward_for_feedback_ep = (
                agent.get_predicted_reward_for_episode(interval_best_episode)
                if hasattr(agent, "get_predicted_reward_for_episode")
                else None
            )
            guessed_feedback_for_feedback_ep = (
                agent.get_predicted_feedback_for_episode(interval_best_episode)
                if hasattr(agent, "get_predicted_feedback_for_episode")
                else ""
            )
            guessed_reward_feedback_ep_value = "" if guessed_reward_for_feedback_ep is None else guessed_reward_for_feedback_ep

            feedback_log_writer.writerow(
                {
                    "Iteration": interval_best_episode,
                    "Actual Reward": interval_best_reward,
                    "Guessed Reward": guessed_reward_feedback_ep_value,
                    "Actual Feedback": current_feedback,
                    "Guessed Feedback": guessed_feedback_for_feedback_ep,
                }
            )
            feedback_log_file.flush()
            
            # Reset interval tracking
            interval_best_episode = None
            interval_best_reward = float('-inf')
        else:
            # No feedback this episode
            current_feedback = ""
    
    overall_log_file.close()
    feedback_log_file.close()


# ============================================================================
# Agent classes with feedback support
# ============================================================================

class LLMNumOptimSemanticAgentWithFeedback(LLMNumOptimSemanticAgent):
    """Extended agent that supports human feedback in the training loop."""
    
    def __init__(self, *args, include_feedback_with_params=False, param_min=None, param_max=None, **kwargs):
        """Initialize with feedback support.
        
        Args:
            include_feedback_with_params: If True, include human feedback with params in LLM prompts
            param_min: Minimum valid parameter value (for continuous tasks)
            param_max: Maximum valid parameter value (for continuous tasks)
        """
        super().__init__(*args, **kwargs)
        self.include_feedback_with_params = include_feedback_with_params
        self.param_min = param_min
        self.param_max = param_max
        self.feedback_buffer = {}  # Maps episode index to feedback string
        self.predicted_reward_buffer = {}  # Maps episode index to LLM-predicted reward
        self.predicted_feedback_buffer = {}  # Maps episode index to LLM-predicted feedback
        self.episode_numbers = []  # Tracks which episode number each buffer entry corresponds to
        self.last_predicted_reward = None
        self.last_predicted_feedback = ""
        self.last_explanation = ""
    
    def store_feedback(self, episode: int, feedback: str):
        """Store human feedback for a given episode."""
        self.feedback_buffer[episode] = feedback

    def store_predicted_outcomes(self, episode: int, predicted_reward=None, predicted_feedback: str = ""):
        """Store model-predicted reward and feedback for a given episode."""
        self.predicted_reward_buffer[episode] = predicted_reward
        self.predicted_feedback_buffer[episode] = predicted_feedback or ""
    
    def store_episode_number(self, buffer_idx: int, episode: int):
        """Track which episode number corresponds to which buffer index."""
        # Expand list if needed
        while len(self.episode_numbers) <= buffer_idx:
            self.episode_numbers.append(None)
        self.episode_numbers[buffer_idx] = episode
    
    def get_feedback_for_buffer_idx(self, buffer_idx: int) -> str:
        """Retrieve feedback for a buffer index by looking up its episode number."""
        if buffer_idx < len(self.episode_numbers):
            episode = self.episode_numbers[buffer_idx]
            if episode is not None and episode in self.feedback_buffer:
                return self.feedback_buffer[episode]
        return ""
    
    def get_feedback_for_episode(self, episode: int) -> str:
        """Retrieve feedback for an episode, or empty string if not available."""
        return self.feedback_buffer.get(episode, "")

    def get_predicted_reward_for_episode(self, episode: int):
        """Retrieve predicted reward for an episode, or None if not available."""
        return self.predicted_reward_buffer.get(episode, None)

    def get_predicted_feedback_for_episode(self, episode: int) -> str:
        """Retrieve predicted feedback for an episode, or empty string if not available."""
        return self.predicted_feedback_buffer.get(episode, "")

    def get_predicted_reward_for_buffer_idx(self, buffer_idx: int):
        """Retrieve predicted reward for a buffer index by looking up its episode number."""
        if buffer_idx < len(self.episode_numbers):
            episode = self.episode_numbers[buffer_idx]
            if episode is not None:
                return self.get_predicted_reward_for_episode(episode)
        return None

    def get_predicted_feedback_for_buffer_idx(self, buffer_idx: int) -> str:
        """Retrieve predicted feedback for a buffer index by looking up its episode number."""
        if buffer_idx < len(self.episode_numbers):
            episode = self.episode_numbers[buffer_idx]
            if episode is not None:
                return self.get_predicted_feedback_for_episode(episode)
        return ""
    
    def train_policy_with_feedback(self, world, logdir, attempt_idx=0, human_feedback="", last_policy_params="", request_feedback_prediction=False):
        """Train policy with human feedback from previous iteration."""

        def canonicalize_params(params):
            arr = np.array(params, dtype=float).reshape(-1)
            return tuple(np.round(arr, 1).tolist())

        def extract_predicted_reward(input_text):
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', input_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.replace('```', '').replace('`', '')

            explicit_matches = re.findall(
                r'^\s*predicted_reward\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$',
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if explicit_matches:
                try:
                    return float(explicit_matches[-1])
                except ValueError:
                    return None

            fallback_matches = re.findall(
                r'^\s*guessed_reward\s*:\s*([+-]?\d+(?:\.\d+)?)\s*$',
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if fallback_matches:
                try:
                    return float(fallback_matches[-1])
                except ValueError:
                    return None
            return None

        def extract_predicted_feedback(input_text):
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', input_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.replace('```', '').replace('`', '')

            feedback_matches = re.findall(
                r'^\s*predicted_feedback\s*:\s*(.+?)\s*$',
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if feedback_matches:
                return feedback_matches[-1].strip().strip('"').strip()

            fallback_matches = re.findall(
                r'^\s*guessed_feedback\s*:\s*(.+?)\s*$',
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if fallback_matches:
                return fallback_matches[-1].strip().strip('"').strip()
            return ""

        def extract_explanation(input_text):
            import re
            cleaned = re.sub(r'<think>.*?</think>', '', input_text, flags=re.DOTALL | re.IGNORECASE)
            cleaned = cleaned.replace('```', '').replace('`', '')
            explanation_matches = re.findall(
                r'^\s*explanation\s*:\s*(.+?)\s*$',
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if explanation_matches:
                return explanation_matches[-1].strip().strip('"').strip()
            return ""
        
        def parse_parameters(input_text):
            import re
            # Same parsing logic as parent class
            cleaned_text = input_text
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'</?think>', '', cleaned_text)
            cleaned_text = cleaned_text.replace('```', '')
            cleaned_text = cleaned_text.replace('`', '')
            
            first_line = cleaned_text.split("\n")[0] if cleaned_text else ""
            print(blue(f"response: {first_line}"))
            
            pattern1 = re.compile(r"params\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)")
            pattern2 = re.compile(r"params\[(\d+)\]\s*=\s*([+-]?\d+(?:\.\d+)?)")
            
            matches = pattern1.findall(cleaned_text)
            if not matches:
                matches = pattern2.findall(cleaned_text)
            
            param_dict = {}
            for match in matches:
                param_idx = int(match[0])
                param_val = float(match[1])
                if param_idx < self.rank:
                    param_dict[param_idx] = param_val
            
            if not param_dict:
                number_pattern = re.compile(r'[+-]?\d+(?:\.\d+)?')
                for line in cleaned_text.split('\n'):
                    if len(line.strip()) > 150 or line.lower().count('exploration') > 0 or line.lower().count('parameter') > 0:
                        continue
                    numbers = number_pattern.findall(line)
                    if len(numbers) >= self.rank:
                        try:
                            for i in range(self.rank):
                                param_dict[i] = float(numbers[i])
                            break
                        except (ValueError, IndexError):
                            continue
            
            results = []
            for i in range(self.rank):
                if i in param_dict:
                    results.append(param_dict[i])
                else:
                    results.append(None)
            
            results = [r for r in results if r is not None]
            print(results)
            
            if len(results) != self.rank:
                from utils.console import gray
                print(f"\n{'='*80}")
                print(f"ERROR: Expected {self.rank} parameters, got {len(results)}")
                print(f"Parsed param indices: {sorted(param_dict.keys())}")
                print(f"{'='*80}")
                print(gray(f"FULL RESPONSE TEXT:\n{input_text}"))
                print(f"{'='*80}\n")
                assert len(results) == self.rank, f"Expected {self.rank} params, got {len(results)}"
            
            # Validate parameter bounds if configured
            if self.param_min is not None and self.param_max is not None:
                out_of_bounds = []
                for i, val in enumerate(results):
                    if val < self.param_min or val > self.param_max:
                        out_of_bounds.append((i, val))
                
                if out_of_bounds:
                    from utils.console import gray
                    print(f"\n{'='*80}")
                    print(f"ERROR: {len(out_of_bounds)} parameters are OUT OF BOUNDS!")
                    print(f"Valid range: [{self.param_min}, {self.param_max}]")
                    print(f"Out of bounds parameters:")
                    for idx, val in out_of_bounds[:10]:  # Show first 10
                        print(f"  params[{idx}]: {val}")
                    if len(out_of_bounds) > 10:
                        print(f"  ... and {len(out_of_bounds) - 10} more")
                    print(f"{'='*80}")
                    print(gray(f"FULL RESPONSE TEXT:\n{input_text}"))
                    print(f"{'='*80}\n")
                    assert False, f"{len(out_of_bounds)} parameters out of bounds [{self.param_min}, {self.param_max}]"
            
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer, traj_buffer, n):
            from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
            if not replay_buffer.buffer:
                return ""

            # Simply iterate through all episodes in the buffer
            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(replay_buffer.buffer))
            
            for idx, (weights, reward) in enumerate(replay_buffer.buffer):
                params = weights.reshape(-1)
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {params[i]:.5g}; "
                predicted_reward = self.get_predicted_reward_for_buffer_idx(idx)
                if predicted_reward is None:
                    predicted_reward_str = "N/A"
                else:
                    predicted_reward_str = f"{predicted_reward:.2f}"
                l += f"f(params): {reward:.2f}; predicted_reward: {predicted_reward_str}\n"
                text += l
                
                # Add feedback if available and enabled
                if self.include_feedback_with_params:
                    feedback = self.get_feedback_for_buffer_idx(idx)
                    predicted_feedback = self.get_predicted_feedback_for_buffer_idx(idx)
                    
                    if predicted_feedback:
                        text += f"  LLM PREDICTED FEEDBACK: \"{predicted_feedback}\"\n"
                    if feedback:
                        text += f"  HUMAN FEEDBACK: \"{feedback}\"\n"
                
                # Add trajectory if available and enabled
                if self.include_trajectories and idx < len(traj_buffer.buffer):
                    trajectory = traj_buffer.buffer[idx].get_trajectory()
                    if trajectory:
                        text += "Trajectory: "
                        for state, action, reward in trajectory:
                            text += f"({state},{action}) "
                        text += "\n"
            
            return text

        # Update the policy using llm_brain with feedback
        print("Updating the policy with human feedback...")
        new_parameter_list, reasoning, api_time, context_size = self.llm_brain.llm_update_parameters_with_feedback(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            num_episodes=self.num_episodes,
            rank=self.rank,
            optimum=self.optimum,
            search_step_size=self.search_step_size,
            num_evaluation_episodes=self.num_evaluation_episodes,
            attempt_idx=attempt_idx,
            human_feedback=human_feedback,
            last_policy_params=last_policy_params,
            request_feedback_prediction=request_feedback_prediction,
        )
        self.api_call_time += api_time

        llm_response_text = reasoning.split("LLM:\n", 1)[-1] if "LLM:\n" in reasoning else reasoning
        self.last_predicted_reward = extract_predicted_reward(llm_response_text)
        self.last_predicted_feedback = extract_predicted_feedback(llm_response_text)
        self.last_explanation = extract_explanation(llm_response_text)

        # Hard duplicate guard: reject params already present in replay buffer
        seen_params = {
            canonicalize_params(weights)
            for weights, _reward in self.replay_buffer.buffer
        }
        candidate_params = canonicalize_params(new_parameter_list)
        if candidate_params in seen_params:
            print(red("ERROR: Duplicate params proposed by LLM. Retrying..."))
            raise AssertionError("Duplicate params proposed by LLM")

        self.policy.update_policy(new_parameter_list)
        
        # Save reasoning and parameters
        logging_q_filename = f"{logdir}/parameters.txt"
        with open(logging_q_filename, "w") as f:
            f.write(str(self.policy))
        
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        with open(q_reasoning_filename, "w") as f:
            f.write(reasoning)
        
        print(green("Policy updated!"))

        # Run evaluation episodes
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=True)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        logging_file.close()
        
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)
        self.training_episodes += 1

        import time
        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        _parameters = str(new_parameter_list)
        _context_size = context_size
        _num_attempts = attempt_idx + 1
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _parameters, _context_size, _num_attempts


class LLMNumOptimQTableSemanticsAgentWithFeedback(LLMNumOptimQTableSemanticsAgent):
    """Extended Q-table agent that supports human feedback in the training loop."""
    
    def __init__(self, *args, include_feedback_with_params=False, **kwargs):
        """Initialize with feedback support.
        
        Args:
            include_feedback_with_params: If True, include human feedback with params in LLM prompts
        """
        super().__init__(*args, **kwargs)
        self.include_feedback_with_params = include_feedback_with_params
        self.feedback_buffer = {}  # Maps episode index to feedback string
        self.episode_numbers = []  # Tracks which episode number each buffer entry corresponds to
    
    def store_feedback(self, episode: int, feedback: str):
        """Store human feedback for a given episode."""
        self.feedback_buffer[episode] = feedback
    
    def store_episode_number(self, buffer_idx: int, episode: int):
        """Track which episode number corresponds to which buffer index."""
        # Expand list if needed
        while len(self.episode_numbers) <= buffer_idx:
            self.episode_numbers.append(None)
        self.episode_numbers[buffer_idx] = episode
    
    def get_feedback_for_buffer_idx(self, buffer_idx: int) -> str:
        """Retrieve feedback for a buffer index by looking up its episode number."""
        if buffer_idx < len(self.episode_numbers):
            episode = self.episode_numbers[buffer_idx]
            if episode is not None and episode in self.feedback_buffer:
                return self.feedback_buffer[episode]
        return ""
    
    def get_feedback_for_episode(self, episode: int) -> str:
        """Retrieve feedback for an episode, or empty string if not available."""
        return self.feedback_buffer.get(episode, "")
    
    def train_policy_with_feedback(self, world, logdir, attempt_idx=0, human_feedback="", last_policy_params="", request_feedback_prediction=False):
        """Train Q-table policy with human feedback from previous iteration."""
        
        def parse_parameters(input_text):
            import re
            cleaned_text = input_text
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'</?think>', '', cleaned_text)
            cleaned_text = cleaned_text.replace('```', '')
            cleaned_text = cleaned_text.replace('`', '')
            
            first_line = cleaned_text.split("\n")[0] if cleaned_text else ""
            print(blue(f"response: {first_line}"))
            
            pattern1 = re.compile(r"params\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)")
            pattern2 = re.compile(r"params\[(\d+)\]\s*=\s*([+-]?\d+(?:\.\d+)?)")
            
            matches = pattern1.findall(cleaned_text)
            if not matches:
                matches = pattern2.findall(cleaned_text)
            
            param_dict = {}
            for match in matches:
                param_idx = int(match[0])
                param_val = int(float(match[1]))  # Q-table uses integers
                if param_idx < self.rank:
                    param_dict[param_idx] = param_val
            
            results = []
            for i in range(self.rank):
                if i in param_dict:
                    results.append(param_dict[i])
                else:
                    results.append(None)
            
            results = [r for r in results if r is not None]
            print(results)
            
            if len(results) != self.rank:
                from utils.console import gray
                print(f"\n{'='*80}")
                print(f"ERROR: Expected {self.rank} parameters, got {len(results)}")
                print(f"{'='*80}\n")
                assert len(results) == self.rank, f"Expected {self.rank} params, got {len(results)}"
            
            # Validate parameter bounds: all values must be valid actions from action space
            valid_actions = set(self.actions[0])  # For Q-table, actions[0] contains valid action values
            invalid_actions = []
            for i, val in enumerate(results):
                if val not in valid_actions:
                    invalid_actions.append((i, val))
            
            if invalid_actions:
                from utils.console import gray
                print(f"\n{'='*80}")
                print(f"ERROR: {len(invalid_actions)} parameters have INVALID ACTIONS!")
                print(f"Valid actions: {sorted(valid_actions)}")
                print(f"Invalid parameters:")
                for idx, val in invalid_actions[:10]:  # Show first 10
                    print(f"  params[{idx}]: {val}")
                if len(invalid_actions) > 10:
                    print(f"  ... and {len(invalid_actions) - 10} more")
                print(f"{'='*80}")
                print(gray(f"FULL RESPONSE TEXT:\n{input_text}"))
                print(f"{'='*80}\n")
                assert False, f"{len(invalid_actions)} parameters have invalid actions. Valid: {sorted(valid_actions)}"
            
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer, traj_buffer, n):
            if not replay_buffer.buffer:
                return ""

            # Simply iterate through all episodes in the buffer
            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(replay_buffer.buffer))
            
            for idx, (weights, reward) in enumerate(replay_buffer.buffer):
                params = weights.reshape(-1)
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {int(params[i])}; "
                l += f"f(params): {reward:.2f}\n"
                text += l
                
                # Add feedback if available and enabled
                if self.include_feedback_with_params:
                    feedback = self.get_feedback_for_buffer_idx(idx)
                    if feedback:
                        text += f"  FEEDBACK: \"{feedback}\"\n"
                
                # Add trajectory if available and enabled
                if self.include_trajectories and idx < len(traj_buffer.buffer):
                    trajectory = traj_buffer.buffer[idx].get_trajectory()
                    if trajectory:
                        text += "Trajectory: "
                        for state, action, reward in trajectory:
                            text += f"({state},{action}) "
                        text += "\n"
            
            return text

        # Update the policy using llm_brain with feedback
        print("Updating the Q-table policy with human feedback...")
        new_parameter_list, reasoning, api_time, context_size = self.llm_brain.llm_update_parameters_with_feedback(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            num_episodes=self.num_episodes,
            rank=self.rank,
            optimum=self.optimum,
            actions=self.actions,
            num_evaluation_episodes=self.num_evaluation_episodes,
            attempt_idx=attempt_idx,
            human_feedback=human_feedback,
            last_policy_params=last_policy_params,
            request_feedback_prediction=request_feedback_prediction,
        )
        self.api_call_time += api_time

        # Update Q-table mapping
        for i in range(self.rank):
            self.q_table.mapping[i] = int(new_parameter_list[i])
        
        # Save reasoning and parameters
        logging_q_filename = f"{logdir}/q_table.txt"
        with open(logging_q_filename, "w") as f:
            f.write(str(self.q_table.mapping))
        
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        with open(q_reasoning_filename, "w") as f:
            f.write(reasoning)
        
        print(green("Q-table policy updated!"))

        # Run evaluation episodes
        print(f"Rolling out episode {self.training_episodes}...")
        logging_filename = f"{logdir}/training_rollout.txt"
        logging_file = open(logging_filename, "w")
        results = []
        for idx in range(self.num_evaluation_episodes):
            if idx == 0:
                result = self.rollout_episode(world, logging_file, record=True)
            else:
                result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        logging_file.close()
        
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)
        self.training_episodes += 1

        import time
        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        _parameters = str(new_parameter_list)
        _context_size = context_size
        _num_attempts = attempt_idx + 1
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _parameters, _context_size, _num_attempts
