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


def get_human_feedback(episode: int, reward: float, config_path: str, logdir: str) -> str:
    """
    Show the policy execution to the human and collect feedback.
    
    Args:
        episode: Current episode number
        reward: Average reward from evaluation
        config_path: Path to the config file for eval_parallel_policy
        logdir: Log directory where overall_log.csv is stored
        
    Returns:
        Human feedback string (empty string if no feedback)
    """
    print("\n" + "=" * 80)
    print(blue(f"EPISODE {episode} COMPLETE - Average Reward: {reward:.2f}"))
    print("=" * 80)
    
    # Run the visualization
    print(yellow("\nLaunching policy visualization..."))
    print(yellow("Watch the policy execute, then close the window to provide feedback.\n"))
    
    try:
        # Run eval_parallel_policy.py with the current episode
        log_file = os.path.join(logdir, "overall_log.csv")
        cmd = [
            sys.executable,
            "eval_parallel_policy.py",
            "--config", config_path,
            "--episode", str(episode),
            "--render",
            "--num_envs", "5",
        ]
        
        # Run and wait for the visualization to complete
        result = subprocess.run(cmd, capture_output=False)
        
    except Exception as e:
        print(red(f"Warning: Could not launch visualization: {e}"))
        print(yellow("Continuing without visualization..."))
    
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
    include_trajectories=True,
    include_feedback_with_params=False,  # NEW: Toggle for feedback display
    feedback_interval=5,  # NEW: Ask for feedback every N episodes (only for best params)
    param_min=None,  # NEW: Minimum parameter value for validation (continuous tasks)
    param_max=None,  # NEW: Maximum parameter value for validation (continuous tasks)
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

    jinja2_env = Environment(loader=FileSystemLoader(template_dir))
    llm_si_template = jinja2_env.get_template(llm_si_template_name)
    llm_output_conversion_template = jinja2_env.get_template(
        llm_output_conversion_template_name
    )
    
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
            include_trajectories=include_trajectories,
            include_feedback_with_params=include_feedback_with_params,
            param_min=param_min,
            param_max=param_max,
        )

    print('init done')

    if not warmup_dir:
        warmup_dir = f"{logdir}/warmup"
        os.makedirs(warmup_dir, exist_ok=True)
        agent.random_warmup(world, warmup_dir, warmup_episodes)
    else:
        agent.replay_buffer.load(warmup_dir)
    
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

    overall_log_file = open(overall_log_path, "a" if start_episode > 0 else "w")
    if start_episode == 0:
        overall_log_file.write("Iteration, CPU Time, API Time, Total Episodes, Total Steps, Total Reward, Context Size, Num Attempts, Human Feedback, Parameters\n")
        overall_log_file.flush()

    # Initialize feedback log
    feedback_log_file = open(feedback_log_path, "a" if start_episode > 0 else "w")
    if start_episode == 0:
        feedback_log_file.write("Iteration, Reward, Human Feedback\n")
        feedback_log_file.flush()

    # Track feedback for intervals
    interval_best_episode = None
    interval_best_reward = float('-inf')
    
    # Initialize feedback tracking across iterations
    current_feedback = ""
    last_policy_params = ""
    
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
                cpu_time, api_time, total_episodes, total_steps, total_reward, parameters, context_size, num_attempts = agent.train_policy_with_feedback(
                    world, 
                    curr_episode_dir, 
                    attempt_idx=trial_idx,
                    human_feedback=current_feedback,
                    last_policy_params=last_policy_params,
                )
                
                # Format parameters for CSV
                formatted_parameters = format_parameters_for_csv(parameters)
                
                # Escape feedback for CSV (replace quotes and commas)
                escaped_feedback = current_feedback.replace('"', '""').replace(',', ';')
                
                overall_log_file.write(
                    f"{episode}, {cpu_time}, {api_time}, {total_episodes}, {total_steps}, {total_reward}, {context_size}, {num_attempts}, \"{escaped_feedback}\", {formatted_parameters}\n"
                )
                overall_log_file.flush()
                
                # Log feedback separately
                feedback_log_file.write(f"{episode}, {total_reward}, \"{escaped_feedback}\"\n")
                feedback_log_file.flush()
                
                # Save feedback to episode dir
                if current_feedback:
                    feedback_file_path = f"{curr_episode_dir}/human_feedback.txt"
                    with open(feedback_file_path, "w") as f:
                        f.write(current_feedback)
                
                # Update training progress plot
                update_training_plot(logdir)
                print(green(f"{trial_idx + 1}th trial attempt succeeded in training"))
                
                # Store current params as last_policy_params for next iteration
                last_policy_params = formatted_parameters
                
                # Track best in current interval
                if total_reward > interval_best_reward:
                    interval_best_reward = total_reward
                    interval_best_episode = episode
                
                # Track which episode number this buffer entry corresponds to
                # Buffer size = training_episodes - 1 (0-indexed)
                agent.store_episode_number(agent.training_episodes - 1, episode)
                
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
            )
            
            # Store feedback in agent for inclusion with params in future iterations
            if current_feedback:
                print(f"[INFO] Storing feedback for episode {interval_best_episode}")
            agent.store_feedback(interval_best_episode, current_feedback)
            
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
    
    def train_policy_with_feedback(self, world, logdir, attempt_idx=0, human_feedback="", last_policy_params=""):
        """Train policy with human feedback from previous iteration."""
        
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
        )
        self.api_call_time += api_time

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
    
    def train_policy_with_feedback(self, world, logdir, attempt_idx=0, human_feedback="", last_policy_params=""):
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
