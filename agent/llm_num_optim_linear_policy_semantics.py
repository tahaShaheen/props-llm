from agent.policy.linear_policy_no_bias import LinearPolicy as LinearPolicyNoBias
from agent.policy.linear_policy import LinearPolicy
from agent.policy.replay_buffer import EpisodeRewardBufferNoBias
from agent.policy.replay_buffer import ReplayBuffer
from agent.policy.llm_brain_linear_policy import LLMBrain
from world.base_world import BaseWorld
import numpy as np
import re
import time


class LLMNumOptimSemanticAgent:
    def __init__(
        self,
        logdir,
        dim_action,
        dim_state,
        max_traj_count,
        max_traj_length,
        llm_si_template,
        llm_output_conversion_template,
        llm_model_name,
        num_evaluation_episodes,
        bias,
        optimum,
        search_step_size,
        env_desc_file=None,
        num_episodes=400,
        ollama_num_ctx=4096,
        buffer_top_k=15,
        buffer_recent_j=5,
    ):
        self.start_time = time.process_time()
        self.api_call_time = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.bias = bias
        self.optimum = optimum
        self.search_step_size = search_step_size
        self.env_desc_file = env_desc_file
        self.num_episodes = num_episodes
        self.buffer_top_k = buffer_top_k
        self.buffer_recent_j = buffer_recent_j

        if not self.bias:
            param_count = dim_action * dim_state
        else:
            param_count = dim_action * dim_state + dim_action
        self.rank = param_count

        if not self.bias:
            self.policy = LinearPolicyNoBias(
                dim_actions=dim_action, dim_states=dim_state
            )
        else:
            self.policy = LinearPolicy(dim_actions=dim_action, dim_states=dim_state)
        self.replay_buffer = EpisodeRewardBufferNoBias(max_size=max_traj_count)
        self.traj_buffer = ReplayBuffer(max_traj_count, max_traj_length)
        self.llm_brain = LLMBrain(
            llm_si_template,
            llm_output_conversion_template,
            llm_model_name,
            ollama_num_ctx=ollama_num_ctx,
        )
        self.logdir = logdir
        self.num_evaluation_episodes = num_evaluation_episodes
        self.training_episodes = 0

        if self.bias:
            self.dim_state += 1

    def rollout_episode(self, world: BaseWorld, logging_file, record=True):
        state = world.reset()
        state = np.expand_dims(state, axis=0)
        logging_file.write(
            f"{', '.join([str(x) for x in self.policy.get_parameters().reshape(-1)])}\n"
        )
        logging_file.write(f"parameter ends\n\n")
        logging_file.write(f"state | action | reward\n")
        done = False
        step_idx = 0
        if record:
            self.traj_buffer.start_new_trajectory()
        while not done:
            action = self.policy.get_action(state.T)
            action = np.reshape(action, (1, self.dim_action))
            if world.discretize:
                action = np.argmax(action)
                action = np.array([action])
            next_state, reward, done = world.step(action)
            logging_file.write(f"{state.T[0]} | {action[0]} | {reward}\n")
            if record:
                self.traj_buffer.add_step(state, action, reward)
            state = next_state
            step_idx += 1
            self.total_steps += 1
        logging_file.write(f"Total reward: {world.get_accu_reward()}\n")
        self.total_episodes += 1
        return world.get_accu_reward()

    def random_warmup(self, world: BaseWorld, logdir, num_episodes):
        for episode in range(num_episodes):
            self.policy.initialize_policy()
            # Run the episode and collect the trajectory
            print(f"Rolling out warmup episode {episode}...")
            logging_filename = f"{logdir}/warmup_rollout_{episode}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file)
            self.replay_buffer.add(
                np.array(self.policy.get_parameters()).reshape(-1), world.get_accu_reward()
            )
            logging_file.close()
            print(f"Result: {result}")
        # self.replay_buffer.sort()

    def train_policy(self, world: BaseWorld, logdir, attempt_idx=0):

        def parse_parameters(input_text):
            # Clean up the input text - remove common formatting artifacts
            cleaned_text = input_text
            
            # Remove <think> tags (from reasoning models like DeepSeek-R1, Qwen)
            cleaned_text = re.sub(r'<think>.*?</think>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'</?think>', '', cleaned_text)
            
            # Remove code block markers
            cleaned_text = cleaned_text.replace('```', '')
            cleaned_text = cleaned_text.replace('`', '')
            
            # Print first line for debugging
            first_line = cleaned_text.split("\n")[0] if cleaned_text else ""
            print("response:", first_line)
            
            # Try multiple regex patterns to be flexible with format
            # Pattern 1: params[X]: value  (original format)
            pattern1 = re.compile(r"params\[(\d+)\]\s*:\s*([+-]?\d+(?:\.\d+)?)")
            # Pattern 2: params[X] = value  (alternative format)
            pattern2 = re.compile(r"params\[(\d+)\]\s*=\s*([+-]?\d+(?:\.\d+)?)")
            
            matches = pattern1.findall(cleaned_text)
            if not matches:
                matches = pattern2.findall(cleaned_text)
            
            # Convert matched strings to float, indexed by parameter number
            # This ensures we only get params[0] through params[rank-1]
            param_dict = {}
            for match in matches:
                param_idx = int(match[0])
                param_val = float(match[1])
                # Only accept valid parameter indices
                if param_idx < self.rank:
                    param_dict[param_idx] = param_val
            
            # If pattern matching failed, try to extract plain numbers
            if not param_dict:
                # Look for lines with space/comma-separated numbers
                number_pattern = re.compile(r'[+-]?\d+(?:\.\d+)?')
                
                for line in cleaned_text.split('\n'):
                    # Skip lines that look like explanations (too long or contain many words)
                    if len(line.strip()) > 150 or line.lower().count('exploration') > 0 or line.lower().count('parameter') > 0:
                        continue
                    
                    # Try to find numbers in this line
                    numbers = number_pattern.findall(line)
                    
                    # If we found enough numbers, use them
                    if len(numbers) >= self.rank:
                        try:
                            for i in range(self.rank):
                                param_dict[i] = float(numbers[i])
                            break
                        except (ValueError, IndexError):
                            continue
            
            # Build results array in order
            results = []
            for i in range(self.rank):
                if i in param_dict:
                    results.append(param_dict[i])
                else:
                    # Missing parameter - this will trigger the error below
                    results.append(None)
            
            # Remove None values for error checking
            results = [r for r in results if r is not None]
            
            print(results)
            
            # More informative error if parsing fails
            if len(results) != self.rank:
                print(f"\n{'='*80}")
                print(f"ERROR: Expected {self.rank} parameters, got {len(results)}")
                print(f"Parsed param indices: {sorted(param_dict.keys())}")
                print(f"{'='*80}")
                print(f"FULL RESPONSE TEXT:\n{input_text}")
                print(f"{'='*80}\n")
                assert len(results) == self.rank, f"Expected {self.rank} params, got {len(results)}"
            
            return np.array(results).reshape(-1)

        def str_nd_examples(replay_buffer: EpisodeRewardBufferNoBias, traj_buffer: ReplayBuffer, n):
            if not replay_buffer.buffer:
                return ""

            episodes = []
            for idx, (weights, reward) in enumerate(replay_buffer.buffer):
                episodes.append({
                    "idx": idx,
                    "params": weights.reshape(-1),
                    "reward": reward,
                })

            top_k = max(0, int(self.buffer_top_k))
            recent_j = max(0, int(self.buffer_recent_j))

            best_episodes = sorted(episodes, key=lambda x: x["reward"], reverse=True)[:top_k]
            recent_episodes = episodes[-recent_j:] if recent_j > 0 else []

            seen_params = set()
            final_list = []
            source_list = []  # Track which section each example comes from

            def add_unique(items, source):
                for ep in items:
                    param_sig = tuple(round(p, 1) for p in ep["params"])
                    if param_sig not in seen_params:
                        seen_params.add(param_sig)
                        final_list.append(ep)
                        source_list.append(source)

            add_unique(best_episodes, "top")
            if recent_episodes:
                add_unique(recent_episodes[:-1], "recent")
                final_list.append(recent_episodes[-1])
                source_list.append("recent")

            text = ""
            print('Num trajs in buffer:', len(traj_buffer.buffer))
            print('Num params in buffer:', len(episodes))
            
            current_section = None
            for ep, source in zip(final_list, source_list):
                # Add section header when transitioning
                if source != current_section:
                    if source == "top":
                        text += f"--- TOP {len([s for s in source_list if s == 'top'])} BEST PERFORMING PARAMS ---\n"
                    elif source == "recent":
                        text += f"--- MOST RECENT {len([s for s in source_list if s == 'recent'])} PARAMS ---\n"
                    current_section = source
                
                l = ""
                for i in range(n):
                    l += f"params[{i}]: {ep['params'][i]:.5g}; "
                l += f"f(params): {ep['reward']:.2f}\n"
                text += l
            return text

        # Update the policy using llm_brain, q_table and replay_buffer
        print("Updating the policy...")
        new_parameter_list, reasoning, api_time = self.llm_brain.llm_update_parameters_num_optim_semantics(
            str_nd_examples(self.replay_buffer, self.traj_buffer, self.rank),
            parse_parameters,
            self.training_episodes,
            self.env_desc_file,
            self.num_episodes,
            self.rank,
            self.optimum,
            self.search_step_size,
            attempt_idx=attempt_idx,
            buffer_top_k=self.buffer_top_k,
            buffer_recent_j=self.buffer_recent_j,
        )
        self.api_call_time += api_time

        print(self.policy.get_parameters().shape)
        print(new_parameter_list.shape)
        self.policy.update_policy(new_parameter_list)
        print(self.policy.get_parameters().shape)
        
        # Only save reasoning and parameters if parsing was successful
        # (parse_parameters would have raised AssertionError on failure,
        # which is caught by the runner and logged to terminal only)
        logging_q_filename = f"{logdir}/parameters.txt"
        logging_q_file = open(logging_q_filename, "w")
        logging_q_file.write(str(self.policy))
        logging_q_file.close()
        q_reasoning_filename = f"{logdir}/parameters_reasoning.txt"
        q_reasoning_file = open(q_reasoning_filename, "w")
        q_reasoning_file.write(reasoning)
        q_reasoning_file.close()
        print("Policy updated!")

        # Run the episode and collect the trajectory
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
        print(f"Results: {results}")
        result = np.mean(results)
        self.replay_buffer.add(new_parameter_list, result)
        # self.replay_buffer.sort()
        self.training_episodes += 1

        _cpu_time = time.process_time() - self.start_time
        _api_time = self.api_call_time
        _total_episodes = self.total_episodes
        _total_steps = self.total_steps
        _total_reward = result
        _parameters = str(new_parameter_list)
        return _cpu_time, _api_time, _total_episodes, _total_steps, _total_reward, _parameters


    def evaluate_policy(self, world: BaseWorld, logdir):
        results = []
        for idx in range(self.num_evaluation_episodes):
            logging_filename = f"{logdir}/evaluation_rollout_{idx}.txt"
            logging_file = open(logging_filename, "w")
            result = self.rollout_episode(world, logging_file, record=False)
            results.append(result)
        return results
