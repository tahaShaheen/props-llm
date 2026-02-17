#!/usr/bin/env python3
"""
Evaluate a policy on 10 parallel environments.
FEATURES:
 - Fixes "Last Man Standing" label issue
 - Adds Dynamic Mean Reward display at the bottom
 - Distinguishes between DEAD (Failure) and DONE (Success/Timeout)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium import Wrapper
import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import time
from datetime import datetime

# --- WRAPPER ---
class RenderInInfoWrapper(Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            info['render_rgb'] = self.env.render()
        except Exception:
            info['render_rgb'] = None
        return obs, reward, terminated, truncated, info

# --- UTILS ---
def load_params_from_csv(log_file, episode):
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    df = pd.read_csv(log_file, skipinitialspace=True)
    if episode >= len(df):
        raise ValueError(f"Episode {episode} not found")
    params_str = df.loc[episode, 'Parameters']
    return np.fromstring(params_str.strip('[]'), sep=' ')

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_policy(config, params):
    try:
        from agent.policy.linear_policy import LinearPolicy
        from agent.policy.q_table import QTable
    except ImportError:
        return None

    if config.get('task', '').startswith('dist_state'):
        dim_actions = config['dim_actions']
        dim_states = config['dim_states']
        policy = QTable(dim_states, dim_actions)
        policy.mapping = {i: params[i] for i in range(len(params))}
        return policy
    else:
        dim_actions = config['dim_actions']
        dim_states = config['dim_states']
        policy = LinearPolicy(dim_states, dim_actions)
        policy.update_policy(params)
        return policy

def make_env(env_name, render=False):
    def _make():
        # --- SILENCE WARNINGS IN SUBPROCESS ---
        import warnings
        # Filter the specific gym/numpy bool8 deprecation warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        env = gym.make(env_name, render_mode='rgb_array' if render else None)
        if render:
            env = RenderInInfoWrapper(env)
        return env
    return _make

def run_parallel_envs_fast(env_name, policy, num_envs=10, max_steps=500, render=False, save_trajectories=None):
    print(f"Creating {num_envs} parallel environments...")
    
    envs = AsyncVectorEnv([make_env(env_name, render=render) for _ in range(num_envs)])
    
    states, _ = envs.reset()
    rewards = np.zeros(num_envs)
    # Store history of cumulative rewards for the animation
    reward_history = [] 
    
    frames = [[] for _ in range(num_envs)] if render else None
    active_envs = np.ones(num_envs, dtype=bool)
    
    print(f"\n{'='*80}")
    print(f"STARTING EVALUATION (Max Steps: {max_steps})")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for step in range(max_steps):
        if not np.any(active_envs):
            print(f"\n>> All environments terminated at step {step}.")
            break

        # --- ACTION ---
        if policy and hasattr(policy, 'get_action_batch'):
            actions = policy.get_action_batch(states)
        elif policy and hasattr(policy, 'get_action'):
            # Check policy type to handle discrete vs continuous states
            try:
                from agent.policy.q_table import QTable
                is_q_table = isinstance(policy, QTable)
            except ImportError:
                is_q_table = False
            
            if is_q_table:
                # Q-table: discrete states (ints), get_action returns action directly
                actions = np.array([int(policy.get_action(int(s))) for s in states])
            else:
                # LinearPolicy: reshape state and get action values
                action_values = np.array([policy.get_action(s.reshape(-1, 1)).flatten() for s in states])
                # For discrete action spaces, use argmax; for continuous, use values directly
                if isinstance(envs.single_action_space, gym.spaces.Discrete):
                    actions = np.argmax(action_values, axis=1)
                else:
                    actions = action_values
        elif policy: 
            actions = np.array([policy.mapping[int(s)] for s in states])
        else: 
            actions = envs.action_space.sample()

        # Broadcast mask for both discrete (10,) and continuous (10, N) actions
        mask = active_envs if actions.ndim == 1 else active_envs[:, None]
        actions = np.where(mask, actions, 0)

        # --- LOGGING ---
        # (Simplified logging for speed, since visualizer is key now)
        if step % 50 == 0 or step < 5:
             print(f"Step {step:03d} | Alive: {np.sum(active_envs)}/{num_envs} | Mean Reward: {np.mean(rewards):.1f}")

        # --- STEP ---
        next_states, step_rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = terminateds | truncateds

        # --- UPDATE ---
        for i in range(num_envs):
            if not active_envs[i]:
                continue 

            rewards[i] += step_rewards[i]

            if render:
                if 'render_rgb' in infos and infos['render_rgb'][i] is not None:
                    frames[i].append(infos['render_rgb'][i])
                
                if dones[i]:
                    if "_final_info" in infos and infos["_final_info"][i]:
                        final_info = infos["final_info"][i]
                        if "render_rgb" in final_info and final_info["render_rgb"] is not None:
                            frames[i].append(final_info["render_rgb"])

            if dones[i]:
                print(f"  >> Env {i} DIED. Final Reward: {rewards[i]:.1f}")
                if "_final_observation" in infos and infos["_final_observation"][i]:
                     states[i] = infos["final_observation"][i]
                active_envs[i] = False
            else:
                states[i] = next_states[i]
        
        # Save snapshot of current cumulative rewards for the frame
        reward_history.append(rewards.copy())

    elapsed = time.time() - start_time
    print(f"\n✓ Completed in {elapsed:.2f} seconds")
    
    envs.close()
    
    # Pad reward history to match max frames if needed (for animation sync)
    # But usually animation length depends on max frames captured.
    
    results = {
        'rewards': rewards,
        'frames': frames,
        'reward_history': reward_history
    }
    
    if save_trajectories:
        np.savez(save_trajectories, rewards=rewards)
    
    return results

def visualize_parallel(frames, rewards, reward_history, num_envs, max_steps=500, fps=10, restart_flag=None, save_video_path=None):
    if not frames or not any(frames):
        print("No frames captured.")
        return

    print("Generating visualization...")
    if save_video_path:
        print(f"Will save video to: {save_video_path}")
    print("Press SPACEBAR to RE-RUN the simulation!")
    print("Press 'Q' to CLOSE the window.")
    
    grid_cols = min(num_envs, 5)  # Max 5 columns for better layout
    grid_rows = int(np.ceil(num_envs / grid_cols))
    
    fig = plt.figure(figsize=(15, 4 * grid_rows + 1))
    gs = GridSpec(grid_rows, grid_cols, figure=fig, hspace=0.4, wspace=0.1, bottom=0.15)
    
    axes = []
    imgs = []
    
    for i in range(num_envs):
        ax = fig.add_subplot(gs[i])
        ax.set_title(f"Env {i}\nReward: 0", fontsize=10)
        ax.axis('off')
        axes.append(ax)
        
        if frames[i]:
            img = ax.imshow(frames[i][0])
        else:
            img = ax.imshow(np.zeros((100,100,3)))
        imgs.append(img)
    
    mean_reward_text = fig.text(
        0.5, 0.05, f"Step: 0 | Mean Reward: 0.0", 
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
    )
    
    anim_length = max(len(f) for f in frames)
    
    # Simple iterator that pauses at the end
    class FrameIterator:
        def __init__(self, limit):
            self.current = 0
            self.limit = limit
        def __iter__(self): return self
        def __next__(self):
            if self.current >= self.limit: return self.limit - 1
            val = self.current
            self.current += 1
            return val

    frame_gen = FrameIterator(anim_length)

    def update(frame_idx):
        hist_idx = min(frame_idx, len(reward_history) - 1)
        current_rewards = reward_history[hist_idx]
        current_mean = np.mean(current_rewards)
        
        mean_reward_text.set_text(f"Step: {frame_idx} | Mean Reward: {current_mean:.1f}")
        
        for i in range(num_envs):
            ax = axes[i]
            if frames[i] and frame_idx < len(frames[i]) - 1:
                imgs[i].set_data(frames[i][frame_idx])
                ax.set_title(f"Env {i}\nReward: {current_rewards[i]:.0f}", color='black', fontweight='normal')
            elif frames[i]:
                imgs[i].set_data(frames[i][-1])
                final_reward = rewards[i]
                
                # STRICT DONE/DEAD CHECK
                if len(frames[i]) >= max_steps - 2:
                    ax.set_title(f"Env {i} (DONE)\nReward: {final_reward:.0f}", color='green', fontweight='bold')
                else:
                    ax.set_title(f"Env {i} (DEAD)\nReward: {final_reward:.0f}", color='red', fontweight='bold')

        return imgs + [mean_reward_text]
    
    ani = animation.FuncAnimation(
        fig, update, frames=frame_gen, interval=1000/fps, 
        blit=False, repeat=True, save_count=anim_length, cache_frame_data=False
    )
    
    # --- SAVE VIDEO ---
    if save_video_path:
        print("Saving GIF... (this may take a moment)")
        try:
            ani.save(save_video_path, writer='pillow', fps=fps)
            print(f"✓ GIF saved successfully: {save_video_path}")
        except Exception as e:
            print(f"⚠ Could not save GIF: {e}")
    
    # --- KEY HANDLER ---
    def on_key(event):
        if event.key == ' ':
            print("\n>> RESTARTING SIMULATION...\n")
            if restart_flag is not None:
                restart_flag[0] = True # Signal main loop to restart
            plt.close() # Close window to unblock main loop
        elif event.key == 'q':
            print("\n>> CLOSING WINDOW...\n")
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--num_envs', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save_trajectories', action='store_true')
    
    args = parser.parse_args()
    
    if not args.config or args.episode is None:
        print("Please provide --config and --episode")
        return

    config = load_config(args.config)
    
    # Use max_traj_length from config if max_steps not provided
    if args.max_steps is None:
        args.max_steps = config.get('max_traj_length', 500)
    
    if args.log:
        log_file = args.log
    else:
        logdir = config.get('logdir', 'logs')
        log_file = os.path.join(logdir, 'overall_log.csv')
        
    params = load_params_from_csv(log_file, args.episode)
    policy = create_policy(config, params)
    
    save_file = None
    if args.save_trajectories:
        logdir = config.get('logdir', 'logs')
        save_file = os.path.join(logdir, f'episode_{args.episode}_trajectories.npz')
    
    # Setup video saving directory and path
    logdir = config.get('logdir', 'logs')
    # Extract environment name from logdir (e.g., 'logs/frozenlake_propsp' -> 'frozenlake_propsp')
    env_name = os.path.basename(logdir)
    video_dir = os.path.join('videos', env_name)
    os.makedirs(video_dir, exist_ok=True)

    # --- EXECUTION LOOP ---
    while True:
        results = run_parallel_envs_fast(
            config['gym_env_name'],
            policy,
            num_envs=args.num_envs,
            max_steps=args.max_steps,
            render=args.render,
            save_trajectories=save_file
        )
        
        print(f"\nFinal Stats: Mean Reward: {np.mean(results['rewards']):.2f}")

        if args.render:
            # Create a mutable flag to pass into the visualizer
            # [False] means "Do not restart" by default
            restart_simulation = [False]
            
            # Generate timestamp-based video filename (YYMMDDHHMMSS format)
            timestamp = datetime.now().strftime('%y%m%d%H%M%S')
            video_path = os.path.join(video_dir, f'{timestamp}.gif')
            
            visualize_parallel(
                results['frames'], 
                results['rewards'], 
                results['reward_history'], 
                args.num_envs,
                max_steps=args.max_steps,
                restart_flag=restart_simulation,
                save_video_path=video_path
            )
            
            # Check if user pressed Spacebar (restart_flag becomes True)
            if not restart_simulation[0]:
                print("Exiting...")
                break # Break loop if window closed normally
        else:
            break # Run once and exit if not rendering

if __name__ == '__main__':
    main()