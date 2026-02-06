import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os


def update_training_plot(logdir):
    """
    Read the overall_log.txt file and create a live training progress plot.
    Updates the plot after each episode.
    
    Args:
        logdir: Directory containing overall_log.txt
    """
    log_file = os.path.join(logdir, "overall_log.txt")
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"[PLOT] Log file not found: {log_file}")
        return
    
    try:
        print(f"[PLOT] Reading log file: {log_file}")
        # Read the CSV log file (skipinitialspace removes spaces after commas in headers)
        df = pd.read_csv(log_file, skipinitialspace=True)
        
        if df.empty or len(df) == 0:
            print(f"[PLOT] Log file is empty, skipping plot generation")
            return
        
        print(f"[PLOT] Loaded {len(df)} episodes from log file")
        
        # Calculate per-episode metrics by taking differences from cumulative values
        df['API Time Per Episode'] = df['API Time'].diff().fillna(df['API Time'])
        df['CPU Time Per Episode'] = df['CPU Time'].diff().fillna(df['CPU Time'])
        df['Steps Per Episode'] = df['Total Steps'].diff().fillna(df['Total Steps'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Total Reward (main metric)
        ax1 = axes[0, 0]
        ax1.plot(df['Iteration'], df['Total Reward'], 'b-o', linewidth=2, markersize=5)
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Total Reward', fontsize=11)
        ax1.set_title('Total Reward per Episode', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: API Time (per-episode)
        ax2 = axes[0, 1]
        ax2.plot(df['Iteration'], df['API Time Per Episode'], 'g-s', linewidth=2, markersize=5)
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('API Time (seconds)', fontsize=11)
        ax2.set_title('LLM API Call Time per Episode', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CPU Time (per-episode)
        ax3 = axes[1, 0]
        ax3.plot(df['Iteration'], df['CPU Time Per Episode'], 'r-^', linewidth=2, markersize=5)
        ax3.set_xlabel('Episode', fontsize=11)
        ax3.set_ylabel('CPU Time (seconds)', fontsize=11)
        ax3.set_title('CPU Time per Episode', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Steps Per Episode
        ax4 = axes[1, 1]
        ax4.plot(df['Iteration'], df['Steps Per Episode'], 'm-d', linewidth=2, markersize=5)
        ax4.set_xlabel('Episode', fontsize=11)
        ax4.set_ylabel('Steps', fontsize=11)
        ax4.set_title('Steps per Episode', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(logdir, "training_progress.png")
        print(f"[PLOT] Saving plot to: {plot_file}")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"[PLOT] Plot saved successfully!")
        
    except Exception as e:
        import traceback
        print(f"\n[PLOT ERROR] Could not generate plot!")
        print(f"[PLOT ERROR] Error: {e}")
        print(f"[PLOT ERROR] Traceback:")
        traceback.print_exc()
        print()
