import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
import argparse
import sys

# --- CONFIGURATION ---
# How often to check for new data (in milliseconds)
UPDATE_INTERVAL = 2000  

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="""
        -----------------------------------------------------------------------
        LIVE REWARD PLOTTER
        -----------------------------------------------------------------------
        Opens a live, interactive graph of training rewards that updates 
        automatically as your training script writes to 'overall_log.csv'.
        
        FEATURES:
          - Auto-updates every 2 seconds.
          - PRESERSVES ZOOM: You can zoom into a specific Y-range (to hide 
            huge spikes) and the graph will continue scrolling sideways 
            without resetting your view.
        -----------------------------------------------------------------------
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        '--log', 
        type=str, 
        required=True, 
        help="Path to the log file (e.g., 'logs/overall_log.csv') or the folder containing it."
    )
    
    return parser.parse_args()

def get_log_path(user_path):
    """
    Smartly determines if the user passed a folder or a file.
    """
    if os.path.isdir(user_path):
        return os.path.join(user_path, "overall_log.csv")
    return user_path

def get_data(log_file):
    """
    Reads the CSV, handles errors, and returns X and Y arrays.
    """
    if not os.path.exists(log_file):
        return None, None

    try:
        # 'on_bad_lines' skips lines that might be half-written during the read
        df = pd.read_csv(log_file, skipinitialspace=True, on_bad_lines='skip')
        
        # Clean data (ensure numeric, drop NaNs)
        df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
        df = df.dropna(subset=['Iteration'])
        df = df.sort_values('Iteration').reset_index(drop=True)
        
        if df.empty:
            return None, None
            
        return df['Iteration'], df['Total Reward']
    except Exception:
        # Return None if file is currently being written to/locked
        return None, None

def main():
    args = parse_arguments()
    log_file = get_log_path(args.log)

    print(f"--> Looking for log file at: {log_file}")
    
    # Wait for file to exist before popping up window
    if not os.path.exists(log_file):
        print("    Waiting for file to be created...", end='\r')
        # We allow the plot to open anyway, it will just be empty until data arrives

    # Setup the Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title(f'Live Monitor: {os.path.basename(log_file)}')
    
    # Create an empty line object that we will update
    line, = ax.plot([], [], 'b-o', markersize=4, alpha=0.7, label='Total Reward')
    
    ax.set_title('Live Training Rewards', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    # Add a text box for the latest value
    status_text = ax.text(0.02, 0.95, 'Waiting for data...', transform=ax.transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Helper to track if we have set the initial view
    plot_state = {"initial_zoom_set": False}

    def update(frame):
        x, y = get_data(log_file)
        
        if x is None or len(x) == 0:
            return line, status_text

        # Update the data in the line
        line.set_data(x, y)
        
        # --- SMART SCROLLING LOGIC ---
        
        # Handle Axes (Only set limits ONCE, then leave user alone)
        if not plot_state["initial_zoom_set"]:
            # Set Y limits
            margin_y = (y.max() - y.min()) * 0.1
            if margin_y == 0: margin_y = 1.0
            ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
            
            # Set X limits (Just once)
            ax.set_xlim(x.min(), x.max() * 1.05)
            
            plot_state["initial_zoom_set"] = True

        # Update the text box
        latest_ep = int(x.iloc[-1])
        latest_rew = y.iloc[-1]
        status_text.set_text(f"Episode: {latest_ep}\nReward: {latest_rew:.4f}")

        return line, status_text

    # Start the animation
    ani = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL, cache_frame_data=False)
    
    print("--> Plot opened. Close the window to stop.")
    plt.show()

if __name__ == "__main__":
    main()