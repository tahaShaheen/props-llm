import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import re


def update_training_plot(logdir):
	"""
	Read the overall_log.csv file and create a live training progress plot.
	Updates the plot after each episode.
    
	Args:
		logdir: Directory containing overall_log.csv
	"""
	log_file = os.path.join(logdir, "overall_log.csv")
    
	# Check if log file exists
	if not os.path.exists(log_file):
		print(f"[PLOT] Log file not found: {log_file}")
		return
    
	try:
		def parse_parameter_vector(value):
			if pd.isna(value):
				return None
			text = str(value).strip()
			if not text:
				return None
			nums = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
			if not nums:
				return None
			return np.array([float(x) for x in nums], dtype=float)

		print(f"[PLOT] Reading log file: {log_file}")
		# Read the CSV log file (on_bad_lines='skip' handles multiline parameter arrays)
		df = pd.read_csv(log_file, skipinitialspace=True, on_bad_lines='skip')
        
		if df.empty or len(df) == 0:
			print(f"[PLOT] Log file is empty, skipping plot generation")
			return
        
		print(f"[PLOT] Loaded {len(df)} episodes from log file")
		
		# Convert Iteration to numeric and drop any invalid rows
		df['Iteration'] = pd.to_numeric(df['Iteration'], errors='coerce')
		df['Total Reward'] = pd.to_numeric(df['Total Reward'], errors='coerce')
		if 'Guessed Reward' in df.columns:
			df['Guessed Reward'] = pd.to_numeric(df['Guessed Reward'], errors='coerce')
		df = df.dropna(subset=['Iteration'])
		df = df.sort_values('Iteration').reset_index(drop=True)
        
		# Calculate per-episode metrics by taking differences from cumulative values
		df['API Time Per Episode'] = df['API Time'].diff().fillna(df['API Time'])
		df['CPU Time Per Episode'] = df['CPU Time'].diff().fillna(df['CPU Time'])
		df['Steps Per Episode'] = df['Total Steps'].diff().fillna(df['Total Steps'])
        
		# Create figure with subplots (4x2 grid, includes reward prediction error)
		fig, axes = plt.subplots(4, 2, figsize=(14, 19))
		fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
		# Plot 1: Actual vs Predicted Reward (same y-axis)
		ax1 = axes[0, 0]
		ax1.plot(df['Iteration'], df['Total Reward'], 'b-o', linewidth=2, markersize=5, label='Actual Reward')
		if 'Guessed Reward' in df.columns and df['Guessed Reward'].notna().any():
			ax1.plot(df['Iteration'], df['Guessed Reward'], 'g--s', linewidth=2, markersize=4, label='Predicted Reward')
		ax1.set_xlabel('Episode', fontsize=11)
		ax1.set_ylabel('Total Reward', fontsize=11)
		ax1.set_title('Actual vs Predicted Reward per Episode', fontsize=12, fontweight='bold')
		ax1.grid(True, alpha=0.3)
		ax1.legend(loc='best', fontsize=9)
        
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
        
		# Plot 5: Context Size
		if 'Context Size' in df.columns:
			ax5 = axes[2, 0]
			ax5.plot(df['Iteration'], df['Context Size'], 'c-p', linewidth=2, markersize=5)
			ax5.set_xlabel('Episode', fontsize=11)
			ax5.set_ylabel('Context Size (tokens)', fontsize=11)
			ax5.set_title('Context Size per Episode', fontsize=12, fontweight='bold')
			ax5.grid(True, alpha=0.3)
		else:
			axes[2, 0].axis('off')
        
		# Plot 6: Number of Attempts (Failure Rate)
		if 'Num Attempts' in df.columns:
			ax6 = axes[2, 1]
			ax6.plot(df['Iteration'], df['Num Attempts'], 'orange', marker='h', linewidth=2, markersize=5)
			ax6.set_xlabel('Episode', fontsize=11)
			ax6.set_ylabel('Number of Attempts', fontsize=11)
			ax6.set_title('Parse Attempts per Episode', fontsize=12, fontweight='bold')
			ax6.grid(True, alpha=0.3)
			ax6.set_ylim(bottom=0.5)
		else:
			axes[2, 1].axis('off')

		# Plot 7: Reward Prediction Error (signed and absolute)
		ax7 = axes[3, 0]
		if 'Guessed Reward' in df.columns and df['Guessed Reward'].notna().any():
			valid_df = df.dropna(subset=['Total Reward', 'Guessed Reward']).copy()
			if not valid_df.empty:
				valid_df['Reward Error'] = valid_df['Total Reward'] - valid_df['Guessed Reward']
				valid_df['Abs Reward Error'] = valid_df['Reward Error'].abs()
				ax7.plot(valid_df['Iteration'], valid_df['Reward Error'], color='tab:red', marker='o', linewidth=2, markersize=4, label='Signed Error (Actual - Predicted)')
				ax7.plot(valid_df['Iteration'], valid_df['Abs Reward Error'], color='tab:purple', linestyle='--', marker='x', linewidth=1.5, markersize=4, label='Absolute Error |Actual - Predicted|')
				ax7.axhline(0, color='black', linewidth=1, alpha=0.6)
				mae = valid_df['Abs Reward Error'].mean()
				ax7.set_title(f'Reward Prediction Error (MAE: {mae:.2f})', fontsize=12, fontweight='bold')
			else:
				ax7.text(0.5, 0.5, 'No valid reward pairs yet', ha='center', va='center', transform=ax7.transAxes)
				ax7.set_title('Reward Prediction Error', fontsize=12, fontweight='bold')
		else:
			ax7.text(0.5, 0.5, 'Guessed Reward column not available', ha='center', va='center', transform=ax7.transAxes)
			ax7.set_title('Reward Prediction Error', fontsize=12, fontweight='bold')
		ax7.set_xlabel('Episode', fontsize=11)
		ax7.set_ylabel('Error', fontsize=11)
		ax7.grid(True, alpha=0.3)
		if ax7.lines:
			ax7.legend(loc='best', fontsize=8)

		# Plot 8: Reserved for future metrics
		axes[3, 1].axis('off')
        
		# Adjust layout to prevent overlap
		plt.tight_layout()
        
		# Save the plot
		plot_file = os.path.join(logdir, "training_progress.png")
		print(f"[PLOT] Saving plot to: {plot_file}")
		plt.savefig(plot_file, dpi=100, bbox_inches='tight')
		plt.close()
		print(f"[PLOT] Plot saved successfully!")

		# Save a separate parameter-over-time heatmap (independent from training_progress.png)
		if 'Parameters' in df.columns:
			param_rows = []
			for _, row in df.iterrows():
				params = parse_parameter_vector(row.get('Parameters'))
				if params is None:
					continue
				iteration = pd.to_numeric(row.get('Iteration'), errors='coerce')
				if pd.isna(iteration):
					continue
				param_rows.append((int(iteration), params))

			if param_rows:
				param_rows.sort(key=lambda x: x[0])
				max_dim = max(len(p) for _, p in param_rows)
				matrix = np.full((len(param_rows), max_dim), np.nan, dtype=float)
				iterations = []
				for i, (it, p) in enumerate(param_rows):
					matrix[i, :len(p)] = p
					iterations.append(it)

				fig2, ax = plt.subplots(figsize=(14, 6))
				im = ax.imshow(
					matrix.T,
					aspect='auto',
					origin='upper',
					interpolation='nearest',
					cmap='viridis'
				)
				ax.set_title('Parameter History (executed iterations)', fontsize=12, fontweight='bold')
				ax.set_xlabel('Iteration', fontsize=10)
				ax.set_ylabel('Parameter Index', fontsize=10)

				if len(iterations) > 1:
					tick_count = min(10, len(iterations))
					tick_positions = np.linspace(0, len(iterations) - 1, num=tick_count, dtype=int)
					ax.set_xticks(tick_positions)
					ax.set_xticklabels([str(iterations[idx]) for idx in tick_positions])
				else:
					ax.set_xticks([0])
					ax.set_xticklabels([str(iterations[0])])

				cbar = plt.colorbar(im, ax=ax)
				cbar.set_label('Parameter Value', fontsize=10)

				params_plot_file = os.path.join(logdir, "params_over_time.png")
				print(f"[PLOT] Saving params heatmap to: {params_plot_file}")
				plt.tight_layout()
				plt.savefig(params_plot_file, dpi=120, bbox_inches='tight')
				plt.close(fig2)
				print("[PLOT] Params heatmap saved successfully!")
        
	except Exception as e:
		import traceback
		print(f"\n[PLOT ERROR] Could not generate plot!")
		print(f"[PLOT ERROR] Error: {e}")
		print(f"[PLOT ERROR] Traceback:")
		traceback.print_exc()
		print()
