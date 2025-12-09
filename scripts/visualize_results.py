"""
Visualization Script for 5D Chess AI Analysis Results
Creates plots and charts for research findings
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_latest_results(results_dir='../results'):
    """Load the most recent result files"""
    results_dir = Path(results_dir)

    result_files = {
        'performance': sorted(glob.glob(str(results_dir / 'performance_results_*.json'))),
        'learning': sorted(glob.glob(str(results_dir / 'learning_report_*.json'))),
        'optimization': sorted(glob.glob(str(results_dir / 'optimization_results_*.json')))
    }

    loaded = {}
    for key, files in result_files.items():
        if files:
            with open(files[-1], 'r') as f:
                loaded[key] = json.load(f)
                print(f"Loaded {key}: {files[-1]}")

    return loaded

def plot_configuration_comparison(performance_data):
    """Compare different MCTS configurations"""
    if not performance_data:
        return

    configs = list(performance_data.keys())
    metrics = ['white_win_rate', 'avg_moves_per_game', 'avg_search_time_per_move', 'checkmate_rate']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MCTS Configuration Comparison', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        values = [performance_data[cfg]['metrics'][metric] for cfg in configs]

        bars = ax.bar(configs, values, color=sns.color_palette("husl", len(configs)))
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('../results/configuration_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: configuration_comparison.png")
    plt.close()

def plot_search_depth_analysis(learning_data):
    """Visualize search depth vs performance"""
    if not learning_data or 'search_depth_analysis' not in learning_data:
        return

    depth_results = learning_data['search_depth_analysis']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Search Depth Analysis', fontsize=16, fontweight='bold')

    depths = [r['num_searches'] for r in depth_results]

    # Plot 1: Entropy vs Depth
    ax = axes[0, 0]
    entropies = [r['avg_entropy'] for r in depth_results]
    ax.plot(depths, entropies, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Searches')
    ax.set_ylabel('Average Entropy')
    ax.set_title('Decision Uncertainty vs Search Depth')
    ax.grid(True, alpha=0.3)

    # Plot 2: Stability vs Depth
    ax = axes[0, 1]
    stabilities = [r['stability'] for r in depth_results]
    ax.plot(depths, stabilities, marker='s', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Searches')
    ax.set_ylabel('Stability Score')
    ax.set_title('Policy Stability vs Search Depth')
    ax.grid(True, alpha=0.3)

    # Plot 3: Search Time vs Depth
    ax = axes[1, 0]
    times = [r['search_time'] for r in depth_results]
    ax.plot(depths, times, marker='^', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Number of Searches')
    ax.set_ylabel('Search Time (s)')
    ax.set_title('Computational Cost vs Search Depth')
    ax.grid(True, alpha=0.3)

    # Plot 4: Efficiency (Stability/Time)
    ax = axes[1, 1]
    efficiency = [s/t for s, t in zip(stabilities, times)]
    ax.plot(depths, efficiency, marker='D', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Searches')
    ax.set_ylabel('Efficiency (Stability/Time)')
    ax.set_title('Search Efficiency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../results/search_depth_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: search_depth_analysis.png")
    plt.close()

def plot_exploration_exploitation(learning_data):
    """Visualize exploration-exploitation tradeoff"""
    if not learning_data or 'exploration_exploitation_analysis' not in learning_data:
        return

    tradeoff_results = learning_data['exploration_exploitation_analysis']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Exploration-Exploitation Tradeoff', fontsize=16, fontweight='bold')

    c_values = [r['c_value'] for r in tradeoff_results]

    # Plot 1: Explored Positions
    ax = axes[0, 0]
    explored = [r['nonzero_positions'] for r in tradeoff_results]
    ax.plot(c_values, explored, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('C Value (Exploration Parameter)')
    ax.set_ylabel('Positions Explored')
    ax.set_title('Exploration Breadth')
    ax.grid(True, alpha=0.3)

    # Plot 2: Max Probability
    ax = axes[0, 1]
    max_probs = [r['max_prob'] for r in tradeoff_results]
    ax.plot(c_values, max_probs, marker='s', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('C Value')
    ax.set_ylabel('Max Probability')
    ax.set_title('Exploitation Strength')
    ax.grid(True, alpha=0.3)

    # Plot 3: Gini Coefficient
    ax = axes[1, 0]
    ginis = [r['gini_coefficient'] for r in tradeoff_results]
    ax.plot(c_values, ginis, marker='^', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('C Value')
    ax.set_ylabel('Gini Coefficient')
    ax.set_title('Policy Concentration (Higher = More Focused)')
    ax.grid(True, alpha=0.3)

    # Plot 4: Exploration Score
    ax = axes[1, 1]
    scores = [r['exploration_score'] for r in tradeoff_results]
    ax.plot(c_values, scores, marker='D', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('C Value')
    ax.set_ylabel('Exploration Score')
    ax.set_title('Overall Exploration Quality')
    ax.grid(True, alpha=0.3)

    # Highlight optimal
    best_idx = scores.index(max(scores))
    ax.axvline(c_values[best_idx], color='orange', linestyle='--', alpha=0.7, label='Optimal')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../results/exploration_exploitation.png', dpi=300, bbox_inches='tight')
    print("Saved: exploration_exploitation.png")
    plt.close()

def plot_optimization_landscape(optimization_data):
    """Create 2D heatmap of parameter optimization"""
    if not optimization_data or 'all_results' not in optimization_data:
        return

    results = optimization_data['all_results']

    # Extract unique values
    search_depths = sorted(set(r['config']['num_searches'] for r in results))
    c_values = sorted(set(r['config']['C'] for r in results))

    # Create score matrix
    score_matrix = np.zeros((len(c_values), len(search_depths)))

    for r in results:
        i = c_values.index(r['config']['C'])
        j = search_depths.index(r['config']['num_searches'])
        score_matrix[i, j] = r['score']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(score_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(search_depths)))
    ax.set_yticks(range(len(c_values)))
    ax.set_xticklabels(search_depths)
    ax.set_yticklabels([f'{c:.2f}' for c in c_values])

    ax.set_xlabel('Number of Searches', fontsize=12)
    ax.set_ylabel('C Value', fontsize=12)
    ax.set_title('Parameter Optimization Landscape\n(Higher Score = Better)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(c_values)):
        for j in range(len(search_depths)):
            text = ax.text(j, i, f'{score_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    # Mark best configuration
    best_i, best_j = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    ax.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                               fill=False, edgecolor='blue', linewidth=3))

    plt.tight_layout()
    plt.savefig('../results/optimization_landscape.png', dpi=300, bbox_inches='tight')
    print("Saved: optimization_landscape.png")
    plt.close()

def create_summary_dashboard(results):
    """Create comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle('5D Chess AI - Research Summary Dashboard', fontsize=18, fontweight='bold')

    # Add summary text
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')

    summary_text = "COMPREHENSIVE ANALYSIS RESULTS\n\n"

    if 'performance' in results:
        perf = results['performance']
        best_config = max(perf.items(), key=lambda x: x[1]['metrics'].get('checkmate_rate', 0))
        summary_text += f"Best Configuration: {best_config[0]}\n"
        summary_text += f"Checkmate Rate: {best_config[1]['metrics']['checkmate_rate']:.2%}\n"
        summary_text += f"Avg Moves per Game: {best_config[1]['metrics']['avg_moves_per_game']:.1f}\n\n"

    if 'learning' in results and 'recommendations' in results['learning']:
        summary_text += "Key Recommendations:\n"
        for rec in results['learning']['recommendations'][:3]:
            summary_text += f"• {rec}\n"

    ax_text.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig('../results/summary_dashboard.png', dpi=300, bbox_inches='tight')
    print("Saved: summary_dashboard.png")
    plt.close()

def main():
    print("=" * 80)
    print("5D Chess AI - Result Visualization")
    print("=" * 80)

    # Load results
    results = load_latest_results()

    if not results:
        print("No results found. Please run tests first.")
        return

    print("\nGenerating visualizations...")

    # Generate all plots
    if 'performance' in results:
        plot_configuration_comparison(results['performance'])

    if 'learning' in results:
        plot_search_depth_analysis(results['learning'])
        plot_exploration_exploitation(results['learning'])

    if 'optimization' in results:
        plot_optimization_landscape(results['optimization'])

    create_summary_dashboard(results)

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("Check the ../results/ directory for output files.")
    print("=" * 80)

if __name__ == "__main__":
    main()
