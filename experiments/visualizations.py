"""
Generate comprehensive visualizations for research paper
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']


def load_experiment_data(experiment_name):
    """Load experiment data from JSON file"""
    try:
        with open(f'/home/claude/work/repo/results/{experiment_name}.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {experiment_name}.json not found")
        return None


def plot_baseline_comparison():
    """Figure 1: Baseline MCTS performance comparison"""
    data = load_experiment_data('experiment1_baseline')
    if data is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Extract player results
    players = list(data['player_results'].keys())
    wins = [data['player_results'][p]['wins'] for p in players]
    losses = [data['player_results'][p]['losses'] for p in players]
    draws = [data['player_results'][p]['draws'] for p in players]

    # Plot 1: Win rates
    ax = axes[0, 0]
    total_games = [w + l + d for w, l, d in zip(wins, losses, draws)]
    win_rates = [w / t if t > 0 else 0 for w, t in zip(wins, total_games)]
    ax.bar(players, win_rates, color='steelblue', alpha=0.8)
    ax.set_ylabel('Win Rate')
    ax.set_title('(a) Win Rates by Configuration')
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Game length
    ax = axes[0, 1]
    if 'overall_metrics' in data:
        game_length = data['overall_metrics'].get('avg_game_length', 0)
        ax.bar(['Average'], [game_length], color='coral', alpha=0.8)
        ax.set_ylabel('Average Moves per Game')
        ax.set_title('(b) Average Game Length')

    # Plot 3: Search time
    ax = axes[1, 0]
    if 'overall_metrics' in data:
        search_time = data['overall_metrics'].get('avg_search_time', 0)
        ax.bar(['Average'], [search_time * 1000], color='mediumseagreen', alpha=0.8)
        ax.set_ylabel('Time (ms)')
        ax.set_title('(c) Average Search Time')

    # Plot 4: Game outcomes
    ax = axes[1, 1]
    if 'overall_metrics' in data and 'win_rates' in data['overall_metrics']:
        outcomes = data['overall_metrics']['win_rates']
        labels = list(outcomes.keys())
        values = list(outcomes.values())
        colors = ['steelblue', 'coral', 'lightgray']
        ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('(d) Game Outcome Distribution')

    plt.tight_layout()
    plt.savefig('/home/claude/work/repo/docs/figures/figure1_baseline_comparison.pdf', bbox_inches='tight')
    plt.savefig('/home/claude/work/repo/docs/figures/figure1_baseline_comparison.png', bbox_inches='tight')
    plt.close()

    print("Generated Figure 1: Baseline comparison")


def plot_architecture_comparison():
    """Figure 2: Network architecture performance"""
    data = load_experiment_data('experiment2_architecture')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    players = list(data['player_results'].keys())
    wins = [data['player_results'][p]['wins'] for p in players]
    losses = [data['player_results'][p]['losses'] for p in players]
    draws = [data['player_results'][p]['draws'] for p in players]

    # Plot 1: Comparative win rates
    ax = axes[0]
    total = [w + l + d for w, l, d in zip(wins, losses, draws)]
    win_rates = [w / t if t > 0 else 0 for w, t in zip(wins, total)]

    colors = ['steelblue' if 'AlphaZero' in p else 'coral' for p in players]
    bars = ax.bar(range(len(players)), win_rates, color=colors, alpha=0.8)
    ax.set_xticks(range(len(players)))
    ax.set_xticklabels(players, rotation=45, ha='right')
    ax.set_ylabel('Win Rate')
    ax.set_title('(a) Architecture Performance Comparison')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()

    # Plot 2: Performance vs model size
    ax = axes[1]
    model_sizes = []
    model_wins = []

    for p in players:
        if 'small' in p.lower():
            model_sizes.append(5 * 128)  # res_blocks * channels
            model_wins.append(win_rates[players.index(p)])
        elif 'medium' in p.lower():
            model_sizes.append(10 * 256)
            model_wins.append(win_rates[players.index(p)])
        elif 'large' in p.lower():
            model_sizes.append(15 * 512)
            model_wins.append(win_rates[players.index(p)])

    if model_sizes:
        ax.scatter(model_sizes, model_wins, s=200, alpha=0.6, color='steelblue')
        ax.plot(model_sizes, model_wins, 'b--', alpha=0.3)
        ax.set_xlabel('Model Complexity (ResBlocks × Channels)')
        ax.set_ylabel('Win Rate')
        ax.set_title('(b) Performance vs Model Complexity')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/work/repo/docs/figures/figure2_architecture_comparison.pdf', bbox_inches='tight')
    plt.savefig('/home/claude/work/repo/docs/figures/figure2_architecture_comparison.png', bbox_inches='tight')
    plt.close()

    print("Generated Figure 2: Architecture comparison")


def plot_search_benchmark():
    """Figure 3: Search algorithm benchmark results"""
    data = load_experiment_data('experiment4_benchmark')
    if data is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    algorithms = list(data.keys())

    # Plot 1: Search time comparison
    ax = axes[0, 0]
    search_times = [data[algo]['avg_search_time'] * 1000 for algo in algorithms]
    colors = sns.color_palette("husl", len(algorithms))
    ax.bar(algorithms, search_times, color=colors, alpha=0.8)
    ax.set_ylabel('Time (ms)')
    ax.set_title('(a) Average Search Time')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Policy entropy
    ax = axes[0, 1]
    entropies = [data[algo]['avg_entropy'] for algo in algorithms]
    ax.bar(algorithms, entropies, color=colors, alpha=0.8)
    ax.set_ylabel('Entropy (bits)')
    ax.set_title('(b) Policy Entropy')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Max probability
    ax = axes[1, 0]
    max_probs = [data[algo]['avg_max_prob'] for algo in algorithms]
    ax.bar(algorithms, max_probs, color=colors, alpha=0.8)
    ax.set_ylabel('Probability')
    ax.set_title('(c) Maximum Action Probability')
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 4: Efficiency scatter
    ax = axes[1, 1]
    ax.scatter(search_times, entropies, s=200, c=range(len(algorithms)),
              cmap='viridis', alpha=0.6)
    for i, algo in enumerate(algorithms):
        ax.annotate(algo, (search_times[i], entropies[i]),
                   fontsize=8, ha='right')
    ax.set_xlabel('Search Time (ms)')
    ax.set_ylabel('Policy Entropy')
    ax.set_title('(d) Efficiency vs Exploration Tradeoff')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/work/repo/docs/figures/figure3_search_benchmark.pdf', bbox_inches='tight')
    plt.savefig('/home/claude/work/repo/docs/figures/figure3_search_benchmark.png', bbox_inches='tight')
    plt.close()

    print("Generated Figure 3: Search benchmark")


def plot_exploration_exploitation():
    """Figure 4: Exploration-exploitation tradeoff analysis"""
    data = load_experiment_data('experiment5_exploration')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Extract C values and performance
    c_values = []
    win_rates = []

    for player in data['player_results'].keys():
        if 'MCTS_C' in player:
            try:
                c_val = float(player.replace('MCTS_C', ''))
                c_values.append(c_val)
                total = sum(data['player_results'][player].values())
                wins = data['player_results'][player]['wins']
                win_rates.append(wins / total if total > 0 else 0)
            except ValueError:
                continue

    # Sort by C value
    if not c_values:
        print("Warning: No C-value data found for exploration-exploitation plot")
        # Create dummy figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax in axes:
            ax.text(0.5, 0.5, 'Data unavailable', ha='center', va='center')
            ax.set_title('Exploration-Exploitation Analysis')
        plt.tight_layout()
        plt.savefig('/home/claude/work/repo/docs/figures/figure4_exploration_exploitation.pdf', bbox_inches='tight')
        plt.savefig('/home/claude/work/repo/docs/figures/figure4_exploration_exploitation.png', bbox_inches='tight')
        plt.close()
        print("Generated Figure 4: Exploration-exploitation (placeholder)")
        return

    sorted_data = sorted(zip(c_values, win_rates))
    c_values, win_rates = zip(*sorted_data)

    # Plot 1: Win rate vs C parameter
    ax = axes[0]
    ax.plot(c_values, win_rates, 'o-', linewidth=2, markersize=10,
           color='steelblue', alpha=0.8)
    ax.set_xlabel('Exploration Parameter (C)')
    ax.set_ylabel('Win Rate')
    ax.set_title('(a) Performance vs Exploration Parameter')
    ax.axvline(x=1.41, color='red', linestyle='--', alpha=0.5,
              label='Theoretical Optimum (√2)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Exploration regions
    ax = axes[1]
    regions = ['Exploitation\n(C < 1.0)', 'Balanced\n(C ≈ 1.41)', 'Exploration\n(C > 2.0)']
    region_scores = []

    # Calculate region scores
    for c_range in [[0.5], [1.0, 1.41], [2.0, 3.0]]:
        scores = [wr for c, wr in zip(c_values, win_rates) if c in c_range]
        region_scores.append(np.mean(scores) if scores else 0)

    colors = ['coral', 'steelblue', 'mediumseagreen']
    ax.bar(regions, region_scores, color=colors, alpha=0.8)
    ax.set_ylabel('Average Win Rate')
    ax.set_title('(b) Performance by Strategy Region')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('/home/claude/work/repo/docs/figures/figure4_exploration_exploitation.pdf', bbox_inches='tight')
    plt.savefig('/home/claude/work/repo/docs/figures/figure4_exploration_exploitation.png', bbox_inches='tight')
    plt.close()

    print("Generated Figure 4: Exploration-exploitation")


def plot_scalability():
    """Figure 5: Scalability analysis"""
    data = load_experiment_data('experiment6_scalability')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = ['small', 'medium', 'large']
    game_lengths = [data[c]['avg_game_length'] for c in configs if c in data]
    moves_per_turn = [data[c]['avg_moves_per_turn'] for c in configs if c in data]

    # Plot 1: Game length scaling
    ax = axes[0]
    x = np.arange(len(configs[:len(game_lengths)]))
    ax.bar(x, game_lengths, color='steelblue', alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs[:len(game_lengths)])
    ax.set_ylabel('Average Game Length (moves)')
    ax.set_title('(a) Game Length vs Board Complexity')

    # Plot 2: Branching factor
    ax = axes[1]
    ax.bar(x, moves_per_turn, color='coral', alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs[:len(moves_per_turn)])
    ax.set_ylabel('Average Available Moves')
    ax.set_title('(b) Branching Factor vs Board Complexity')

    plt.tight_layout()
    plt.savefig('/home/claude/work/repo/docs/figures/figure5_scalability.pdf', bbox_inches='tight')
    plt.savefig('/home/claude/work/repo/docs/figures/figure5_scalability.png', bbox_inches='tight')
    plt.close()

    print("Generated Figure 5: Scalability")


def generate_all_visualizations():
    """Generate all figures for the research paper"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR RESEARCH PAPER")
    print("="*80)

    plot_baseline_comparison()
    plot_architecture_comparison()
    plot_search_benchmark()
    plot_exploration_exploitation()
    plot_scalability()

    print("\n" + "="*80)
    print("ALL VISUALIZATIONS GENERATED")
    print("Saved to: /home/claude/work/repo/docs/figures/")
    print("="*80)


if __name__ == '__main__':
    generate_all_visualizations()
