"""
Comprehensive Visualization System for Learning Analysis
Generates publication-quality graphs and charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class LearningVisualizer:
    """Create comprehensive visualizations of learning progress"""

    def __init__(self, save_dir="results/graphs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_learning_curves(self, data, save_name="learning_curves"):
        """Plot training loss curves"""
        if 'training_stats' not in data or not data['training_stats']:
            print("No training data available")
            return

        df = pd.DataFrame(data['training_stats'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Learning Curves', fontsize=16, fontweight='bold')

        # Total loss
        axes[0, 0].plot(df['epoch'], df['loss'], linewidth=2, color='blue', alpha=0.7)
        axes[0, 0].set_title('Total Loss Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)

        # Policy loss
        if 'policy_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['policy_loss'], linewidth=2, color='green', alpha=0.7)
            axes[0, 1].set_title('Policy Loss Over Time')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Policy Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Value loss
        if 'value_loss' in df.columns:
            axes[1, 0].plot(df['epoch'], df['value_loss'], linewidth=2, color='red', alpha=0.7)
            axes[1, 0].set_title('Value Loss Over Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Value Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # Learning rate
        if 'learning_rate' in df.columns:
            axes[1, 1].plot(df['epoch'], df['learning_rate'], linewidth=2, color='purple', alpha=0.7)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved learning curves to {save_path}")

    def plot_game_statistics(self, data, save_name="game_statistics"):
        """Plot game outcome and performance statistics"""
        if 'game_history' not in data or not data['game_history']:
            print("No game data available")
            return

        games = data['game_history']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Game Performance Statistics', fontsize=16, fontweight='bold')

        # Outcome distribution
        outcomes = [g['outcome'] for g in games]
        outcome_counts = pd.Series(outcomes).value_counts()
        axes[0, 0].bar(outcome_counts.index, outcome_counts.values, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Game Outcomes Distribution')
        axes[0, 0].set_xlabel('Outcome Type')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Game length distribution
        move_counts = [g['total_moves'] for g in games]
        axes[0, 1].hist(move_counts, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Game Length Distribution')
        axes[0, 1].set_xlabel('Number of Moves')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(move_counts), color='red', linestyle='--',
                           label=f'Mean: {np.mean(move_counts):.1f}')
        axes[0, 1].legend()

        # Game duration
        durations = [g['duration'] for g in games]
        axes[0, 2].hist(durations, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 2].set_title('Game Duration Distribution')
        axes[0, 2].set_xlabel('Duration (seconds)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(durations), color='blue', linestyle='--',
                          label=f'Mean: {np.mean(durations):.1f}s')
        axes[0, 2].legend()

        # Game length over time
        axes[1, 0].plot(range(len(move_counts)), move_counts, alpha=0.5, color='gray')
        window = min(10, len(move_counts) // 10 + 1)
        moving_avg = pd.Series(move_counts).rolling(window=window, min_periods=1).mean()
        axes[1, 0].plot(range(len(moving_avg)), moving_avg, linewidth=2, color='blue',
                       label=f'{window}-game moving average')
        axes[1, 0].set_title('Game Length Over Time')
        axes[1, 0].set_xlabel('Game Number')
        axes[1, 0].set_ylabel('Moves per Game')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Winner distribution (if available)
        winners = [g.get('winner', 'unknown') for g in games]
        winner_counts = pd.Series(winners).value_counts()
        axes[1, 1].pie(winner_counts.values, labels=winner_counts.index, autopct='%1.1f%%',
                      colors=['gold', 'silver', 'lightgray'])
        axes[1, 1].set_title('Winner Distribution')

        # Cumulative games over time
        cumulative = list(range(1, len(games) + 1))
        times = [(g['end_time'] - games[0]['start_time']) / 60 for g in games]  # Minutes
        axes[1, 2].plot(times, cumulative, linewidth=2, color='purple')
        axes[1, 2].set_title('Cumulative Games Played')
        axes[1, 2].set_xlabel('Time (minutes)')
        axes[1, 2].set_ylabel('Total Games')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved game statistics to {save_path}")

    def plot_mcts_analysis(self, data, save_name="mcts_analysis"):
        """Plot MCTS search behavior analysis"""
        if 'move_history' not in data or not data['move_history']:
            print("No move data available")
            return

        moves = data['move_history']
        df = pd.DataFrame(moves)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MCTS Search Analysis', fontsize=16, fontweight='bold')

        # Value predictions over time
        if 'value' in df.columns:
            axes[0, 0].plot(df['move_num'], df['value'], alpha=0.6, linewidth=1)
            window = min(20, len(df) // 10 + 1)
            moving_avg = df['value'].rolling(window=window, min_periods=1).mean()
            axes[0, 0].plot(df['move_num'], moving_avg, linewidth=2, color='red',
                           label=f'{window}-move average')
            axes[0, 0].set_title('Value Predictions Over Time')
            axes[0, 0].set_xlabel('Move Number')
            axes[0, 0].set_ylabel('Predicted Value')
            axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Search time distribution
        if 'search_time' in df.columns:
            search_times = df['search_time'].dropna()
            if len(search_times) > 0:
                axes[0, 1].hist(search_times, bins=30, color='lightblue',
                               edgecolor='black', alpha=0.7)
                axes[0, 1].set_title('Search Time Distribution')
                axes[0, 1].set_xlabel('Search Time (seconds)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].axvline(search_times.mean(), color='red', linestyle='--',
                                  label=f'Mean: {search_times.mean():.3f}s')
                axes[0, 1].legend()

        # Uncertainty over time (if available)
        if 'uncertainty' in df.columns:
            uncertainties = df['uncertainty'].dropna()
            if len(uncertainties) > 0:
                axes[1, 0].plot(range(len(uncertainties)), uncertainties, alpha=0.6)
                window = min(20, len(uncertainties) // 10 + 1)
                moving_avg = uncertainties.rolling(window=window, min_periods=1).mean()
                axes[1, 0].plot(range(len(moving_avg)), moving_avg, linewidth=2,
                               color='red', label=f'{window}-move average')
                axes[1, 0].set_title('Prediction Uncertainty Over Time')
                axes[1, 0].set_xlabel('Move Index')
                axes[1, 0].set_ylabel('Uncertainty')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

        # Value distribution by player
        if 'player' in df.columns and 'value' in df.columns:
            white_values = df[df['player'] == 'white']['value']
            black_values = df[df['player'] == 'black']['value']

            axes[1, 1].hist([white_values, black_values], bins=20,
                           label=['White', 'Black'], alpha=0.7,
                           color=['white', 'gray'], edgecolor='black')
            axes[1, 1].set_title('Value Distribution by Player')
            axes[1, 1].set_xlabel('Predicted Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved MCTS analysis to {save_path}")

    def plot_performance_comparison(self, experiments, save_name="performance_comparison"):
        """Compare performance across multiple experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Experiment Performance Comparison', fontsize=16, fontweight='bold')

        colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

        for idx, (name, data) in enumerate(experiments.items()):
            color = colors[idx]

            # Training loss comparison
            if 'training_stats' in data and data['training_stats']:
                df = pd.DataFrame(data['training_stats'])
                axes[0, 0].plot(df['epoch'], df['loss'], label=name,
                               color=color, linewidth=2, alpha=0.7)

            # Game length comparison
            if 'game_history' in data and data['game_history']:
                move_counts = [g['total_moves'] for g in data['game_history']]
                window = min(10, len(move_counts) // 10 + 1)
                moving_avg = pd.Series(move_counts).rolling(window=window, min_periods=1).mean()
                axes[0, 1].plot(range(len(moving_avg)), moving_avg, label=name,
                               color=color, linewidth=2, alpha=0.7)

                # Win rate over time
                outcomes = [1 if g['outcome'] == 'checkmate' and g.get('winner') == 'white'
                           else 0 for g in data['game_history']]
                win_rate = pd.Series(outcomes).rolling(window=window, min_periods=1).mean()
                axes[1, 0].plot(range(len(win_rate)), win_rate, label=name,
                               color=color, linewidth=2, alpha=0.7)

        axes[0, 0].set_title('Training Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title('Average Game Length Comparison')
        axes[0, 1].set_xlabel('Game Number')
        axes[0, 1].set_ylabel('Moves per Game')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title('Win Rate Comparison')
        axes[1, 0].set_xlabel('Game Number')
        axes[1, 0].set_ylabel('Win Rate (Moving Average)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Summary statistics table
        summary_data = []
        for name, data in experiments.items():
            if 'summary' in data:
                summary = data['summary']
                summary_data.append([
                    name,
                    summary.get('total_games', 0),
                    f"{summary.get('game_length', {}).get('mean', 0):.1f}",
                    f"{summary.get('game_duration', {}).get('mean', 0):.2f}s"
                ])

        if summary_data:
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=summary_data,
                                    colLabels=['Experiment', 'Games', 'Avg Moves', 'Avg Duration'],
                                    cellLoc='center',
                                    loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 1].set_title('Summary Statistics')

        plt.tight_layout()
        save_path = self.save_dir / f"{save_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance comparison to {save_path}")

    def create_interactive_dashboard(self, data, save_name="interactive_dashboard"):
        """Create interactive Plotly dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Game Outcomes',
                          'Move Values', 'Game Length Over Time'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        # Training loss
        if 'training_stats' in data and data['training_stats']:
            df = pd.DataFrame(data['training_stats'])
            fig.add_trace(
                go.Scatter(x=df['epoch'], y=df['loss'], mode='lines',
                          name='Loss', line=dict(color='blue')),
                row=1, col=1
            )

        # Game outcomes
        if 'game_history' in data and data['game_history']:
            outcomes = [g['outcome'] for g in data['game_history']]
            outcome_counts = pd.Series(outcomes).value_counts()
            fig.add_trace(
                go.Bar(x=outcome_counts.index, y=outcome_counts.values,
                      name='Outcomes'),
                row=1, col=2
            )

            # Game length over time
            move_counts = [g['total_moves'] for g in data['game_history']]
            fig.add_trace(
                go.Scatter(x=list(range(len(move_counts))), y=move_counts,
                          mode='lines+markers', name='Game Length'),
                row=2, col=2
            )

        # Move values
        if 'move_history' in data and data['move_history']:
            df_moves = pd.DataFrame(data['move_history'])
            if 'value' in df_moves.columns:
                fig.add_trace(
                    go.Scatter(x=df_moves['move_num'], y=df_moves['value'],
                              mode='markers', name='Values',
                              marker=dict(size=3, opacity=0.6)),
                    row=2, col=1
                )

        fig.update_layout(height=800, showlegend=True,
                         title_text="5D Chess Learning Dashboard")

        save_path = self.save_dir / f"{save_name}.html"
        fig.write_html(str(save_path))
        print(f"Saved interactive dashboard to {save_path}")

    def generate_all_visualizations(self, data, prefix=""):
        """Generate all available visualizations"""
        print(f"\nGenerating visualizations for: {prefix}")
        self.plot_learning_curves(data, f"{prefix}learning_curves")
        self.plot_game_statistics(data, f"{prefix}game_statistics")
        self.plot_mcts_analysis(data, f"{prefix}mcts_analysis")
        self.create_interactive_dashboard(data, f"{prefix}dashboard")
        print("All visualizations generated successfully!")


def visualize_from_file(filepath):
    """Load data and create visualizations"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    visualizer = LearningVisualizer()
    visualizer.generate_all_visualizations(data)


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded. Use visualize_from_file(path) to create graphs.")
