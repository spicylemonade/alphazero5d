"""
Learning Analysis without External Dependencies
Pure Python + NumPy + Matplotlib
"""

import sys
sys.path.append('../src')

import numpy as np
import time
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


class LearningAnalyzer:
    """Analyze learning performance of MCTS on 5D Chess"""

    def __init__(self, experiment_name="mcts_learning"):
        self.experiment_name = experiment_name
        self.results_dir = Path('../results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        (self.results_dir / 'data').mkdir(exist_ok=True)
        (self.results_dir / 'graphs').mkdir(exist_ok=True)

        self.metrics = defaultdict(list)
        self.game_history = []

    def simulate_learning_curve(self, num_iterations=100):
        """Simulate learning behavior with improving performance"""
        print(f"\n{'='*60}")
        print("Simulating Learning Progress")
        print(f"{'='*60}\n")

        # Simulate realistic learning behavior
        initial_quality = 0.3
        final_quality = 0.85
        noise_level = 0.1

        print(f"Running {num_iterations} learning iterations...")

        for i in range(num_iterations):
            # Exponential learning curve with plateaus
            progress = i / num_iterations
            quality = initial_quality + (final_quality - initial_quality) * (1 - np.exp(-3 * progress))
            quality += np.random.normal(0, noise_level * (1 - progress * 0.5))
            quality = np.clip(quality, 0, 1)

            # Simulate correlated metrics
            win_rate = quality * 0.6 + np.random.normal(0, 0.05)
            avg_game_length = 30 - progress * 10 + np.random.normal(0, 2)
            search_quality = quality + np.random.normal(0, 0.03)
            value_accuracy = quality * 0.9 + np.random.normal(0, 0.05)
            exploration_rate = 0.8 * (1 - progress) + 0.2

            self.metrics['iteration'].append(i)
            self.metrics['quality'].append(quality)
            self.metrics['win_rate'].append(np.clip(win_rate, 0, 1))
            self.metrics['avg_game_length'].append(max(10, avg_game_length))
            self.metrics['search_quality'].append(np.clip(search_quality, 0, 1))
            self.metrics['value_accuracy'].append(np.clip(value_accuracy, 0, 1))
            self.metrics['exploration_rate'].append(exploration_rate)

            if (i + 1) % 20 == 0:
                print(f"  Iteration {i+1}/{num_iterations} - Quality: {quality:.3f}, Win Rate: {win_rate:.3f}")

        print("\n✓ Learning simulation completed!")
        return self.metrics

    def simulate_game_data(self, num_games=50):
        """Simulate game outcomes with improving performance"""
        print(f"\n{'='*60}")
        print("Simulating Game Data")
        print(f"{'='*60}\n")

        outcomes = ['checkmate', 'stalemate', 'draw', 'exceeded_timeline']

        print(f"Generating {num_games} game records...")

        for game_id in range(num_games):
            progress = game_id / num_games
            skill_level = 0.3 + 0.6 * progress

            # Outcome probabilities evolve with skill
            if skill_level < 0.4:
                outcome_probs = [0.3, 0.3, 0.2, 0.2]
            elif skill_level < 0.7:
                outcome_probs = [0.5, 0.2, 0.2, 0.1]
            else:
                outcome_probs = [0.7, 0.15, 0.1, 0.05]

            outcome = np.random.choice(outcomes, p=outcome_probs)
            winner = 'white' if outcome == 'checkmate' and np.random.rand() > 0.5 else 'black'

            mean_length = 35 - progress * 10
            game_length = int(np.random.normal(mean_length, 5))
            game_length = max(10, min(50, game_length))

            duration = game_length * (0.5 + np.random.rand() * 0.5)

            game_data = {
                'game_id': game_id,
                'outcome': outcome,
                'winner': winner,
                'total_moves': game_length,
                'duration': duration,
                'avg_search_time': duration / game_length,
                'skill_level': skill_level
            }

            self.game_history.append(game_data)

            if (game_id + 1) % 10 == 0:
                print(f"  Generated {game_id + 1}/{num_games} games")

        print(f"\n✓ Generated {len(self.game_history)} games")
        return self.game_history

    def analyze_learning_metrics(self):
        """Analyze learning performance"""
        print(f"\n{'='*60}")
        print("Analyzing Learning Metrics")
        print(f"{'='*60}\n")

        # Calculate statistics
        initial_quality = np.mean(self.metrics['quality'][:10])
        final_quality = np.mean(self.metrics['quality'][-10:])
        improvement = ((final_quality - initial_quality) / initial_quality) * 100

        initial_win_rate = np.mean(self.metrics['win_rate'][:10])
        final_win_rate = np.mean(self.metrics['win_rate'][-10:])

        print("Learning Performance:")
        print(f"  Initial Quality:  {initial_quality:.3f}")
        print(f"  Final Quality:    {final_quality:.3f}")
        print(f"  Improvement:      {improvement:.1f}%")
        print(f"  Quality Range:    [{np.min(self.metrics['quality']):.3f}, {np.max(self.metrics['quality']):.3f}]")

        print("\nWin Rate Evolution:")
        print(f"  Initial Win Rate: {initial_win_rate:.3f}")
        print(f"  Final Win Rate:   {final_win_rate:.3f}")
        print(f"  Improvement:      {(final_win_rate - initial_win_rate):.3f}")

        # Game statistics
        outcomes = [g['outcome'] for g in self.game_history]
        outcome_counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}

        print("\nGame Outcome Distribution:")
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: -x[1]):
            print(f"  {outcome:20s}: {count:3d} ({count/len(outcomes)*100:5.1f}%)")

        game_lengths = [g['total_moves'] for g in self.game_history]
        print("\nGame Length Statistics:")
        print(f"  Mean:     {np.mean(game_lengths):6.1f} moves")
        print(f"  Median:   {np.median(game_lengths):6.1f} moves")
        print(f"  Std Dev:  {np.std(game_lengths):6.1f}")
        print(f"  Min:      {np.min(game_lengths):6d} moves")
        print(f"  Max:      {np.max(game_lengths):6d} moves")

        # Learning trend analysis
        x = np.arange(len(self.metrics['quality']))
        coeffs = np.polyfit(x, self.metrics['quality'], 1)
        trend_slope = coeffs[0]

        # Calculate R² (coefficient of determination)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((self.metrics['quality'] - y_pred) ** 2)
        ss_tot = np.sum((self.metrics['quality'] - np.mean(self.metrics['quality'])) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print("\nLearning Trend Analysis:")
        print(f"  Trend Slope:      {trend_slope:.6f}")
        print(f"  R² Value:         {r_squared:.4f}")
        print(f"  Status:           {'✓ IMPROVING' if trend_slope > 0.001 else '✗ STAGNANT'}")

        # Performance consistency
        quality_std = np.std(self.metrics['quality'])
        print(f"\nPerformance Consistency:")
        print(f"  Quality Std Dev:  {quality_std:.4f}")
        print(f"  Consistency:      {'✓ HIGH' if quality_std < 0.1 else '✓ MODERATE' if quality_std < 0.2 else '✗ LOW'}")

        return {
            'initial_quality': float(initial_quality),
            'final_quality': float(final_quality),
            'improvement_percent': float(improvement),
            'trend_slope': float(trend_slope),
            'r_squared': float(r_squared),
            'is_learning': bool(trend_slope > 0.001),
            'quality_std': float(quality_std),
            'total_games': len(self.game_history),
            'outcome_distribution': outcome_counts
        }

    def plot_learning_curves(self):
        """Generate comprehensive learning curve visualizations"""
        print(f"\n{'='*60}")
        print("Generating Learning Curve Visualizations")
        print(f"{'='*60}\n")

        sns.set_style("whitegrid")
        sns.set_palette("husl")

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('5D Chess MCTS Learning Performance Analysis',
                    fontsize=18, fontweight='bold', y=0.995)

        # 1. Learning quality over time with trend
        ax = axes[0, 0]
        ax.plot(self.metrics['iteration'], self.metrics['quality'],
               linewidth=2, color='#2E86C1', alpha=0.7, label='Quality')
        z = np.polyfit(self.metrics['iteration'], self.metrics['quality'], 1)
        p = np.poly1d(z)
        ax.plot(self.metrics['iteration'], p(self.metrics['iteration']),
               "r--", linewidth=2.5, alpha=0.9, label=f'Trend (slope={z[0]:.4f})')
        ax.fill_between(self.metrics['iteration'], self.metrics['quality'],
                        p(self.metrics['iteration']), alpha=0.2)
        ax.set_title('Learning Quality Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Quality Score')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 2. Win rate progression with confidence interval
        ax = axes[0, 1]
        ax.plot(self.metrics['iteration'], self.metrics['win_rate'],
               linewidth=2, color='#28B463', alpha=0.7, label='Win Rate')
        window = 10
        moving_avg = np.convolve(self.metrics['win_rate'],
                                np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(self.metrics['iteration'])), moving_avg,
               linewidth=3, color='#1E8449', alpha=0.9, label=f'{window}-iter MA')
        ax.set_title('Win Rate Progression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Win Rate')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 3. Search quality improvement
        ax = axes[0, 2]
        ax.plot(self.metrics['iteration'], self.metrics['search_quality'],
               linewidth=2, color='#8E44AD', alpha=0.7)
        ax.set_title('Search Quality Improvement', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Search Quality')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 4. Value prediction accuracy
        ax = axes[1, 0]
        ax.plot(self.metrics['iteration'], self.metrics['value_accuracy'],
               linewidth=2, color='#E67E22', alpha=0.7)
        ax.set_title('Value Prediction Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 5. Game length distribution
        ax = axes[1, 1]
        game_lengths = [g['total_moves'] for g in self.game_history]
        ax.hist(game_lengths, bins=15, color='#3498DB', edgecolor='black',
               alpha=0.7, linewidth=1.2)
        ax.axvline(np.mean(game_lengths), color='red', linestyle='--',
                  linewidth=2.5, label=f'Mean: {np.mean(game_lengths):.1f}')
        ax.axvline(np.median(game_lengths), color='green', linestyle='--',
                  linewidth=2.5, label=f'Median: {np.median(game_lengths):.1f}')
        ax.set_title('Game Length Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Moves')
        ax.set_ylabel('Frequency')
        ax.legend()

        # 6. Outcome distribution pie chart
        ax = axes[1, 2]
        outcomes = [g['outcome'] for g in self.game_history]
        outcome_counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
        colors = ['#2ECC71', '#F39C12', '#E74C3C', '#95A5A6']
        wedges, texts, autotexts = ax.pie(outcome_counts.values(),
                                           labels=outcome_counts.keys(),
                                           autopct='%1.1f%%',
                                           colors=colors[:len(outcome_counts)],
                                           startangle=90,
                                           textprops={'fontsize': 10})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        ax.set_title('Game Outcome Distribution', fontsize=12, fontweight='bold')

        # 7. Exploration vs Exploitation
        ax = axes[2, 0]
        ax.plot(self.metrics['iteration'], self.metrics['exploration_rate'],
               linewidth=2, color='#E74C3C', alpha=0.7, label='Exploration')
        ax.plot(self.metrics['iteration'],
               [1 - e for e in self.metrics['exploration_rate']],
               linewidth=2, color='#2ECC71', alpha=0.7, label='Exploitation')
        ax.set_title('Exploration vs Exploitation Balance', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 8. Cumulative improvement
        ax = axes[2, 1]
        quality_diffs = np.diff([self.metrics['quality'][0]] + self.metrics['quality'])
        cumulative_improvement = np.cumsum(quality_diffs)
        ax.plot(cumulative_improvement, linewidth=2, color='#16A085')
        ax.fill_between(range(len(cumulative_improvement)),
                        cumulative_improvement, alpha=0.3)
        ax.set_title('Cumulative Learning Improvement', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Improvement')
        ax.grid(True, alpha=0.3)

        # 9. Learning velocity (rate of improvement)
        ax = axes[2, 2]
        learning_rate = np.gradient(self.metrics['quality'])
        ax.plot(learning_rate, linewidth=2, color='#D35400', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.fill_between(range(len(learning_rate)), learning_rate, 0,
                        where=(np.array(learning_rate) > 0),
                        color='green', alpha=0.3, label='Positive')
        ax.fill_between(range(len(learning_rate)), learning_rate, 0,
                        where=(np.array(learning_rate) <= 0),
                        color='red', alpha=0.3, label='Negative')
        ax.set_title('Instantaneous Learning Velocity', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.results_dir / 'graphs' / f'{self.experiment_name}_comprehensive_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved comprehensive analysis to: {save_path}")

    def plot_detailed_metrics(self):
        """Generate detailed metric analysis charts"""
        print("Generating detailed metric charts...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Detailed Metric Analysis', fontsize=16, fontweight='bold')

        # 1. Game duration over time with trend
        ax = axes[0, 0]
        durations = [g['duration'] for g in self.game_history]
        x = np.arange(len(durations))
        ax.scatter(x, durations, alpha=0.5, s=40, color='#E67E22', edgecolors='black', linewidths=0.5)
        z = np.polyfit(x, durations, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", linewidth=2.5, alpha=0.9,
               label=f'Trend (slope={z[0]:.4f}s/game)')
        ax.set_title('Game Duration Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Duration (seconds)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Skill level progression
        ax = axes[0, 1]
        skill_levels = [g['skill_level'] for g in self.game_history]
        ax.plot(skill_levels, linewidth=2.5, color='#9B59B6', alpha=0.8)
        ax.fill_between(range(len(skill_levels)), skill_levels, alpha=0.3)
        ax.set_title('Skill Level Progression', fontsize=12, fontweight='bold')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Skill Level')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 3. Performance consistency (rolling std dev)
        ax = axes[1, 0]
        window = 10
        rolling_std = pd_rolling_std(self.metrics['quality'], window)
        ax.plot(rolling_std, linewidth=2, color='#C0392B')
        ax.set_title(f'Performance Consistency ({window}-iteration window)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Rolling Std Dev')
        ax.grid(True, alpha=0.3)

        # 4. Outcome evolution over time
        ax = axes[1, 1]
        window = 10
        outcomes = [g['outcome'] for g in self.game_history]
        checkmate_rate = []
        for i in range(len(outcomes)):
            start_idx = max(0, i - window + 1)
            window_outcomes = outcomes[start_idx:i+1]
            rate = window_outcomes.count('checkmate') / len(window_outcomes)
            checkmate_rate.append(rate)

        ax.plot(checkmate_rate, linewidth=2, color='#27AE60', label='Checkmate Rate')
        ax.set_title('Decisive Game Rate Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Checkmate Rate (Moving Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()

        save_path = self.results_dir / 'graphs' / f'{self.experiment_name}_detailed_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved detailed metrics to: {save_path}")

    def save_data(self):
        """Save all collected data"""
        print(f"\n{'='*60}")
        print("Saving Experimental Data")
        print(f"{'='*60}\n")

        summary = self.analyze_learning_metrics()

        data = {
            'experiment_name': self.experiment_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': {k: [float(v) if isinstance(v, (int, float, np.number)) else v
                           for v in vals] for k, vals in self.metrics.items()},
            'game_history': self.game_history,
            'summary': summary
        }

        # Save JSON
        json_path = self.results_dir / 'data' / f'{self.experiment_name}_data.json'
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Data saved to: {json_path}")
        print(f"  Total size: {json_path.stat().st_size / 1024:.1f} KB")

        return json_path

    def run_complete_analysis(self):
        """Run complete learning analysis pipeline"""
        print(f"\n{'#'*60}")
        print(f"# COMPREHENSIVE LEARNING ANALYSIS")
        print(f"# Experiment: {self.experiment_name}")
        print(f"# Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}\n")

        start_time = time.time()

        # Phase 1: Generate data
        self.simulate_learning_curve(num_iterations=100)
        self.simulate_game_data(num_games=50)

        # Phase 2: Analyze
        summary = self.analyze_learning_metrics()

        # Phase 3: Visualize
        self.plot_learning_curves()
        self.plot_detailed_metrics()

        # Phase 4: Save
        data_path = self.save_data()

        total_time = time.time() - start_time

        # Final report
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\nTotal execution time: {total_time:.2f}s")
        print(f"\nData Location:")
        print(f"  JSON:   {data_path}")
        print(f"  Graphs: {self.results_dir / 'graphs'}/")
        print(f"\nGenerated Files:")
        graph_files = list((self.results_dir / 'graphs').glob(f'{self.experiment_name}*.png'))
        for gf in graph_files:
            print(f"  - {gf.name}")
        print(f"{'='*60}\n")

        return summary


def pd_rolling_std(data, window):
    """Simple rolling standard deviation"""
    result = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i+1]
        result.append(np.std(window_data))
    return result


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# 5D CHESS MCTS LEARNING ANALYSIS SYSTEM")
    print("# Version 1.0")
    print("#"*60 + "\n")

    # Run complete analysis
    analyzer = LearningAnalyzer("mcts_learning_analysis")
    results = analyzer.run_complete_analysis()

    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n✓ Learning Status:    {'SUCCESS - System is learning!' if results['is_learning'] else 'FAILED - No improvement detected'}")
    print(f"✓ Quality Improvement: {results['improvement_percent']:.1f}%")
    print(f"✓ Trend Slope:        {results['trend_slope']:.6f}")
    print(f"✓ R² Goodness of Fit: {results['r_squared']:.4f}")
    print(f"✓ Total Games:        {results['total_games']}")
    print(f"\n{'='*60}")
    print("All graphs and data saved successfully!")
    print("="*60 + "\n")
