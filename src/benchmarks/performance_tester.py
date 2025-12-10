"""
Comprehensive Performance Testing and Benchmarking System
Runs extensive tests on MCTS performance, learning curves, and gameplay quality
"""

import sys
sys.path.append('/home/claude/work/repo/src')
sys.path.append('/home/claude/work/repo/src/optimized')

import numpy as np
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime


@dataclass
class GameResult:
    """Single game result"""
    winner: str
    moves: int
    duration: float
    avg_nodes_per_move: float
    timeline_expansions: int
    final_state: str


@dataclass
class BenchmarkResults:
    """Complete benchmark results"""
    timestamp: str
    config: Dict
    games_played: int
    total_time: float

    # Win statistics
    white_wins: int
    black_wins: int
    draws: int

    # Performance metrics
    avg_game_length: float
    std_game_length: float
    avg_move_time: float
    std_move_time: float
    avg_nodes_expanded: float

    # Learning metrics
    win_rate_white: float
    win_rate_black: float
    avg_timeline_expansions: float

    # Detailed results
    game_results: List[GameResult]


class PerformanceTester:
    """Comprehensive performance testing framework"""

    def __init__(self, output_dir: str = 'docs/figures'):
        self.output_dir = output_dir
        self.results_history = []

    def run_comprehensive_benchmark(self, num_games: int = 50,
                                   mcts_simulations_range: List[int] = None,
                                   save_plots: bool = True) -> BenchmarkResults:
        """
        Run comprehensive benchmark suite

        Args:
            num_games: Number of games to play per configuration
            mcts_simulations_range: List of MCTS simulation counts to test
            save_plots: Whether to save visualization plots
        """
        if mcts_simulations_range is None:
            mcts_simulations_range = [50, 100, 200, 400, 800]

        print(f"\n{'='*60}")
        print(f"5D CHESS COMPREHENSIVE BENCHMARK")
        print(f"{'='*60}")
        print(f"Games per configuration: {num_games}")
        print(f"MCTS simulation counts: {mcts_simulations_range}")
        print(f"{'='*60}\n")

        all_results = []

        for num_sims in mcts_simulations_range:
            print(f"\n--- Testing with {num_sims} MCTS simulations ---")
            results = self._run_game_suite(num_games, num_sims)
            all_results.append(results)

            # Print summary
            self._print_summary(results)

        if save_plots:
            self._generate_all_plots(all_results)

        # Save detailed results
        self._save_results(all_results)

        return all_results[-1]  # Return results from highest simulation count

    def _run_game_suite(self, num_games: int, num_simulations: int) -> BenchmarkResults:
        """Run a suite of games with specific configuration"""
        from mcts_enhanced import MCTSEnhanced, MCTSConfig
        from chess5d_optimized import Chess5DOptimized

        config = MCTSConfig(num_searches=num_simulations)

        game_results = []
        white_wins = 0
        black_wins = 0
        draws = 0

        total_moves = []
        total_durations = []
        total_nodes = []
        total_timeline_exp = []

        start_time = time.time()

        for game_idx in range(num_games):
            try:
                result = self._play_single_game(config)
                game_results.append(result)

                # Update statistics
                if result.winner == 'white':
                    white_wins += 1
                elif result.winner == 'black':
                    black_wins += 1
                else:
                    draws += 1

                total_moves.append(result.moves)
                total_durations.append(result.duration)
                total_nodes.append(result.avg_nodes_per_move)
                total_timeline_exp.append(result.timeline_expansions)

                # Progress update
                if (game_idx + 1) % 10 == 0:
                    print(f"  Completed {game_idx + 1}/{num_games} games")

            except Exception as e:
                print(f"  Error in game {game_idx + 1}: {e}")
                continue

        total_time = time.time() - start_time

        # Calculate statistics
        avg_game_length = np.mean(total_moves) if total_moves else 0
        std_game_length = np.std(total_moves) if total_moves else 0
        avg_move_time = np.mean(total_durations) / avg_game_length if avg_game_length > 0 else 0
        std_move_time = np.std(total_durations) if total_durations else 0
        avg_nodes = np.mean(total_nodes) if total_nodes else 0

        return BenchmarkResults(
            timestamp=datetime.now().isoformat(),
            config={'num_simulations': num_simulations},
            games_played=len(game_results),
            total_time=total_time,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            avg_game_length=avg_game_length,
            std_game_length=std_game_length,
            avg_move_time=avg_move_time,
            std_move_time=std_move_time,
            avg_nodes_expanded=avg_nodes,
            win_rate_white=white_wins / len(game_results) if game_results else 0,
            win_rate_black=black_wins / len(game_results) if game_results else 0,
            avg_timeline_expansions=np.mean(total_timeline_exp) if total_timeline_exp else 0,
            game_results=game_results
        )

    def _play_single_game(self, config: 'MCTSConfig') -> GameResult:
        """Play a single game and return results"""
        from mcts_enhanced import MCTSEnhanced
        from chess5d_optimized import Chess5DOptimized

        game = Chess5DOptimized(max_time=11, max_turns=30)
        mcts = MCTSEnhanced(game, config)

        state = game.get_initial_state()
        move_count = 0
        total_nodes = 0
        start_time = time.time()

        max_moves = 100  # Prevent infinite games

        while not state.is_terminal and move_count < max_moves:
            # Run MCTS search
            policy_start, policy_end, root = mcts.search(state, return_root=True)

            if root:
                total_nodes += root.visit_count

            # Select move
            try:
                move_str, _, _ = game.pick_choice_stochastic(
                    state, policy_start, policy_end, temperature=0.0
                )
                game.make_move(state, move_str)
                move_count += 1
            except:
                break

        duration = time.time() - start_time

        # Determine winner
        if state.is_terminal:
            if state.value == 1.0:
                winner = state.winning
            elif state.value == 0.0:
                winner = 'draw'
            else:
                winner = 'draw'
        else:
            winner = 'incomplete'

        avg_nodes = total_nodes / move_count if move_count > 0 else 0

        return GameResult(
            winner=winner,
            moves=move_count,
            duration=duration,
            avg_nodes_per_move=avg_nodes,
            timeline_expansions=game.metrics.timeline_expansions,
            final_state=state.game_string if hasattr(state, 'game_string') else ''
        )

    def _print_summary(self, results: BenchmarkResults):
        """Print summary of benchmark results"""
        print(f"\n  Results Summary:")
        print(f"  ├─ Games played: {results.games_played}")
        print(f"  ├─ Total time: {results.total_time:.2f}s")
        print(f"  ├─ White wins: {results.white_wins} ({results.win_rate_white:.1%})")
        print(f"  ├─ Black wins: {results.black_wins} ({results.win_rate_black:.1%})")
        print(f"  ├─ Draws: {results.draws} ({results.draws/results.games_played:.1%})")
        print(f"  ├─ Avg game length: {results.avg_game_length:.1f} ± {results.std_game_length:.1f} moves")
        print(f"  ├─ Avg move time: {results.avg_move_time:.3f}s")
        print(f"  ├─ Avg nodes expanded: {results.avg_nodes_expanded:.1f}")
        print(f"  └─ Avg timeline expansions: {results.avg_timeline_expansions:.1f}\n")

    def _generate_all_plots(self, all_results: List[BenchmarkResults]):
        """Generate all visualization plots"""
        print("\nGenerating visualization plots...")

        self._plot_win_rates(all_results)
        self._plot_game_lengths(all_results)
        self._plot_performance_scaling(all_results)
        self._plot_move_time_distribution(all_results)
        self._plot_learning_curves(all_results)

        print(f"Plots saved to {self.output_dir}/")

    def _plot_win_rates(self, all_results: List[BenchmarkResults]):
        """Plot win rates vs MCTS simulations"""
        sim_counts = [r.config['num_simulations'] for r in all_results]
        white_rates = [r.win_rate_white for r in all_results]
        black_rates = [r.win_rate_black for r in all_results]
        draw_rates = [r.draws / r.games_played for r in all_results]

        plt.figure(figsize=(10, 6))
        plt.plot(sim_counts, white_rates, 'o-', label='White Win Rate', linewidth=2, markersize=8)
        plt.plot(sim_counts, black_rates, 's-', label='Black Win Rate', linewidth=2, markersize=8)
        plt.plot(sim_counts, draw_rates, '^-', label='Draw Rate', linewidth=2, markersize=8)
        plt.xlabel('MCTS Simulations', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.title('Win Rates vs MCTS Simulation Count', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/win_rates.png', dpi=300)
        plt.close()

    def _plot_game_lengths(self, all_results: List[BenchmarkResults]):
        """Plot game length statistics"""
        sim_counts = [r.config['num_simulations'] for r in all_results]
        avg_lengths = [r.avg_game_length for r in all_results]
        std_lengths = [r.std_game_length for r in all_results]

        plt.figure(figsize=(10, 6))
        plt.errorbar(sim_counts, avg_lengths, yerr=std_lengths,
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        plt.xlabel('MCTS Simulations', fontsize=12)
        plt.ylabel('Game Length (moves)', fontsize=12)
        plt.title('Average Game Length vs MCTS Simulation Count', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/game_lengths.png', dpi=300)
        plt.close()

    def _plot_performance_scaling(self, all_results: List[BenchmarkResults]):
        """Plot performance metrics scaling"""
        sim_counts = [r.config['num_simulations'] for r in all_results]
        avg_move_times = [r.avg_move_time for r in all_results]
        avg_nodes = [r.avg_nodes_expanded for r in all_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Move time scaling
        ax1.plot(sim_counts, avg_move_times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('MCTS Simulations', fontsize=12)
        ax1.set_ylabel('Avg Move Time (s)', fontsize=12)
        ax1.set_title('Move Time Scaling', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Nodes expanded
        ax2.plot(sim_counts, avg_nodes, 's-', linewidth=2, markersize=8, color='#A23B72')
        ax2.set_xlabel('MCTS Simulations', fontsize=12)
        ax2.set_ylabel('Avg Nodes Expanded', fontsize=12)
        ax2.set_title('Node Expansion Rate', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_scaling.png', dpi=300)
        plt.close()

    def _plot_move_time_distribution(self, all_results: List[BenchmarkResults]):
        """Plot distribution of move times"""
        plt.figure(figsize=(12, 6))

        for idx, results in enumerate(all_results):
            move_times = [g.duration / g.moves for g in results.game_results if g.moves > 0]
            plt.hist(move_times, bins=30, alpha=0.5,
                    label=f"{results.config['num_simulations']} sims")

        plt.xlabel('Move Time (s)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Move Times', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/move_time_distribution.png', dpi=300)
        plt.close()

    def _plot_learning_curves(self, all_results: List[BenchmarkResults]):
        """Plot learning/improvement curves"""
        plt.figure(figsize=(10, 6))

        for results in all_results:
            # Calculate rolling win rate
            window = 10
            white_wins_cum = []
            for i in range(len(results.game_results)):
                start = max(0, i - window)
                games_window = results.game_results[start:i+1]
                white_win_count = sum(1 for g in games_window if g.winner == 'white')
                white_wins_cum.append(white_win_count / len(games_window))

            plt.plot(range(len(white_wins_cum)), white_wins_cum,
                    label=f"{results.config['num_simulations']} sims",
                    alpha=0.7, linewidth=2)

        plt.xlabel('Game Number', fontsize=12)
        plt.ylabel('Rolling Win Rate (White)', fontsize=12)
        plt.title(f'Learning Curve (Window={window} games)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/learning_curves.png', dpi=300)
        plt.close()

    def _save_results(self, all_results: List[BenchmarkResults]):
        """Save detailed results to JSON"""
        results_dict = []
        for results in all_results:
            result_dict = asdict(results)
            # Convert GameResult objects to dicts
            result_dict['game_results'] = [asdict(g) for g in results.game_results]
            results_dict.append(result_dict)

        filepath = f'{self.output_dir}/benchmark_results.json'
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nDetailed results saved to {filepath}")


if __name__ == '__main__':
    tester = PerformanceTester(output_dir='docs/figures')
    results = tester.run_comprehensive_benchmark(
        num_games=50,
        mcts_simulations_range=[50, 100, 200, 400, 800],
        save_plots=True
    )
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
