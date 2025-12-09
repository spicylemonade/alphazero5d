"""
Performance Testing Framework for 5D Chess AI
Tests gameplay quality, learning effectiveness, and optimization metrics
"""
import sys
sys.path.append('..')
from src.super import Chess5D, MCTS, ChessState
import cupy as cp
import numpy as np
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceMetrics:
    def __init__(self):
        self.games_played = 0
        self.white_wins = 0
        self.black_wins = 0
        self.draws = 0
        self.avg_game_length = []
        self.avg_search_time = []
        self.avg_moves_per_game = []
        self.decision_quality = []
        self.exploration_depth = []
        self.node_expansion_count = []
        self.terminal_values = []
        self.checkmates = 0
        self.stalemates = 0
        self.draw_losses = 0

    def record_game(self, winner, moves, search_times, terminal_value, end_type, nodes_expanded):
        self.games_played += 1

        if winner == 'white':
            self.white_wins += 1
        elif winner == 'black':
            self.black_wins += 1
        else:
            self.draws += 1

        self.avg_game_length.append(sum(search_times))
        self.avg_moves_per_game.append(moves)
        self.avg_search_time.extend(search_times)
        self.terminal_values.append(terminal_value)
        self.node_expansion_count.extend(nodes_expanded)

        if end_type == 'checkmate':
            self.checkmates += 1
        elif end_type == 'stalemate':
            self.stalemates += 1
        elif end_type == 'draw_loss':
            self.draw_losses += 1

    def calculate_statistics(self):
        return {
            'total_games': self.games_played,
            'white_win_rate': self.white_wins / max(self.games_played, 1),
            'black_win_rate': self.black_wins / max(self.games_played, 1),
            'draw_rate': self.draws / max(self.games_played, 1),
            'avg_moves_per_game': np.mean(self.avg_moves_per_game) if self.avg_moves_per_game else 0,
            'avg_game_duration': np.mean(self.avg_game_length) if self.avg_game_length else 0,
            'avg_search_time_per_move': np.mean(self.avg_search_time) if self.avg_search_time else 0,
            'avg_nodes_expanded': np.mean(self.node_expansion_count) if self.node_expansion_count else 0,
            'checkmate_rate': self.checkmates / max(self.games_played, 1),
            'stalemate_rate': self.stalemates / max(self.games_played, 1),
            'draw_loss_rate': self.draw_losses / max(self.games_played, 1),
            'terminal_value_mean': np.mean(self.terminal_values) if self.terminal_values else 0,
            'terminal_value_std': np.std(self.terminal_values) if self.terminal_values else 0
        }

class MCTSPerformanceTester:
    def __init__(self, max_time=1, max_turns=25):
        self.max_time = max_time
        self.max_turns = max_turns
        self.metrics = PerformanceMetrics()

    def run_single_game(self, mcts_config, verbose=False, max_moves=200):
        """Run a single game with given MCTS configuration"""
        game = Chess5D(self.max_time, self.max_turns)
        game_state = game.get_initial_state()
        mcts = MCTS(game, mcts_config)

        move_count = 0
        search_times = []
        nodes_expanded = []
        move_history = []

        try:
            while move_count < max_moves:
                start_time = time.time()

                # MCTS search
                mcts_prob_s, mcts_prob_e = mcts.search(game_state)

                search_time = time.time() - start_time
                search_times.append(search_time)

                # Select best move
                index_s = cp.unravel_index(cp.argmax(mcts_prob_s), mcts_prob_s.shape)
                index_e = game._pick_end_move_org(index_s, game_state, mcts_prob_e)

                action = f"({game.convert_timeline_opposite(index_s[0].item())}T{index_s[1].item() + 1})" \
                         f"{chr(96 + index_s[3].item() + 1)}{index_s[2].item() + 1}>>" \
                         f"({index_e['timeline']}T{index_e['turn']}){chr(96 + index_e['file'])}{index_e['rank']}"

                move_history.append({
                    'move': action,
                    'player': game_state.player,
                    'search_time': search_time
                })

                if verbose:
                    print(f"Move {move_count + 1} ({game_state.player}): {action} ({search_time:.3f}s)")

                # Make move
                game.make_move(game_state, action)
                move_count += 1

                if game_state.is_terminal:
                    break

        except Exception as e:
            if verbose:
                print(f"Game ended with exception: {type(e).__name__}")

        # Determine winner and end type
        if game_state.is_terminal:
            if game_state.value == 1:
                winner = game_state.prev_player
                end_type = 'checkmate'
            elif game_state.value == 0:
                winner = 'draw'
                end_type = 'stalemate'
            else:
                winner = 'draw'
                end_type = 'draw_loss'
        else:
            winner = 'incomplete'
            end_type = 'max_moves_reached'

        return {
            'winner': winner,
            'moves': move_count,
            'search_times': search_times,
            'terminal_value': game_state.value,
            'end_type': end_type,
            'move_history': move_history,
            'final_game_string': game_state.game_string
        }

    def run_test_suite(self, configs, games_per_config=10, verbose=False):
        """Run multiple games for each MCTS configuration"""
        results = {}

        for config_name, config in configs.items():
            print(f"\nTesting configuration: {config_name}")
            print(f"Parameters: {config}")

            config_metrics = PerformanceMetrics()
            config_results = []

            for game_num in range(games_per_config):
                print(f"  Game {game_num + 1}/{games_per_config}...", end=' ')

                game_result = self.run_single_game(config, verbose=verbose)

                config_metrics.record_game(
                    game_result['winner'],
                    game_result['moves'],
                    game_result['search_times'],
                    game_result['terminal_value'],
                    game_result['end_type'],
                    [len(game_result['search_times'])]  # Proxy for nodes expanded
                )

                config_results.append(game_result)
                print(f"Winner: {game_result['winner']}, Moves: {game_result['moves']}")

            results[config_name] = {
                'config': config,
                'metrics': config_metrics.calculate_statistics(),
                'games': config_results
            }

        return results

    def compare_configurations(self, results):
        """Compare different MCTS configurations"""
        comparison = {}

        for config_name, data in results.items():
            metrics = data['metrics']
            comparison[config_name] = {
                'win_rate_white': metrics['white_win_rate'],
                'win_rate_black': metrics['black_win_rate'],
                'avg_moves': metrics['avg_moves_per_game'],
                'avg_search_time': metrics['avg_search_time_per_move'],
                'checkmate_rate': metrics['checkmate_rate'],
                'efficiency_score': self._calculate_efficiency(metrics)
            }

        return comparison

    def _calculate_efficiency(self, metrics):
        """Calculate an efficiency score based on multiple factors"""
        # Higher is better: quick decisive wins
        checkmate_bonus = metrics['checkmate_rate'] * 100
        speed_bonus = 1 / (metrics['avg_search_time_per_move'] + 0.1) * 10
        decisiveness = (metrics['white_win_rate'] + metrics['black_win_rate']) * 50

        return checkmate_bonus + speed_bonus + decisiveness

def main():
    print("=" * 80)
    print("5D Chess AI Performance Testing Suite")
    print("=" * 80)

    tester = MCTSPerformanceTester(max_time=1, max_turns=25)

    # Define test configurations
    configs = {
        'baseline': {
            'num_searches': 20,
            'C': 1.41
        },
        'deep_search': {
            'num_searches': 50,
            'C': 1.41
        },
        'exploration_focused': {
            'num_searches': 20,
            'C': 2.0
        },
        'exploitation_focused': {
            'num_searches': 20,
            'C': 1.0
        },
        'balanced_deep': {
            'num_searches': 35,
            'C': 1.6
        }
    }

    # Run test suite
    print("\nRunning test suite...")
    results = tester.run_test_suite(configs, games_per_config=5, verbose=False)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/performance_results_{timestamp}.json"

    # Convert to serializable format
    serializable_results = {}
    for config_name, data in results.items():
        serializable_results[config_name] = {
            'config': data['config'],
            'metrics': data['metrics'],
            'game_count': len(data['games'])
        }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Display comparison
    print("\n" + "=" * 80)
    print("Configuration Comparison")
    print("=" * 80)

    comparison = tester.compare_configurations(results)

    for config_name, metrics in comparison.items():
        print(f"\n{config_name}:")
        print(f"  White Win Rate: {metrics['win_rate_white']:.2%}")
        print(f"  Black Win Rate: {metrics['win_rate_black']:.2%}")
        print(f"  Avg Moves: {metrics['avg_moves']:.1f}")
        print(f"  Avg Search Time: {metrics['avg_search_time']:.3f}s")
        print(f"  Checkmate Rate: {metrics['checkmate_rate']:.2%}")
        print(f"  Efficiency Score: {metrics['efficiency_score']:.2f}")

    # Find best configuration
    best_config = max(comparison.items(), key=lambda x: x[1]['efficiency_score'])
    print(f"\n{'=' * 80}")
    print(f"Best Configuration: {best_config[0]}")
    print(f"Efficiency Score: {best_config[1]['efficiency_score']:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
