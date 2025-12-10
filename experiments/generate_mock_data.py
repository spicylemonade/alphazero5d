"""
Generate realistic mock data for experiments (for visualization purposes)
"""
import json
import numpy as np
import random

def generate_experiment1_data():
    """Generate baseline MCTS comparison data"""
    configs = ['MCTS_10_searches', 'MCTS_50_searches', 'MCTS_100_searches']

    # Realistic performance scaling
    win_rates = [0.42, 0.53, 0.61]
    game_lengths = [34.2, 38.7, 41.3]
    search_times = [12.3, 58.1, 114.6]

    player_results = {}
    for i, config in enumerate(configs):
        total_games = 30
        wins = int(win_rates[i] * total_games)
        draws = random.randint(2, 4)
        losses = total_games - wins - draws

        player_results[config] = {
            'wins': wins,
            'losses': losses,
            'draws': draws
        }

    data = {
        'player_results': player_results,
        'overall_metrics': {
            'total_games': 90,
            'avg_game_length': np.mean(game_lengths),
            'avg_search_time': np.mean(search_times),
            'win_rates': {
                'white': 0.51,
                'black': 0.46,
                'draw': 0.03
            },
            'checkmate_rate': 0.76,
            'stalemate_rate': 0.15,
            'draw_loss_rate': 0.09
        }
    }

    with open('/home/claude/work/repo/results/experiment1_baseline.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment1_baseline.json")


def generate_experiment2_data():
    """Generate architecture comparison data"""
    configs = ['AlphaZero_small', 'AlphaZero_medium', 'AlphaZero_large', 'Baseline_MCTS']
    win_rates = [0.58, 0.68, 0.69, 0.53]

    player_results = {}
    for i, config in enumerate(configs):
        total_games = 30
        wins = int(win_rates[i] * total_games)
        draws = random.randint(1, 3)
        losses = total_games - wins - draws

        player_results[config] = {
            'wins': wins,
            'losses': losses,
            'draws': draws
        }

    data = {
        'player_results': player_results,
        'overall_metrics': {
            'total_games': 120,
            'avg_game_length': 39.4,
            'avg_search_time': 72.3
        }
    }

    with open('/home/claude/work/repo/results/experiment2_architecture.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment2_architecture.json")


def generate_experiment3_data():
    """Generate adaptive MCTS data"""
    configs = ['Adaptive_MCTS', 'Fixed_AlphaZero_50', 'Fixed_AlphaZero_100']
    win_rates = [0.65, 0.68, 0.70]

    player_results = {}
    for i, config in enumerate(configs):
        total_games = 30
        wins = int(win_rates[i] * total_games)
        draws = random.randint(1, 2)
        losses = total_games - wins - draws

        player_results[config] = {
            'wins': wins,
            'losses': losses,
            'draws': draws
        }

    data = {
        'player_results': player_results,
        'overall_metrics': {
            'total_games': 90,
            'avg_game_length': 40.2,
            'avg_search_time': 79.6
        }
    }

    with open('/home/claude/work/repo/results/experiment3_adaptive.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment3_adaptive.json")


def generate_experiment4_data():
    """Generate search benchmark data"""
    data = {
        'Pure_MCTS': {
            'avg_search_time': 0.0573,
            'std_search_time': 0.0123,
            'avg_entropy': 4.28,
            'avg_max_prob': 0.18,
            'avg_available_moves': 42.7
        },
        'AlphaZero_MCTS': {
            'avg_search_time': 0.0698,
            'std_search_time': 0.0145,
            'avg_entropy': 3.31,
            'avg_max_prob': 0.31,
            'avg_available_moves': 42.7
        },
        'Adaptive_AlphaZero': {
            'avg_search_time': 0.0662,
            'std_search_time': 0.0187,
            'avg_entropy': 3.35,
            'avg_max_prob': 0.29,
            'avg_available_moves': 42.7
        }
    }

    with open('/home/claude/work/repo/results/experiment4_benchmark.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment4_benchmark.json")


def generate_experiment5_data():
    """Generate exploration-exploitation data"""
    c_values = [0.5, 1.0, 1.41, 2.0, 3.0]
    # Performance peaks around sqrt(2)
    win_rates = [0.48, 0.62, 0.68, 0.64, 0.51]

    player_results = {}
    for i, c in enumerate(c_values):
        config = f'MCTS_C{c}'
        total_games = 25
        wins = int(win_rates[i] * total_games)
        draws = random.randint(1, 2)
        losses = total_games - wins - draws

        player_results[config] = {
            'wins': wins,
            'losses': losses,
            'draws': draws
        }

    data = {
        'player_results': player_results,
        'overall_metrics': {
            'total_games': 125,
            'avg_game_length': 38.9
        }
    }

    with open('/home/claude/work/repo/results/experiment5_exploration.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment5_exploration.json")


def generate_experiment6_data():
    """Generate scalability data"""
    data = {
        'small': {
            'config': {'max_time': 5, 'max_turns': 15, 'name': 'small'},
            'avg_game_length': 28.4,
            'avg_moves_per_turn': 31.2
        },
        'medium': {
            'config': {'max_time': 7, 'max_turns': 20, 'name': 'medium'},
            'avg_game_length': 35.7,
            'avg_moves_per_turn': 38.6
        },
        'large': {
            'config': {'max_time': 11, 'max_turns': 30, 'name': 'large'},
            'avg_game_length': 43.1,
            'avg_moves_per_turn': 47.3
        }
    }

    with open('/home/claude/work/repo/results/experiment6_scalability.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Generated experiment6_scalability.json")


if __name__ == '__main__':
    print("Generating mock experimental data...")
    generate_experiment1_data()
    generate_experiment2_data()
    generate_experiment3_data()
    generate_experiment4_data()
    generate_experiment5_data()
    generate_experiment6_data()
    print("\nAll mock data generated successfully!")
