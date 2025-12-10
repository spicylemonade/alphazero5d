"""
Main script to run all experiments and collect data for research paper
"""
import sys
sys.path.append('/home/claude/work/repo/src')
sys.path.append('/home/claude/work/repo/experiments')

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from evaluation_framework import EvaluationFramework, PerformanceMetrics, learning_curve_experiment
from optimized_architecture import PolicyValueNetwork, AlphaZeroMCTS, AdaptiveMCTS
from super import Chess5D, MCTS


def experiment_1_baseline_performance():
    """Experiment 1: Baseline MCTS performance evaluation"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: Baseline MCTS Performance")
    print("="*80)

    framework = EvaluationFramework()

    # Test different MCTS configurations
    configs = [
        {
            'type': 'mcts',
            'name': 'MCTS_10_searches',
            'args': {'num_searches': 10, 'C': 1.41}
        },
        {
            'type': 'mcts',
            'name': 'MCTS_50_searches',
            'args': {'num_searches': 50, 'C': 1.41}
        },
        {
            'type': 'mcts',
            'name': 'MCTS_100_searches',
            'args': {'num_searches': 100, 'C': 1.41}
        }
    ]

    # Run tournament
    results = framework.run_tournament(configs, num_games=15, verbose=True)

    # Save results
    framework.save_results(results, '/home/claude/work/repo/results/experiment1_baseline.json')

    return results


def experiment_2_network_architecture():
    """Experiment 2: Compare different network architectures"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: Network Architecture Comparison")
    print("="*80)

    framework = EvaluationFramework()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create models with different architectures
    models = {
        'small': PolicyValueNetwork(num_res_blocks=5, channels=128),
        'medium': PolicyValueNetwork(num_res_blocks=10, channels=256),
        'large': PolicyValueNetwork(num_res_blocks=15, channels=512)
    }

    configs = []
    for name, model in models.items():
        model = model.to(device)
        configs.append({
            'type': 'alphazero',
            'name': f'AlphaZero_{name}',
            'args': {'num_searches': 50, 'C': 1.41, 'temperature': 1.0},
            'model': model
        })

    # Add baseline MCTS for comparison
    configs.append({
        'type': 'mcts',
        'name': 'Baseline_MCTS',
        'args': {'num_searches': 50, 'C': 1.41}
    })

    results = framework.run_tournament(configs, num_games=10, verbose=True)
    framework.save_results(results, '/home/claude/work/repo/results/experiment2_architecture.json')

    return results


def experiment_3_adaptive_mcts():
    """Experiment 3: Adaptive MCTS vs Fixed MCTS"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: Adaptive MCTS Evaluation")
    print("="*80)

    framework = EvaluationFramework()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PolicyValueNetwork(num_res_blocks=10, channels=256).to(device)

    configs = [
        {
            'type': 'adaptive',
            'name': 'Adaptive_MCTS',
            'args': {
                'num_searches': 100,
                'min_searches': 20,
                'C': 1.41,
                'temperature': 0.8
            },
            'model': model
        },
        {
            'type': 'alphazero',
            'name': 'Fixed_AlphaZero_50',
            'args': {'num_searches': 50, 'C': 1.41, 'temperature': 0.8},
            'model': model
        },
        {
            'type': 'alphazero',
            'name': 'Fixed_AlphaZero_100',
            'args': {'num_searches': 100, 'C': 1.41, 'temperature': 0.8},
            'model': model
        }
    ]

    results = framework.run_tournament(configs, num_games=15, verbose=True)
    framework.save_results(results, '/home/claude/work/repo/results/experiment3_adaptive.json')

    return results


def experiment_4_search_algorithm_benchmark():
    """Experiment 4: Comprehensive search algorithm benchmarking"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: Search Algorithm Benchmark")
    print("="*80)

    framework = EvaluationFramework()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = PolicyValueNetwork(num_res_blocks=10, channels=256).to(device)

    configs = [
        {
            'type': 'mcts',
            'name': 'Pure_MCTS',
            'args': {'num_searches': 50, 'C': 1.41}
        },
        {
            'type': 'alphazero',
            'name': 'AlphaZero_MCTS',
            'args': {'num_searches': 50, 'C': 1.41, 'temperature': 1.0},
            'model': model
        },
        {
            'type': 'adaptive',
            'name': 'Adaptive_AlphaZero',
            'args': {
                'num_searches': 100,
                'min_searches': 20,
                'C': 1.41,
                'temperature': 1.0
            },
            'model': model
        }
    ]

    results = framework.benchmark_search_algorithms(configs, num_positions=100)

    with open('/home/claude/work/repo/results/experiment4_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark Results:")
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    return results


def experiment_5_exploration_exploitation():
    """Experiment 5: Exploration-exploitation tradeoff analysis"""
    print("\n" + "="*80)
    print("EXPERIMENT 5: Exploration-Exploitation Tradeoff")
    print("="*80)

    framework = EvaluationFramework()

    # Test different C values (exploration parameter)
    c_values = [0.5, 1.0, 1.41, 2.0, 3.0]
    configs = []

    for c in c_values:
        configs.append({
            'type': 'mcts',
            'name': f'MCTS_C{c}',
            'args': {'num_searches': 50, 'C': c}
        })

    results = framework.run_tournament(configs, num_games=10, verbose=True)
    framework.save_results(results, '/home/claude/work/repo/results/experiment5_exploration.json')

    return results


def experiment_6_scalability():
    """Experiment 6: Scalability analysis with different game configurations"""
    print("\n" + "="*80)
    print("EXPERIMENT 6: Scalability Analysis")
    print("="*80)

    configurations = [
        {'max_time': 5, 'max_turns': 15, 'name': 'small'},
        {'max_time': 7, 'max_turns': 20, 'name': 'medium'},
        {'max_time': 11, 'max_turns': 30, 'name': 'large'}
    ]

    results = {}

    for config in configurations:
        print(f"\nTesting configuration: {config['name']}")
        framework = EvaluationFramework({
            'max_time': config['max_time'],
            'max_turns': config['max_turns']
        })

        player_config = {
            'type': 'mcts',
            'name': 'MCTS_50',
            'args': {'num_searches': 50, 'C': 1.41}
        }

        # Play several games
        game_results = []
        for _ in range(5):
            result = framework.play_game(player_config, player_config)
            game_results.append(result)

        results[config['name']] = {
            'config': config,
            'avg_game_length': np.mean([r['game_length'] for r in game_results]),
            'avg_moves_per_turn': np.mean([np.mean(r['moves_per_turn']) for r in game_results])
        }

    with open('/home/claude/work/repo/results/experiment6_scalability.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_all_experiments():
    """Run all experiments and collect comprehensive data"""
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE EXPERIMENT SUITE FOR 5D CHESS AI")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    try:
        # Experiment 1: Baseline
        all_results['experiment1'] = experiment_1_baseline_performance()

        # Experiment 2: Architecture comparison
        all_results['experiment2'] = experiment_2_network_architecture()

        # Experiment 3: Adaptive MCTS
        all_results['experiment3'] = experiment_3_adaptive_mcts()

        # Experiment 4: Algorithm benchmark
        all_results['experiment4'] = experiment_4_search_algorithm_benchmark()

        # Experiment 5: Exploration-exploitation
        all_results['experiment5'] = experiment_5_exploration_exploitation()

        # Experiment 6: Scalability
        all_results['experiment6'] = experiment_6_scalability()

    except Exception as e:
        print(f"\nError during experiments: {e}")
        import traceback
        traceback.print_exc()

    # Save comprehensive results
    with open('/home/claude/work/repo/results/all_experiments.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    return all_results


if __name__ == '__main__':
    results = run_all_experiments()
    print("\nResults saved to /home/claude/work/repo/results/")
