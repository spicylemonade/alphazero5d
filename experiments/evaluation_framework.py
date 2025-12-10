"""
Comprehensive Evaluation Framework for 5D Chess AI
"""
import sys
sys.path.append('/home/claude/work/repo/src')

import torch
import numpy as np
import cupy as cp
from collections import defaultdict
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from super import Chess5D, MCTS
from optimized_architecture import (
    PolicyValueNetwork, AlphaZeroMCTS, AdaptiveMCTS,
    create_training_sample
)


class PerformanceMetrics:
    """Track comprehensive performance metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.games_played = 0
        self.wins = {'white': 0, 'black': 0, 'draw': 0}
        self.avg_game_length = []
        self.avg_moves_per_turn = []
        self.search_times = []
        self.position_evaluations = []
        self.move_quality_scores = []
        self.timeline_usage = []
        self.checkmate_count = 0
        self.stalemate_count = 0
        self.draw_loss_count = 0
        self.exploration_depth = []
        self.branching_factors = []

    def record_game(self, winner, game_length, moves_per_turn, checkmate=False,
                   stalemate=False, draw_loss=False):
        self.games_played += 1
        self.wins[winner] += 1
        self.avg_game_length.append(game_length)
        self.avg_moves_per_turn.extend(moves_per_turn)

        if checkmate:
            self.checkmate_count += 1
        if stalemate:
            self.stalemate_count += 1
        if draw_loss:
            self.draw_loss_count += 1

    def record_search(self, search_time, evaluations, branching_factor, depth):
        self.search_times.append(search_time)
        self.position_evaluations.append(evaluations)
        self.branching_factors.append(branching_factor)
        self.exploration_depth.append(depth)

    def get_summary(self):
        return {
            'total_games': self.games_played,
            'win_rates': {
                'white': self.wins['white'] / max(self.games_played, 1),
                'black': self.wins['black'] / max(self.games_played, 1),
                'draw': self.wins['draw'] / max(self.games_played, 1)
            },
            'avg_game_length': np.mean(self.avg_game_length) if self.avg_game_length else 0,
            'avg_moves_per_turn': np.mean(self.avg_moves_per_turn) if self.avg_moves_per_turn else 0,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'avg_evaluations': np.mean(self.position_evaluations) if self.position_evaluations else 0,
            'avg_branching_factor': np.mean(self.branching_factors) if self.branching_factors else 0,
            'avg_exploration_depth': np.mean(self.exploration_depth) if self.exploration_depth else 0,
            'checkmate_rate': self.checkmate_count / max(self.games_played, 1),
            'stalemate_rate': self.stalemate_count / max(self.games_played, 1),
            'draw_loss_rate': self.draw_loss_count / max(self.games_played, 1)
        }


class EvaluationFramework:
    """Comprehensive evaluation framework for testing AI performance"""

    def __init__(self, game_config={'max_time': 11, 'max_turns': 30}):
        self.game_config = game_config
        self.metrics = PerformanceMetrics()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def play_game(self, player1_config, player2_config, verbose=False):
        """
        Play a single game between two agents

        Args:
            player1_config: {'type': 'mcts'/'alphazero'/'adaptive', 'args': {...}, 'model': model}
            player2_config: Same as player1_config
            verbose: Print game moves

        Returns:
            Game result dictionary
        """
        game = Chess5D(**self.game_config)
        state = game.get_initial_state()

        # Initialize players
        player1 = self._create_player(player1_config, game)
        player2 = self._create_player(player2_config, game)

        move_history = []
        moves_per_turn = []
        game_length = 0

        current_player = player1
        player_name = 'white'

        try:
            while not state.is_terminal:
                turn_start = time.time()

                # Get move from current player
                if player1_config['type'] == 'mcts' or current_player == player1:
                    policy_start, policy_end = current_player.search(state)
                else:
                    policy_start, policy_end = current_player.search(state)

                # Select best move
                index_s = cp.unravel_index(cp.argmax(policy_start), policy_start.shape)
                index_e = game._pick_end_move_org(index_s, state, policy_end)

                action = (
                    f"({game.convert_timeline_opposite(index_s[0].item())}T{index_s[1].item() + 1})"
                    f"{chr(96 + index_s[3].item() + 1)}{index_s[2].item() + 1}>>"
                    f"({index_e['timeline']}T{index_e['turn']})"
                    f"{chr(96 + index_e['file'])}{index_e['rank']}"
                )

                if verbose:
                    print(f"{player_name} move: {action}")

                # Make move
                game.make_move(state, action)

                turn_time = time.time() - turn_start
                moves_per_turn.append(len(state.moves) if state.moves else 0)
                game_length += 1

                move_history.append({
                    'player': player_name,
                    'action': action,
                    'time': turn_time,
                    'available_moves': len(state.moves) if state.moves else 0
                })

                # Switch players
                current_player = player2 if current_player == player1 else player1
                player_name = 'black' if player_name == 'white' else 'white'

                # Safety limit
                if game_length > 200:
                    state.is_terminal = True
                    state.value = 0
                    break

        except Exception as e:
            print(f"Game error: {e}")
            state.is_terminal = True
            state.value = 0

        # Determine winner
        if state.value > 0:
            winner = player_name  # Last player to move won
        elif state.value < 0:
            winner = 'black' if player_name == 'white' else 'white'
        else:
            winner = 'draw'

        return {
            'winner': winner,
            'game_length': game_length,
            'moves_per_turn': moves_per_turn,
            'move_history': move_history,
            'final_value': state.value
        }

    def _create_player(self, config, game):
        """Create player agent from configuration"""
        player_type = config['type']
        args = config['args']
        model = config.get('model', None)

        if player_type == 'mcts':
            return MCTS(game, args)
        elif player_type == 'alphazero':
            return AlphaZeroMCTS(game, args, model, self.device)
        elif player_type == 'adaptive':
            return AdaptiveMCTS(game, args, model, self.device)
        else:
            raise ValueError(f"Unknown player type: {player_type}")

    def run_tournament(self, players_configs, num_games=10, verbose=False):
        """
        Run round-robin tournament between multiple players

        Args:
            players_configs: List of player configurations
            num_games: Games per matchup
            verbose: Print progress

        Returns:
            Tournament results
        """
        results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0})
        matchup_results = []

        # Round-robin tournament
        for i, p1_config in enumerate(players_configs):
            for j, p2_config in enumerate(players_configs):
                if i >= j:  # Skip self-play and duplicate matchups
                    continue

                p1_name = p1_config.get('name', f'Player{i}')
                p2_name = p2_config.get('name', f'Player{j}')

                if verbose:
                    print(f"\n{p1_name} vs {p2_name}")

                matchup_wins = {p1_name: 0, p2_name: 0, 'draw': 0}

                for game_num in tqdm(range(num_games), desc=f"{p1_name} vs {p2_name}"):
                    result = self.play_game(p1_config, p2_config, verbose=False)

                    # Record results
                    if result['winner'] == 'white':
                        results[p1_name]['wins'] += 1
                        results[p2_name]['losses'] += 1
                        matchup_wins[p1_name] += 1
                    elif result['winner'] == 'black':
                        results[p2_name]['wins'] += 1
                        results[p1_name]['losses'] += 1
                        matchup_wins[p2_name] += 1
                    else:
                        results[p1_name]['draws'] += 1
                        results[p2_name]['draws'] += 1
                        matchup_wins['draw'] += 1

                    self.metrics.record_game(
                        result['winner'],
                        result['game_length'],
                        result['moves_per_turn']
                    )

                matchup_results.append({
                    'player1': p1_name,
                    'player2': p2_name,
                    'results': matchup_wins
                })

        return {
            'player_results': dict(results),
            'matchup_results': matchup_results,
            'overall_metrics': self.metrics.get_summary()
        }

    def benchmark_search_algorithms(self, configs, num_positions=50):
        """
        Benchmark different search algorithms on random positions

        Args:
            configs: List of algorithm configurations
            num_positions: Number of test positions

        Returns:
            Benchmark results
        """
        game = Chess5D(**self.game_config)
        results = defaultdict(list)

        for pos_num in tqdm(range(num_positions), desc="Benchmarking positions"):
            # Generate random position by playing random moves
            state = game.get_initial_state()
            for _ in range(np.random.randint(5, 20)):
                if state.is_terminal:
                    break
                try:
                    action, _, _ = game.pick_choice(state, state.choices_start, state.choices_end, False)
                    game.make_move(state, action)
                except:
                    break

            if state.is_terminal:
                continue

            # Test each algorithm on this position
            for config in configs:
                algo_name = config.get('name', 'unnamed')
                player = self._create_player(config, game)

                start_time = time.time()
                policy_start, policy_end = player.search(state.copy())
                search_time = time.time() - start_time

                # Calculate move quality metrics
                entropy = -np.sum(cp.asnumpy(policy_start) * np.log(cp.asnumpy(policy_start) + 1e-10))
                max_prob = float(cp.max(policy_start))

                results[algo_name].append({
                    'search_time': search_time,
                    'entropy': entropy,
                    'max_probability': max_prob,
                    'available_moves': len(state.moves) if state.moves else 0
                })

        # Aggregate results
        summary = {}
        for algo_name, data in results.items():
            summary[algo_name] = {
                'avg_search_time': np.mean([d['search_time'] for d in data]),
                'std_search_time': np.std([d['search_time'] for d in data]),
                'avg_entropy': np.mean([d['entropy'] for d in data]),
                'avg_max_prob': np.mean([d['max_probability'] for d in data]),
                'avg_available_moves': np.mean([d['available_moves'] for d in data])
            }

        return summary

    def save_results(self, results, filename):
        """Save evaluation results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filename}")


def learning_curve_experiment(num_iterations=10, games_per_iteration=20):
    """
    Track learning progress over self-play iterations

    Returns:
        Learning curve data
    """
    game_config = {'max_time': 11, 'max_turns': 30}
    model = PolicyValueNetwork(**game_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    learning_data = []

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")

        # Self-play games
        framework = EvaluationFramework(game_config)

        config = {
            'type': 'alphazero',
            'name': f'AlphaZero_iter{iteration}',
            'args': {
                'num_searches': 50,
                'C': 1.41,
                'temperature': 1.0 if iteration < 3 else 0.5
            },
            'model': model
        }

        # Play games against baseline MCTS
        baseline_config = {
            'type': 'mcts',
            'name': 'Baseline_MCTS',
            'args': {
                'num_searches': 50,
                'C': 1.41
            }
        }

        tournament_results = framework.run_tournament(
            [config, baseline_config],
            num_games=games_per_iteration
        )

        learning_data.append({
            'iteration': iteration,
            'metrics': framework.metrics.get_summary(),
            'tournament_results': tournament_results
        })

    return learning_data
