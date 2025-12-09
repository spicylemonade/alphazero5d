"""
Comprehensive Data Collection System for Learning Analysis
Tracks and stores metrics during training and gameplay
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import pickle


class MetricsCollector:
    """Collects and organizes training and gameplay metrics"""

    def __init__(self, experiment_name=None, save_dir="results/data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        # Initialize storage
        self.metrics = defaultdict(list)
        self.game_history = []
        self.move_history = []
        self.mcts_stats = []
        self.training_stats = []

        # Performance tracking
        self.start_time = time.time()
        self.episode_times = []

    def log_game_start(self, game_id, config):
        """Log the start of a new game"""
        self.current_game = {
            'game_id': game_id,
            'config': config,
            'start_time': time.time(),
            'moves': [],
            'mcts_iterations': []
        }

    def log_move(self, move_num, player, move, mcts_probs_start, mcts_probs_end,
                 value, uncertainty=None, search_time=None):
        """Log a single move with MCTS statistics"""
        move_data = {
            'move_num': move_num,
            'player': player,
            'move': str(move[0]) if isinstance(move, tuple) else str(move),
            'value': float(value),
            'search_time': search_time,
            'timestamp': time.time()
        }

        if uncertainty is not None:
            move_data['uncertainty'] = float(uncertainty)

        # Store top-k move probabilities
        if mcts_probs_start is not None:
            flat_probs = mcts_probs_start.flatten()
            top_k = min(10, len(flat_probs))
            top_indices = np.argsort(flat_probs)[-top_k:]
            move_data['top_start_probs'] = {
                int(idx): float(flat_probs[idx]) for idx in top_indices
            }

        self.current_game['moves'].append(move_data)
        self.move_history.append(move_data)

    def log_mcts_search(self, node_count, depth, avg_value, exploration_rate):
        """Log MCTS search statistics"""
        mcts_data = {
            'node_count': node_count,
            'depth': depth,
            'avg_value': avg_value,
            'exploration_rate': exploration_rate,
            'timestamp': time.time()
        }
        self.mcts_stats.append(mcts_data)

    def log_game_end(self, winner, outcome, total_moves, reason=None):
        """Log game completion"""
        self.current_game['end_time'] = time.time()
        self.current_game['duration'] = self.current_game['end_time'] - self.current_game['start_time']
        self.current_game['winner'] = winner
        self.current_game['outcome'] = outcome
        self.current_game['total_moves'] = total_moves
        self.current_game['reason'] = reason

        self.game_history.append(self.current_game)
        self.episode_times.append(self.current_game['duration'])

        # Update aggregate metrics
        self.metrics['total_games'].append(len(self.game_history))
        self.metrics['avg_game_length'].append(np.mean([g['total_moves'] for g in self.game_history]))
        self.metrics['avg_game_duration'].append(np.mean(self.episode_times))

    def log_training_step(self, epoch, loss, policy_loss, value_loss, learning_rate):
        """Log training step metrics"""
        training_data = {
            'epoch': epoch,
            'loss': float(loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'learning_rate': float(learning_rate),
            'timestamp': time.time()
        }
        self.training_stats.append(training_data)

    def log_evaluation(self, win_rate, avg_moves, avg_value, confidence_interval=None):
        """Log evaluation metrics"""
        eval_data = {
            'win_rate': win_rate,
            'avg_moves': avg_moves,
            'avg_value': avg_value,
            'timestamp': time.time()
        }
        if confidence_interval:
            eval_data['confidence_interval'] = confidence_interval

        self.metrics['evaluations'].append(eval_data)

    def get_summary_statistics(self):
        """Generate summary statistics"""
        if not self.game_history:
            return {}

        outcomes = [g['outcome'] for g in self.game_history]
        move_counts = [g['total_moves'] for g in self.game_history]
        durations = [g['duration'] for g in self.game_history]

        summary = {
            'total_games': len(self.game_history),
            'total_runtime': time.time() - self.start_time,
            'outcomes': {
                'checkmate': outcomes.count('checkmate'),
                'stalemate': outcomes.count('stalemate'),
                'draw': outcomes.count('draw'),
                'exceeded_timeline': outcomes.count('exceeded_timeline')
            },
            'game_length': {
                'mean': np.mean(move_counts),
                'std': np.std(move_counts),
                'min': np.min(move_counts),
                'max': np.max(move_counts),
                'median': np.median(move_counts)
            },
            'game_duration': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations)
            }
        }

        if self.training_stats:
            losses = [s['loss'] for s in self.training_stats]
            summary['training'] = {
                'total_steps': len(self.training_stats),
                'final_loss': losses[-1],
                'min_loss': np.min(losses),
                'avg_loss': np.mean(losses)
            }

        return summary

    def save(self, suffix=""):
        """Save all collected data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_path = self.save_dir / f"{self.experiment_name}_{timestamp}{suffix}"

        # Save as JSON
        data_to_save = {
            'experiment_name': self.experiment_name,
            'summary': self.get_summary_statistics(),
            'game_history': self.game_history,
            'move_history': self.move_history,
            'mcts_stats': self.mcts_stats,
            'training_stats': self.training_stats,
            'metrics': dict(self.metrics)
        }

        with open(f"{base_path}.json", 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)

        # Save as pickle for faster loading
        with open(f"{base_path}.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)

        # Save CSV files for easy analysis
        if self.game_history:
            df_games = pd.DataFrame(self.game_history)
            df_games.to_csv(f"{base_path}_games.csv", index=False)

        if self.move_history:
            df_moves = pd.DataFrame(self.move_history)
            df_moves.to_csv(f"{base_path}_moves.csv", index=False)

        if self.training_stats:
            df_training = pd.DataFrame(self.training_stats)
            df_training.to_csv(f"{base_path}_training.csv", index=False)

        print(f"Data saved to {base_path}.*")
        return str(base_path)

    @classmethod
    def load(cls, filepath):
        """Load saved metrics from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        collector = cls(experiment_name=data['experiment_name'])
        collector.game_history = data['game_history']
        collector.move_history = data['move_history']
        collector.mcts_stats = data['mcts_stats']
        collector.training_stats = data['training_stats']
        collector.metrics = defaultdict(list, data['metrics'])

        return collector


class LearningProgressTracker:
    """Track learning progress over time"""

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics_history = defaultdict(list)

    def update(self, **metrics):
        """Update tracked metrics"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

    def get_trend(self, metric_name, window=None):
        """Calculate trend for a metric"""
        if window is None:
            window = self.window_size

        values = self.metrics_history.get(metric_name, [])
        if len(values) < 2:
            return 0.0

        recent = values[-window:]
        if len(recent) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(recent))
        coeffs = np.polyfit(x, recent, 1)
        return coeffs[0]  # Slope

    def get_moving_average(self, metric_name, window=None):
        """Calculate moving average"""
        if window is None:
            window = self.window_size

        values = self.metrics_history.get(metric_name, [])
        if not values:
            return []

        return pd.Series(values).rolling(window=window, min_periods=1).mean().tolist()

    def is_improving(self, metric_name, threshold=0.01):
        """Check if metric is improving"""
        trend = self.get_trend(metric_name)
        return trend > threshold

    def get_statistics(self, metric_name):
        """Get statistics for a metric"""
        values = self.metrics_history.get(metric_name, [])
        if not values:
            return {}

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'current': values[-1],
            'trend': self.get_trend(metric_name)
        }
