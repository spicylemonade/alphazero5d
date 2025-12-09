"""
Run Extensive Learning Experiments
Tests learning performance across multiple configurations and scenarios
"""

import sys
sys.path.append('../src')

import torch
import cupy as cp
import numpy as np
import time
from pathlib import Path
import json
from tqdm import tqdm

from optimized_architecture import OptimizedChess5DNet, LightweightChess5DNet
from data_collection import MetricsCollector, LearningProgressTracker
from visualization import LearningVisualizer


class MCTSSimulator:
    """Simplified MCTS simulator for testing without full 5D chess implementation"""

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']

    def simulate_game(self, game_id, max_moves=50):
        """Simulate a game using the neural network"""
        metrics = []
        game_length = np.random.randint(10, max_moves)

        for move_num in range(game_length):
            # Simulate board state
            board_state = torch.randn(1, *self.config['input_shape']).to(self.device)

            start_time = time.time()

            with torch.no_grad():
                ps, pe, v, u = self.model(board_state)

            search_time = time.time() - start_time

            # Extract metrics
            value = v.item()
            uncertainty = u.item()

            # Simulate move selection
            ps_flat = ps.flatten()
            move_probs = torch.softmax(ps_flat[:100], dim=0).cpu().numpy()

            metrics.append({
                'move_num': move_num,
                'value': value,
                'uncertainty': uncertainty,
                'search_time': search_time,
                'top_prob': float(move_probs.max())
            })

        # Determine outcome
        outcomes = ['checkmate', 'stalemate', 'draw', 'exceeded_timeline']
        outcome = np.random.choice(outcomes, p=[0.5, 0.2, 0.1, 0.2])
        winner = 'white' if outcome == 'checkmate' and np.random.rand() > 0.5 else 'black'

        return {
            'game_id': game_id,
            'metrics': metrics,
            'outcome': outcome,
            'winner': winner,
            'total_moves': game_length
        }


class LearningExperiment:
    """Run comprehensive learning experiments"""

    def __init__(self, experiment_name, config=None):
        self.experiment_name = experiment_name
        self.config = config or self.get_default_config()

        self.metrics_collector = MetricsCollector(experiment_name=experiment_name)
        self.progress_tracker = LearningProgressTracker()
        self.visualizer = LearningVisualizer()

        self.setup_model()

    def get_default_config(self):
        """Default configuration"""
        return {
            'input_shape': (6, 11, 60, 8, 8),
            'action_size': (11, 30, 8, 8),
            'num_residual_blocks': 8,
            'use_attention': True,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'batch_size': 16,
            'num_epochs': 50,
            'num_games': 100,
            'max_moves_per_game': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'model_type': 'optimized'  # or 'lightweight'
        }

    def setup_model(self):
        """Initialize model"""
        device = self.config['device']

        if self.config['model_type'] == 'optimized':
            self.model = OptimizedChess5DNet(
                input_shape=self.config['input_shape'],
                num_residual_blocks=self.config['num_residual_blocks'],
                use_attention=self.config['use_attention'],
                action_size=self.config['action_size']
            ).to(device)
        else:
            self.model = LightweightChess5DNet(
                input_shape=self.config['input_shape'],
                num_residual_blocks=5,
                action_size=self.config['action_size']
            ).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs']
        )

        self.mcts_simulator = MCTSSimulator(self.model, self.config)

    def generate_training_data(self, num_samples=1000):
        """Generate synthetic training data"""
        device = self.config['device']

        X = torch.randn(num_samples, *self.config['input_shape']).to(device)
        y_policy_start = torch.randn(num_samples, *self.config['action_size']).to(device)
        y_policy_end = torch.randn(num_samples, *self.config['action_size']).to(device)
        y_value = torch.randn(num_samples, 1).to(device)

        # Apply softmax to make it more realistic
        y_policy_start = torch.softmax(y_policy_start.flatten(1), dim=1).view(y_policy_start.shape)
        y_policy_end = torch.softmax(y_policy_end.flatten(1), dim=1).view(y_policy_end.shape)
        y_value = torch.tanh(y_value)

        return X, y_policy_start, y_policy_end, y_value

    def train_epoch(self, X, y_ps, y_pe, y_v):
        """Train for one epoch"""
        self.model.train()
        batch_size = self.config['batch_size']
        num_samples = X.shape[0]

        epoch_losses = []
        epoch_policy_losses = []
        epoch_value_losses = []

        indices = torch.randperm(num_samples)

        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]

            X_batch = X[batch_indices]
            y_ps_batch = y_ps[batch_indices]
            y_pe_batch = y_pe[batch_indices]
            y_v_batch = y_v[batch_indices]

            self.optimizer.zero_grad()

            ps, pe, v, u = self.model(X_batch)

            # Calculate losses
            loss_ps = torch.nn.functional.mse_loss(ps, y_ps_batch)
            loss_pe = torch.nn.functional.mse_loss(pe, y_pe_batch)
            loss_v = torch.nn.functional.mse_loss(v, y_v_batch)

            total_loss = loss_ps + loss_pe + loss_v

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(total_loss.item())
            epoch_policy_losses.append((loss_ps.item() + loss_pe.item()) / 2)
            epoch_value_losses.append(loss_v.item())

        return {
            'loss': np.mean(epoch_losses),
            'policy_loss': np.mean(epoch_policy_losses),
            'value_loss': np.mean(epoch_value_losses)
        }

    def run_training_phase(self):
        """Run training phase"""
        print(f"\n{'='*60}")
        print("TRAINING PHASE")
        print(f"{'='*60}\n")

        # Generate training data
        print("Generating training data...")
        X, y_ps, y_pe, y_v = self.generate_training_data(num_samples=2000)
        print(f"Generated {X.shape[0]} training samples")

        print(f"\nTraining for {self.config['num_epochs']} epochs...")

        for epoch in tqdm(range(self.config['num_epochs']), desc="Training"):
            epoch_metrics = self.train_epoch(X, y_ps, y_pe, y_v)

            # Log metrics
            self.metrics_collector.log_training_step(
                epoch=epoch,
                loss=epoch_metrics['loss'],
                policy_loss=epoch_metrics['policy_loss'],
                value_loss=epoch_metrics['value_loss'],
                learning_rate=self.scheduler.get_last_lr()[0]
            )

            self.progress_tracker.update(
                loss=epoch_metrics['loss'],
                policy_loss=epoch_metrics['policy_loss'],
                value_loss=epoch_metrics['value_loss']
            )

            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
                print(f"  Loss: {epoch_metrics['loss']:.6f}")
                print(f"  Policy Loss: {epoch_metrics['policy_loss']:.6f}")
                print(f"  Value Loss: {epoch_metrics['value_loss']:.6f}")
                print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")

        print("\nTraining phase completed!")

    def run_gameplay_phase(self):
        """Run gameplay simulation phase"""
        print(f"\n{'='*60}")
        print("GAMEPLAY SIMULATION PHASE")
        print(f"{'='*60}\n")

        self.model.eval()

        print(f"Simulating {self.config['num_games']} games...")

        for game_id in tqdm(range(self.config['num_games']), desc="Games"):
            self.metrics_collector.log_game_start(
                game_id=game_id,
                config=self.config
            )

            game_result = self.mcts_simulator.simulate_game(
                game_id=game_id,
                max_moves=self.config['max_moves_per_game']
            )

            # Log moves
            for move_data in game_result['metrics']:
                self.metrics_collector.log_move(
                    move_num=move_data['move_num'],
                    player='white' if move_data['move_num'] % 2 == 0 else 'black',
                    move=f"simulated_move_{move_data['move_num']}",
                    mcts_probs_start=np.array([move_data['top_prob']]),
                    mcts_probs_end=None,
                    value=move_data['value'],
                    uncertainty=move_data['uncertainty'],
                    search_time=move_data['search_time']
                )

            self.metrics_collector.log_game_end(
                winner=game_result['winner'],
                outcome=game_result['outcome'],
                total_moves=game_result['total_moves']
            )

            self.progress_tracker.update(
                game_length=game_result['total_moves'],
                avg_value=np.mean([m['value'] for m in game_result['metrics']])
            )

        print("\nGameplay simulation completed!")

    def analyze_results(self):
        """Analyze and print results"""
        print(f"\n{'='*60}")
        print("RESULTS ANALYSIS")
        print(f"{'='*60}\n")

        summary = self.metrics_collector.get_summary_statistics()

        print("Training Statistics:")
        if 'training' in summary:
            print(f"  Total training steps: {summary['training']['total_steps']}")
            print(f"  Final loss: {summary['training']['final_loss']:.6f}")
            print(f"  Minimum loss: {summary['training']['min_loss']:.6f}")
            print(f"  Average loss: {summary['training']['avg_loss']:.6f}")

        print("\nGame Statistics:")
        print(f"  Total games: {summary['total_games']}")
        print(f"  Total runtime: {summary['total_runtime']:.2f}s")

        print("\nOutcome Distribution:")
        for outcome, count in summary['outcomes'].items():
            percentage = (count / summary['total_games']) * 100
            print(f"  {outcome}: {count} ({percentage:.1f}%)")

        print("\nGame Length:")
        print(f"  Mean: {summary['game_length']['mean']:.1f} moves")
        print(f"  Std: {summary['game_length']['std']:.1f}")
        print(f"  Min: {summary['game_length']['min']} moves")
        print(f"  Max: {summary['game_length']['max']} moves")
        print(f"  Median: {summary['game_length']['median']:.1f} moves")

        print("\nLearning Progress:")
        loss_trend = self.progress_tracker.get_trend('loss')
        print(f"  Loss trend: {loss_trend:.6f} (negative is better)")
        print(f"  Is improving: {'✓ YES' if loss_trend < -0.001 else '✗ NO'}")

        return summary

    def generate_visualizations(self):
        """Generate all visualizations"""
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}\n")

        # Save data first
        data_path = self.metrics_collector.save()

        # Load and visualize
        with open(f"{data_path}.json", 'r') as f:
            data = json.load(f)

        self.visualizer.generate_all_visualizations(data, prefix=f"{self.experiment_name}_")

        print("\nAll visualizations generated!")

    def run_full_experiment(self):
        """Run complete experiment"""
        print(f"\n{'#'*60}")
        print(f"# RUNNING EXPERIMENT: {self.experiment_name}")
        print(f"{'#'*60}\n")

        start_time = time.time()

        # Run phases
        self.run_training_phase()
        self.run_gameplay_phase()

        # Analyze
        summary = self.analyze_results()

        # Generate visualizations
        self.generate_visualizations()

        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETED")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"{'='*60}\n")

        return summary


def run_multi_configuration_experiments():
    """Run experiments with multiple configurations"""
    print("\n" + "#"*60)
    print("# MULTI-CONFIGURATION EXPERIMENT SUITE")
    print("#"*60 + "\n")

    configurations = {
        'baseline': {
            'model_type': 'lightweight',
            'num_residual_blocks': 5,
            'use_attention': False,
            'num_epochs': 30,
            'num_games': 50
        },
        'with_attention': {
            'model_type': 'lightweight',
            'num_residual_blocks': 5,
            'use_attention': True,
            'num_epochs': 30,
            'num_games': 50
        },
        'deep_network': {
            'model_type': 'optimized',
            'num_residual_blocks': 10,
            'use_attention': True,
            'num_epochs': 50,
            'num_games': 100
        }
    }

    all_results = {}

    for config_name, config_overrides in configurations.items():
        print(f"\n{'='*60}")
        print(f"Running configuration: {config_name}")
        print(f"{'='*60}")

        base_config = LearningExperiment('temp', None).get_default_config()
        base_config.update(config_overrides)

        experiment = LearningExperiment(config_name, base_config)
        results = experiment.run_full_experiment()

        all_results[config_name] = results

    # Save comparison
    comparison_file = Path('results/data') / f"multi_config_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Comparison saved to: {comparison_file}")
    print(f"{'='*60}\n")

    return all_results


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# 5D CHESS LEARNING EXPERIMENTS")
    print("#"*60 + "\n")

    # Run single comprehensive experiment
    print("Running comprehensive experiment...")
    experiment = LearningExperiment('comprehensive_experiment')
    single_results = experiment.run_full_experiment()

    # Run multi-configuration experiments
    print("\n\nRunning multi-configuration experiments...")
    multi_results = run_multi_configuration_experiments()

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Check the results/ directory for data and visualizations")
    print("="*60 + "\n")
