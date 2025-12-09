"""
Comprehensive Testing Framework for Learning System
Tests neural network, MCTS, and learning capabilities
"""

import sys
sys.path.append('../src')

import torch
import cupy as cp
import numpy as np
import time
from pathlib import Path
import json

# Import modules
from optimized_architecture import OptimizedChess5DNet, LightweightChess5DNet, count_parameters
from data_collection import MetricsCollector, LearningProgressTracker
from visualization import LearningVisualizer


class LearningExperiment:
    """Run controlled experiments to test learning capabilities"""

    def __init__(self, experiment_name, config=None):
        self.experiment_name = experiment_name
        self.config = config or self.get_default_config()
        self.metrics = MetricsCollector(experiment_name=experiment_name)
        self.tracker = LearningProgressTracker()

        # Setup models
        self.setup_models()

    def get_default_config(self):
        """Default experiment configuration"""
        return {
            'input_shape': (6, 11, 60, 8, 8),
            'action_size': (11, 30, 8, 8),
            'num_residual_blocks': 8,
            'use_attention': True,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_simulations': 20,
            'num_games': 50,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def setup_models(self):
        """Initialize neural network models"""
        print(f"\n{'='*60}")
        print(f"Setting up models for experiment: {self.experiment_name}")
        print(f"{'='*60}")

        device = self.config['device']
        print(f"Using device: {device}")

        # Create models
        self.model_optimized = OptimizedChess5DNet(
            input_shape=self.config['input_shape'],
            num_residual_blocks=self.config['num_residual_blocks'],
            use_attention=self.config['use_attention'],
            action_size=self.config['action_size']
        ).to(device)

        self.model_lightweight = LightweightChess5DNet(
            input_shape=self.config['input_shape'],
            num_residual_blocks=5,
            action_size=self.config['action_size']
        ).to(device)

        # Count parameters
        opt_params = count_parameters(self.model_optimized)
        light_params = count_parameters(self.model_lightweight)

        print(f"\nOptimized Model Parameters: {opt_params:,}")
        print(f"Lightweight Model Parameters: {light_params:,}")
        print(f"Parameter Ratio: {opt_params / light_params:.2f}x")

    def test_forward_pass(self):
        """Test neural network forward pass"""
        print(f"\n{'='*60}")
        print("Testing Forward Pass Performance")
        print(f"{'='*60}")

        device = self.config['device']
        batch_sizes = [1, 4, 16, 32]

        results = {}

        for batch_size in batch_sizes:
            # Create dummy input
            dummy_input = torch.randn(
                batch_size, *self.config['input_shape']
            ).to(device)

            # Test optimized model
            start_time = time.time()
            with torch.no_grad():
                ps, pe, v, u = self.model_optimized(dummy_input)
            opt_time = time.time() - start_time

            # Test lightweight model
            start_time = time.time()
            with torch.no_grad():
                ps_l, pe_l, v_l, u_l = self.model_lightweight(dummy_input)
            light_time = time.time() - start_time

            results[batch_size] = {
                'optimized_time': opt_time,
                'lightweight_time': light_time,
                'speedup': opt_time / light_time
            }

            print(f"\nBatch Size: {batch_size}")
            print(f"  Optimized:   {opt_time*1000:.2f} ms")
            print(f"  Lightweight: {light_time*1000:.2f} ms")
            print(f"  Speedup:     {light_time/opt_time:.2f}x faster (lightweight)")

        return results

    def test_memory_usage(self):
        """Test GPU memory consumption"""
        print(f"\n{'='*60}")
        print("Testing Memory Usage")
        print(f"{'='*60}")

        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory test")
            return {}

        device = self.config['device']
        results = {}

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        # Test optimized model
        dummy_input = torch.randn(1, *self.config['input_shape']).to(device)
        with torch.no_grad():
            _ = self.model_optimized(dummy_input)

        opt_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

        # Reset and test lightweight
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

        with torch.no_grad():
            _ = self.model_lightweight(dummy_input)

        light_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

        results = {
            'optimized_mb': opt_memory,
            'lightweight_mb': light_memory,
            'memory_ratio': opt_memory / light_memory
        }

        print(f"\nOptimized Model:   {opt_memory:.2f} MB")
        print(f"Lightweight Model: {light_memory:.2f} MB")
        print(f"Memory Ratio:      {opt_memory/light_memory:.2f}x")

        return results

    def test_gradient_flow(self):
        """Test gradient flow through the network"""
        print(f"\n{'='*60}")
        print("Testing Gradient Flow")
        print(f"{'='*60}")

        device = self.config['device']
        dummy_input = torch.randn(4, *self.config['input_shape']).to(device)
        dummy_target = torch.randn(4, 1).to(device)

        # Optimized model
        self.model_optimized.train()
        ps, pe, v, u = self.model_optimized(dummy_input)
        loss = F.mse_loss(v, dummy_target)
        loss.backward()

        # Check gradients
        grad_norms = []
        for name, param in self.model_optimized.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        print(f"\nGradient Statistics:")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        print(f"  Max gradient norm:  {np.max(grad_norms):.6f}")
        print(f"  Min gradient norm:  {np.min(grad_norms):.6f}")
        print(f"  Std gradient norm:  {np.std(grad_norms):.6f}")

        # Check for vanishing/exploding gradients
        vanishing = sum(1 for g in grad_norms if g < 1e-6)
        exploding = sum(1 for g in grad_norms if g > 100)

        print(f"\n  Vanishing gradients: {vanishing}/{len(grad_norms)}")
        print(f"  Exploding gradients: {exploding}/{len(grad_norms)}")

        return {
            'mean_grad': np.mean(grad_norms),
            'max_grad': np.max(grad_norms),
            'min_grad': np.min(grad_norms),
            'vanishing_count': vanishing,
            'exploding_count': exploding
        }

    def test_learning_capability(self, num_iterations=100):
        """Test if the network can learn a simple pattern"""
        print(f"\n{'='*60}")
        print("Testing Learning Capability (Overfitting Test)")
        print(f"{'='*60}")

        device = self.config['device']
        self.model_optimized.train()

        # Create a simple synthetic dataset
        batch_size = 16
        X = torch.randn(batch_size, *self.config['input_shape']).to(device)
        y_policy_start = torch.randint(0, 2, (batch_size, *self.config['action_size'])).float().to(device)
        y_policy_end = torch.randint(0, 2, (batch_size, *self.config['action_size'])).float().to(device)
        y_value = torch.randn(batch_size, 1).to(device)

        # Setup optimizer
        optimizer = torch.optim.Adam(self.model_optimized.parameters(), lr=0.001)

        losses = []
        print("\nTraining on synthetic data...")

        for epoch in range(num_iterations):
            optimizer.zero_grad()

            ps, pe, v, u = self.model_optimized(X)

            # Calculate losses
            loss_ps = F.mse_loss(ps, y_policy_start)
            loss_pe = F.mse_loss(pe, y_policy_end)
            loss_v = F.mse_loss(v, y_value)

            total_loss = loss_ps + loss_pe + loss_v

            total_loss.backward()
            optimizer.step()

            losses.append(total_loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_iterations} - Loss: {total_loss.item():.6f}")

        # Check if loss decreased
        initial_loss = np.mean(losses[:10])
        final_loss = np.mean(losses[-10:])
        improvement = (initial_loss - final_loss) / initial_loss * 100

        print(f"\nLearning Results:")
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Improvement:  {improvement:.2f}%")

        can_learn = improvement > 50  # At least 50% improvement
        print(f"\n  Can Learn: {'✓ YES' if can_learn else '✗ NO'}")

        return {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'improvement_percent': improvement,
            'can_learn': can_learn,
            'loss_history': losses
        }

    def test_mcts_integration(self):
        """Test MCTS integration with neural network"""
        print(f"\n{'='*60}")
        print("Testing MCTS Integration")
        print(f"{'='*60}")

        device = self.config['device']
        self.model_optimized.eval()

        # Simulate MCTS search
        num_simulations = 10
        search_times = []

        print(f"\nRunning {num_simulations} MCTS simulations...")

        for i in range(num_simulations):
            # Create random board state
            board_state = torch.randn(1, *self.config['input_shape']).to(device)

            start_time = time.time()

            with torch.no_grad():
                ps, pe, v, u = self.model_optimized(board_state)

                # Simulate MCTS search behavior
                for _ in range(20):  # 20 MCTS iterations
                    _ = self.model_optimized(board_state)

            search_time = time.time() - start_time
            search_times.append(search_time)

            if (i + 1) % 5 == 0:
                print(f"  Simulation {i+1}/{num_simulations} - Time: {search_time:.3f}s")

        print(f"\nMCTS Performance:")
        print(f"  Mean search time: {np.mean(search_times):.3f}s")
        print(f"  Min search time:  {np.min(search_times):.3f}s")
        print(f"  Max search time:  {np.max(search_times):.3f}s")
        print(f"  Std search time:  {np.std(search_times):.3f}s")

        return {
            'mean_time': np.mean(search_times),
            'min_time': np.min(search_times),
            'max_time': np.max(search_times),
            'std_time': np.std(search_times)
        }

    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"\n{'#'*60}")
        print(f"# RUNNING COMPREHENSIVE LEARNING TESTS")
        print(f"# Experiment: {self.experiment_name}")
        print(f"{'#'*60}\n")

        results = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Run tests
        results['forward_pass'] = self.test_forward_pass()
        results['memory_usage'] = self.test_memory_usage()
        results['gradient_flow'] = self.test_gradient_flow()
        results['learning_capability'] = self.test_learning_capability()
        results['mcts_integration'] = self.test_mcts_integration()

        # Save results
        results_dir = Path('results/data')
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"test_results_{self.experiment_name}_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"Test results saved to: {results_file}")
        print(f"{'='*60}\n")

        return results


def run_architecture_comparison():
    """Compare different architecture configurations"""
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON EXPERIMENTS")
    print("="*60 + "\n")

    configs = {
        'baseline': {
            'num_residual_blocks': 5,
            'use_attention': False
        },
        'with_attention': {
            'num_residual_blocks': 5,
            'use_attention': True
        },
        'deep_network': {
            'num_residual_blocks': 10,
            'use_attention': False
        },
        'deep_with_attention': {
            'num_residual_blocks': 10,
            'use_attention': True
        }
    }

    results = {}

    for name, config_overrides in configs.items():
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {name}")
        print(f"{'='*60}")

        config = LearningExperiment('baseline', None).get_default_config()
        config.update(config_overrides)

        experiment = LearningExperiment(name, config)
        results[name] = experiment.run_all_tests()

    # Save comparison
    comparison_file = Path('results/data') / f"architecture_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Comparison results saved to: {comparison_file}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import torch.nn.functional as F

    print("\n" + "#"*60)
    print("# 5D CHESS LEARNING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("#"*60 + "\n")

    # Run single experiment
    experiment = LearningExperiment('comprehensive_test')
    single_results = experiment.run_all_tests()

    # Run architecture comparison
    comparison_results = run_architecture_comparison()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")
