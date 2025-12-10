"""
Comprehensive benchmarking suite for 5D Chess Engine
Collects performance metrics for research paper
"""

import sys
sys.path.insert(0, '/home/claude/work/repo/src')
sys.path.insert(0, '/home/claude/work/repo/src/optimized')

import time
import json
import cupy as cp
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from chess5d_optimized import Chess5D, MCTS, TranspositionTable

# Import original for comparison
import super as original_module


class BenchmarkSuite:
    """Comprehensive benchmarking for research paper"""

    def __init__(self):
        self.results = defaultdict(list)
        self.game_optimized = Chess5D(max_time=11, max_turns=25)
        self.game_original = original_module.Chess5D(max_time=11, max_turns=25)

    def benchmark_state_copy(self, iterations=500):
        """Benchmark state copying performance"""
        print("\n[1/8] Benchmarking state copy performance...")

        # Optimized version
        state_opt = self.game_optimized.get_initial_state()
        times_opt = []

        for i in range(iterations):
            start = time.perf_counter()
            copy_state = state_opt.copy()
            elapsed = time.perf_counter() - start
            times_opt.append(elapsed * 1000)  # Convert to ms

        # Original version
        state_orig = self.game_original.get_initial_state()
        times_orig = []

        for i in range(iterations):
            start = time.perf_counter()
            copy_state = state_orig.copy()
            elapsed = time.perf_counter() - start
            times_orig.append(elapsed * 1000)

        self.results['state_copy'] = {
            'optimized': times_opt,
            'original': times_orig,
            'speedup': np.mean(times_orig) / np.mean(times_opt)
        }

        print(f"  Optimized: {np.mean(times_opt):.3f}ms ± {np.std(times_opt):.3f}ms")
        print(f"  Original:  {np.mean(times_orig):.3f}ms ± {np.std(times_orig):.3f}ms")
        print(f"  Speedup:   {self.results['state_copy']['speedup']:.2f}x")

    def benchmark_move_generation(self, iterations=200):
        """Benchmark move generation performance"""
        print("\n[2/8] Benchmarking move generation...")

        times_opt = []
        move_counts = []

        for i in range(iterations):
            state = self.game_optimized.get_initial_state()
            start = time.perf_counter()
            self.game_optimized.convert_moves(state)
            elapsed = time.perf_counter() - start
            times_opt.append(elapsed * 1000)
            move_counts.append(len(state.moves))

        self.results['move_generation'] = {
            'times': times_opt,
            'move_counts': move_counts,
            'avg_time': np.mean(times_opt),
            'avg_moves': np.mean(move_counts)
        }

        print(f"  Average time: {np.mean(times_opt):.3f}ms ± {np.std(times_opt):.3f}ms")
        print(f"  Average moves generated: {np.mean(move_counts):.1f}")

    def benchmark_mcts_search(self, search_counts=[10, 20, 50, 100]):
        """Benchmark MCTS search with varying iteration counts"""
        print("\n[3/8] Benchmarking MCTS search performance...")

        results = {}

        for num_searches in search_counts:
            args = {
                'num_searches': num_searches,
                'C': 1.41,
                'max_rollout_depth': 20,
                'tt_size': 50000
            }
            mcts = MCTS(self.game_optimized, args)
            state = self.game_optimized.get_initial_state()

            start = time.perf_counter()
            action_probs_start, action_probs_end = mcts.search(state)
            elapsed = time.perf_counter() - start

            stats = mcts.get_stats()

            results[num_searches] = {
                'time': elapsed,
                'nodes_searched': stats['nodes_searched'],
                'tt_hit_rate': stats['transposition_table']['hit_rate'],
                'time_per_node': elapsed / max(stats['nodes_searched'], 1)
            }

            print(f"  {num_searches} searches: {elapsed:.3f}s, "
                  f"{stats['nodes_searched']} nodes, "
                  f"TT hit rate: {stats['transposition_table']['hit_rate']:.2%}")

        self.results['mcts_search'] = results

    def benchmark_transposition_table(self, sizes=[1000, 10000, 50000, 100000]):
        """Benchmark transposition table performance"""
        print("\n[4/8] Benchmarking transposition table...")

        results = {}

        for size in sizes:
            tt = TranspositionTable(max_size=size)

            # Fill with data
            for i in range(min(size * 2, 10000)):
                hash_key = f"state_{i}"
                tt.put(hash_key, np.random.random(), np.random.randint(1, 10))

            # Benchmark lookups
            lookup_times = []
            for i in range(1000):
                hash_key = f"state_{np.random.randint(0, min(size * 2, 10000))}"
                start = time.perf_counter()
                entry = tt.get(hash_key)
                elapsed = time.perf_counter() - start
                lookup_times.append(elapsed * 1000000)  # Convert to microseconds

            stats = tt.get_stats()
            results[size] = {
                'avg_lookup_time': np.mean(lookup_times),
                'hit_rate': stats['hit_rate'],
                'size': stats['size']
            }

            print(f"  Size {size}: {np.mean(lookup_times):.2f}µs lookup, "
                  f"hit rate: {stats['hit_rate']:.2%}")

        self.results['transposition_table'] = results

    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        print("\n[5/8] Benchmarking memory usage...")

        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple states
        states = []
        for i in range(100):
            state = self.game_optimized.get_initial_state()
            states.append(state)

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_per_state = (mem_after - mem_before) / 100

        self.results['memory_usage'] = {
            'per_state_mb': mem_per_state,
            'base_memory_mb': mem_before,
            'total_memory_mb': mem_after
        }

        print(f"  Memory per state: {mem_per_state:.2f}MB")
        print(f"  Total memory: {mem_after:.2f}MB")

    def benchmark_scalability(self, max_time_values=[5, 7, 9, 11]):
        """Benchmark scalability with different board sizes"""
        print("\n[6/8] Benchmarking scalability...")

        results = {}

        for max_time in max_time_values:
            game = Chess5D(max_time=max_time, max_turns=25)
            state = game.get_initial_state()

            # Time state creation
            start = time.perf_counter()
            for _ in range(10):
                test_state = game.get_initial_state()
            elapsed = time.perf_counter() - start
            init_time = elapsed / 10

            # Time move generation
            start = time.perf_counter()
            game.convert_moves(state)
            move_gen_time = time.perf_counter() - start

            # Memory size
            tensor_size = state.board.nbytes / 1024 / 1024  # MB

            results[max_time] = {
                'init_time': init_time,
                'move_gen_time': move_gen_time,
                'tensor_size_mb': tensor_size,
                'num_moves': len(state.moves)
            }

            print(f"  max_time={max_time}: init={init_time*1000:.2f}ms, "
                  f"moves={move_gen_time*1000:.2f}ms, "
                  f"size={tensor_size:.2f}MB")

        self.results['scalability'] = results

    def benchmark_rollout_depth(self, depths=[10, 20, 30, 50]):
        """Benchmark MCTS with different rollout depths"""
        print("\n[7/8] Benchmarking rollout depth impact...")

        results = {}

        for depth in depths:
            args = {
                'num_searches': 30,
                'C': 1.41,
                'max_rollout_depth': depth,
                'tt_size': 10000
            }
            mcts = MCTS(self.game_optimized, args)
            state = self.game_optimized.get_initial_state()

            start = time.perf_counter()
            action_probs_start, action_probs_end = mcts.search(state)
            elapsed = time.perf_counter() - start

            stats = mcts.get_stats()

            results[depth] = {
                'time': elapsed,
                'nodes_searched': stats['nodes_searched'],
                'terminal_nodes': stats['terminal_nodes']
            }

            print(f"  Depth {depth}: {elapsed:.3f}s, "
                  f"{stats['nodes_searched']} nodes, "
                  f"{stats['terminal_nodes']} terminals")

        self.results['rollout_depth'] = results

    def benchmark_full_game(self, num_games=5):
        """Benchmark full game playthrough"""
        print("\n[8/8] Benchmarking full game scenarios...")

        game_results = []

        for game_num in range(num_games):
            args = {
                'num_searches': 20,
                'C': 1.41,
                'max_rollout_depth': 20,
                'tt_size': 10000
            }
            mcts = MCTS(self.game_optimized, args)
            state = self.game_optimized.get_initial_state()

            moves = []
            move_times = []
            start_game = time.perf_counter()

            max_moves = 20  # Limit for benchmark
            for move_num in range(max_moves):
                if state.is_terminal:
                    break

                start = time.perf_counter()
                action_probs_start, action_probs_end = mcts.search(state)
                search_time = time.perf_counter() - start

                # Make best move
                index_s = cp.unravel_index(cp.argmax(action_probs_start),
                                           action_probs_start.shape)
                result = self.game_optimized.pick_choice(state,
                                                        state.choices_start,
                                                        state.choices_end)
                if result is None or result[0] is None:
                    break

                action, _, _ = result
                self.game_optimized.make_move(state, action)

                moves.append(str(action))
                move_times.append(search_time)

            game_time = time.perf_counter() - start_game

            game_results.append({
                'moves': len(moves),
                'total_time': game_time,
                'avg_move_time': np.mean(move_times) if move_times else 0,
                'terminal': state.is_terminal
            })

            print(f"  Game {game_num+1}: {len(moves)} moves, "
                  f"{game_time:.2f}s total, "
                  f"{np.mean(move_times):.3f}s per move")

        self.results['full_game'] = game_results

    def save_results(self, filename='research_artifacts/benchmark_results.json'):
        """Save results to JSON"""
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        results_converted = convert(dict(self.results))

        with open(filename, 'w') as f:
            json.dump(results_converted, f, indent=2)

        print(f"\nResults saved to {filename}")

    def generate_visualizations(self):
        """Generate all visualizations for research paper"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # 1. State copy performance comparison
        self._plot_state_copy()

        # 2. MCTS scaling
        self._plot_mcts_scaling()

        # 3. Transposition table performance
        self._plot_transposition_table()

        # 4. Memory usage
        self._plot_memory_usage()

        # 5. Scalability analysis
        self._plot_scalability()

        # 6. Rollout depth analysis
        self._plot_rollout_depth()

        # 7. Move time distribution
        self._plot_move_distribution()

        print("\nAll visualizations generated!")

    def _plot_state_copy(self):
        """Plot state copy performance"""
        data = self.results['state_copy']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot comparison
        box_data = [data['original'], data['optimized']]
        ax1.boxplot(box_data, labels=['Original', 'Optimized'])
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('State Copy Performance Comparison')
        ax1.grid(True, alpha=0.3)

        # Speedup bar
        ax2.bar(['Speedup'], [data['speedup']], color='green', alpha=0.7)
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title(f"Performance Improvement: {data['speedup']:.2f}x")
        ax2.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('research_artifacts/figure1_state_copy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 1: State copy performance")

    def _plot_mcts_scaling(self):
        """Plot MCTS search scaling"""
        data = self.results['mcts_search']

        search_counts = sorted(data.keys())
        times = [data[k]['time'] for k in search_counts]
        nodes = [data[k]['nodes_searched'] for k in search_counts]
        tt_rates = [data[k]['tt_hit_rate'] * 100 for k in search_counts]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Time scaling
        ax1.plot(search_counts, times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of MCTS Searches')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('MCTS Search Time Scaling')
        ax1.grid(True, alpha=0.3)

        # Nodes explored
        ax2.plot(search_counts, nodes, 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Number of MCTS Searches')
        ax2.set_ylabel('Nodes Explored')
        ax2.set_title('MCTS Node Exploration')
        ax2.grid(True, alpha=0.3)

        # Transposition table hit rate
        ax3.plot(search_counts, tt_rates, '^-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Number of MCTS Searches')
        ax3.set_ylabel('TT Hit Rate (%)')
        ax3.set_title('Transposition Table Efficiency')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('research_artifacts/figure2_mcts_scaling.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 2: MCTS scaling analysis")

    def _plot_transposition_table(self):
        """Plot transposition table performance"""
        data = self.results['transposition_table']

        sizes = sorted(data.keys())
        lookup_times = [data[k]['avg_lookup_time'] for k in sizes]
        hit_rates = [data[k]['hit_rate'] * 100 for k in sizes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Lookup time
        ax1.plot(sizes, lookup_times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Table Size')
        ax1.set_ylabel('Lookup Time (µs)')
        ax1.set_title('Transposition Table Lookup Performance')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # Hit rate
        ax2.plot(sizes, hit_rates, 's-', linewidth=2, markersize=8, color='purple')
        ax2.set_xlabel('Table Size')
        ax2.set_ylabel('Hit Rate (%)')
        ax2.set_title('Cache Hit Rate vs Table Size')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('research_artifacts/figure3_transposition_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 3: Transposition table performance")

    def _plot_memory_usage(self):
        """Plot memory usage"""
        data = self.results['memory_usage']

        fig, ax = plt.subplots(figsize=(8, 6))

        categories = ['Per State', 'Base Memory', 'Total (100 states)']
        values = [data['per_state_mb'], data['base_memory_mb'], data['total_memory_mb']]
        colors = ['skyblue', 'lightcoral', 'lightgreen']

        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Analysis')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}MB',
                   ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('research_artifacts/figure4_memory_usage.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 4: Memory usage analysis")

    def _plot_scalability(self):
        """Plot scalability analysis"""
        data = self.results['scalability']

        max_times = sorted(data.keys())
        init_times = [data[k]['init_time'] * 1000 for k in max_times]
        move_gen_times = [data[k]['move_gen_time'] * 1000 for k in max_times]
        tensor_sizes = [data[k]['tensor_size_mb'] for k in max_times]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time scaling
        ax1.plot(max_times, init_times, 'o-', label='Initialization', linewidth=2, markersize=8)
        ax1.plot(max_times, move_gen_times, 's-', label='Move Generation', linewidth=2, markersize=8)
        ax1.set_xlabel('Timeline Dimension Size')
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Performance Scaling with Board Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Memory scaling
        ax2.plot(max_times, tensor_sizes, '^-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Timeline Dimension Size')
        ax2.set_ylabel('Tensor Size (MB)')
        ax2.set_title('Memory Scaling with Board Size')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('research_artifacts/figure5_scalability.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 5: Scalability analysis")

    def _plot_rollout_depth(self):
        """Plot rollout depth analysis"""
        data = self.results['rollout_depth']

        depths = sorted(data.keys())
        times = [data[k]['time'] for k in depths]
        nodes = [data[k]['nodes_searched'] for k in depths]
        terminals = [data[k]['terminal_nodes'] for k in depths]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Time vs depth
        ax1.plot(depths, times, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Max Rollout Depth')
        ax1.set_ylabel('Search Time (seconds)')
        ax1.set_title('Impact of Rollout Depth on Search Time')
        ax1.grid(True, alpha=0.3)

        # Node statistics
        width = 3
        x = np.array(depths)
        ax2.bar(x - width/2, nodes, width, label='Total Nodes', alpha=0.7)
        ax2.bar(x + width/2, terminals, width, label='Terminal Nodes', alpha=0.7)
        ax2.set_xlabel('Max Rollout Depth')
        ax2.set_ylabel('Node Count')
        ax2.set_title('Node Exploration vs Rollout Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('research_artifacts/figure6_rollout_depth.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 6: Rollout depth analysis")

    def _plot_move_distribution(self):
        """Plot move time distribution"""
        if 'full_game' not in self.results:
            return

        games = self.results['full_game']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Move counts per game
        game_nums = list(range(1, len(games) + 1))
        move_counts = [g['moves'] for g in games]
        game_times = [g['total_time'] for g in games]

        ax1.bar(game_nums, move_counts, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Game Number')
        ax1.set_ylabel('Number of Moves')
        ax1.set_title('Moves per Game')
        ax1.grid(True, alpha=0.3, axis='y')

        # Time distribution
        avg_move_times = [g['avg_move_time'] for g in games]
        ax2.bar(game_nums, avg_move_times, alpha=0.7, color='coral')
        ax2.set_xlabel('Game Number')
        ax2.set_ylabel('Average Move Time (seconds)')
        ax2.set_title('Average Decision Time per Move')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('research_artifacts/figure7_move_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Figure 7: Move distribution analysis")


def main():
    """Run all benchmarks"""
    print("="*70)
    print("5D CHESS ENGINE BENCHMARK SUITE")
    print("="*70)

    suite = BenchmarkSuite()

    # Run all benchmarks
    suite.benchmark_state_copy(iterations=500)
    suite.benchmark_move_generation(iterations=200)
    suite.benchmark_mcts_search(search_counts=[10, 20, 50, 100])
    suite.benchmark_transposition_table(sizes=[1000, 10000, 50000, 100000])
    suite.benchmark_memory_usage()
    suite.benchmark_scalability(max_time_values=[5, 7, 9, 11])
    suite.benchmark_rollout_depth(depths=[10, 20, 30, 50])
    suite.benchmark_full_game(num_games=5)

    # Save results
    suite.save_results()

    # Generate visualizations
    suite.generate_visualizations()

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print("\nAll results saved to research_artifacts/")


if __name__ == '__main__':
    main()
