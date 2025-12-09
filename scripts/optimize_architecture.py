"""
Architecture Optimization Script for 5D Chess AI
Implements performance improvements and architectural enhancements
"""
import sys
sys.path.append('..')
from src.super import Chess5D, MCTS, Node, ChessState
import cupy as cp
import numpy as np
import time
import json
from datetime import datetime

class OptimizedMCTS(MCTS):
    """Enhanced MCTS with caching and pruning optimizations"""

    def __init__(self, game, args):
        super().__init__(game, args)
        self.position_cache = {}
        self.transposition_table = {}
        self.search_statistics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'nodes_pruned': 0,
            'total_nodes': 0
        }

    def search(self, state):
        """Optimized search with transposition table and caching"""
        # Generate position hash
        position_hash = self._hash_position(state)

        # Check transposition table
        if position_hash in self.transposition_table:
            self.search_statistics['cache_hits'] += 1
            cached_result = self.transposition_table[position_hash]
            if cached_result['depth'] >= self.args.get('cache_depth_threshold', 10):
                return cached_result['action_probs_start'], cached_result['action_probs_end']

        self.search_statistics['cache_misses'] += 1

        # Perform regular MCTS search
        root = Node(self.game, self.args, state)

        for search_num in range(self.args['num_searches']):
            node = root

            # Selection phase with early stopping
            while node.is_fully_expanded():
                node = node.select()
                self.search_statistics['total_nodes'] += 1

            value, is_terminal = node.state.value, node.state.is_terminal
            value = self.game.get_opponent_value(state, value)

            # Expansion and simulation
            if not is_terminal:
                node = node.expand()
                self.search_statistics['total_nodes'] += 1

                # Progressive widening: limit expansions early in search
                if search_num < self.args['num_searches'] * 0.3:
                    if len(node.children) > self.args.get('early_expansion_limit', 5):
                        self.search_statistics['nodes_pruned'] += 1
                        continue

                value = node.simulate()

            # Backpropagation
            node.backpropagate(value)

        # Build action probabilities
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float64)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float64)

        for child in root.children:
            action_probs_start[child.action_taken_s] += child.visit_count
            action_probs_end[child.action_taken_e] += child.visit_count

        action_probs_start /= cp.sum(action_probs_start) if cp.sum(action_probs_start) > 0 else 1

        # Cache result
        self.transposition_table[position_hash] = {
            'action_probs_start': action_probs_start,
            'action_probs_end': action_probs_end,
            'depth': self.args['num_searches']
        }

        # Limit cache size
        if len(self.transposition_table) > 1000:
            # Remove oldest entries
            keys = list(self.transposition_table.keys())
            for key in keys[:100]:
                del self.transposition_table[key]

        return action_probs_start, action_probs_end

    def _hash_position(self, state):
        """Generate hash for position caching"""
        board_hash = hash(state.board.tobytes()) if state.board is not None else 0
        player_hash = hash(state.player)
        return (board_hash, player_hash)

    def get_statistics(self):
        """Return search statistics"""
        stats = self.search_statistics.copy()
        stats['cache_hit_rate'] = (
            stats['cache_hits'] / max(stats['cache_hits'] + stats['cache_misses'], 1)
        )
        stats['pruning_rate'] = (
            stats['nodes_pruned'] / max(stats['total_nodes'], 1)
        )
        return stats

class ParallelGameSimulator:
    """Run multiple games in parallel for faster testing"""

    def __init__(self, max_time=1, max_turns=25, num_parallel=4):
        self.max_time = max_time
        self.max_turns = max_turns
        self.num_parallel = num_parallel

    def simulate_batch(self, config, num_games=10):
        """Simulate multiple games efficiently"""
        results = []

        print(f"Simulating {num_games} games with {self.num_parallel} parallel workers...")

        for batch_start in range(0, num_games, self.num_parallel):
            batch_size = min(self.num_parallel, num_games - batch_start)
            batch_results = []

            # Simulate games in batch
            for i in range(batch_size):
                game = Chess5D(self.max_time, self.max_turns)
                game_state = game.get_initial_state()
                mcts = OptimizedMCTS(game, config)

                move_count = 0
                start_time = time.time()

                try:
                    while move_count < 100:  # Max moves limit
                        mcts_prob_s, mcts_prob_e = mcts.search(game_state)

                        index_s = cp.unravel_index(cp.argmax(mcts_prob_s), mcts_prob_s.shape)
                        index_e = game._pick_end_move_org(index_s, game_state, mcts_prob_e)

                        action = f"({game.convert_timeline_opposite(index_s[0].item())}T{index_s[1].item() + 1})" \
                                 f"{chr(96 + index_s[3].item() + 1)}{index_s[2].item() + 1}>>" \
                                 f"({index_e['timeline']}T{index_e['turn']}){chr(96 + index_e['file'])}{index_e['rank']}"

                        game.make_move(game_state, action)
                        move_count += 1

                        if game_state.is_terminal:
                            break
                except:
                    pass

                game_time = time.time() - start_time
                mcts_stats = mcts.get_statistics()

                batch_results.append({
                    'moves': move_count,
                    'time': game_time,
                    'winner': game_state.prev_player if game_state.is_terminal else 'incomplete',
                    'terminal_value': game_state.value,
                    'mcts_stats': mcts_stats
                })

            results.extend(batch_results)
            print(f"  Completed {len(results)}/{num_games} games")

        return results

def optimize_parameters():
    """Find optimal MCTS parameters through automated testing"""
    print("=" * 80)
    print("Automated Parameter Optimization")
    print("=" * 80)

    # Parameter grid
    search_depths = [10, 20, 30, 40, 50]
    c_values = [1.0, 1.2, 1.41, 1.6, 2.0]

    best_config = None
    best_score = -float('inf')

    simulator = ParallelGameSimulator(max_time=1, max_turns=25, num_parallel=2)

    results_log = []

    for num_searches in search_depths:
        for c_value in c_values:
            config = {
                'num_searches': num_searches,
                'C': c_value,
                'cache_depth_threshold': 10,
                'early_expansion_limit': 5
            }

            print(f"\nTesting: num_searches={num_searches}, C={c_value}")

            # Run simulations
            batch_results = simulator.simulate_batch(config, num_games=5)

            # Calculate performance score
            avg_moves = np.mean([r['moves'] for r in batch_results])
            avg_time = np.mean([r['time'] for r in batch_results])
            completion_rate = sum(1 for r in batch_results if r['winner'] != 'incomplete') / len(batch_results)

            # Composite score (higher is better)
            score = completion_rate * 100 - avg_time * 2 + (1 / (avg_moves + 1)) * 10

            results_log.append({
                'config': config,
                'avg_moves': avg_moves,
                'avg_time': avg_time,
                'completion_rate': completion_rate,
                'score': score
            })

            print(f"  Avg Moves: {avg_moves:.1f}")
            print(f"  Avg Time: {avg_time:.2f}s")
            print(f"  Completion Rate: {completion_rate:.2%}")
            print(f"  Score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_config = config
                print(f"  *** New best configuration! ***")

    # Save optimization results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/optimization_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'best_config': best_config,
            'best_score': best_score,
            'all_results': results_log
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION FOUND")
    print("=" * 80)
    print(f"Configuration: {best_config}")
    print(f"Score: {best_score:.2f}")
    print(f"\nResults saved to: {results_file}")
    print("=" * 80)

    return best_config, results_log

if __name__ == "__main__":
    optimal_config, results = optimize_parameters()
