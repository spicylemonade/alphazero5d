"""
Comprehensive test suite for 5D Chess Engine
Tests cover:
- State management and copying
- Move generation and validation
- MCTS search algorithm
- Performance benchmarks
"""

import sys
sys.path.insert(0, '/home/claude/work/repo/src')
sys.path.insert(0, '/home/claude/work/repo/src/optimized')

import unittest
import cupy as cp
import numpy as np
import time
from chess5d_optimized import (
    Chess5D, ChessState, Node, MCTS,
    TranspositionTable, Checkmate, Stalemate, DrawLoss
)


class TestChessState(unittest.TestCase):
    """Test suite for ChessState class"""

    def setUp(self):
        self.state = ChessState()

    def test_initialization(self):
        """Test state initialization"""
        self.assertIsNotNone(self.state.chess)
        self.assertEqual(self.state.value, 0)
        self.assertEqual(self.state.player, 'white')
        self.assertFalse(self.state.is_terminal)

    def test_copy(self):
        """Test state copying"""
        self.state.value = 1
        self.state.player = 'black'
        self.state.choices_start = cp.ones((11, 25, 8, 8))

        copy_state = self.state.copy()

        self.assertEqual(copy_state.value, 1)
        self.assertEqual(copy_state.player, 'black')
        self.assertIsNotNone(copy_state.choices_start)

        # Verify deep copy
        copy_state.value = 2
        self.assertEqual(self.state.value, 1)

    def test_hash_generation(self):
        """Test hash generation for transposition table"""
        hash1 = self.state.get_hash()
        self.assertIsNotNone(hash1)

        # Same state should generate same hash
        hash2 = self.state.get_hash()
        self.assertEqual(hash1, hash2)


class TestChess5D(unittest.TestCase):
    """Test suite for Chess5D game logic"""

    def setUp(self):
        self.game = Chess5D(max_time=11, max_turns=25)

    def test_initialization(self):
        """Test game initialization"""
        self.assertEqual(self.game.max_time, 11)
        self.assertEqual(self.game.max_turns, 25)
        self.assertEqual(self.game.action_size, (11, 25, 8, 8))

    def test_initial_state(self):
        """Test initial game state"""
        state = self.game.get_initial_state()

        self.assertIsNotNone(state.chess)
        self.assertIsNotNone(state.raw_board)
        self.assertIsNotNone(state.board)
        self.assertIsNotNone(state.choices_start)
        self.assertIsNotNone(state.choices_end)
        self.assertEqual(state.player, 'white')
        self.assertFalse(state.is_terminal)

    def test_timeline_conversion(self):
        """Test timeline coordinate conversion"""
        self.assertEqual(Chess5D.convert_timeline(0), 0)
        self.assertEqual(Chess5D.convert_timeline(1), 2)
        self.assertEqual(Chess5D.convert_timeline(-1), -3)

        self.assertEqual(Chess5D.convert_timeline_opposite(0), 0)
        self.assertEqual(Chess5D.convert_timeline_opposite(2), 1)
        self.assertEqual(Chess5D.convert_timeline_opposite(-3), -1)

    def test_move_generation(self):
        """Test move generation"""
        state = self.game.get_initial_state()

        # Should have valid moves
        self.assertIsNotNone(state.moves)
        self.assertGreater(len(state.moves), 0)

        # Should have valid start choices
        self.assertGreater(cp.sum(state.choices_start), 0)

    def test_tensor_conversion(self):
        """Test board to tensor conversion"""
        state = self.game.get_initial_state()
        tensor = state.board

        self.assertEqual(tensor.shape, (11, 50, 6, 8, 8))
        self.assertEqual(tensor.dtype, cp.int8)

    def test_make_move(self):
        """Test making a move"""
        state = self.game.get_initial_state()
        original_player = state.player

        # Pick a move
        result = self.game.pick_choice(state, state.choices_start, state.choices_end)
        if result is not None and result[0] is not None:
            action, _, _ = result

            # Make the move
            self.game.make_move(state, action)

            # Player should switch (or stay if multi-move turn)
            self.assertIsNotNone(state.player)


class TestTranspositionTable(unittest.TestCase):
    """Test suite for transposition table"""

    def setUp(self):
        self.tt = TranspositionTable(max_size=100)

    def test_put_and_get(self):
        """Test storing and retrieving values"""
        self.tt.put("hash1", 0.5, 5)

        entry = self.tt.get("hash1")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['value'], 0.5)
        self.assertEqual(entry['depth'], 5)

    def test_cache_miss(self):
        """Test cache miss"""
        entry = self.tt.get("nonexistent")
        self.assertIsNone(entry)

    def test_max_size_eviction(self):
        """Test eviction when max size reached"""
        for i in range(150):
            self.tt.put(f"hash{i}", i * 0.01, 1)

        self.assertLessEqual(len(self.tt.table), 100)

    def test_statistics(self):
        """Test statistics tracking"""
        self.tt.put("hash1", 0.5, 5)
        self.tt.get("hash1")  # hit
        self.tt.get("hash2")  # miss

        stats = self.tt.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertAlmostEqual(stats['hit_rate'], 0.5)


class TestMCTS(unittest.TestCase):
    """Test suite for MCTS algorithm"""

    def setUp(self):
        self.game = Chess5D(max_time=11, max_turns=25)
        self.args = {
            'num_searches': 10,
            'C': 1.41,
            'max_rollout_depth': 20,
            'tt_size': 1000,
            'tt_depth_threshold': 3
        }
        self.mcts = MCTS(self.game, self.args)

    def test_initialization(self):
        """Test MCTS initialization"""
        self.assertIsNotNone(self.mcts.game)
        self.assertIsNotNone(self.mcts.transposition_table)
        self.assertEqual(self.mcts.nodes_searched, 0)

    def test_search(self):
        """Test MCTS search"""
        state = self.game.get_initial_state()

        action_probs_start, action_probs_end = self.mcts.search(state)

        # Should return valid probability distributions
        self.assertEqual(action_probs_start.shape, (11, 25, 8, 8))
        self.assertEqual(action_probs_end.shape, (11, 25, 8, 8))

        # Probabilities should sum to ~1
        total_start = cp.sum(action_probs_start)
        if total_start > 0:
            self.assertAlmostEqual(float(total_start), 1.0, places=5)

    def test_node_expansion(self):
        """Test node expansion"""
        state = self.game.get_initial_state()
        node = Node(self.game, self.args, state)

        # Expand node
        child = node.expand()

        if child is not None:
            self.assertEqual(len(node.children), 1)
            self.assertIsNotNone(child.state)

    def test_ucb_calculation(self):
        """Test UCB calculation"""
        state = self.game.get_initial_state()
        parent = Node(self.game, self.args, state)
        parent.visit_count = 10

        child = Node(self.game, self.args, state, parent=parent)
        child.visit_count = 5
        child.value_sum = 2.5

        ucb = parent.get_ucb(child)
        self.assertIsInstance(ucb, (int, float, cp.ndarray))
        self.assertGreater(ucb, 0)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""

    def setUp(self):
        self.game = Chess5D(max_time=11, max_turns=25)
        self.state = self.game.get_initial_state()

    def test_state_copy_performance(self):
        """Benchmark state copying"""
        start_time = time.time()

        for _ in range(100):
            copy_state = self.state.copy()

        elapsed = time.time() - start_time
        avg_time = elapsed / 100

        print(f"\nAverage state copy time: {avg_time*1000:.3f}ms")
        self.assertLess(avg_time, 0.1)  # Should be under 100ms

    def test_move_generation_performance(self):
        """Benchmark move generation"""
        start_time = time.time()

        for _ in range(50):
            state = self.game.get_initial_state()
            self.game.convert_moves(state)

        elapsed = time.time() - start_time
        avg_time = elapsed / 50

        print(f"Average move generation time: {avg_time*1000:.3f}ms")
        self.assertLess(avg_time, 0.5)  # Should be under 500ms

    def test_mcts_search_performance(self):
        """Benchmark MCTS search"""
        args = {
            'num_searches': 50,
            'C': 1.41,
            'max_rollout_depth': 20,
            'tt_size': 10000
        }
        mcts = MCTS(self.game, args)

        start_time = time.time()
        action_probs_start, action_probs_end = mcts.search(self.state)
        elapsed = time.time() - start_time

        print(f"MCTS search time (50 iterations): {elapsed:.3f}s")
        print(f"Time per iteration: {elapsed/50*1000:.1f}ms")

        stats = mcts.get_stats()
        print(f"Nodes searched: {stats['nodes_searched']}")
        print(f"TT hit rate: {stats['transposition_table']['hit_rate']:.2%}")


def run_all_tests():
    """Run all tests and return results"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestChessState))
    suite.addTests(loader.loadTestsFromTestCase(TestChess5D))
    suite.addTests(loader.loadTestsFromTestCase(TestTranspositionTable))
    suite.addTests(loader.loadTestsFromTestCase(TestMCTS))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_all_tests()

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
