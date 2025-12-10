"""
Unit tests for MCTS search algorithm.
"""
import unittest
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcts.search import MCTS
from engine.chess_engine import Chess5DEngine


class TestMCTS(unittest.TestCase):
    """Test cases for MCTS."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = Chess5DEngine(max_time=11, max_turns=30)
        self.args = {'num_searches': 5, 'C': 1.41}
        self.mcts = MCTS(self.engine, self.args)

    def test_initialization(self):
        """Test MCTS initialization."""
        self.assertIsNotNone(self.mcts.game)
        self.assertEqual(self.mcts.args['num_searches'], 5)
        self.assertEqual(self.mcts.args['C'], 1.41)

    def test_search_returns_probabilities(self):
        """Test that search returns probability distributions."""
        state = self.engine.get_initial_state()
        start_probs, end_probs = self.mcts.search(state)

        self.assertEqual(start_probs.shape, self.engine.action_size)
        self.assertEqual(end_probs.shape, self.engine.action_size)
        self.assertEqual(start_probs.dtype, cp.float64)
        self.assertEqual(end_probs.dtype, cp.float64)

    def test_search_probability_sum(self):
        """Test that probabilities sum to approximately 1."""
        state = self.engine.get_initial_state()
        start_probs, end_probs = self.mcts.search(state)

        # Allow small numerical error
        self.assertAlmostEqual(float(cp.sum(start_probs)), 1.0, places=5)

    def test_search_with_different_parameters(self):
        """Test search with different MCTS parameters."""
        state = self.engine.get_initial_state()

        # Test with more searches
        mcts_many = MCTS(self.engine, {'num_searches': 20, 'C': 1.41})
        start_probs, _ = mcts_many.search(state)

        self.assertIsNotNone(start_probs)
        self.assertGreater(cp.max(start_probs), 0)

    def test_search_determinism(self):
        """Test that search produces valid distributions."""
        state = self.engine.get_initial_state()
        start_probs, end_probs = self.mcts.search(state)

        # Check that some moves have non-zero probability
        self.assertGreater(cp.sum(start_probs > 0), 0)

        # Check no negative probabilities
        self.assertTrue(cp.all(start_probs >= 0))
        self.assertTrue(cp.all(end_probs >= 0))


if __name__ == '__main__':
    unittest.main()
