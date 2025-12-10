"""
Unit tests for MCTSNode class.
"""
import unittest
import math
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcts.node import MCTSNode
from engine.chess_engine import Chess5DEngine


class TestMCTSNode(unittest.TestCase):
    """Test cases for MCTSNode."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = Chess5DEngine(max_time=11, max_turns=30)
        self.state = self.engine.get_initial_state()
        self.args = {'num_searches': 10, 'C': 1.41}
        self.node = MCTSNode(self.engine, self.args, self.state)

    def test_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.node.visit_count, 0)
        self.assertEqual(self.node.value_sum, 0)
        self.assertEqual(len(self.node.children), 0)
        self.assertEqual(self.node.player, 'white')
        self.assertIsNone(self.node.parent)

    def test_not_fully_expanded_initially(self):
        """Test that node is not fully expanded initially."""
        self.assertFalse(self.node.is_fully_expanded())

    def test_ucb_with_unvisited_child(self):
        """Test UCB calculation with unvisited child."""
        child_state = self.state.copy()
        child = MCTSNode(self.engine, self.args, child_state, self.node)

        # Unvisited child should have infinite UCB
        self.node.visit_count = 10
        ucb = self.node.get_ucb(child)
        self.assertEqual(ucb, float('inf'))

    def test_ucb_calculation(self):
        """Test UCB calculation."""
        # Setup parent with visits
        self.node.visit_count = 100

        # Create child with some visits
        child_state = self.state.copy()
        child = MCTSNode(self.engine, self.args, child_state, self.node)
        child.visit_count = 10
        child.value_sum = 5

        ucb = self.node.get_ucb(child)
        self.assertIsInstance(ucb, float)
        self.assertGreater(ucb, 0)

    def test_backpropagation(self):
        """Test backpropagation of values."""
        value = 1.0
        self.node.backpropagate(value)

        self.assertEqual(self.node.visit_count, 1)
        self.assertEqual(self.node.value_sum, value)

        # Backpropagate again
        self.node.backpropagate(0.5)
        self.assertEqual(self.node.visit_count, 2)
        self.assertEqual(self.node.value_sum, 1.5)

    def test_parent_child_relationship(self):
        """Test parent-child relationship."""
        child_state = self.state.copy()
        child = MCTSNode(
            self.engine,
            self.args,
            child_state,
            parent=self.node,
            action_taken_s=(0, 0, 0, 0),
            action_taken_e=(0, 0, 1, 0)
        )

        self.assertEqual(child.parent, self.node)
        self.assertIsNotNone(child.action_taken_s)
        self.assertIsNotNone(child.action_taken_e)


if __name__ == '__main__':
    unittest.main()
