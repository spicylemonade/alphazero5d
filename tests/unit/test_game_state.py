"""
Unit tests for ChessState class.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from engine.game_state import ChessState, GameException, Checkmate, Stalemate, DrawLoss


class TestChessState(unittest.TestCase):
    """Test cases for ChessState."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = ChessState()

    def test_initialization(self):
        """Test that ChessState initializes correctly."""
        self.assertIsNotNone(self.state.chess)
        self.assertEqual(self.state.value, 0)
        self.assertIsNone(self.state.piece)
        self.assertIsNone(self.state.game_string)
        self.assertFalse(self.state.is_terminal)
        self.assertEqual(self.state.player, 'white')
        self.assertEqual(self.state.winning, 'white')

    def test_copy(self):
        """Test that copy creates independent state."""
        self.state.value = 5
        self.state.player = 'black'
        self.state.is_terminal = True

        copied_state = self.state.copy()

        self.assertEqual(copied_state.value, 5)
        self.assertEqual(copied_state.player, 'black')
        self.assertTrue(copied_state.is_terminal)

        # Modify original
        self.state.value = 10
        self.assertEqual(copied_state.value, 5)  # Copy should be unchanged

    def test_str_representation(self):
        """Test string representation."""
        str_repr = str(self.state)
        self.assertIn('ChessState', str_repr)
        self.assertIn('player=white', str_repr)
        self.assertIn('value=0', str_repr)

    def test_exceptions(self):
        """Test game exception types."""
        self.assertTrue(issubclass(Checkmate, GameException))
        self.assertTrue(issubclass(Stalemate, GameException))
        self.assertTrue(issubclass(DrawLoss, GameException))


if __name__ == '__main__':
    unittest.main()
