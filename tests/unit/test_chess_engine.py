"""
Unit tests for Chess5DEngine class.
"""
import unittest
import cupy as cp
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from engine.chess_engine import Chess5DEngine
from engine.game_state import ChessState


class TestChess5DEngine(unittest.TestCase):
    """Test cases for Chess5DEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = Chess5DEngine(max_time=11, max_turns=30)

    def test_initialization(self):
        """Test engine initialization."""
        self.assertEqual(self.engine.max_time, 11)
        self.assertEqual(self.engine.max_turns, 30)
        self.assertEqual(self.engine.action_size, (11, 30, 8, 8))

    def test_initial_state(self):
        """Test initial state creation."""
        state = self.engine.get_initial_state()

        self.assertIsInstance(state, ChessState)
        self.assertEqual(state.player, 'white')
        self.assertFalse(state.is_terminal)
        self.assertIsNotNone(state.board)
        self.assertIsNotNone(state.choices_start)
        self.assertIsNotNone(state.choices_end)

    def test_timeline_conversion(self):
        """Test timeline conversion functions."""
        # Test positive timelines
        self.assertEqual(Chess5DEngine.convert_timeline(0), 0)
        self.assertEqual(Chess5DEngine.convert_timeline(1), 2)
        self.assertEqual(Chess5DEngine.convert_timeline(2), 4)

        # Test negative timelines
        self.assertEqual(Chess5DEngine.convert_timeline(-1), -3)
        self.assertEqual(Chess5DEngine.convert_timeline(-2), -5)

        # Test round-trip conversion
        for i in range(-5, 6):
            converted = Chess5DEngine.convert_timeline(i)
            back = Chess5DEngine.convert_timeline_opposite(converted)
            self.assertEqual(i, back)

    def test_board_tensor_shape(self):
        """Test raw board to tensor conversion."""
        state = self.engine.get_initial_state()
        board = state.board

        expected_shape = (11, 60, 6, 8, 8)
        self.assertEqual(board.shape, expected_shape)
        self.assertEqual(board.dtype, cp.int8)

    def test_opponent_value(self):
        """Test opponent value calculation."""
        state = ChessState()
        state.player = 'white'
        state.prev_player = 'white'

        # Same player - value unchanged
        self.assertEqual(self.engine.get_opponent_value(state, 1.0), 1.0)

        # Different player - value inverted
        state.prev_player = 'black'
        self.assertEqual(self.engine.get_opponent_value(state, 1.0), -1.0)

    def test_piece_map(self):
        """Test piece mapping."""
        self.assertEqual(Chess5DEngine.piece_map[0], 'P')
        self.assertEqual(Chess5DEngine.piece_map[5], 'K')
        self.assertEqual(len(Chess5DEngine.piece_map), 6)

    def test_player_map(self):
        """Test player mapping."""
        self.assertEqual(Chess5DEngine.player_map['white'], 1)
        self.assertEqual(Chess5DEngine.player_map['black'], -1)
        self.assertEqual(Chess5DEngine.player_map_opp['white'], 'black')
        self.assertEqual(Chess5DEngine.player_map_opp['black'], 'white')


if __name__ == '__main__':
    unittest.main()
