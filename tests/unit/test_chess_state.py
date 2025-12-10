"""Unit tests for ChessState class."""

import pytest
import cupy as cp
from src.chess_state import ChessState


class TestChessState:
    """Test ChessState class functionality."""

    def test_initial_state_creation(self):
        """Test creating a new chess state."""
        state = ChessState()
        assert state.value == 0.0
        assert state.player == 'white'
        assert state.is_terminal is False
        assert state.piece is None
        assert state.game_string is None

    def test_state_string_representation(self):
        """Test string representation of chess state."""
        state = ChessState()
        state.value = 0.5
        state.moves = [1, 2, 3]

        str_repr = str(state)
        assert 'ChessState' in str_repr
        assert 'player=white' in str_repr
        assert 'value=0.5' in str_repr
        assert 'moves_count=3' in str_repr

    def test_state_copy_independence(self):
        """Test that copied state is independent."""
        state1 = ChessState()
        state1.value = 1.0
        state1.player = 'black'
        state1.is_terminal = True
        state1.choices_start = cp.array([1, 2, 3])

        state2 = state1.copy()

        # Modify state2
        state2.value = 2.0
        state2.player = 'white'
        state2.is_terminal = False
        state2.choices_start[0] = 999

        # Verify state1 unchanged
        assert state1.value == 1.0
        assert state1.player == 'black'
        assert state1.is_terminal is True
        assert state1.choices_start[0] == 1

    def test_is_game_over(self):
        """Test game over detection."""
        state = ChessState()
        assert not state.is_game_over()

        state.is_terminal = True
        assert state.is_game_over()

    def test_get_winner_not_terminal(self):
        """Test get_winner returns None when game not over."""
        state = ChessState()
        assert state.get_winner() is None

    def test_get_winner_white_wins(self):
        """Test get_winner when white wins."""
        state = ChessState()
        state.is_terminal = True
        state.value = 1.0
        state.winning = 'white'
        assert state.get_winner() == 'white'

    def test_get_winner_black_wins(self):
        """Test get_winner when black wins."""
        state = ChessState()
        state.is_terminal = True
        state.value = -1.0
        state.winning = 'white'
        assert state.get_winner() == 'black'

    def test_get_winner_draw(self):
        """Test get_winner returns None for draw."""
        state = ChessState()
        state.is_terminal = True
        state.value = 0.0
        assert state.get_winner() is None

    def test_copy_with_none_arrays(self):
        """Test copying state with None arrays."""
        state1 = ChessState()
        state2 = state1.copy()

        assert state2.choices_start is None
        assert state2.choices_end is None
        assert state2.raw_board is None
        assert state2.board is None

    def test_copy_with_arrays(self):
        """Test copying state with cupy arrays."""
        state1 = ChessState()
        state1.choices_start = cp.array([[1, 2], [3, 4]])
        state1.board = cp.array([5, 6, 7])

        state2 = state1.copy()

        # Verify arrays are copied
        assert cp.array_equal(state2.choices_start, state1.choices_start)
        assert cp.array_equal(state2.board, state1.board)

        # Verify independence
        state2.choices_start[0, 0] = 999
        assert state1.choices_start[0, 0] == 1
