"""Unit tests for Chess5D engine."""

import pytest
import cupy as cp
from src.chess_engine import Chess5D
from src.chess_state import ChessState
from src.exceptions import Checkmate, Stalemate, DrawLoss
from config.game_config import TestConfig


class TestChess5D:
    """Test Chess5D game engine."""

    @pytest.fixture
    def game_engine(self):
        """Create a test game engine."""
        config = TestConfig.get_test_game_config()
        return Chess5D(config['max_time'], config['max_turns'])

    def test_initialization(self, game_engine):
        """Test game engine initialization."""
        assert game_engine.max_time == TestConfig.TEST_MAX_TIMELINES
        assert game_engine.max_turns == TestConfig.TEST_MAX_TURNS
        assert game_engine.action_size == (
            TestConfig.TEST_MAX_TIMELINES,
            TestConfig.TEST_MAX_TURNS,
            8,
            8
        )

    def test_timeline_conversion(self):
        """Test timeline number conversion."""
        assert Chess5D.convert_timeline(0) == 0
        assert Chess5D.convert_timeline(1) == 2
        assert Chess5D.convert_timeline(-1) == 1
        assert Chess5D.convert_timeline(2) == 4
        assert Chess5D.convert_timeline(-2) == 3

    def test_timeline_opposite_conversion(self):
        """Test reverse timeline conversion."""
        assert Chess5D.convert_timeline_opposite(0) == 0
        assert Chess5D.convert_timeline_opposite(2) == 1
        assert Chess5D.convert_timeline_opposite(1) == -1
        assert Chess5D.convert_timeline_opposite(4) == 2
        assert Chess5D.convert_timeline_opposite(3) == -2

    def test_timeline_conversion_roundtrip(self):
        """Test timeline conversion is reversible."""
        for i in range(-5, 6):
            converted = Chess5D.convert_timeline(i)
            back = Chess5D.convert_timeline_opposite(converted)
            assert back == i

    def test_get_initial_state(self, game_engine):
        """Test getting initial game state."""
        state = game_engine.get_initial_state()

        assert isinstance(state, ChessState)
        assert state.player == 'white'
        assert not state.is_terminal
        assert state.choices_start is not None
        assert state.choices_end is not None
        assert state.board is not None

    def test_raw_board_to_tensor_shape(self, game_engine):
        """Test board tensor conversion shape."""
        raw_board = cp.zeros((game_engine.max_time, game_engine.max_turns, 8, 8))
        tensor = game_engine.raw_board_to_tensor(raw_board)

        expected_shape = (
            game_engine.max_time,
            game_engine.max_turns * 2,
            6,
            8,
            8
        )
        assert tensor.shape == expected_shape

    def test_move_to_string_format(self):
        """Test move string formatting."""
        start = {'timeline': 0, 'turn': 1, 'rank': 2, 'file': 1}
        end = {'timeline': 0, 'turn': 1, 'rank': 4, 'file': 1}

        move_str, start_pos, end_pos = Chess5D.move_to_string(start, end)

        assert '(0T1)' in move_str
        assert 'a2' in move_str
        assert 'a4' in move_str
        assert isinstance(start_pos, tuple)
        assert isinstance(end_pos, tuple)
        assert len(start_pos) == 4
        assert len(end_pos) == 4

    def test_get_end_moves_filtering(self):
        """Test filtering of end moves."""
        moves = [
            {
                'start': {'timeline': 0, 'turn': 1, 'rank': 2, 'file': 1},
                'end': {'timeline': 0, 'turn': 1, 'rank': 3, 'file': 1}
            },
            {
                'start': {'timeline': 0, 'turn': 1, 'rank': 2, 'file': 1},
                'end': {'timeline': 0, 'turn': 1, 'rank': 4, 'file': 1}
            },
            {
                'start': {'timeline': 0, 'turn': 1, 'rank': 2, 'file': 2},
                'end': {'timeline': 0, 'turn': 1, 'rank': 3, 'file': 2}
            }
        ]

        start_value = {'timeline': 0, 'turn': 1, 'rank': 2, 'file': 1}
        end_moves = Chess5D.get_end_moves(moves, start_value)

        assert len(end_moves) == 2
        assert all(move['rank'] in [3, 4] for move in end_moves)

    def test_check_timelines_within_bounds(self, game_engine):
        """Test timeline bounds checking."""
        move = {
            'realEnd': {'timeline': 0, 'turn': 1}
        }
        assert not game_engine.check_timelines(move)

    def test_check_timelines_out_of_bounds(self, game_engine):
        """Test timeline exceeding bounds."""
        max_timeline = (game_engine.max_time - 1) // 2
        move = {
            'realEnd': {'timeline': max_timeline + 1, 'turn': 1}
        }
        assert game_engine.check_timelines(move)

    def test_check_turns_within_bounds(self, game_engine):
        """Test turn bounds checking."""
        move = {
            'realEnd': {'timeline': 0, 'turn': 1}
        }
        assert not game_engine.check_turns(move)

    def test_check_turns_out_of_bounds(self, game_engine):
        """Test turn exceeding bounds."""
        move = {
            'realEnd': {'timeline': 0, 'turn': game_engine.max_turns + 1}
        }
        assert game_engine.check_turns(move)

    def test_get_opponent_value_same_player(self, game_engine):
        """Test opponent value when player hasn't changed."""
        state = ChessState()
        state.player = 'white'
        state.prev_player = 'white'

        value = game_engine.get_opponent_value(state, 1.0)
        assert value == 1.0

    def test_get_opponent_value_different_player(self, game_engine):
        """Test opponent value when player changed."""
        state = ChessState()
        state.player = 'black'
        state.prev_player = 'white'

        value = game_engine.get_opponent_value(state, 1.0)
        assert value == -1.0

    def test_player_map_values(self):
        """Test player mapping to numeric values."""
        assert Chess5D.player_map['white'] == 1
        assert Chess5D.player_map['black'] == -1

    def test_player_map_opponent(self):
        """Test opponent mapping."""
        assert Chess5D.player_map_opp['white'] == 'black'
        assert Chess5D.player_map_opp['black'] == 'white'
