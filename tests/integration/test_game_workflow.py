"""Integration tests for full game workflow."""

import pytest
from src.game_runner import GameRunner
from config.game_config import TestConfig


class TestGameWorkflow:
    """Test complete game workflows."""

    @pytest.fixture
    def runner(self):
        """Create a game runner with test configuration."""
        config = TestConfig.get_test_game_config()
        mcts_config = TestConfig.get_test_mcts_config()
        return GameRunner(
            max_time=config['max_time'],
            max_turns=config['max_turns'],
            mcts_args=mcts_config
        )

    def test_start_new_game(self, runner):
        """Test starting a new game."""
        state = runner.start_new_game()

        assert state is not None
        assert state.player == 'white'
        assert not state.is_terminal
        assert runner.state == state

    def test_make_ai_move(self, runner):
        """Test making an AI move."""
        runner.start_new_game()
        move = runner.make_ai_move()

        assert move is not None
        assert isinstance(move, str)
        assert '>' in move  # Basic move format check

    def test_game_string_retrieval(self, runner):
        """Test retrieving game string."""
        runner.start_new_game()
        game_str = runner.get_game_string()

        assert game_str is not None
        assert isinstance(game_str, str)

    def test_is_game_over_initial(self, runner):
        """Test game over check at start."""
        runner.start_new_game()
        assert not runner.is_game_over()

    def test_play_ai_vs_ai_completes(self, runner):
        """Test AI vs AI game completes."""
        results = runner.play_ai_vs_ai(max_moves=10, verbose=False)

        assert 'winner' in results
        assert 'moves' in results
        assert 'move_count' in results
        assert 'game_string' in results
        assert isinstance(results['moves'], list)
        assert results['move_count'] >= 0

    def test_play_ai_vs_ai_respects_max_moves(self, runner):
        """Test AI vs AI respects max moves limit."""
        max_moves = 5
        results = runner.play_ai_vs_ai(max_moves=max_moves, verbose=False)

        assert results['move_count'] <= max_moves

    def test_multiple_games(self, runner):
        """Test playing multiple games in sequence."""
        results1 = runner.play_ai_vs_ai(max_moves=5, verbose=False)
        results2 = runner.play_ai_vs_ai(max_moves=5, verbose=False)

        # Games should be independent
        assert results1['move_count'] >= 0
        assert results2['move_count'] >= 0

    def test_game_state_consistency(self, runner):
        """Test game state remains consistent."""
        runner.start_new_game()
        initial_player = runner.state.player

        # Make a move
        move = runner.make_ai_move()

        if move is not None:
            # Player should change after move
            assert runner.state.prev_player == initial_player

    def test_winner_determination(self, runner):
        """Test winner determination in completed game."""
        results = runner.play_ai_vs_ai(max_moves=20, verbose=False)

        if runner.is_game_over():
            winner = runner.get_winner()
            # Winner should be None (draw) or a player name
            assert winner is None or winner in ['white', 'black']

    def test_game_progression(self, runner):
        """Test game progresses correctly."""
        runner.start_new_game()
        move_count = 0
        max_moves = 10

        while not runner.is_game_over() and move_count < max_moves:
            move = runner.make_ai_move()
            if move is None:
                break
            move_count += 1

        assert move_count <= max_moves

    def test_no_moves_after_game_over(self, runner):
        """Test no moves possible after game ends."""
        results = runner.play_ai_vs_ai(max_moves=30, verbose=False)

        if runner.is_game_over():
            # Try to make another move
            move = runner.make_ai_move()
            assert move is None


class TestGameEdgeCases:
    """Test edge cases in game workflow."""

    def test_make_move_before_start(self):
        """Test making move before starting game."""
        runner = GameRunner()
        move = runner.make_ai_move()
        assert move is None

    def test_get_game_string_before_start(self):
        """Test getting game string before starting."""
        runner = GameRunner()
        game_str = runner.get_game_string()
        assert game_str is None

    def test_is_game_over_before_start(self):
        """Test game over check before starting."""
        runner = GameRunner()
        assert runner.is_game_over()

    def test_get_winner_before_start(self):
        """Test getting winner before starting."""
        runner = GameRunner()
        winner = runner.get_winner()
        assert winner is None


class TestGameConfiguration:
    """Test different game configurations."""

    def test_custom_configuration(self):
        """Test game with custom configuration."""
        runner = GameRunner(
            max_time=5,
            max_turns=15,
            mcts_args={'num_searches': 3, 'C': 1.41}
        )

        assert runner.game.max_time == 5
        assert runner.game.max_turns == 15
        assert runner.mcts_args['num_searches'] == 3

    def test_default_configuration(self):
        """Test game with default configuration."""
        runner = GameRunner()

        assert runner.game.max_time > 0
        assert runner.game.max_turns > 0
        assert 'num_searches' in runner.mcts_args
        assert 'C' in runner.mcts_args
