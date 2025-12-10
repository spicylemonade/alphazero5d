"""Configuration settings for 5D Chess game."""

from typing import Dict, Any


class GameConfig:
    """Configuration for 5D Chess game settings."""

    # Board dimensions
    MAX_TIMELINES = 11
    MAX_TURNS = 30
    BOARD_SIZE = 8

    # MCTS parameters
    MCTS_NUM_SEARCHES = 20
    MCTS_EXPLORATION_CONSTANT = 1.41

    # Player settings
    DEFAULT_PLAYER = 'white'
    PLAYERS = ['white', 'black']

    # Game values
    WIN_VALUE = 1.0
    LOSS_VALUE = -1.0
    DRAW_VALUE = 0.0
    DRAW_LOSS_VALUE = 0.3

    @classmethod
    def get_mcts_config(cls) -> Dict[str, Any]:
        """
        Get MCTS configuration dictionary.

        Returns:
            Dict[str, Any]: MCTS configuration parameters
        """
        return {
            'num_searches': cls.MCTS_NUM_SEARCHES,
            'C': cls.MCTS_EXPLORATION_CONSTANT
        }

    @classmethod
    def get_game_config(cls) -> Dict[str, Any]:
        """
        Get game configuration dictionary.

        Returns:
            Dict[str, Any]: Game configuration parameters
        """
        return {
            'max_time': cls.MAX_TIMELINES,
            'max_turns': cls.MAX_TURNS,
            'board_size': cls.BOARD_SIZE
        }


class TestConfig:
    """Configuration for testing."""

    # Smaller dimensions for faster tests
    TEST_MAX_TIMELINES = 3
    TEST_MAX_TURNS = 10
    TEST_MCTS_SEARCHES = 5

    @classmethod
    def get_test_mcts_config(cls) -> Dict[str, Any]:
        """Get MCTS config for testing."""
        return {
            'num_searches': cls.TEST_MCTS_SEARCHES,
            'C': GameConfig.MCTS_EXPLORATION_CONSTANT
        }

    @classmethod
    def get_test_game_config(cls) -> Dict[str, Any]:
        """Get game config for testing."""
        return {
            'max_time': cls.TEST_MAX_TIMELINES,
            'max_turns': cls.TEST_MAX_TURNS,
            'board_size': GameConfig.BOARD_SIZE
        }
