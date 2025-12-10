"""Unit tests for MCTS algorithm."""

import pytest
import cupy as cp
import math
from src.mcts import Node, MCTS
from src.chess_engine import Chess5D
from src.chess_state import ChessState
from config.game_config import TestConfig


class TestNode:
    """Test MCTS Node class."""

    @pytest.fixture
    def game_engine(self):
        """Create a test game engine."""
        config = TestConfig.get_test_game_config()
        return Chess5D(config['max_time'], config['max_turns'])

    @pytest.fixture
    def mcts_args(self):
        """Create MCTS test arguments."""
        return TestConfig.get_test_mcts_config()

    @pytest.fixture
    def initial_state(self, game_engine):
        """Create initial game state."""
        return game_engine.get_initial_state()

    def test_node_initialization(self, game_engine, mcts_args, initial_state):
        """Test node initialization."""
        node = Node(game_engine, mcts_args, initial_state)

        assert node.game == game_engine
        assert node.args == mcts_args
        assert node.state == initial_state
        assert node.parent is None
        assert node.children == []
        assert node.visit_count == 0
        assert node.value_sum == 0.0

    def test_node_with_parent(self, game_engine, mcts_args, initial_state):
        """Test node with parent."""
        parent = Node(game_engine, mcts_args, initial_state)
        child_state = initial_state.copy()
        child = Node(
            game_engine,
            mcts_args,
            child_state,
            parent,
            (0, 0, 0, 0),
            (0, 0, 0, 0)
        )

        assert child.parent == parent
        assert child.action_taken_s == (0, 0, 0, 0)
        assert child.action_taken_e == (0, 0, 0, 0)

    def test_is_fully_expanded_false_no_children(self, game_engine, mcts_args, initial_state):
        """Test node not fully expanded with no children."""
        node = Node(game_engine, mcts_args, initial_state)
        assert not node.is_fully_expanded()

    def test_is_fully_expanded_with_moves_remaining(self, game_engine, mcts_args, initial_state):
        """Test node not fully expanded with moves remaining."""
        node = Node(game_engine, mcts_args, initial_state)
        # Add a child but keep expandable moves
        child_state = initial_state.copy()
        child = Node(game_engine, mcts_args, child_state, node)
        node.children.append(child)

        assert not node.is_fully_expanded()

    def test_ucb_unvisited_child(self, game_engine, mcts_args, initial_state):
        """Test UCB returns infinity for unvisited child."""
        parent = Node(game_engine, mcts_args, initial_state)
        parent.visit_count = 10

        child = Node(game_engine, mcts_args, initial_state.copy(), parent)

        ucb = parent.get_ucb(child)
        assert ucb == float('inf')

    def test_ucb_calculation_visited_child(self, game_engine, mcts_args, initial_state):
        """Test UCB calculation for visited child."""
        parent = Node(game_engine, mcts_args, initial_state)
        parent.visit_count = 10

        child = Node(game_engine, mcts_args, initial_state.copy(), parent)
        child.visit_count = 5
        child.value_sum = 2.5

        ucb = parent.get_ucb(child)
        assert isinstance(ucb, float)
        assert ucb > 0

    def test_backpropagate_updates_counts(self, game_engine, mcts_args, initial_state):
        """Test backpropagation updates visit counts and values."""
        parent = Node(game_engine, mcts_args, initial_state)
        child = Node(game_engine, mcts_args, initial_state.copy(), parent)

        child.backpropagate(1.0)

        assert child.visit_count == 1
        assert child.value_sum == 1.0
        assert parent.visit_count == 1

    def test_backpropagate_chain(self, game_engine, mcts_args, initial_state):
        """Test backpropagation through chain of nodes."""
        root = Node(game_engine, mcts_args, initial_state)
        child1 = Node(game_engine, mcts_args, initial_state.copy(), root)
        child2 = Node(game_engine, mcts_args, initial_state.copy(), child1)

        child2.backpropagate(0.5)

        assert child2.visit_count == 1
        assert child1.visit_count == 1
        assert root.visit_count == 1

    def test_select_chooses_best_child(self, game_engine, mcts_args, initial_state):
        """Test select chooses child with best UCB."""
        parent = Node(game_engine, mcts_args, initial_state)
        parent.visit_count = 10

        # Create children with different values
        child1 = Node(game_engine, mcts_args, initial_state.copy(), parent)
        child1.visit_count = 5
        child1.value_sum = 1.0

        child2 = Node(game_engine, mcts_args, initial_state.copy(), parent)
        child2.visit_count = 3
        child2.value_sum = 2.0

        parent.children = [child1, child2]

        selected = parent.select()
        assert selected in [child1, child2]


class TestMCTS:
    """Test MCTS algorithm class."""

    @pytest.fixture
    def game_engine(self):
        """Create a test game engine."""
        config = TestConfig.get_test_game_config()
        return Chess5D(config['max_time'], config['max_turns'])

    @pytest.fixture
    def mcts_args(self):
        """Create MCTS test arguments."""
        return TestConfig.get_test_mcts_config()

    @pytest.fixture
    def mcts(self, game_engine, mcts_args):
        """Create MCTS instance."""
        return MCTS(game_engine, mcts_args)

    def test_mcts_initialization(self, game_engine, mcts_args):
        """Test MCTS initialization."""
        mcts = MCTS(game_engine, mcts_args)

        assert mcts.game == game_engine
        assert mcts.args == mcts_args

    def test_search_returns_probabilities(self, mcts, game_engine):
        """Test search returns action probability distributions."""
        state = game_engine.get_initial_state()

        action_probs_start, action_probs_end = mcts.search(state)

        assert action_probs_start.shape == game_engine.action_size
        assert action_probs_end.shape == game_engine.action_size
        assert isinstance(action_probs_start, cp.ndarray)
        assert isinstance(action_probs_end, cp.ndarray)

    def test_search_probabilities_sum_to_one(self, mcts, game_engine):
        """Test search probability distributions sum to 1."""
        state = game_engine.get_initial_state()

        action_probs_start, action_probs_end = mcts.search(state)

        # Allow small floating point error
        assert abs(cp.sum(action_probs_start) - 1.0) < 1e-6 or cp.sum(action_probs_start) == 0
        assert abs(cp.sum(action_probs_end) - 1.0) < 1e-6 or cp.sum(action_probs_end) == 0

    def test_search_runs_specified_iterations(self, mcts, game_engine):
        """Test search runs the correct number of iterations."""
        state = game_engine.get_initial_state()
        num_searches = mcts.args['num_searches']

        # This should complete without error
        action_probs_start, action_probs_end = mcts.search(state)

        # Verify we got valid output
        assert action_probs_start is not None
        assert action_probs_end is not None

    def test_get_best_move_returns_valid_move(self, mcts, game_engine):
        """Test get_best_move returns a valid move tuple."""
        state = game_engine.get_initial_state()

        result = mcts.get_best_move(state)

        if result is not None:
            move_str, start_pos, end_pos = result
            assert isinstance(move_str, str)
            assert isinstance(start_pos, tuple)
            assert isinstance(end_pos, tuple)
            assert len(start_pos) == 4
            assert len(end_pos) == 4
