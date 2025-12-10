"""
Monte Carlo Tree Search Algorithm
Implements MCTS for move selection in 5D chess.
"""
import cupy as cp
from typing import Tuple
from .node import MCTSNode


class MCTS:
    """
    Monte Carlo Tree Search implementation.

    Attributes:
        game: Game engine instance
        args: MCTS configuration parameters
    """

    def __init__(self, game, args: dict):
        """
        Initialize MCTS.

        Args:
            game: Game engine instance
            args: Dictionary with 'num_searches' and 'C' parameters
        """
        self.game = game
        self.args = args

    def search(self, state) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Perform MCTS to find best move distribution.

        Args:
            state: Current game state

        Returns:
            Tuple of (start_probs, end_probs) as CuPy arrays
        """
        root = MCTSNode(self.game, self.args, state)

        # Perform specified number of MCTS iterations
        for search_iteration in range(self.args['num_searches']):
            node = root

            # Selection: traverse tree using UCB
            while node.is_fully_expanded():
                node = node.select()

            # Check if node is terminal
            value = self.game.get_opponent_value(state, state.value)
            is_terminal = state.is_terminal

            # Expansion and simulation
            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            # Backpropagation
            node.backpropagate(value)

        # Build action probability distributions from visit counts
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float64)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float64)

        for child in root.children:
            action_probs_start[child.action_taken_s] += child.visit_count
            action_probs_end[child.action_taken_e] += child.visit_count

        # Normalize to create probability distribution
        if cp.sum(action_probs_start) > 0:
            action_probs_start /= cp.sum(action_probs_start)

        return action_probs_start, action_probs_end
