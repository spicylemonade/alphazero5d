"""Monte Carlo Tree Search implementation for 5D Chess."""

from typing import Optional, Tuple, Dict, Any
import math
import cupy as cp

from src.chess_state import ChessState
from src.chess_engine import Chess5D


class Node:
    """
    A node in the Monte Carlo Tree Search tree.

    Attributes:
        game: Reference to the game engine
        args: MCTS configuration arguments
        state: Current game state at this node
        parent: Parent node in the tree
        action_taken_s: Starting position action taken to reach this node
        action_taken_e: Ending position action taken to reach this node
        children: List of child nodes
        visit_count: Number of times this node has been visited
        value_sum: Sum of values from all simulations through this node
    """

    def __init__(
        self,
        game: Chess5D,
        args: Dict[str, Any],
        state: ChessState,
        parent: Optional['Node'] = None,
        action_taken_s: Optional[Tuple[int, ...]] = None,
        action_taken_e: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize a new MCTS node.

        Args:
            game: Game engine instance
            args: MCTS configuration parameters
            state: Game state at this node
            parent: Parent node (None for root)
            action_taken_s: Start position of action taken
            action_taken_e: End position of action taken
        """
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken_s = action_taken_s
        self.action_taken_e = action_taken_e
        self.children = []
        self.expandable_moves_start = state.choices_start.copy()
        self.expandable_moves_end = state.choices_end.copy()
        self.player = state.player

        self.visit_count = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        """
        Check if all possible moves from this node have been explored.

        Returns:
            bool: True if node is fully expanded
        """
        return cp.sum(self.expandable_moves_start) == 0 and len(self.children) > 0

    def select(self) -> 'Node':
        """
        Select the best child node using UCB1 formula.

        Returns:
            Node: Child node with highest UCB value
        """
        best_child = None
        best_ucb = -cp.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child: 'Node') -> float:
        """
        Calculate Upper Confidence Bound (UCB1) value for a child node.

        Args:
            child: Child node to evaluate

        Returns:
            float: UCB value
        """
        if child.visit_count == 0:
            return float('inf')

        # Calculate Q-value (exploitation term)
        if self.parent is not None:
            if self.parent.player == child.player:
                # Same player continuing turn
                q_value = ((child.value_sum / child.visit_count) + 1) / 2
            else:
                # Opponent's turn
                q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        # Exploration term
        exploration = self.args['C'] * math.sqrt(
            math.log(self.visit_count) / child.visit_count
        )

        return q_value + exploration

    def expand(self) -> 'Node':
        """
        Expand the node by creating a new child with an unexplored action.

        Returns:
            Node: Newly created child node
        """
        result = self.game.pick_choice(
            self.state,
            self.expandable_moves_start,
            self.expandable_moves_end,
            False
        )

        if result is None:
            # Stalemate case
            return self

        action, rl_action_s, rl_action_e = result
        child_state = self.state.copy()
        self.game.make_move(child_state, action)

        child = Node(
            self.game,
            self.args,
            child_state,
            self,
            rl_action_s,
            rl_action_e
        )
        self.children.append(child)

        return child

    def simulate(self) -> float:
        """
        Simulate a random playout from this node to a terminal state.

        Returns:
            float: Value of the terminal state from current player's perspective
        """
        value, is_terminal = self.state.value, self.state.is_terminal
        value = self.game.get_opponent_value(self.state, value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        initial_player = rollout_state.player

        while True:
            result = self.game.pick_choice(
                rollout_state,
                rollout_state.choices_start,
                rollout_state.choices_end,
                True
            )

            if result is None:
                # Stalemate during simulation
                return 0.3

            action_m, _, _ = result
            self.game.make_move(rollout_state, action_m)

            value, is_terminal = rollout_state.value, rollout_state.is_terminal

            if is_terminal:
                # Return value from initial player's perspective
                if initial_player != rollout_state.player:
                    return value
                else:
                    return -value

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate the simulation result up the tree.

        Args:
            value: Value to backpropagate
        """
        self.value_sum += value
        self.visit_count += 1

        # Flip value based on player perspective
        if self.state.player == "white":
            value = abs(value)
        else:
            value = -abs(value)

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    """
    Monte Carlo Tree Search algorithm for game tree search.

    Attributes:
        game: Game engine instance
        args: Configuration parameters for MCTS
    """

    def __init__(self, game: Chess5D, args: Dict[str, Any]):
        """
        Initialize MCTS algorithm.

        Args:
            game: Game engine instance
            args: Configuration with 'num_searches' and 'C' parameters
        """
        self.game = game
        self.args = args

    def search(
        self,
        state: ChessState
    ) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Perform MCTS search from the given state.

        Args:
            state: Current game state to search from

        Returns:
            Tuple[cp.ndarray, cp.ndarray]: Action probability distributions
                for start and end positions
        """
        root = Node(self.game, self.args, state)

        for search_iteration in range(self.args['num_searches']):
            node = root

            # Selection: traverse tree using UCB
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = node.state.value, node.state.is_terminal
            value = self.game.get_opponent_value(node.state, value)

            # Expansion and simulation
            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            # Backpropagation
            node.backpropagate(value)

        # Create action probability distributions based on visit counts
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float64)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float64)

        for child in root.children:
            if child.action_taken_s is not None:
                action_probs_start[child.action_taken_s] += child.visit_count
            if child.action_taken_e is not None:
                action_probs_end[child.action_taken_e] += child.visit_count

        # Normalize probabilities
        if cp.sum(action_probs_start) > 0:
            action_probs_start /= cp.sum(action_probs_start)
        if cp.sum(action_probs_end) > 0:
            action_probs_end /= cp.sum(action_probs_end)

        return action_probs_start, action_probs_end

    def get_best_move(
        self,
        state: ChessState
    ) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
        """
        Get the best move from the current state using MCTS.

        Args:
            state: Current game state

        Returns:
            Optional[Tuple]: Best move string and positions, or None if no moves
        """
        mcts_prob_s, mcts_prob_e = self.search(state)

        if cp.sum(mcts_prob_s) == 0:
            return None

        # Get best start position
        index_s = cp.unravel_index(cp.argmax(mcts_prob_s), mcts_prob_s.shape)

        # Get best end position
        start_move = {
            "timeline": self.game.convert_timeline_opposite(index_s[0].item()),
            "turn": index_s[1].item() + 1,
            "rank": index_s[2].item() + 1,
            "file": index_s[3].item() + 1,
        }

        end_moves = self.game.get_end_moves(state.moves, start_move)
        if not end_moves:
            return None

        self.game.convert_moves_end(end_moves, state)

        # Find best end move
        temp = state.choices_end * mcts_prob_e
        if cp.sum(temp) == 0:
            return None

        index_e = cp.unravel_index(cp.argmax(temp), temp.shape)

        end_move = {
            "timeline": self.game.convert_timeline_opposite(index_e[0].item()),
            "turn": index_e[1].item() + 1,
            "rank": index_e[2].item() + 1,
            "file": index_e[3].item() + 1,
        }

        return self.game.move_to_string(start_move, end_move)
