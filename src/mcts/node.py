"""
Monte Carlo Tree Search Node
Represents a node in the MCTS tree with UCB selection.
"""
import math
import cupy as cp
from typing import Optional, Tuple


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search tree.

    Attributes:
        game: Reference to game engine
        args: MCTS parameters
        state: Game state at this node
        parent: Parent node
        action_taken_s: Start position of action taken
        action_taken_e: End position of action taken
        children: List of child nodes
        expandable_moves_start: Valid start positions
        expandable_moves_end: Valid end positions
        player: Current player at this node
        visit_count: Number of times node visited
        value_sum: Sum of values from rollouts
    """

    __slots__ = ['game', 'args', 'state', 'parent', 'action_taken_s',
                 'action_taken_e', 'children', 'expandable_moves_start',
                 'expandable_moves_end', 'player', 'visit_count', 'value_sum']

    def __init__(
        self,
        game,
        args: dict,
        state,
        parent: Optional['MCTSNode'] = None,
        action_taken_s: Optional[Tuple] = None,
        action_taken_e: Optional[Tuple] = None
    ):
        """
        Initialize MCTS node.

        Args:
            game: Game engine instance
            args: MCTS parameters dictionary
            state: Game state at this node
            parent: Parent node (None for root)
            action_taken_s: Start position of action
            action_taken_e: End position of action
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
        self.value_sum = 0

    def is_fully_expanded(self) -> bool:
        """
        Check if node is fully expanded.

        Returns:
            bool: True if all children have been generated
        """
        return cp.sum(self.expandable_moves_start) == 0 and len(self.children) > 0

    def select(self) -> 'MCTSNode':
        """
        Select best child using UCB formula.

        Returns:
            MCTSNode: Child node with highest UCB value
        """
        best_child = None
        best_ucb = -cp.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child: 'MCTSNode') -> float:
        """
        Calculate Upper Confidence Bound for child node.

        Args:
            child: Child node to evaluate

        Returns:
            float: UCB value
        """
        # Avoid division by zero
        if child.visit_count == 0:
            return float('inf')

        # Calculate Q-value based on perspective
        if self.parent is not None and self.parent.player == child.player:
            # Same player - use value directly
            q_value = ((child.value_sum / child.visit_count) + 1) / 2
        else:
            # Opponent - invert value
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2

        # UCB formula: Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))
        exploration = self.args['C'] * math.sqrt(
            math.log(self.visit_count) / child.visit_count
        )

        return q_value + exploration

    def expand(self) -> 'MCTSNode':
        """
        Expand node by creating a new child.

        Returns:
            MCTSNode: Newly created child node
        """
        action, rl_action_s, rl_action_e = self.game.pick_random_move(
            self.state,
            self.expandable_moves_start,
            self.expandable_moves_end,
            modify_probs=False
        )

        child_state = self.state.copy()
        self.game.make_move(child_state, action)

        child = MCTSNode(
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
        Simulate random playout from this node.

        Returns:
            float: Value of terminal state
        """
        value, is_terminal = self.state.value, self.state.is_terminal
        value = self.game.get_opponent_value(self.state, value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        initial_player = rollout_state.player

        # Perform random rollout
        while True:
            try:
                action, _, _ = self.game.pick_random_move(
                    rollout_state,
                    rollout_state.choices_start,
                    rollout_state.choices_end,
                    modify_probs=True
                )

                self.game.make_move(rollout_state, action)

                value, is_terminal = rollout_state.value, rollout_state.is_terminal

                if is_terminal:
                    # Return value from initial player's perspective
                    if initial_player != rollout_state.player:
                        return value
                    else:
                        return -value

            except Exception as e:
                # Handle unexpected errors during rollout
                return 0

    def backpropagate(self, value: float) -> None:
        """
        Backpropagate value up the tree.

        Args:
            value: Value to backpropagate
        """
        self.value_sum += value
        self.visit_count += 1

        # Adjust value based on player perspective
        value = abs(value) if self.state.player == "white" else -abs(value)

        if self.parent is not None:
            self.parent.backpropagate(value)
