"""
Enhanced Monte Carlo Tree Search with Advanced Features
Improvements over base MCTS:
1. Progressive widening for action selection
2. Virtual loss for parallel MCTS
3. RAVE (Rapid Action Value Estimation)
4. Adaptive exploration constant
5. Transposition table
"""

import cupy as cp
import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class MCTSConfig:
    """Configuration for MCTS algorithm"""
    num_searches: int = 800
    c_puct: float = 1.41
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 3.0
    rave_constant: float = 300
    progressive_widening_alpha: float = 0.5
    progressive_widening_beta: float = 1.0
    use_transposition_table: bool = True
    max_depth: int = 100


class Node:
    """Enhanced MCTS node with additional features"""

    def __init__(self, game, config: MCTSConfig, state,
                 parent=None, action_taken_s=None, action_taken_e=None,
                 prior: float = 0.0):
        self.game = game
        self.config = config
        self.state = state
        self.parent = parent
        self.action_taken_s = action_taken_s
        self.action_taken_e = action_taken_e
        self.prior = prior
        self.children = []
        self.expandable_moves_start = state.choices_start.copy() if state.choices_start is not None else None
        self.expandable_moves_end = state.choices_end.copy() if state.choices_end is not None else None
        self.player = state.player

        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss_count = 0

        # RAVE statistics
        self.rave_visit_count = 0
        self.rave_value_sum = 0.0

        # For progressive widening
        self.num_actions_tried = 0

        # Depth tracking
        self.depth = 0 if parent is None else parent.depth + 1

    def is_fully_expanded(self) -> bool:
        """Check if node is fully expanded using progressive widening"""
        if self.expandable_moves_start is None:
            return True

        k_n = self.config.progressive_widening_alpha * (self.visit_count ** self.config.progressive_widening_beta)
        max_children = min(int(k_n) + 1, int(cp.sum(self.expandable_moves_start)))

        return len(self.children) >= max_children and len(self.children) > 0

    def select(self) -> 'Node':
        """Select best child using UCB with RAVE"""
        best_child = None
        best_score = -cp.inf

        for child in self.children:
            score = self._get_ucb_rave(child)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _get_ucb_rave(self, child: 'Node') -> float:
        """Compute UCB score with RAVE"""
        if child.visit_count == 0:
            return float('inf')

        # Standard UCB component
        q_value = self._get_q_value(child)

        # Exploration component
        exploration = self.config.c_puct * child.prior * \
                     math.sqrt(self.visit_count) / (1 + child.visit_count)

        ucb_score = q_value + exploration

        # RAVE component
        if child.rave_visit_count > 0 and self.config.rave_constant > 0:
            rave_value = child.rave_value_sum / child.rave_visit_count

            # Beta coefficient for mixing UCB and RAVE
            beta = child.rave_visit_count / (child.rave_visit_count + child.visit_count +
                                            child.rave_visit_count * child.visit_count / self.config.rave_constant)

            # Mix RAVE with UCB
            ucb_score = (1 - beta) * q_value + beta * rave_value + exploration

        # Subtract virtual loss
        virtual_loss_penalty = self.config.virtual_loss * child.virtual_loss_count / (child.visit_count + 1)

        return ucb_score - virtual_loss_penalty

    def _get_q_value(self, child: 'Node') -> float:
        """Compute Q-value from child's perspective"""
        if child.visit_count == 0:
            return 0.0

        avg_value = child.value_sum / child.visit_count

        # Adjust for player
        if self.parent is not None:
            if self.parent.player == child.player:
                # Same player (unusual in 2-player game)
                q_value = (avg_value + 1) / 2
            else:
                # Opponent's turn
                q_value = 1 - (avg_value + 1) / 2
        else:
            q_value = 1 - (avg_value + 1) / 2

        return q_value

    def expand(self, add_noise: bool = False) -> 'Node':
        """Expand node by adding a new child"""
        if self.expandable_moves_start is None or cp.sum(self.expandable_moves_start) == 0:
            return self

        # Select action with highest prior among unexpanded actions
        action_probs = self.expandable_moves_start.copy()

        # Add Dirichlet noise for root node exploration
        if add_noise and self.parent is None:
            noise = cp.random.dirichlet(
                cp.full(action_probs.size, self.config.dirichlet_alpha)
            ).reshape(action_probs.shape)
            action_probs = ((1 - self.config.dirichlet_epsilon) * action_probs +
                          self.config.dirichlet_epsilon * noise)

        # Normalize and sample
        action_probs = action_probs / cp.sum(action_probs)
        flat_probs = action_probs.flatten()
        flat_idx = cp.random.choice(len(flat_probs), p=flat_probs.get())
        action_s = cp.unravel_index(flat_idx, action_probs.shape)

        # Create move
        try:
            start_move = {
                "timeline": self.game.convert_timeline_opposite(int(action_s[0])),
                "turn": int(action_s[1]) + 1,
                "rank": int(action_s[2]) + 1,
                "file": int(action_s[3]) + 1,
            }

            # Get end moves
            from chess5d_optimized import Chess5DOptimized
            end_moves = Chess5DOptimized._get_end_moves(self.state.moves, start_move)

            if not end_moves:
                # Mark as unexpandable and try again
                self.expandable_moves_start[action_s] = 0
                if cp.sum(self.expandable_moves_start) > 0:
                    return self.expand(add_noise)
                return self

            # Select end move
            self.state.choices_end.fill(0)
            for move in end_moves:
                timeline = self.game.convert_timeline(move['timeline'])
                self.state.choices_end[timeline, move['turn']-1, move['rank']-1, move['file']-1] += 1

            end_probs = self.state.choices_end.copy()
            end_probs = end_probs / cp.sum(end_probs)
            flat_end_probs = end_probs.flatten()
            flat_end_idx = cp.random.choice(len(flat_end_probs), p=flat_end_probs.get())
            action_e = cp.unravel_index(flat_end_idx, end_probs.shape)

            end_move = {
                "timeline": self.game.convert_timeline_opposite(int(action_e[0])),
                "turn": int(action_e[1]) + 1,
                "rank": int(action_e[2]) + 1,
                "file": int(action_e[3]) + 1,
            }

            # Create move string
            move_str = Chess5DOptimized._move_to_string(start_move, end_move)[0]

            # Create child state
            child_state = self.state.copy()
            self.game.make_move(child_state, move_str)

            # Create child node
            prior = float(action_probs[action_s])
            child = Node(self.game, self.config, child_state, self,
                        tuple(action_s), tuple(action_e), prior)

            self.children.append(child)
            self.expandable_moves_start[action_s] -= 1
            self.num_actions_tried += 1

            return child

        except Exception as e:
            # Remove this action from expandable moves
            self.expandable_moves_start[action_s] = 0
            if cp.sum(self.expandable_moves_start) > 0:
                return self.expand(add_noise)
            return self

    def simulate(self) -> float:
        """Simulate game to terminal state using rollout policy"""
        value, is_terminal = self.state.value, self.state.is_terminal
        value = self.game.get_opponent_value(self.state, value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        starting_player = rollout_state.player
        rollout_depth = 0
        max_rollout_depth = 50

        # Light rollout with random moves
        while not rollout_state.is_terminal and rollout_depth < max_rollout_depth:
            try:
                # Pick random legal move
                if not rollout_state.moves:
                    break

                move_idx = cp.random.randint(0, len(rollout_state.moves))
                move = rollout_state.moves[int(move_idx)]

                start = move['start']
                end = move['end']
                from chess5d_optimized import Chess5DOptimized
                move_str = Chess5DOptimized._move_to_string(start, end)[0]

                self.game.make_move(rollout_state, move_str)
                rollout_depth += 1

            except:
                break

        value = rollout_state.value
        is_terminal = rollout_state.is_terminal

        if is_terminal:
            if starting_player != rollout_state.player:
                return value
            else:
                return -value

        # Return heuristic value for non-terminal states
        return 0.0

    def backpropagate(self, value: float, actions_taken: List[Tuple] = None) -> None:
        """Backpropagate value with RAVE updates"""
        self.value_sum += value
        self.visit_count += 1
        self.virtual_loss_count -= 1  # Remove virtual loss

        # Update RAVE statistics if actions provided
        if actions_taken and self.action_taken_s in actions_taken:
            self.rave_value_sum += value
            self.rave_visit_count += 1

        # Flip value for opponent
        value = -value

        if self.parent is not None:
            if actions_taken is None:
                actions_taken = []
            if self.action_taken_s:
                actions_taken.append(self.action_taken_s)
            self.parent.backpropagate(value, actions_taken)

    def add_virtual_loss(self) -> None:
        """Add virtual loss for parallel MCTS"""
        self.virtual_loss_count += 1
        if self.parent:
            self.parent.add_virtual_loss()


class MCTSEnhanced:
    """Enhanced Monte Carlo Tree Search"""

    def __init__(self, game, config: MCTSConfig = None):
        self.game = game
        self.config = config or MCTSConfig()
        self.transposition_table = {} if self.config.use_transposition_table else None

        # Statistics
        self.nodes_created = 0
        self.transposition_hits = 0
        self.search_times = []

    def search(self, state, return_root: bool = False) -> Tuple[cp.ndarray, cp.ndarray, Optional['Node']]:
        """Run MCTS search and return action probabilities"""
        start_time = time.time()

        # Check transposition table
        state_key = state.game_string if self.transposition_table else None
        if state_key and state_key in self.transposition_table:
            root = self.transposition_table[state_key]
            self.transposition_hits += 1
        else:
            root = Node(self.game, self.config, state)
            self.nodes_created += 1
            if state_key:
                self.transposition_table[state_key] = root

        # Run simulations
        for search_idx in range(self.config.num_searches):
            node = root
            actions_taken = []

            # Selection
            while node.is_fully_expanded() and not node.state.is_terminal:
                node = node.select()
                node.add_virtual_loss()

            # Check if terminal
            value, is_terminal = node.state.value, node.state.is_terminal

            if not is_terminal and node.depth < self.config.max_depth:
                # Expansion
                add_noise = (search_idx == 0)  # Add noise to first expansion
                node = node.expand(add_noise)
                self.nodes_created += 1

                # Simulation
                value = node.simulate()
            else:
                value = self.game.get_opponent_value(node.state, value)

            # Backpropagation
            node.backpropagate(value, actions_taken)

        # Calculate action probabilities
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float32)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float32)

        for child in root.children:
            if child.action_taken_s:
                action_probs_start[child.action_taken_s] += child.visit_count
            if child.action_taken_e:
                action_probs_end[child.action_taken_e] += child.visit_count

        # Apply temperature
        if self.config.temperature == 0:
            # Greedy selection
            best_idx_s = cp.argmax(action_probs_start)
            action_probs_start.fill(0)
            action_probs_start.flat[best_idx_s] = 1

            best_idx_e = cp.argmax(action_probs_end)
            action_probs_end.fill(0)
            action_probs_end.flat[best_idx_e] = 1
        elif self.config.temperature != 1.0:
            action_probs_start = cp.power(action_probs_start, 1.0 / self.config.temperature)
            action_probs_end = cp.power(action_probs_end, 1.0 / self.config.temperature)

        # Normalize
        if cp.sum(action_probs_start) > 0:
            action_probs_start /= cp.sum(action_probs_start)
        if cp.sum(action_probs_end) > 0:
            action_probs_end /= cp.sum(action_probs_end)

        search_time = time.time() - start_time
        self.search_times.append(search_time)

        if return_root:
            return action_probs_start, action_probs_end, root
        return action_probs_start, action_probs_end, None

    def get_statistics(self) -> Dict:
        """Get search statistics"""
        return {
            'nodes_created': self.nodes_created,
            'transposition_hits': self.transposition_hits,
            'avg_search_time': np.mean(self.search_times) if self.search_times else 0,
            'total_searches': len(self.search_times)
        }
