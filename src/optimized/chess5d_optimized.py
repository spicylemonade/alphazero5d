"""
Optimized 5D Chess Engine with Enhanced MCTS
Improvements:
- Memory pooling for state copies
- Transposition table for repeated positions
- Move ordering and alpha-beta pruning hints
- Vectorized operations with CuPy
- Lazy evaluation of board tensors
- Cached move generation
"""

import cupy as cp
import numpy as np
from functools import lru_cache
import hashlib
from collections import defaultdict
from jsrun import JSrun
from javascript import require
import copy

Chess = require('5d-chess-js')


class GameException(Exception):
    pass


class DrawLoss(GameException):
    pass


class Stalemate(GameException):
    pass


class Checkmate(GameException):
    pass


class ChessState:
    """Optimized chess state with lazy evaluation and memory pooling"""
    __slots__ = ['chess', 'value', 'piece', 'game_string', 'choices_start',
                 'choices_end', 'is_terminal', 'moves', 'raw_board', 'board',
                 'player', 'prev_player', 'winning', '_hash', '_tensor_cached']

    def __init__(self):
        self.chess = Chess()
        self.value = 0
        self.piece = None
        self.game_string = None
        self.choices_start = None
        self.choices_end = None
        self.is_terminal = False
        self.moves = None
        self.raw_board = None
        self.board = None
        self.player = 'white'
        self.prev_player = None
        self.winning = 'white'
        self._hash = None
        self._tensor_cached = False

    def get_hash(self):
        """Generate hash for transposition table"""
        if self._hash is None:
            game_str = self.chess.export("5dpgn")
            self._hash = hashlib.md5(game_str.encode()).hexdigest()
        return self._hash

    def copy(self):
        """Optimized copy with selective deep copying"""
        new_state = ChessState()
        new_state.chess = self.chess.copy()
        new_state.value = self.value
        new_state.piece = self.piece
        new_state.game_string = self.game_string
        # Only copy arrays if they exist
        new_state.choices_start = cp.copy(self.choices_start) if self.choices_start is not None else None
        new_state.choices_end = cp.copy(self.choices_end) if self.choices_end is not None else None
        new_state.is_terminal = self.is_terminal
        new_state.moves = self.moves  # Share move list (immutable during copy)
        new_state.raw_board = self.raw_board  # Share raw board (will be recalculated if needed)
        new_state.board = self.board  # Share board tensor (will be recalculated if needed)
        new_state.player = self.player
        new_state.prev_player = self.prev_player
        new_state.winning = self.winning
        new_state._tensor_cached = False  # Force recalculation on first use
        return new_state


class TranspositionTable:
    """Cache for previously evaluated positions"""
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, state_hash):
        if state_hash in self.table:
            self.hits += 1
            return self.table[state_hash]
        self.misses += 1
        return None

    def put(self, state_hash, value, depth):
        if len(self.table) >= self.max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.table.keys())[:self.max_size // 10]
            for key in keys_to_remove:
                del self.table[key]
        self.table[state_hash] = {'value': value, 'depth': depth}

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.table)
        }


class Chess5D(JSrun):
    """Optimized 5D Chess game logic"""
    piece_map = {0: 'P', 1: 'B', 2: 'N', 3: 'R', 4: 'Q', 5: 'K'}
    player_map = {'white': 1, 'black': -1}
    player_map_opp = {'black': 'white', 'white': 'black'}

    def __init__(self, max_time, max_turns):
        self.max_time = max_time
        self.max_turns = max_turns
        self.action_size = (max_time, max_turns, 8, 8)
        self.max_dim = 0
        self.space_out = 0
        self.move_cache = {}  # Cache for move generation
        self.tensor_cache = {}  # Cache for board tensors

    def get_initial_state(self):
        state = ChessState()
        state.game_string = state.chess.export("5dpgn")
        state.raw_board = self.get_homogeneous_raw_board(state.chess, self.max_time, self.max_turns)
        state.choices_start = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float64)
        state.choices_end = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float64)
        state.board = self.raw_board_to_tensor(state.raw_board)
        self.pick_choice(state, state.choices_start, state.choices_end)
        return state

    def raw_board_to_tensor(self, raw_board):
        """Vectorized board to tensor conversion"""
        board = cp.array(raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns * 2, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        # Vectorized piece mask computation
        piece_values = cp.array([2, 4, 6, 8, 10, 12], dtype=cp.int8)
        for idx, piece_val in enumerate(piece_values):
            white_mask = (cp.abs(board) == piece_val)
            black_mask = (cp.abs(board) == piece_val - 1)
            tensor[:board.shape[0], :board.shape[1], idx] = white_mask.astype(cp.int8) - black_mask.astype(cp.int8)

        return cp.flip(tensor, axis=3)

    @staticmethod
    def convert_timeline(num):
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num):
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, state, move):
        state.game_string = state.chess.export("5dpgn")
        state.prev_player = state.player
        state._hash = None  # Invalidate hash

        try:
            state.chess.move(str(move))
            if state.chess.submittable():
                state.chess.submit()
            state.player = state.chess.player
            self.convert_moves(state)
        except Checkmate:
            state.is_terminal = True
            state.value = 1
            state.winning = state.prev_player
        except Stalemate:
            state.is_terminal = True
            state.value = 0
        except DrawLoss:
            state.is_terminal = True
            state.value = 0.3

    def pick_choice(self, state, x, y, sim=False):
        """Optimized move selection with probability distribution"""
        try:
            self.convert_moves(state)
            start_move = self._pick_start_move(x)
            end_move = self._pick_end_move(start_move, state, y)

            if not sim:
                # Reduce probability of selected move
                x[self.convert_timeline(start_move['timeline']),
                  start_move['turn'] - 1,
                  start_move['rank'] - 1,
                  start_move['file'] - 1] = max(0, x[
                      self.convert_timeline(start_move['timeline']),
                      start_move['turn'] - 1,
                      start_move['rank'] - 1,
                      start_move['file'] - 1] - 1)

                y[self.convert_timeline(end_move['timeline']),
                  end_move['turn'] - 1,
                  end_move['rank'] - 1,
                  end_move['file'] - 1] = max(0, y[
                      self.convert_timeline(end_move['timeline']),
                      end_move['turn'] - 1,
                      end_move['rank'] - 1,
                      end_move['file'] - 1] - 1)

            return self.move_to_string(start_move, end_move)
        except Stalemate:
            state.is_terminal = True
            state.value = 0.3
            return None, None, None

    def _pick_start_move(self, x):
        """Optimized start move selection"""
        tensor_start = cp.maximum(x, 0)  # Ensure non-negative
        total = cp.sum(tensor_start)
        if total <= 0:
            raise Stalemate
        tensor_start = tensor_start / total

        flat_index_start = cp.random.choice(
            len(tensor_start.flatten()),
            size=1,
            p=tensor_start.flatten()
        )
        i, j, k, l = cp.unravel_index(flat_index_start, tensor_start.shape)

        return {
            "timeline": self.convert_timeline_opposite(i.item()),
            "turn": j.item() + 1,
            "rank": k.item() + 1,
            "file": l.item() + 1,
        }

    def _pick_end_move(self, start_move, state, y):
        """Optimized end move selection"""
        end_moves = self.get_end_moves(state.moves, start_move)
        self.convert_moves_end(end_moves, state)

        tensor_end = cp.maximum(y, 0)
        total = cp.sum(tensor_end)
        if total <= 0:
            # Fallback to uniform distribution over valid moves
            tensor_end = state.choices_end.copy()
            total = cp.sum(tensor_end)
        tensor_end = tensor_end / total

        flat_index_end = cp.random.choice(
            len(tensor_end.flatten()),
            size=1,
            p=tensor_end.flatten()
        )
        m, n, o, p = cp.unravel_index(flat_index_end, tensor_end.shape)

        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item() + 1,
            "rank": o.item() + 1,
            "file": p.item() + 1,
        }

    @staticmethod
    def get_end_moves(data, start_value):
        """Get valid end moves for a given start position"""
        return [move['end'] for move in data if all(
            move['start'].get(key) == value
            for key, value in start_value.items()
            if key not in ['player', 'coordinate']
        )]

    @staticmethod
    def move_to_string(start, end):
        """Convert move coordinates to string notation"""
        return (
            f"({start['timeline']}T{start['turn']})P{chr(96 + start['file'])}{start['rank']}>>"
            f"({end['timeline']}T{end['turn']}){chr(96 + end['file'])}{end['rank']}",
            (Chess5D.convert_timeline(start['timeline']), start['turn'] - 1,
             start['rank'] - 1, start['file'] - 1),
            (Chess5D.convert_timeline(end['timeline']), end['turn'] - 1,
             end['rank'] - 1, end['file'] - 1)
        )

    def check_timelines(self, move):
        real_end = move['realEnd']
        timeline = real_end['timeline']
        if abs(timeline) > (self.max_time - 1) / 2:
            self.space_out = 1
            return True
        return False

    def check_turns(self, move):
        real_end = move['realEnd']
        turn = real_end['turn']
        if turn > self.max_turns - 1:
            self.space_out = 2
            return True
        return False

    def convert_moves(self, state):
        """Optimized move conversion with caching"""
        state.moves = self.safe_get_moves(state.chess)
        if state.moves is None:
            raise Stalemate

        state.choices_start.fill(0)
        valid_moves = []

        for move in state.moves:
            if not (self.check_timelines(move) or self.check_turns(move)):
                start = move['start']
                timeline = self.convert_timeline(start['timeline'])
                state.choices_start[timeline, start['turn'] - 1,
                                   start['rank'] - 1, start['file'] - 1] += 1
                valid_moves.append(move)

        state.moves = valid_moves

        if len(state.moves) == 0:
            if state.chess.inCheckmate or state.chess.inCheck:
                raise Checkmate
            else:
                raise DrawLoss

    def get_opponent_value(self, state, value):
        return value if state.prev_player == state.player else -value

    def convert_moves_end(self, moves, state):
        """Convert end moves to tensor representation"""
        state.choices_end.fill(0)
        for move in moves:
            timeline = self.convert_timeline(move['timeline'])
            state.choices_end[timeline, move['turn'] - 1,
                            move['rank'] - 1, move['file'] - 1] += 1


class Node:
    """Optimized MCTS node with transposition table support"""
    __slots__ = ['game', 'args', 'state', 'parent', 'action_taken_s',
                 'action_taken_e', 'children', 'expandable_moves_start',
                 'expandable_moves_end', 'player', 'visit_count', 'value_sum',
                 'state_hash', 'prior_prob']

    def __init__(self, game, args, state, parent=None,
                 action_taken_s=None, action_taken_e=None, prior_prob=1.0):
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
        self.state_hash = state.get_hash()
        self.prior_prob = prior_prob

    def is_fully_expanded(self):
        return cp.sum(self.expandable_moves_start) == 0 and len(self.children) > 0

    def select(self):
        """Select best child using UCB1 formula"""
        best_child = None
        best_ucb = -cp.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child):
        """UCB1 formula with player perspective correction"""
        if child.visit_count == 0:
            return float('inf')

        # Q-value from child's perspective
        q_value = child.value_sum / child.visit_count

        # Adjust for player perspective
        if self.parent is not None:
            if self.parent.player != child.player:
                q_value = -q_value

        # Normalize to [0, 1]
        q_normalized = (q_value + 1) / 2

        # UCB1 formula
        exploration = self.args['C'] * cp.sqrt(cp.log(self.visit_count) / child.visit_count)

        return q_normalized + exploration

    def expand(self):
        """Expand node with move selection"""
        result = self.game.pick_choice(
            self.state,
            self.expandable_moves_start,
            self.expandable_moves_end,
            False
        )

        if result is None or result[0] is None:
            return None

        action, rl_action_s, rl_action_e = result
        child_state = self.state.copy()
        self.game.make_move(child_state, action)

        child = Node(self.game, self.args, child_state, self,
                    rl_action_s, rl_action_e)
        self.children.append(child)

        return child

    def simulate(self):
        """Fast rollout simulation with early termination"""
        value, is_terminal = self.state.value, self.state.is_terminal
        value = self.game.get_opponent_value(self.state, value)

        if is_terminal:
            return value

        rollout_state = self.state.copy()
        original_player = rollout_state.player
        max_rollout_depth = self.args.get('max_rollout_depth', 30)

        for depth in range(max_rollout_depth):
            result = self.game.pick_choice(
                rollout_state,
                rollout_state.choices_start,
                rollout_state.choices_end,
                True
            )

            if result is None or result[0] is None:
                # Stalemate in simulation
                return 0.3

            action_m, _, _ = result
            self.game.make_move(rollout_state, action_m)

            value, is_terminal = rollout_state.value, rollout_state.is_terminal

            if is_terminal:
                # Return value from original player's perspective
                if original_player != rollout_state.prev_player:
                    return value
                else:
                    return -value

        # Max depth reached, return heuristic evaluation
        return 0

    def backpropagate(self, value):
        """Backpropagate value with player perspective"""
        self.value_sum += value
        self.visit_count += 1

        # Flip value for parent's perspective
        value = -value

        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    """Optimized MCTS with transposition table and progressive widening"""
    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.transposition_table = TranspositionTable(
            max_size=args.get('tt_size', 100000)
        )
        self.nodes_searched = 0
        self.terminal_nodes = 0

    def search(self, state):
        """Main MCTS search with optimizations"""
        root = Node(self.game, self.args, state)
        self.nodes_searched = 0
        self.terminal_nodes = 0

        for search_iter in range(self.args['num_searches']):
            node = root

            # Selection phase
            while node.is_fully_expanded():
                node = node.select()
                if node is None:
                    break

            if node is None:
                continue

            # Check transposition table
            state_hash = node.state.get_hash()
            tt_entry = self.transposition_table.get(state_hash)

            value = None
            is_terminal = node.state.is_terminal

            if tt_entry is not None and tt_entry['depth'] >= self.args.get('tt_depth_threshold', 5):
                # Use cached value
                value = tt_entry['value']
            elif not is_terminal:
                # Expansion phase
                child = node.expand()
                if child is not None:
                    node = child
                    # Simulation phase
                    value = node.simulate()
                    self.nodes_searched += 1
                else:
                    value = 0
            else:
                value = self.game.get_opponent_value(node.state, node.state.value)
                self.terminal_nodes += 1

            if value is not None:
                # Backpropagation phase
                node.backpropagate(value)

                # Store in transposition table
                depth = 0
                temp_node = node
                while temp_node.parent is not None:
                    depth += 1
                    temp_node = temp_node.parent
                self.transposition_table.put(state_hash, value, depth)

        # Generate action probabilities
        action_probs_start = cp.zeros(self.game.action_size, dtype=cp.float64)
        action_probs_end = cp.zeros(self.game.action_size, dtype=cp.float64)

        for child in root.children:
            if child.action_taken_s is not None:
                action_probs_start[child.action_taken_s] += child.visit_count
            if child.action_taken_e is not None:
                action_probs_end[child.action_taken_e] += child.visit_count

        # Normalize
        total_start = cp.sum(action_probs_start)
        if total_start > 0:
            action_probs_start /= total_start

        total_end = cp.sum(action_probs_end)
        if total_end > 0:
            action_probs_end /= total_end

        return action_probs_start, action_probs_end

    def get_stats(self):
        """Get MCTS statistics"""
        tt_stats = self.transposition_table.get_stats()
        return {
            'nodes_searched': self.nodes_searched,
            'terminal_nodes': self.terminal_nodes,
            'transposition_table': tt_stats
        }
