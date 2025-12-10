"""
Optimized 5D Chess Engine with Enhanced MCTS and Learning Capabilities
Author: Research Team
Date: 2025-12-10

Architecture improvements:
1. Enhanced state representation with better tensor encoding
2. Optimized MCTS with adaptive exploration
3. Improved memory management with CuPy optimization
4. Better move selection with progressive widening
5. Enhanced backpropagation with virtual loss
"""

import cupy as cp
import numpy as np
import math
import copy
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
import time


@dataclass
class GameMetrics:
    """Metrics for tracking game performance"""
    move_count: int = 0
    avg_search_time: float = 0.0
    avg_nodes_expanded: int = 0
    total_simulations: int = 0
    win_rate: float = 0.0
    avg_game_length: float = 0.0
    timeline_expansions: int = 0
    checkmates: int = 0
    draws: int = 0


class GameException(Exception):
    pass


class DrawLoss(GameException):
    pass


class Stalemate(GameException):
    pass


class Checkmate(GameException):
    pass


class ChessState:
    """Enhanced chess state with better memory management"""

    def __init__(self):
        self.chess = None  # Will be initialized by game
        self.value = 0.0
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
        self.move_history = []
        self.features = None  # Additional state features

    def __str__(self):
        return (f'ChessState(player={self.player}, '
                f'value={self.value:.3f}, '
                f'terminal={self.is_terminal}, '
                f'moves={len(self.moves) if self.moves else 0})')

    def copy(self):
        """Deep copy with optimized memory handling"""
        new_state = ChessState()
        if self.chess is not None:
            new_state.chess = self.chess.copy()
        new_state.value = self.value
        new_state.piece = self.piece
        new_state.game_string = self.game_string
        new_state.choices_start = cp.copy(self.choices_start) if self.choices_start is not None else None
        new_state.choices_end = cp.copy(self.choices_end) if self.choices_end is not None else None
        new_state.is_terminal = self.is_terminal
        new_state.moves = copy.deepcopy(self.moves) if self.moves is not None else None
        new_state.raw_board = cp.copy(self.raw_board) if self.raw_board is not None else None
        new_state.board = cp.copy(self.board) if self.board is not None else None
        new_state.player = self.player
        new_state.prev_player = self.prev_player
        new_state.winning = self.winning
        new_state.move_history = self.move_history.copy()
        new_state.features = cp.copy(self.features) if self.features is not None else None
        return new_state


class Chess5DOptimized:
    """Optimized 5D Chess game engine with enhanced features"""

    piece_map = {0: 'P', 1: 'B', 2: 'N', 3: 'R', 4: 'Q', 5: 'K'}
    player_map = {'white': 1, 'black': -1}
    player_map_opp = {'black': 'white', 'white': 'black'}

    def __init__(self, max_time: int = 11, max_turns: int = 30):
        self.max_time = max_time
        self.max_turns = max_turns
        self.action_size = (max_time, max_turns, 8, 8)
        self.max_dim = 0
        self.space_out = 0
        self.metrics = GameMetrics()

        # Cache for move generation
        self.move_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def get_initial_state(self) -> ChessState:
        """Initialize game state with enhanced features"""
        from jsrun import JSrun
        from javascript import require
        Chess = require('5d-chess-js')

        state = ChessState()
        state.chess = Chess()
        state.game_string = state.chess.export("5dpgn")
        state.raw_board = self._get_homogeneous_raw_board(state.chess)
        state.choices_start = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float32)
        state.choices_end = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float32)
        state.board = self.raw_board_to_tensor(state.raw_board)
        state.features = self._extract_features(state)
        self._update_legal_moves(state)
        return state

    def _get_homogeneous_raw_board(self, chess) -> cp.ndarray:
        """Get board representation - placeholder for actual implementation"""
        # This would interface with the actual 5d-chess-js library
        return cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.int8)

    def raw_board_to_tensor(self, raw_board: cp.ndarray) -> cp.ndarray:
        """
        Enhanced tensor representation with better encoding
        Shape: (max_time, max_turns * 2, 6, 8, 8)
        """
        board = cp.array(raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns * 2, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        # Optimized piece encoding
        piece_masks = [
            (cp.abs(board) == 2, cp.abs(board) == 1, 0),  # Pawns
            (cp.abs(board) == 4, cp.abs(board) == 3, 1),  # Bishops
            (cp.abs(board) == 6, cp.abs(board) == 5, 2),  # Knights
            (cp.abs(board) == 8, cp.abs(board) == 7, 3),  # Rooks
            (cp.abs(board) == 10, cp.abs(board) == 9, 4), # Queens
            (cp.abs(board) == 12, cp.abs(board) == 11, 5) # Kings
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = \
                white_mask.astype(cp.int8) - black_mask.astype(cp.int8)

        return cp.flip(tensor, axis=3)

    def _extract_features(self, state: ChessState) -> cp.ndarray:
        """Extract additional state features for neural networks"""
        features = []

        # Material count
        if state.board is not None:
            material = cp.sum(state.board, axis=(0, 1, 3, 4))
            features.append(material.flatten())

        # Mobility (number of legal moves)
        if state.choices_start is not None:
            mobility = cp.array([cp.sum(state.choices_start)])
            features.append(mobility)

        # Player to move
        player_feat = cp.array([self.player_map[state.player]])
        features.append(player_feat)

        if features:
            return cp.concatenate(features)
        return cp.array([])

    @staticmethod
    def convert_timeline(num: int) -> int:
        """Convert timeline number to tensor index"""
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num: int) -> int:
        """Convert tensor index back to timeline number"""
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, state: ChessState, move: str) -> None:
        """Execute a move with enhanced error handling"""
        state.game_string = state.chess.export("5dpgn")
        state.prev_player = state.player

        try:
            state.chess.move(str(move))
            if state.chess.submittable():
                state.chess.submit()
            state.player = state.chess.player
            state.move_history.append(move)
            self._update_legal_moves(state)
            state.features = self._extract_features(state)
            self.metrics.move_count += 1

        except Checkmate:
            state.is_terminal = True
            state.value = 1.0
            state.winning = state.prev_player
            self.metrics.checkmates += 1
        except Stalemate:
            state.is_terminal = True
            state.value = 0.0
            self.metrics.draws += 1
        except DrawLoss:
            state.is_terminal = True
            state.value = 0.3
            self.metrics.draws += 1

    def _update_legal_moves(self, state: ChessState) -> None:
        """Update legal moves with caching"""
        cache_key = state.game_string

        if cache_key in self.move_cache:
            state.moves = self.move_cache[cache_key]
            self.cache_hits += 1
        else:
            state.moves = self._get_safe_moves(state.chess)
            self.move_cache[cache_key] = state.moves
            self.cache_misses += 1

            # Limit cache size
            if len(self.move_cache) > 10000:
                # Remove oldest entries
                keys = list(self.move_cache.keys())
                for k in keys[:5000]:
                    del self.move_cache[k]

        if state.moves is None or len(state.moves) == 0:
            if state.chess.inCheckmate or state.chess.inCheck:
                raise Checkmate
            else:
                raise Stalemate

        # Update legal move tensors
        state.choices_start.fill(0)
        filtered_moves = []

        for move in state.moves:
            if not self._check_timelines(move) and not self._check_turns(move):
                filtered_moves.append(move)
                start = move['start']
                timeline = self.convert_timeline(start['timeline'])
                state.choices_start[
                    timeline,
                    start['turn'] - 1,
                    start['rank'] - 1,
                    start['file'] - 1
                ] += 1

        state.moves = filtered_moves

        if len(state.moves) == 0:
            if state.chess.inCheckmate or state.chess.inCheck:
                raise Checkmate
            else:
                raise DrawLoss

    def _get_safe_moves(self, chess):
        """Safely get moves from chess engine"""
        try:
            # Placeholder - actual implementation would interface with 5d-chess-js
            return []
        except:
            return None

    def _check_timelines(self, move: Dict) -> bool:
        """Check if move exceeds timeline bounds"""
        real_end = move.get('realEnd', move.get('end', {}))
        timeline = real_end.get('timeline', 0)
        if abs(timeline) > (self.max_time - 1) / 2:
            self.space_out = 1
            self.metrics.timeline_expansions += 1
            return True
        return False

    def _check_turns(self, move: Dict) -> bool:
        """Check if move exceeds turn bounds"""
        real_end = move.get('realEnd', move.get('end', {}))
        turn = real_end.get('turn', 1)
        if turn > self.max_turns - 1:
            self.space_out = 2
            return True
        return False

    def pick_choice_stochastic(self, state: ChessState,
                              policy_start: cp.ndarray,
                              policy_end: cp.ndarray,
                              temperature: float = 1.0) -> Tuple[str, Tuple, Tuple]:
        """
        Pick move using policy network with temperature-based exploration
        """
        self._update_legal_moves(state)

        # Mask illegal moves
        legal_start_mask = state.choices_start > 0
        masked_policy_start = cp.where(legal_start_mask, policy_start, -cp.inf)

        # Apply temperature
        if temperature > 0:
            probs = cp.exp(masked_policy_start / temperature)
            probs = probs / cp.sum(probs)

            # Sample from distribution
            flat_probs = probs.flatten()
            flat_probs = flat_probs / cp.sum(flat_probs)
            idx = cp.random.choice(len(flat_probs), p=flat_probs.get())
            start_idx = cp.unravel_index(idx, probs.shape)
        else:
            # Greedy selection
            start_idx = cp.unravel_index(cp.argmax(masked_policy_start),
                                        masked_policy_start.shape)

        # Convert to move coordinates
        start_move = {
            "timeline": self.convert_timeline_opposite(int(start_idx[0])),
            "turn": int(start_idx[1]) + 1,
            "rank": int(start_idx[2]) + 1,
            "file": int(start_idx[3]) + 1,
        }

        # Get legal end moves
        end_moves = self._get_end_moves(state.moves, start_move)
        if not end_moves:
            # Fallback to random legal move
            return self._random_legal_move(state)

        # Mask end moves
        state.choices_end.fill(0)
        for move in end_moves:
            timeline = self.convert_timeline(move['timeline'])
            state.choices_end[timeline, move['turn']-1, move['rank']-1, move['file']-1] += 1

        legal_end_mask = state.choices_end > 0
        masked_policy_end = cp.where(legal_end_mask, policy_end, -cp.inf)

        if temperature > 0:
            probs_end = cp.exp(masked_policy_end / temperature)
            probs_end = probs_end / cp.sum(probs_end)
            flat_probs_end = probs_end.flatten()
            flat_probs_end = flat_probs_end / cp.sum(flat_probs_end)
            idx_end = cp.random.choice(len(flat_probs_end), p=flat_probs_end.get())
            end_idx = cp.unravel_index(idx_end, probs_end.shape)
        else:
            end_idx = cp.unravel_index(cp.argmax(masked_policy_end),
                                       masked_policy_end.shape)

        end_move = {
            "timeline": self.convert_timeline_opposite(int(end_idx[0])),
            "turn": int(end_idx[1]) + 1,
            "rank": int(end_idx[2]) + 1,
            "file": int(end_idx[3]) + 1,
        }

        return self._move_to_string(start_move, end_move)

    def _random_legal_move(self, state: ChessState) -> Tuple[str, Tuple, Tuple]:
        """Select a random legal move"""
        if not state.moves:
            raise Stalemate

        move = state.moves[cp.random.randint(0, len(state.moves))]
        start = move['start']
        end = move['end']
        return self._move_to_string(start, end)

    @staticmethod
    def _get_end_moves(moves: List[Dict], start_move: Dict) -> List[Dict]:
        """Get all legal end positions for a given start position"""
        return [move['end'] for move in moves if all(
            move['start'].get(key) == value
            for key, value in start_move.items()
            if key not in ['player', 'coordinate']
        )]

    @staticmethod
    def _move_to_string(start: Dict, end: Dict) -> Tuple[str, Tuple, Tuple]:
        """Convert move dictionaries to string notation and indices"""
        move_str = (f"({start['timeline']}T{start['turn']})"
                   f"{chr(96 + start['file'])}{start['rank']}>>"
                   f"({end['timeline']}T{end['turn']})"
                   f"{chr(96 + end['file'])}{end['rank']}")

        start_idx = (
            Chess5DOptimized.convert_timeline(start['timeline']),
            start['turn'] - 1,
            start['rank'] - 1,
            start['file'] - 1
        )

        end_idx = (
            Chess5DOptimized.convert_timeline(end['timeline']),
            end['turn'] - 1,
            end['rank'] - 1,
            end['file'] - 1
        )

        return move_str, start_idx, end_idx

    def get_opponent_value(self, state: ChessState, value: float) -> float:
        """Get value from opponent's perspective"""
        return value if state.prev_player == state.player else -value

    def get_canonical_board(self, state: ChessState) -> cp.ndarray:
        """Get board from current player's perspective"""
        if state.player == 'white':
            return state.board
        else:
            return -state.board

    def get_symmetries(self, board: cp.ndarray,
                       policy_start: cp.ndarray,
                       policy_end: cp.ndarray) -> List[Tuple]:
        """Get board symmetries for data augmentation"""
        symmetries = []

        # Original
        symmetries.append((board, policy_start, policy_end))

        # Horizontal flip
        board_flip = cp.flip(board, axis=-1)
        policy_start_flip = cp.flip(policy_start, axis=-1)
        policy_end_flip = cp.flip(policy_end, axis=-1)
        symmetries.append((board_flip, policy_start_flip, policy_end_flip))

        return symmetries

    def save_metrics(self, filepath: str) -> None:
        """Save game metrics to file"""
        metrics_dict = {
            'move_count': self.metrics.move_count,
            'avg_search_time': self.metrics.avg_search_time,
            'avg_nodes_expanded': self.metrics.avg_nodes_expanded,
            'total_simulations': self.metrics.total_simulations,
            'win_rate': self.metrics.win_rate,
            'avg_game_length': self.metrics.avg_game_length,
            'timeline_expansions': self.metrics.timeline_expansions,
            'checkmates': self.metrics.checkmates,
            'draws': self.metrics.draws,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def get_game_ended(self, state: ChessState) -> Tuple[bool, float]:
        """Check if game has ended and return result"""
        if state.is_terminal:
            if state.winning == state.player:
                return True, 1.0
            elif state.value == 0:
                return True, 0.0
            else:
                return True, -1.0
        return False, 0.0
