"""
5D Chess Engine
Core game logic for 5D chess including move generation and board representation.
"""
import cupy as cp
import numpy as np
from typing import List, Dict, Tuple, Optional
from .game_state import ChessState, Checkmate, Stalemate, DrawLoss
from ..utils.js_interface import JSInterface


class Chess5DEngine(JSInterface):
    """
    5D Chess game engine with GPU-accelerated board representation.

    Attributes:
        piece_map: Mapping from piece indices to piece names
        player_map: Mapping from player names to numeric values
        player_map_opp: Mapping from player to opponent
        max_time: Maximum number of timelines
        max_turns: Maximum number of turns per timeline
        action_size: Shape of action space (timelines, turns, rank, file)
    """

    piece_map = {0: 'P', 1: 'B', 2: 'N', 3: 'R', 4: 'Q', 5: 'K'}
    player_map = {'white': 1, 'black': -1}
    player_map_opp = {'black': 'white', 'white': 'black'}

    def __init__(self, max_time: int = 11, max_turns: int = 30):
        """
        Initialize the 5D Chess engine.

        Args:
            max_time: Maximum number of timelines to support
            max_turns: Maximum number of turns per timeline
        """
        super().__init__()
        self.max_time = max_time
        self.max_turns = max_turns
        self.action_size = (max_time, max_turns, 8, 8)
        self.max_dim = 0
        self.space_out = 0

    def get_initial_state(self) -> ChessState:
        """
        Create and return the initial game state.

        Returns:
            ChessState: Initial state of the game
        """
        state = ChessState()
        state.game_string = state.chess.export("5dpgn")
        state.raw_board = self.get_homogeneous_raw_board(
            state.chess, self.max_time, self.max_turns
        )
        state.choices_start = cp.zeros(
            (self.max_time, self.max_turns, 8, 8), dtype=cp.float64
        )
        state.choices_end = cp.zeros(
            (self.max_time, self.max_turns, 8, 8), dtype=cp.float64
        )
        state.board = self.raw_board_to_tensor(state.raw_board)
        self._update_legal_moves(state)
        return state

    def raw_board_to_tensor(self, raw_board: cp.ndarray) -> cp.ndarray:
        """
        Convert raw board representation to tensor format.

        Args:
            raw_board: Raw board array

        Returns:
            cp.ndarray: Tensor representation of board
        """
        board = cp.array(raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns * 2, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        # Map pieces to their tensor indices
        piece_masks = [
            (cp.abs(board) == 2, cp.abs(board) == 1, 0),  # Pawns
            (cp.abs(board) == 4, cp.abs(board) == 3, 1),  # Bishops
            (cp.abs(board) == 6, cp.abs(board) == 5, 2),  # Knights
            (cp.abs(board) == 8, cp.abs(board) == 7, 3),  # Rooks
            (cp.abs(board) == 10, cp.abs(board) == 9, 4), # Queens
            (cp.abs(board) == 12, cp.abs(board) == 11, 5) # Kings
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = (
                white_mask.astype(cp.int8) - black_mask.astype(cp.int8)
            )

        return cp.flip(tensor, axis=3)

    @staticmethod
    def convert_timeline(num: int) -> int:
        """Convert timeline number to array index."""
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num: int) -> int:
        """Convert array index to timeline number."""
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, state: ChessState, move: str) -> None:
        """
        Execute a move on the given state.

        Args:
            state: Current game state
            move: Move string to execute

        Raises:
            Checkmate: If move results in checkmate
            Stalemate: If move results in stalemate
            DrawLoss: If move results in draw
        """
        state.game_string = state.chess.export("5dpgn")
        state.prev_player = state.player

        try:
            state.chess.move(str(move))
            if state.chess.submittable():
                state.chess.submit()
            state.player = state.chess.player
            self._update_legal_moves(state)

        except Checkmate:
            state.is_terminal = True
            state.value = 1
            raise
        except Stalemate:
            state.is_terminal = True
            state.value = 0
            raise
        except DrawLoss:
            state.is_terminal = True
            state.value = 0.3
            raise

    def _update_legal_moves(self, state: ChessState) -> None:
        """
        Update the legal moves for the current state.

        Args:
            state: Current game state

        Raises:
            Checkmate: If no moves and in checkmate
            DrawLoss: If no moves and exceeded timeline
            Stalemate: If no legal moves available
        """
        state.moves = self.safe_get_moves(state.chess)

        if state.moves is None:
            raise Stalemate

        state.choices_start.fill(0)

        # Filter moves that exceed board boundaries
        for i in range(len(state.moves) - 1, -1, -1):
            move = state.moves[i]
            if self._check_timelines(move) or self._check_turns(move):
                state.moves.pop(i)
            else:
                start = move['start']
                timeline = self.convert_timeline(start['timeline'])
                state.choices_start[
                    timeline,
                    start['turn'] - 1,
                    start['rank'] - 1,
                    start['file'] - 1
                ] += 1

        if len(state.moves) == 0:
            if state.chess.inCheckmate or state.chess.inCheck:
                print(f"Checkmate: {self.player_map[state.chess.player] * -1}")
                raise Checkmate
            else:
                print(f"Exceeded Timeline (Draw/Loss): {state.chess.player}")
                raise DrawLoss

    def _check_timelines(self, move: Dict) -> bool:
        """Check if move exceeds timeline boundaries."""
        real_end = move['realEnd']
        timeline = real_end['timeline']
        if abs(timeline) > (self.max_time - 1) / 2:
            self.space_out = 1
            return True
        return False

    def _check_turns(self, move: Dict) -> bool:
        """Check if move exceeds turn boundaries."""
        real_end = move['realEnd']
        turn = real_end['turn']
        if turn > self.max_turns - 1:
            self.space_out = 2
            return True
        return False

    def get_opponent_value(self, state: ChessState, value: float) -> float:
        """
        Get value from opponent's perspective.

        Args:
            state: Current game state
            value: Value to convert

        Returns:
            float: Value from opponent's perspective
        """
        return value if state.prev_player == state.player else -value

    def pick_random_move(
        self,
        state: ChessState,
        start_probs: cp.ndarray,
        end_probs: cp.ndarray,
        modify_probs: bool = False
    ) -> Tuple[str, Tuple, Tuple]:
        """
        Pick a random move based on probability distributions.

        Args:
            state: Current game state
            start_probs: Probability distribution over start positions
            end_probs: Probability distribution over end positions
            modify_probs: Whether to modify probability tensors

        Returns:
            Tuple containing move string and position tuples
        """
        try:
            self._update_legal_moves(state)
            start_move = self._pick_start_move(start_probs)
            end_move = self._pick_end_move(start_move, state, end_probs)

            if modify_probs:
                start_probs[
                    self.convert_timeline(start_move['timeline']),
                    start_move['turn'] - 1,
                    start_move['rank'] - 1,
                    start_move['file'] - 1
                ] -= 1

                end_probs[
                    self.convert_timeline(end_move['timeline']),
                    end_move['turn'] - 1,
                    end_move['rank'] - 1,
                    end_move['file'] - 1
                ] -= 1

            return self._move_to_string(start_move, end_move)

        except Stalemate:
            state.is_terminal = True
            state.value = 0.3
            raise

    def _pick_start_move(self, probs: cp.ndarray) -> Dict:
        """Pick starting position from probability distribution."""
        tensor_start = probs.copy()
        tensor_start /= cp.sum(tensor_start)
        flat_index = cp.random.choice(
            len(tensor_start.flatten()),
            size=1,
            p=tensor_start.flatten()
        )
        i, j, k, l = cp.unravel_index(flat_index, tensor_start.shape)

        return {
            "timeline": self.convert_timeline_opposite(i.item()),
            "turn": j.item() + 1,
            "rank": k.item() + 1,
            "file": l.item() + 1,
        }

    def _pick_end_move(
        self,
        start_move: Dict,
        state: ChessState,
        probs: cp.ndarray
    ) -> Dict:
        """Pick ending position from probability distribution."""
        end_moves = self._get_end_moves(state.moves, start_move)
        self._update_end_move_probs(end_moves, state)

        tensor_end = probs.copy()
        tensor_end /= cp.sum(tensor_end)
        flat_index = cp.random.choice(
            len(tensor_end.flatten()),
            size=1,
            p=tensor_end.flatten()
        )
        m, n, o, p = cp.unravel_index(flat_index, tensor_end.shape)

        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item() + 1,
            "rank": o.item() + 1,
            "file": p.item() + 1,
        }

    @staticmethod
    def _get_end_moves(moves: List[Dict], start_value: Dict) -> List[Dict]:
        """Get all possible end positions for a given start position."""
        return [
            move['end'] for move in moves
            if all(
                move['start'].get(key) == value
                for key, value in start_value.items()
                if key not in ['player', 'coordinate']
            )
        ]

    def _update_end_move_probs(self, moves: List[Dict], state: ChessState) -> None:
        """Update probability tensor for end moves."""
        state.choices_end.fill(0)
        for move in moves:
            timeline = self.convert_timeline(move['timeline'])
            state.choices_end[
                timeline,
                move['turn'] - 1,
                move['rank'] - 1,
                move['file'] - 1
            ] += 1

    @staticmethod
    def _move_to_string(start: Dict, end: Dict) -> Tuple[str, Tuple, Tuple]:
        """Convert move dictionaries to string representation."""
        move_str = (
            f"({start['timeline']}T{start['turn']})"
            f"{chr(96 + start['file'])}{start['rank']}>>"
            f"({end['timeline']}T{end['turn']})"
            f"{chr(96 + end['file'])}{end['rank']}"
        )

        start_tuple = (
            Chess5DEngine.convert_timeline(start['timeline']),
            start['turn'] - 1,
            start['rank'] - 1,
            start['file'] - 1
        )

        end_tuple = (
            Chess5DEngine.convert_timeline(end['timeline']),
            end['turn'] - 1,
            end['rank'] - 1,
            end['file'] - 1
        )

        return move_str, start_tuple, end_tuple
