"""5D Chess game engine implementation."""

from typing import Tuple, List, Dict, Any, Optional
import cupy as cp
from jsrun import JSrun

from src.chess_state import ChessState
from src.exceptions import Checkmate, Stalemate, DrawLoss


class Chess5D(JSrun):
    """
    5D Chess game engine with multi-timeline and multi-turn support.

    Attributes:
        piece_map: Mapping of piece indices to piece symbols
        player_map: Mapping of player names to numeric values
        player_map_opp: Mapping of players to their opponents
    """

    piece_map = {0: 'P', 1: 'B', 2: 'N', 3: 'R', 4: 'Q', 5: 'K'}
    player_map = {'white': 1, 'black': -1}
    player_map_opp = {'black': 'white', 'white': 'black'}

    def __init__(self, max_time: int, max_turns: int):
        """
        Initialize the 5D Chess engine.

        Args:
            max_time: Maximum number of timelines
            max_turns: Maximum number of turns per timeline
        """
        self.max_time = max_time
        self.max_turns = max_turns
        self.action_size = (max_time, max_turns, 8, 8)
        self.max_dim = 0
        self.space_out = 0

    def get_initial_state(self) -> ChessState:
        """
        Create and initialize a new game state.

        Returns:
            ChessState: Initialized game state ready for play
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
        self.pick_choice(state, state.choices_start, state.choices_end)
        return state

    def raw_board_to_tensor(self, raw_board: cp.ndarray) -> cp.ndarray:
        """
        Convert raw board representation to tensor format.

        Args:
            raw_board: Raw board array

        Returns:
            cp.ndarray: Tensor representation of the board
        """
        board = cp.array(raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns * 2, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        piece_masks = [
            (cp.abs(board) == 2, cp.abs(board) == 1, 0),
            (cp.abs(board) == 4, cp.abs(board) == 3, 1),
            (cp.abs(board) == 6, cp.abs(board) == 5, 2),
            (cp.abs(board) == 8, cp.abs(board) == 7, 3),
            (cp.abs(board) == 10, cp.abs(board) == 9, 4),
            (cp.abs(board) == 12, cp.abs(board) == 11, 5)
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = (
                white_mask.astype(cp.int8) - black_mask.astype(cp.int8)
            )

        return cp.flip(tensor, axis=3)

    @staticmethod
    def convert_timeline(num: int) -> int:
        """
        Convert timeline number to internal representation.

        Args:
            num: Timeline number

        Returns:
            int: Converted timeline number
        """
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num: int) -> int:
        """
        Convert internal timeline representation back to timeline number.

        Args:
            num: Internal timeline number

        Returns:
            int: Original timeline number
        """
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, state: ChessState, move: str) -> None:
        """
        Execute a move on the game state.

        Args:
            state: Current game state (modified in place)
            move: Move string to execute

        Raises:
            Checkmate: If move results in checkmate
            Stalemate: If move results in stalemate
            DrawLoss: If move results in draw/loss
        """
        state.game_string = state.chess.export("5dpgn")
        state.prev_player = state.player

        try:
            state.chess.move(str(move))
            if state.chess.submittable():
                state.chess.submit()
            state.player = state.chess.player
            self.convert_moves(state)
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

    def pick_choice(
        self,
        state: ChessState,
        x: cp.ndarray,
        y: cp.ndarray,
        sim: bool = False
    ) -> Optional[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
        """
        Pick a move based on probability distributions.

        Args:
            state: Current game state
            x: Start position probability distribution
            y: End position probability distribution
            sim: Whether this is a simulation (no probability updates)

        Returns:
            Optional[Tuple]: Move string and position tuples, or None if stalemate
        """
        try:
            self.convert_moves(state)
            start_move = self._pick_start_move(x)
            end_move = self._pick_end_move(start_move, state, y)

            if not sim:
                x[
                    self.convert_timeline(start_move['timeline']),
                    start_move['turn'] - 1,
                    start_move['rank'] - 1,
                    start_move['file'] - 1
                ] -= 1

                y[
                    self.convert_timeline(end_move['timeline']),
                    end_move['turn'] - 1,
                    end_move['rank'] - 1,
                    end_move['file'] - 1
                ] -= 1

            return self.move_to_string(start_move, end_move)
        except Stalemate:
            state.is_terminal = True
            state.value = 0.3
            return None

    def _pick_start_move(self, x: cp.ndarray) -> Dict[str, int]:
        """
        Select a starting position from probability distribution.

        Args:
            x: Start position probability distribution

        Returns:
            Dict[str, int]: Starting position coordinates
        """
        tensor_start = x.copy()
        tensor_start /= cp.sum(tensor_start)
        flat_index_start = cp.random.choice(
            len(tensor_start.flatten()), size=1, p=tensor_start.flatten()
        )
        i, j, k, l = cp.unravel_index(flat_index_start, tensor_start.shape)

        return {
            "timeline": self.convert_timeline_opposite(i.item()),
            "turn": j.item() + 1,
            "rank": k.item() + 1,
            "file": l.item() + 1,
        }

    def _pick_end_move(
        self,
        start_move: Dict[str, int],
        state: ChessState,
        y: cp.ndarray
    ) -> Dict[str, int]:
        """
        Select an ending position from probability distribution.

        Args:
            start_move: Starting position
            state: Current game state
            y: End position probability distribution

        Returns:
            Dict[str, int]: Ending position coordinates
        """
        end_moves = self.get_end_moves(state.moves, start_move)
        self.convert_moves_end(end_moves, state)

        tensor_end = y.copy()
        tensor_end /= cp.sum(tensor_end)
        flat_index_end = cp.random.choice(
            len(tensor_end.flatten()), size=1, p=tensor_end.flatten()
        )
        m, n, o, p = cp.unravel_index(flat_index_end, tensor_end.shape)

        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item() + 1,
            "rank": o.item() + 1,
            "file": p.item() + 1,
        }

    @staticmethod
    def get_end_moves(
        data: List[Dict[str, Any]],
        start_value: Dict[str, int]
    ) -> List[Dict[str, int]]:
        """
        Get all valid end positions for a given start position.

        Args:
            data: List of all available moves
            start_value: Starting position to filter by

        Returns:
            List[Dict[str, int]]: Valid ending positions
        """
        return [
            move['end'] for move in data
            if all(
                move['start'].get(key) == value
                for key, value in start_value.items()
                if key not in ['player', 'coordinate']
            )
        ]

    @staticmethod
    def move_to_string(
        start: Dict[str, int],
        end: Dict[str, int]
    ) -> Tuple[str, Tuple[int, ...], Tuple[int, ...]]:
        """
        Convert move coordinates to string notation.

        Args:
            start: Starting position
            end: Ending position

        Returns:
            Tuple: Move string and position tuples
        """
        move_str = (
            f"({start['timeline']}T{start['turn']})P"
            f"{chr(96 + start['file'])}{start['rank']}>>"
            f"({end['timeline']}T{end['turn']})"
            f"{chr(96 + end['file'])}{end['rank']}"
        )
        start_tuple = (
            Chess5D.convert_timeline(start['timeline']),
            start['turn'] - 1,
            start['rank'] - 1,
            start['file'] - 1
        )
        end_tuple = (
            Chess5D.convert_timeline(end['timeline']),
            end['turn'] - 1,
            end['rank'] - 1,
            end['file'] - 1
        )
        return move_str, start_tuple, end_tuple

    def check_timelines(self, move: Dict[str, Any]) -> bool:
        """
        Check if move exceeds timeline limits.

        Args:
            move: Move to check

        Returns:
            bool: True if timeline limit exceeded
        """
        real_end = move['realEnd']
        timeline = real_end['timeline']
        if abs(timeline) > (self.max_time - 1) / 2:
            self.space_out = 1
            return True
        return False

    def check_turns(self, move: Dict[str, Any]) -> bool:
        """
        Check if move exceeds turn limits.

        Args:
            move: Move to check

        Returns:
            bool: True if turn limit exceeded
        """
        real_end = move['realEnd']
        turn = real_end['turn']
        if turn > self.max_turns - 1:
            self.space_out = 2
            return True
        return False

    def convert_moves(self, state: ChessState) -> None:
        """
        Convert available moves to choice tensors.

        Args:
            state: Current game state (modified in place)

        Raises:
            Stalemate: If no moves available
            Checkmate: If in checkmate
            DrawLoss: If exceeded timelines
        """
        state.moves = self.safe_get_moves(state.chess)

        if state.moves is None:
            raise Stalemate

        state.choices_start.fill(0)

        for i in range(len(state.moves) - 1, -1, -1):
            move = state.moves[i]
            if self.check_timelines(move) or self.check_turns(move):
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

        if len(state.moves) == 0 and (state.chess.inCheckmate or state.chess.inCheck):
            print(f"checkmate: {Chess5D.player_map[state.chess.player] * -1}")
            raise Checkmate
        elif len(state.moves) == 0:
            print(f"Exceeded Timeline (Draw_loss): {state.chess.player}")
            raise DrawLoss

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

    def convert_moves_end(self, moves: List[Dict[str, int]], state: ChessState) -> None:
        """
        Convert ending moves to choice tensor.

        Args:
            moves: List of ending positions
            state: Current game state (modified in place)
        """
        state.choices_end.fill(0)
        for move in moves:
            timeline = self.convert_timeline(move['timeline'])
            state.choices_end[
                timeline,
                move['turn'] - 1,
                move['rank'] - 1,
                move['file'] - 1
            ] += 1
