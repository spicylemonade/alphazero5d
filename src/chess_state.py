"""Chess state management for 5D Chess."""

from typing import Optional, List, Dict, Any
import copy
import cupy as cp
from javascript import require

Chess = require('5d-chess-js')


class ChessState:
    """Represents the state of a 5D chess game at a given point."""

    def __init__(self):
        """Initialize a new chess state."""
        self.chess = Chess()
        self.value: float = 0.0
        self.piece: Optional[str] = None
        self.game_string: Optional[str] = None
        self.choices_start: Optional[cp.ndarray] = None
        self.choices_end: Optional[cp.ndarray] = None
        self.is_terminal: bool = False
        self.moves: Optional[List[Dict[str, Any]]] = None
        self.raw_board: Optional[cp.ndarray] = None
        self.board: Optional[cp.ndarray] = None
        self.player: str = 'white'
        self.prev_player: Optional[str] = None
        self.winning: str = 'white'

    def __str__(self) -> str:
        """Return string representation of chess state."""
        return (
            f'ChessState('
            f'player={self.player}, '
            f'value={self.value}, '
            f'is_terminal={self.is_terminal}, '
            f'moves_count={len(self.moves) if self.moves else 0})'
        )

    def __repr__(self) -> str:
        """Return detailed representation of chess state."""
        return self.__str__()

    def copy(self) -> 'ChessState':
        """
        Create a deep copy of the chess state.

        Returns:
            ChessState: A new independent copy of the state
        """
        new_state = ChessState()
        new_state.chess = self.chess.copy()
        new_state.value = copy.copy(self.value)
        new_state.piece = copy.copy(self.piece)
        new_state.game_string = copy.copy(self.game_string)
        new_state.choices_start = cp.copy(self.choices_start) if self.choices_start is not None else None
        new_state.choices_end = cp.copy(self.choices_end) if self.choices_end is not None else None
        new_state.is_terminal = self.is_terminal
        new_state.moves = copy.deepcopy(self.moves) if self.moves is not None else None
        new_state.raw_board = cp.copy(self.raw_board) if self.raw_board is not None else None
        new_state.board = cp.copy(self.board) if self.board is not None else None
        new_state.player = copy.copy(self.player)
        new_state.prev_player = copy.copy(self.prev_player)
        new_state.winning = copy.copy(self.winning)
        return new_state

    def is_game_over(self) -> bool:
        """
        Check if the game is over.

        Returns:
            bool: True if game is terminal, False otherwise
        """
        return self.is_terminal

    def get_winner(self) -> Optional[str]:
        """
        Get the winner of the game if it's over.

        Returns:
            Optional[str]: 'white', 'black', or None if not terminal or draw
        """
        if not self.is_terminal:
            return None
        if self.value > 0:
            return self.winning
        elif self.value < 0:
            return 'black' if self.winning == 'white' else 'white'
        return None  # Draw
