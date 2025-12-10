"""
5D Chess Game State Management
Handles state representation and copying for the chess engine.
"""
import copy
import cupy as cp
from javascript import require

Chess = require('5d-chess-js')


class GameException(Exception):
    """Base exception for game-related errors."""
    pass


class DrawLoss(GameException):
    """Exception raised when game ends in a draw."""
    pass


class Stalemate(GameException):
    """Exception raised when game reaches stalemate."""
    pass


class Checkmate(GameException):
    """Exception raised when checkmate occurs."""
    pass


class ChessState:
    """
    Represents the complete state of a 5D chess game.

    Attributes:
        chess: JavaScript Chess object instance
        value: Numeric value of the position
        piece: Current piece being moved
        game_string: String representation of game
        choices_start: Valid start positions tensor
        choices_end: Valid end positions tensor
        is_terminal: Whether game has ended
        moves: List of available moves
        raw_board: Raw board representation
        board: Tensor board representation
        player: Current player ('white' or 'black')
        prev_player: Previous player
        winning: Winning player
    """

    __slots__ = ['chess', 'value', 'piece', 'game_string', 'choices_start',
                 'choices_end', 'is_terminal', 'moves', 'raw_board', 'board',
                 'player', 'prev_player', 'winning']

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

    def __str__(self):
        return (f'ChessState('
                f'player={self.player}, '
                f'value={self.value}, '
                f'is_terminal={self.is_terminal}, '
                f'num_moves={len(self.moves) if self.moves else 0})')

    def __repr__(self):
        return self.__str__()

    def copy(self):
        """
        Create a deep copy of the chess state.

        Returns:
            ChessState: A new independent chess state
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
