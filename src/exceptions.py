"""Custom exceptions for 5D Chess game."""


class GameException(Exception):
    """Base exception for all game-related errors."""
    pass


class DrawLoss(GameException):
    """Raised when game ends in a draw or loss due to exceeded timelines."""
    pass


class Stalemate(GameException):
    """Raised when game reaches a stalemate position."""
    pass


class Checkmate(GameException):
    """Raised when a player is checkmated."""
    pass


class InvalidMoveError(GameException):
    """Raised when an invalid move is attempted."""
    pass
