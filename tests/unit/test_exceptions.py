"""Unit tests for custom exceptions."""

import pytest
from src.exceptions import (
    GameException,
    DrawLoss,
    Stalemate,
    Checkmate,
    InvalidMoveError
)


class TestExceptions:
    """Test custom exception classes."""

    def test_game_exception_is_exception(self):
        """Test that GameException is an Exception."""
        assert issubclass(GameException, Exception)

    def test_draw_loss_inherits_game_exception(self):
        """Test that DrawLoss inherits from GameException."""
        assert issubclass(DrawLoss, GameException)

    def test_stalemate_inherits_game_exception(self):
        """Test that Stalemate inherits from GameException."""
        assert issubclass(Stalemate, GameException)

    def test_checkmate_inherits_game_exception(self):
        """Test that Checkmate inherits from GameException."""
        assert issubclass(Checkmate, GameException)

    def test_invalid_move_error_inherits_game_exception(self):
        """Test that InvalidMoveError inherits from GameException."""
        assert issubclass(InvalidMoveError, GameException)

    def test_can_raise_game_exception(self):
        """Test that GameException can be raised and caught."""
        with pytest.raises(GameException):
            raise GameException("Test exception")

    def test_can_raise_draw_loss(self):
        """Test that DrawLoss can be raised and caught."""
        with pytest.raises(DrawLoss):
            raise DrawLoss("Draw loss test")

    def test_can_raise_stalemate(self):
        """Test that Stalemate can be raised and caught."""
        with pytest.raises(Stalemate):
            raise Stalemate("Stalemate test")

    def test_can_raise_checkmate(self):
        """Test that Checkmate can be raised and caught."""
        with pytest.raises(Checkmate):
            raise Checkmate("Checkmate test")

    def test_can_catch_all_with_game_exception(self):
        """Test that all game exceptions can be caught with GameException."""
        for exc_class in [DrawLoss, Stalemate, Checkmate, InvalidMoveError]:
            with pytest.raises(GameException):
                raise exc_class("Test")
