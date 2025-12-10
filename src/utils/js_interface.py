"""
JavaScript Interface Utilities
Handles communication with JavaScript chess library.
"""
import json
import threading
from functools import wraps
from typing import Optional, Any
from javascript import require

Chess = require('5d-chess-js')
json_manage = require('./src/json_manage.js')


def timeout(seconds: int):
    """
    Decorator to add timeout functionality to functions.

    Args:
        seconds: Maximum execution time in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                return None

            if exception[0] is not None:
                raise exception[0]

            return result[0]

        return wrapper
    return decorator


class JSInterface:
    """Base class for interfacing with JavaScript chess library."""

    @timeout(3)
    def get_moves(self, chess) -> Any:
        """
        Get available moves from chess object with timeout.

        Args:
            chess: JavaScript Chess object

        Returns:
            List of available moves in JSON format
        """
        json_state = chess.moves('json')
        py_json = json.loads(json_state)
        return py_json

    def safe_get_moves(self, chess) -> Optional[Any]:
        """
        Safely get moves with exception handling.

        Args:
            chess: JavaScript Chess object

        Returns:
            List of moves or None if error occurs
        """
        try:
            return self.get_moves(chess)
        except Exception as e:
            print(f"Error getting moves: {e}")
            return None

    def get_homogeneous_raw_board(
        self,
        chess,
        fixed_timelines: int = 11,
        fixed_turns: int = 30
    ):
        """
        Convert chess state to homogeneous 4D array.

        Args:
            chess: JavaScript Chess object
            fixed_timelines: Number of timelines in array
            fixed_turns: Number of turns per timeline

        Returns:
            CuPy array representing the board state
        """
        import cupy as cp

        json_state = json_manage.stringify(chess.state()['rawBoard'])
        python_board = json.loads(json_state)

        board_shape = (fixed_timelines, fixed_turns, 8, 8)
        board = cp.zeros(board_shape, dtype=int)

        for t, timeline in enumerate(python_board):
            if t >= fixed_timelines:
                break
            if timeline is not None:
                for turn, state in enumerate(timeline):
                    if turn >= fixed_turns:
                        break
                    if state is not None:
                        board[t, turn] = cp.array(state)

        return board
