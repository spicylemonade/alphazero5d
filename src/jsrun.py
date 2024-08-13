import numpy as cp
import json
import cupy as cp
import numpy as np
import threading
from functools import wraps
from javascript import require

Chess = require('5d-chess-js')
json_manage = require('./json_manage.js')

def timeout(seconds):
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
            thread.daemon = True  # Set the thread as a daemon thread
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                return None  # Return None on timeout

            if exception[0] is not None:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator

class JSrun:
    @timeout(3)
    def get_moves(self, chess):
        json_state = chess.moves('json')

        py_json = json.loads(json_state)
        return py_json

    def safe_get_moves(self, chess):
        try:
            return self.get_moves(chess)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    def get_homogeneous_raw_board(self,chess,fixed_timelines=11, fixed_turns=30):
        """Convert the chess state to a 4D numpy array."""
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
