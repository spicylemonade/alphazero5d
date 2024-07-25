import numpy as cp
import json
import cupy as cp
from javascript import require

# Import the 5D Chess JS library and json_manage module
Chess = require('5d-chess-js')
json_manage = require('./json_manage.js')

class JSrun:
    def get_moves(self,chess):
        json_state = chess.moves('json')
        py_json = json.loads(json_state)
        return py_json
    # Define a function to get the homogeneous raw board
    def get_homogeneous_raw_board(self,chess,fixed_timelines=11, fixed_turns=50):
        """Convert the chess state to a 4D numpy array."""
        json_state = json_manage.stringify(chess.state()['rawBoard'])
        python_board = json.loads(json_state)

        # Create a 4D numpy array filled with zeros
        board_shape = (fixed_timelines, fixed_turns, 8, 8)
        board = cp.zeros(board_shape, dtype=int)

        # Fill the board with actual data
        for t, timeline in enumerate(python_board):
            if t >= fixed_timelines:
                break
            for turn, state in enumerate(timeline):
                if turn >= fixed_turns:
                    break
                if state is not None:
                    board[t, turn] = cp.array(state)

        return board
