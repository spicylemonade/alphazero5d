import numpy as np
import json
from javascript import require

# Import the 5D Chess JS library and json_manage module
Chess = require('5d-chess-js')
json_manage = require('./json_manage.js')

# Create a new chess instance and enable console output
chess = Chess()
chess.enableConsole = True

# Make some moves and submit them
moves = [
    '(0T1)Pc2 >> (0T1)c4',
    '(0T1)Pg7 >> (0T1)g5',
    '(0T2)Nb1 >> (0T1)b3',
    '(1T1)Ng8 >> (0T1)g6'
]
for move in moves:
    chess.move(move)
    chess.submit()

# Define a function to get the homogeneous raw board
def get_homogeneous_raw_board(fixed_timelines=11, fixed_turns=50):
    """Convert the chess state to a 4D numpy array."""
    json_state = json_manage.stringify(chess.state()['rawBoard'])
    python_board = json.loads(json_state)

    # Create a 4D numpy array filled with zeros
    board_shape = (fixed_timelines, fixed_turns, 8, 8)
    board = np.zeros(board_shape, dtype=int)

    # Fill the board with actual data
    for t, timeline in enumerate(python_board):
        if t >= fixed_timelines:
            break
        for turn, state in enumerate(timeline):
            if turn >= fixed_turns:
                break
            if state is not None:
                board[t, turn] = np.array(state)

    return board

# Get the current board state
state = chess.state()