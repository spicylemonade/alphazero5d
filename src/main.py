import numpy as np
import jsrun


class chess5D:
    def __init__(self,max_time,max_turns):
        self.max_turns = max_turns
        self.max_time = max_time
        self.raw_board = jsrun.get_homogeneous_raw_board(max_time, max_turns)
        self.Board = np.flip(self.raw_board_to_tensor(),axis=3)
    def raw_board_to_tensor(self):
        """Convert raw board to tensor representation."""
        board = np.array(self.raw_board, dtype=np.int8)
        tensor_shape = (self.max_time, self.max_turns, 6, 8, 8)
        tensor = np.zeros(tensor_shape, dtype=np.int8)

        piece_masks = [
            ((board == 2), (board == -2), 0),  # Pawns
            ((board == 4), (board == -4), 1),  # Bishops
            ((board == 6), (board == -6), 2),  # Knights
            ((board == 8), (board == -8), 3),  # Rooks
            ((board == 10), (board == -10), 4),  # Queens
            ((board == 12), (board == -12), 5)  # Kings
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = white_mask.astype(np.int8) - black_mask.astype(np.int8)

        return tensor


game = chess5D(11,50)


print(game.Board.shape)
print(game.Board[1, 2])