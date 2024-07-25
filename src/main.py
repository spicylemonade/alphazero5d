import cupy as cp
from jsrun import JSrun
from javascript import require

# Import the 5D Chess JS library and json_manage module
Chess = require('5d-chess-js')


class Chess5D(JSrun):
    def __init__(self, max_time, max_turns):
        self.chess = Chess()
        self.max_time = max_time
        self.max_turns = max_turns
        self.choices_start = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.int8)
        self.choices_end = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.int8)
        self.raw_board = self.get_homogeneous_raw_board(self.chess, max_time, max_turns)
        self.board = cp.flip(self.raw_board_to_tensor(), axis=3)
        self.moves = None
        self.player = 1
        self.convert_moves()

    def raw_board_to_tensor(self):
        """Convert raw board to tensor representation."""
        board = cp.array(self.raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        piece_masks = [
            (cp.abs(board) == 2, cp.abs(board) == 1, 0),  # Pawns
            (cp.abs(board) == 4, cp.abs(board) == 3, 1),  # Bishops
            (cp.abs(board) == 6, cp.abs(board) == 5, 2),  # Knights
            (cp.abs(board) == 8, cp.abs(board) == 7, 3),  # Rooks
            (cp.abs(board) == 10, cp.abs(board) == 9, 4),  # Queens
            (cp.abs(board) == 12, cp.abs(board) == 11, 5)  # Kings
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = white_mask.astype(cp.int8) - black_mask.astype(cp.int8)

        return tensor

    @staticmethod
    def convert_timeline(num):
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num):
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, move):
        self.chess.move(str(move))
        self.chess.submit()
        self.convert_moves()

    def pick_choice(self):
        start_move = self._pick_start_move()
        end_move = self._pick_end_move(start_move)
        return self.move_to_string(start_move, end_move)

    def _pick_start_move(self):
        tensor_start = self.choices_start.astype(cp.float64)
        tensor_start /= cp.sum(tensor_start)
        flat_index_start = cp.random.choice(len(tensor_start.flatten()),size=1, p=tensor_start.flatten())
        i, j, k, l = cp.unravel_index(flat_index_start, tensor_start.shape)
        return {
            "timeline": self.convert_timeline_opposite(i.item()),
            "turn": j.item(),
            "rank": k.item() + 1,
            "file": l.item() + 1
        }

    def _pick_end_move(self, start_move):
        end_moves = self.get_end_moves(self.moves, start_move)
        self.convert_moves_end(end_moves)
        tensor_end = self.choices_end.astype(cp.float64)
        tensor_end /= cp.sum(tensor_end)
        flat_index_end = cp.random.choice(len(tensor_end.flatten()),size=1, p=tensor_end.flatten())
        m, n, o, p = cp.unravel_index(flat_index_end, tensor_end.shape)
        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item(),
            "rank": o.item() + 1,
            "file": p.item() + 1
        }

    @staticmethod
    def get_end_moves(data, start_value):
        return [move['end'] for move in data if all(
            move['start'].get(key) == value
            for key, value in start_value.items()
            if key not in ['player', 'coordinate']
        )]

    @staticmethod
    def move_to_string(start, end):
        return f"({start['timeline']}T{start['turn']}){chr(96 + start['file'])}{start['rank']} >> " \
               f"({end['timeline']}T{end['turn']}){chr(96 + end['file'])}{end['rank']}"

    def convert_moves(self):
        self.moves = self.get_moves(self.chess)
        self.choices_start.fill(0)
        for move in self.moves:
            start = move['start']
            timeline = self.convert_timeline(start['timeline'])
            self.choices_start[timeline, start['turn'], start['rank'] - 1, start['file'] - 1] = 1

    def convert_moves_end(self, moves):
        self.choices_end.fill(0)
        for move in moves:
            timeline = self.convert_timeline(move['timeline'])
            self.choices_end[timeline, move['turn'], move['rank'] - 1, move['file'] - 1] = 1


# Game loop
game = Chess5D(11, 50)

while True:
    print(f"Current player: {game.chess.player}")
    ai_move = game.pick_choice()
    print(f"AI move: {ai_move}")
    #game.make_move(ai_move)

    user_move = input("Your move (or 'q' to quit): ")
    if user_move.lower() == 'q':
        break
    game.make_move(user_move)