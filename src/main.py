import cupy as cp
from jsrun import JSrun
from javascript import require
import copy
Chess = require('5d-chess-js')

class GameException(Exception):
    pass

class DrawLoss(GameException):
    pass

class Stalemate(GameException):
    pass

class Checkmate(GameException):
    pass

class ChessState:
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
    def copy(self):
        new_state = ChessState()
        new_state.chess = Chess(self.chess.export("5dpgn"))  # Create a new Chess object with the same state
        new_state.value = self.value
        new_state.piece = self.piece
        new_state.game_string = self.game_string
        new_state.choices_start = cp.copy(self.choices_start) if self.choices_start is not None else None
        new_state.choices_end = cp.copy(self.choices_end) if self.choices_end is not None else None
        new_state.is_terminal = self.is_terminal
        new_state.moves = copy.deepcopy(self.moves) if self.moves is not None else None
        new_state.raw_board = cp.copy(self.raw_board) if self.raw_board is not None else None
        new_state.board = cp.copy(self.board) if self.board is not None else None
        new_state.player = self.player
        new_state.prev_player = self.prev_player
        return new_state

class Chess5D(JSrun):
    piece_map = {0: 'P', 1: 'B', 2: 'N', 3: 'R', 4: 'Q', 5: 'K'}
    #thing
    player_map = {'white': 1, 'black': -1}

    def __init__(self, max_time, max_turns):
        self.max_time = max_time
        self.max_turns = max_turns
        self.action_size =(max_time, max_turns,8,8)
        self.max_dim = 0
        self.space_out = 0

    def get_initial_state(self):
        state = ChessState()
        state.game_string = state.chess.export("5dpgn")
        state.raw_board = self.get_homogeneous_raw_board(state.chess, self.max_time, self.max_turns)
        state.choices_start = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float64)
        state.choices_end = cp.zeros((self.max_time, self.max_turns, 8, 8), dtype=cp.float64)
        state.board = self.raw_board_to_tensor(state.raw_board)
        self.pick_choice(state,state.choices_start,state.choices_end)
        return state

    def raw_board_to_tensor(self, raw_board):
        board = cp.array(raw_board, dtype=cp.int8)
        tensor_shape = (self.max_time, self.max_turns * 2, 6, 8, 8)
        tensor = cp.zeros(tensor_shape, dtype=cp.int8)

        piece_masks = [
            (cp.abs(board) == 2, cp.abs(board) == 1, 0),
            (cp.abs(board) == 4, cp.abs(board) == 3, 1),
            (cp.abs(board) == 6, cp.abs(board) == 5, 2),
            (cp.abs(board) == 8, cp.abs(board) == 7, 3),
            (cp.abs(board) == 10, cp.abs(board) == 9, 4),
            (cp.abs(board) == 12, cp.abs(board) == 11, 5)
        ]

        for white_mask, black_mask, index in piece_masks:
            tensor[:board.shape[0], :board.shape[1], index] = white_mask.astype(cp.int8) - black_mask.astype(cp.int8)

        return cp.flip(tensor, axis=3)

    @staticmethod
    def convert_timeline(num):
        return -2 * num - 1 if num < 0 else 2 * num

    @staticmethod
    def convert_timeline_opposite(num):
        return num // 2 if num % 2 == 0 else -(num + 1) // 2

    def make_move(self, state, move):
        state.game_string = state.chess.export("5dpgn")

        try:
            state.prev_player = state.player
            state.player = state.chess.player
            state.chess.move(str(move))
            if state.chess.submittable():
                state.chess.submit()

            self.convert_moves(state)

        except Checkmate:
            state.is_terminal = True
            state.value = 1
        except Stalemate:
            state.is_terminal = True
            state.value = 0
        except DrawLoss:
            state.is_terminal = True
            state.value = 0.3

        return state

    def pick_choice(self, state,x,y):
        self.convert_moves(state)
        start_move = self._pick_start_move(x)
        end_move = self._pick_end_move(start_move, state,y)
        x[self.convert_timeline(start_move['timeline']), start_move['turn']-1, start_move['rank']-1, start_move['file']-1] = 0
        y[self.convert_timeline(start_move['timeline']), end_move['turn']-1, end_move['rank']-1, end_move['file']-1] = 0
        return self.move_to_string(start_move, end_move)

    def _pick_start_move(self, x):
        tensor_start = x
        tensor_start /= cp.sum(tensor_start)
        flat_index_start = cp.random.choice(len(tensor_start.flatten()), size=1, p=tensor_start.flatten())
        i, j, k, l = cp.unravel_index(flat_index_start, tensor_start.shape)

        return {
            "timeline": self.convert_timeline_opposite(i.item()),
            "turn": j.item() + 1,
            "rank": k.item() + 1,
            "file": l.item() + 1,
        }

    def _pick_end_move(self, start_move, state, y):
        end_moves = self.get_end_moves(state.moves, start_move)
        self.convert_moves_end(end_moves, state)
        tensor_end = y
        tensor_end /= cp.sum(tensor_end)
        flat_index_end = cp.random.choice(len(tensor_end.flatten()), size=1, p=tensor_end.flatten())
        m, n, o, p = cp.unravel_index(flat_index_end, tensor_end.shape)
        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item() + 1,
            "rank": o.item() + 1,
            "file": p.item() + 1,
        }
    def _pick_end_move_org(self, start_move, state,y):
        start_move ={
            "timeline": self.convert_timeline_opposite(start_move[0].item()),
            "turn": start_move[1].item()+1,
            "rank": start_move[2].item()+1,
            "file": start_move[3].item()+1,
        }
        end_moves = self.get_end_moves(state.moves, start_move)
        self.convert_moves_end(end_moves, state)

        m, n, o, p = cp.unravel_index(cp.argmax(y), y.shape)
        return {
            "timeline": self.convert_timeline_opposite(m.item()),
            "turn": n.item() + 1,
            "rank": o.item() + 1,
            "file": p.item() + 1,
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
               f"({end['timeline']}T{end['turn']}){chr(96 + end['file'])}{end['rank']}", \
                (Chess5D.convert_timeline(start['timeline']), start['turn']-1, start['rank']-1, start['file']-1), \
                (Chess5D.convert_timeline(end['timeline']), end['turn'] - 1, end['rank'] - 1, end['file'] - 1)


    def check_timelines(self, move):
        real_end = move['realEnd']
        timeline = real_end['timeline']
        if abs(timeline) > (self.max_time - 1) / 2:
            self.space_out = 1
            return True
        return False

    def check_turns(self, move):
        real_end = move['realEnd']
        turn = real_end['turn']
        if turn > self.max_turns - 1:
            self.space_out = 2
            return True
        return False

    def convert_moves(self, state):
        state.moves = self.safe_get_moves(state.chess)
        if state.moves == 0:
            raise Stalemate
        state.choices_start.fill(0)
        for i in range(len(state.moves) - 1, -1, -1):
            move = state.moves[i]
            if self.check_timelines(move) or self.check_turns(move):
                state.moves.pop(i)
            else:
                start = move['start']
                timeline = self.convert_timeline(start['timeline'])
                state.choices_start[timeline, start['turn'] - 1, start['rank'] - 1, start['file'] - 1] = 1
        if len(state.moves) == 0 and (state.chess.inCheckmate or state.chess.inCheck):
            print(f"checkmate: {Chess5D.player_map[state.chess.player]*-1}")
            raise Checkmate
        elif len(state.moves) == 0:
            print(f"Exceeded Timeline (Draw_loss): {state.chess.player}")
            raise DrawLoss

    def get_opponent_value(self,state, value):
        return value if state.prev_player == state.player else -value

    def convert_moves_end(self, moves, state):
        state.choices_end.fill(0)
        for move in moves:
            timeline = self.convert_timeline(move['timeline'])
            state.choices_end[timeline, move['turn']-1, move['rank'] - 1, move['file'] - 1] = 1


# # Game loop
# def main():
#     game = Chess5D(11, 30)
#     game_play = game.get_initial_state()
#
#     while True:
#         ai_move = game.pick_choice(game_play)
#         print(f"AI move: {ai_move}")
#         game_play = game.make_move(game_play, ai_move)
#         if game_play.is_terminal:
#             print(game_play.game_string)
#             break
#
#
#
# if __name__ == "__main__":
#     main()