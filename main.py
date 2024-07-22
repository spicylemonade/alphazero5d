import numpy as np


class Chess5DConverter:
    def __init__(self, MAX_TIMELINES, MAX_TURNS, NUM_PIECE_TYPES, BOARD_SIZE):
        self.MAX_TIMELINES = MAX_TIMELINES
        self.MAX_TURNS = MAX_TURNS
        self.NUM_PIECE_TYPES = NUM_PIECE_TYPES
        self.BOARD_SIZE = BOARD_SIZE
        self.piece_map = {
            'P': 0,  # Pawn
            'N': 1,  # Knight
            'B': 2,  # Bishop
            'R': 3,  # Rook
            'Q': 4,  # Queen
            'K': 5  # King
        }

    def notation_to_indices(self, notation):
        # Extract parts from the notation
        start, end = notation.split(' >> ')
        start = start[1:]
        end = end[1:]


        # Parse start part
        timeline_turn_piece_start = start.split('T')

        timeline_start = int(timeline_turn_piece_start[0])
        turn_piece_start = timeline_turn_piece_start[1]

        turn_start = int(turn_piece_start[0])
        piece_type_char = turn_piece_start[-3]

        piece_type = self.piece_map[piece_type_char]
        start_pos = turn_piece_start[-2:]

        x_start = ord(start_pos[0]) - ord('a')  # Convert 'a'-'h' to 0-7
        y_start = 8-int(start_pos[1])  # Convert '1'-'8' to 0-7

        # Parse end part
        timeline_turn_end = end.split('T')

        timeline_end = int(timeline_turn_end[0])

        turn_end_pos = timeline_turn_end[1]
        turn_end = int(turn_end_pos[0])

        end_pos = turn_end_pos[-2:]
        x_end = ord(end_pos[0]) - ord('a')  # Convert 'a'-'h' to 0-7
        y_end = 8-int(end_pos[1])   # Convert '1'-'8' to 0-7


        return timeline_start, turn_start, piece_type, x_start, y_start, timeline_end, turn_end, x_end, y_end


    def notation_to_tensor(self,tensor, notations,player, is_init = False):

        #tensor = np.zeros((self.MAX_TIMELINES, self.MAX_TURNS, self.NUM_PIECE_TYPES, self.BOARD_SIZE, self.BOARD_SIZE))


        for notation in notations:
            t_start, turn_start, p_type, x_start, y_start, t_end, turn_end, x_end, y_end = self.notation_to_indices(
                notation)
            if not is_init:
                #print(f"player: {player}, ", tensor[0, 0])
                tensor[t_end, turn_end] = tensor[t_end, turn_end - 1]



            tensor[t_start, turn_start, p_type, y_start,x_start] = 0  # Piece's start position

            tensor[t_end, turn_end, p_type, y_end, x_end] = player  # Piece's end position

        return tensor
    def init_board(self):
        tensor = np.zeros((self.MAX_TIMELINES, self.MAX_TURNS, self.NUM_PIECE_TYPES, self.BOARD_SIZE, self.BOARD_SIZE))

        initial_positions_white = [
            "(0T0)Ra1 >> (0T0)a1", "(0T0)Nb1 >> (0T0)b1", "(0T0)Bc1 >> (0T0)c1", "(0T0)Qd1 >> (0T0)d1",
            "(0T0)Ke1 >> (0T0)e1", "(0T0)Bf1 >> (0T0)f1", "(0T0)Ng1 >> (0T0)g1", "(0T0)Rh1 >> (0T0)h1",
            "(0T0)Pa2 >> (0T0)a2", "(0T0)Pb2 >> (0T0)b2", "(0T0)Pc2 >> (0T0)c2", "(0T0)Pd2 >> (0T0)d2",
            "(0T0)Pe2 >> (0T0)e2", "(0T0)Pf2 >> (0T0)f2", "(0T0)Pg2 >> (0T0)g2", "(0T0)Ph2 >> (0T0)h2",

        ]
        initial_positions_black = [
            "(0T0)Ra8 >> (0T0)a8", "(0T0)Nb8 >> (0T0)b8", "(0T0)Bc8 >> (0T0)c8", "(0T0)Qd8 >> (0T0)d8",
            "(0T0)Ke8 >> (0T0)e8", "(0T0)Bf8 >> (0T0)f8", "(0T0)Ng8 >> (0T0)g8", "(0T0)Rh8 >> (0T0)h8",
            "(0T0)Pa7 >> (0T0)a7", "(0T0)Pb7 >> (0T0)b7", "(0T0)Pc7 >> (0T0)c7", "(0T0)Pd7 >> (0T0)d7",
            "(0T0)Pe7 >> (0T0)e7", "(0T0)Pf7 >> (0T0)f7", "(0T0)Pg7 >> (0T0)g7", "(0T0)Ph7 >> (0T0)h7"
        ]
        tensor = self.notation_to_tensor(tensor, initial_positions_white, 1,is_init = True)
        tensor = self.notation_to_tensor(tensor, initial_positions_black, -1,is_init = True)
        return tensor

    def point_to_notation(self, start, end):
        return f'({start[0]}T{start[1]}){chr(ord('@')+start[2])}{chr(ord('`')+start[3])}{start[4]} >> ({end[0]}T{end[1]}){chr(ord('`')+end[2])}{end[3]}'



MAX_TIMELINES = 1
MAX_TURNS = 2
NUM_PIECE_TYPES = 6
BOARD_SIZE = 8

converter = Chess5DConverter(MAX_TIMELINES, MAX_TURNS, NUM_PIECE_TYPES, BOARD_SIZE)


notations = [
    "(0T1)Pc2 >> (0T1)c4",

    # "(1T2)Nb1 >> (1T2)c3",
]

tensor= converter.init_board()
tensor = converter.notation_to_tensor(tensor, notations, 1)
print("Tensor shape:", tensor.shape)
print(tensor[0,1])
#print(converter.point_to_notation([1,2,3,4,5],[5,6,7,8]))

