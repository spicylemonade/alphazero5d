"""
Main Game Runner
Entry point for playing 5D chess with MCTS AI.
"""
import cupy as cp
from engine import Chess5DEngine
from mcts import MCTS


class Game5DChess:
    """Main game controller for 5D chess with MCTS."""

    def __init__(
        self,
        max_timelines: int = 11,
        max_turns: int = 30,
        num_searches: int = 20,
        c_param: float = 1.41
    ):
        """
        Initialize game.

        Args:
            max_timelines: Maximum number of timelines
            max_turns: Maximum turns per timeline
            num_searches: Number of MCTS simulations per move
            c_param: Exploration parameter for UCB
        """
        self.engine = Chess5DEngine(max_timelines, max_turns)
        self.mcts_args = {
            'num_searches': num_searches,
            'C': c_param
        }
        self.mcts = MCTS(self.engine, self.mcts_args)

    def play_game(self, verbose: bool = True) -> dict:
        """
        Play a complete game.

        Args:
            verbose: Whether to print moves

        Returns:
            dict: Game statistics
        """
        state = self.engine.get_initial_state()
        move_count = 0
        move_history = []

        while not state.is_terminal:
            if verbose:
                print(f"\n=== Move {move_count + 1} ===")
                print(f"Player: {state.player}")

            # Perform MCTS search
            start_probs, end_probs = self.mcts.search(state)

            # Select best move
            start_idx = cp.unravel_index(cp.argmax(start_probs), start_probs.shape)
            end_move_dict = self._get_best_end_move(start_idx, state, end_probs)

            # Format move string
            move_str = self._format_move(start_idx, end_move_dict)

            if verbose:
                print(f"Move: {move_str}")

            move_history.append({
                'move': move_str,
                'player': state.player,
                'move_number': move_count + 1
            })

            # Execute move
            try:
                self.engine.make_move(state, move_str)
                move_count += 1
            except Exception as e:
                print(f"Error making move: {e}")
                break

            if move_count > 200:  # Safety limit
                print("Move limit reached")
                break

        if verbose:
            print(f"\n=== Game Over ===")
            print(f"Final state: {state.value}")
            print(f"Winner: {state.winning}")
            print(f"Total moves: {move_count}")

        return {
            'moves': move_count,
            'winner': state.winning,
            'value': state.value,
            'history': move_history
        }

    def _get_best_end_move(self, start_idx: tuple, state, end_probs: cp.ndarray) -> dict:
        """Get best end move given start position."""
        start_move = {
            "timeline": self.engine.convert_timeline_opposite(start_idx[0].item()),
            "turn": start_idx[1].item() + 1,
            "rank": start_idx[2].item() + 1,
            "file": start_idx[3].item() + 1,
        }

        # Get valid end moves
        end_moves = self.engine._get_end_moves(state.moves, start_move)
        self.engine._update_end_move_probs(end_moves, state)

        # Mask invalid moves
        masked_probs = state.choices_end * end_probs
        if cp.sum(masked_probs) == 0:
            masked_probs = state.choices_end

        masked_probs /= cp.sum(masked_probs)

        # Select best end position
        end_idx = cp.unravel_index(cp.argmax(masked_probs), masked_probs.shape)

        return {
            "timeline": self.engine.convert_timeline_opposite(end_idx[0].item()),
            "turn": end_idx[1].item() + 1,
            "rank": end_idx[2].item() + 1,
            "file": end_idx[3].item() + 1,
        }

    def _format_move(self, start_idx: tuple, end_dict: dict) -> str:
        """Format move as string."""
        start_timeline = self.engine.convert_timeline_opposite(start_idx[0].item())
        start_turn = start_idx[1].item() + 1
        start_file = chr(96 + start_idx[3].item() + 1)
        start_rank = start_idx[2].item() + 1

        end_file = chr(96 + end_dict['file'])
        end_rank = end_dict['rank']

        return (
            f"({start_timeline}T{start_turn}){start_file}{start_rank}>>"
            f"({end_dict['timeline']}T{end_dict['turn']}){end_file}{end_rank}"
        )


def main():
    """Main entry point."""
    print("=== 5D Chess with MCTS ===")
    print("Initializing game...")

    game = Game5DChess(
        max_timelines=11,
        max_turns=30,
        num_searches=20,
        c_param=1.41
    )

    print("Starting game...\n")
    result = game.play_game(verbose=True)

    print(f"\n=== Final Result ===")
    print(f"Total moves: {result['moves']}")
    print(f"Winner: {result['winner']}")
    print(f"Final value: {result['value']}")


if __name__ == "__main__":
    main()
