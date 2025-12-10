"""Main game runner for 5D Chess with MCTS."""

from typing import Optional
import cupy as cp

from src.chess_engine import Chess5D
from src.chess_state import ChessState
from src.mcts import MCTS
from config.game_config import GameConfig


class GameRunner:
    """Manages game execution and player turns."""

    def __init__(
        self,
        max_time: int = GameConfig.MAX_TIMELINES,
        max_turns: int = GameConfig.MAX_TURNS,
        mcts_args: Optional[dict] = None
    ):
        """
        Initialize the game runner.

        Args:
            max_time: Maximum number of timelines
            max_turns: Maximum number of turns
            mcts_args: MCTS configuration (uses defaults if None)
        """
        self.game = Chess5D(max_time, max_turns)
        self.mcts_args = mcts_args or GameConfig.get_mcts_config()
        self.mcts = MCTS(self.game, self.mcts_args)
        self.state: Optional[ChessState] = None

    def start_new_game(self) -> ChessState:
        """
        Start a new game.

        Returns:
            ChessState: Initial game state
        """
        self.state = self.game.get_initial_state()
        return self.state

    def make_ai_move(self) -> Optional[str]:
        """
        Make an AI move using MCTS.

        Returns:
            Optional[str]: Move string, or None if game is over
        """
        if self.state is None or self.state.is_terminal:
            return None

        result = self.mcts.get_best_move(self.state)
        if result is None:
            return None

        move_str, _, _ = result
        self.game.make_move(self.state, move_str)
        return move_str

    def make_move(self, move_str: str) -> bool:
        """
        Make a move from move string.

        Args:
            move_str: Move in string notation

        Returns:
            bool: True if move successful, False otherwise
        """
        if self.state is None or self.state.is_terminal:
            return False

        try:
            self.game.make_move(self.state, move_str)
            return True
        except Exception as e:
            print(f"Invalid move: {e}")
            return False

    def get_game_string(self) -> Optional[str]:
        """
        Get current game state as string.

        Returns:
            Optional[str]: Game state string
        """
        return self.state.game_string if self.state else None

    def is_game_over(self) -> bool:
        """
        Check if game is over.

        Returns:
            bool: True if game is terminal
        """
        return self.state.is_terminal if self.state else True

    def get_winner(self) -> Optional[str]:
        """
        Get the winner of the game.

        Returns:
            Optional[str]: Winner name or None
        """
        return self.state.get_winner() if self.state else None

    def play_ai_vs_ai(self, max_moves: int = 100, verbose: bool = True) -> dict:
        """
        Run a full AI vs AI game.

        Args:
            max_moves: Maximum number of moves before stopping
            verbose: Whether to print moves

        Returns:
            dict: Game results including winner, moves, and final state
        """
        self.start_new_game()
        moves = []
        move_count = 0

        if verbose:
            print("Starting AI vs AI game...")
            print(f"Initial state: {self.state.player} to move\n")

        while not self.is_game_over() and move_count < max_moves:
            move_str = self.make_ai_move()

            if move_str is None:
                break

            moves.append(move_str)
            move_count += 1

            if verbose:
                print(f"Move {move_count}: {self.state.prev_player} plays {move_str}")

        winner = self.get_winner()
        game_string = self.get_game_string()

        if verbose:
            print(f"\nGame Over!")
            print(f"Winner: {winner if winner else 'Draw'}")
            print(f"Total moves: {move_count}")
            print(f"Final game state:\n{game_string}")

        return {
            'winner': winner,
            'moves': moves,
            'move_count': move_count,
            'game_string': game_string,
            'final_state': self.state
        }


def main():
    """Run a sample game."""
    print("5D Chess with MCTS AI")
    print("=" * 50)

    runner = GameRunner()
    results = runner.play_ai_vs_ai(max_moves=50, verbose=True)

    print("\n" + "=" * 50)
    print("Game Statistics:")
    print(f"Total moves: {results['move_count']}")
    print(f"Winner: {results['winner'] if results['winner'] else 'Draw'}")


if __name__ == "__main__":
    main()
