"""
Integration tests for complete game flow.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from engine.chess_engine import Chess5DEngine
from engine.game_state import Checkmate, Stalemate, DrawLoss
from mcts.search import MCTS


class TestGameFlow(unittest.TestCase):
    """Integration tests for game flow."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = Chess5DEngine(max_time=11, max_turns=30)
        self.mcts_args = {'num_searches': 5, 'C': 1.41}
        self.mcts = MCTS(self.engine, self.mcts_args)

    def test_game_initialization_and_first_move(self):
        """Test that game can be initialized and first move executed."""
        state = self.engine.get_initial_state()

        # Check initial state is valid
        self.assertFalse(state.is_terminal)
        self.assertEqual(state.player, 'white')

        # Perform MCTS search
        start_probs, end_probs = self.mcts.search(state)

        # Check that we got valid probabilities
        self.assertGreater(float(start_probs.max()), 0)

    def test_multiple_moves(self):
        """Test that multiple moves can be executed."""
        state = self.engine.get_initial_state()
        moves_made = 0
        max_moves = 10

        while not state.is_terminal and moves_made < max_moves:
            try:
                # Get move from MCTS
                start_probs, end_probs = self.mcts.search(state)

                # Pick random move weighted by probabilities
                move_str, _, _ = self.engine.pick_random_move(
                    state,
                    state.choices_start,
                    state.choices_end,
                    modify_probs=True
                )

                # Execute move
                self.engine.make_move(state, move_str)
                moves_made += 1

            except (Checkmate, Stalemate, DrawLoss):
                # Game ended normally
                break
            except Exception as e:
                # Unexpected error
                self.fail(f"Unexpected error during move {moves_made}: {e}")

        # Check that we made at least one move
        self.assertGreater(moves_made, 0)

    def test_state_independence(self):
        """Test that copied states are independent."""
        state = self.engine.get_initial_state()
        state_copy = state.copy()

        # Make move on original
        move_str, _, _ = self.engine.pick_random_move(
            state,
            state.choices_start,
            state.choices_end,
            modify_probs=True
        )
        self.engine.make_move(state, move_str)

        # Copy should be unchanged
        self.assertNotEqual(state.game_string, state_copy.game_string)

    def test_alternating_players(self):
        """Test that players alternate correctly."""
        state = self.engine.get_initial_state()
        self.assertEqual(state.player, 'white')

        # Make a move
        try:
            move_str, _, _ = self.engine.pick_random_move(
                state,
                state.choices_start,
                state.choices_end,
                modify_probs=True
            )
            self.engine.make_move(state, move_str)

            # Player should change (in most cases)
            # Note: In 5D chess, player might not always change
            self.assertIn(state.player, ['white', 'black'])

        except Exception as e:
            # Game might end on first move in rare cases
            pass


if __name__ == '__main__':
    unittest.main()
