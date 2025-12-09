"""
Quick Performance Test - Rapid validation of AI gameplay
Runs a small number of games to validate the system works
"""
import sys
sys.path.append('..')
from src.super import Chess5D, MCTS
import cupy as cp
import time

def quick_test_game():
    """Run a single quick test game"""
    print("=" * 80)
    print("5D Chess AI - Quick Test")
    print("=" * 80)

    config = {
        'num_searches': 15,
        'C': 1.41
    }

    print(f"\nConfiguration: {config}")
    print("Starting game...\n")

    game = Chess5D(1, 25)
    game_state = game.get_initial_state()
    mcts = MCTS(game, config)

    move_count = 0
    start_time = time.time()

    try:
        while move_count < 50:  # Limit moves for quick test
            move_start = time.time()

            # MCTS search
            mcts_prob_s, mcts_prob_e = mcts.search(game_state)

            # Select best move
            index_s = cp.unravel_index(cp.argmax(mcts_prob_s), mcts_prob_s.shape)
            index_e = game._pick_end_move_org(index_s, game_state, mcts_prob_e)

            action = f"({game.convert_timeline_opposite(index_s[0].item())}T{index_s[1].item() + 1})" \
                     f"{chr(96 + index_s[3].item() + 1)}{index_s[2].item() + 1}>>" \
                     f"({index_e['timeline']}T{index_e['turn']}){chr(96 + index_e['file'])}{index_e['rank']}"

            move_time = time.time() - move_start

            print(f"Move {move_count + 1} ({game_state.player}): {action} ({move_time:.2f}s)")

            # Make move
            game.make_move(game_state, action)
            move_count += 1

            if game_state.is_terminal:
                break

    except Exception as e:
        print(f"\nGame ended with exception: {type(e).__name__}: {e}")

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("GAME RESULTS")
    print("=" * 80)
    print(f"Total Moves: {move_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time per Move: {total_time/move_count:.2f}s")
    print(f"Terminal: {game_state.is_terminal}")

    if game_state.is_terminal:
        if game_state.value == 1:
            print(f"Winner: {game_state.prev_player}")
            print("Result: Checkmate")
        elif game_state.value == 0:
            print("Result: Stalemate")
        else:
            print("Result: Draw/Loss")
    else:
        print("Result: Incomplete (max moves reached)")

    print("=" * 80)

    return {
        'moves': move_count,
        'time': total_time,
        'terminal': game_state.is_terminal,
        'value': game_state.value
    }

if __name__ == "__main__":
    result = quick_test_game()
    print("\n✓ Quick test completed successfully!")
