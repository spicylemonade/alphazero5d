# Quick Start Guide

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "from src.game_runner import GameRunner; print('Installation successful!')"
```

## Basic Usage

### Example 1: Simple AI vs AI Game

```python
from src.game_runner import GameRunner

# Create game runner
runner = GameRunner()

# Play a quick game
results = runner.play_ai_vs_ai(max_moves=20, verbose=True)

# Check results
print(f"\nWinner: {results['winner']}")
print(f"Total moves: {results['move_count']}")
```

### Example 2: Custom Configuration

```python
from src.game_runner import GameRunner

# Create runner with custom settings
runner = GameRunner(
    max_time=7,           # Fewer timelines for faster games
    max_turns=20,         # Fewer turns
    mcts_args={
        'num_searches': 10,  # Fewer searches for speed
        'C': 1.41
    }
)

# Play game
results = runner.play_ai_vs_ai(max_moves=30, verbose=False)
print(f"Game finished in {results['move_count']} moves")
```

### Example 3: Step-by-Step Game

```python
from src.game_runner import GameRunner

runner = GameRunner()
runner.start_new_game()

move_count = 0
while not runner.is_game_over() and move_count < 10:
    # Get AI move
    move = runner.make_ai_move()

    if move is None:
        break

    move_count += 1
    print(f"Move {move_count}: {runner.state.prev_player} plays {move}")
    print(f"Current player: {runner.state.player}")

print(f"\nGame ended after {move_count} moves")
if runner.is_game_over():
    print(f"Winner: {runner.get_winner()}")
```

### Example 4: Using Lower-Level API

```python
from src.chess_engine import Chess5D
from src.mcts import MCTS
from config.game_config import GameConfig

# Create game engine
game = Chess5D(
    max_time=GameConfig.MAX_TIMELINES,
    max_turns=GameConfig.MAX_TURNS
)

# Get initial state
state = game.get_initial_state()

# Create MCTS
mcts = MCTS(game, GameConfig.get_mcts_config())

# Get best move
result = mcts.get_best_move(state)
if result:
    move_str, start_pos, end_pos = result
    print(f"Best move: {move_str}")
    print(f"Start position: {start_pos}")
    print(f"End position: {end_pos}")

    # Make the move
    game.make_move(state, move_str)
    print(f"New player: {state.player}")
```

### Example 5: Testing Configuration

```python
from src.game_runner import GameRunner
from config.game_config import TestConfig

# Use test configuration for faster execution
runner = GameRunner(
    max_time=TestConfig.TEST_MAX_TIMELINES,
    max_turns=TestConfig.TEST_MAX_TURNS,
    mcts_args=TestConfig.get_test_mcts_config()
)

# Run quick test game
results = runner.play_ai_vs_ai(max_moves=10, verbose=False)
print(f"Test game completed: {results['move_count']} moves")
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_chess_engine.py

# Specific test
pytest tests/unit/test_chess_engine.py::TestChess5D::test_initialization
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run with Verbose Output

```bash
pytest -v -s
```

## Configuration Options

### Game Configuration

```python
from config.game_config import GameConfig

# Access configuration values
print(f"Max timelines: {GameConfig.MAX_TIMELINES}")
print(f"Max turns: {GameConfig.MAX_TURNS}")
print(f"MCTS searches: {GameConfig.MCTS_NUM_SEARCHES}")

# Get configuration dictionaries
game_config = GameConfig.get_game_config()
mcts_config = GameConfig.get_mcts_config()
```

### MCTS Tuning

```python
# Faster, lower quality
fast_config = {
    'num_searches': 5,
    'C': 1.0
}

# Balanced
balanced_config = {
    'num_searches': 20,
    'C': 1.41
}

# Slower, higher quality
strong_config = {
    'num_searches': 100,
    'C': 1.41
}
```

## Performance Tips

### 1. Adjust MCTS Searches

Lower `num_searches` for faster games:
```python
runner = GameRunner(mcts_args={'num_searches': 10, 'C': 1.41})
```

### 2. Reduce Board Size

Use smaller timeline/turn limits:
```python
runner = GameRunner(max_time=5, max_turns=15)
```

### 3. GPU Memory

Monitor GPU memory usage:
```python
import cupy as cp

mempool = cp.get_default_memory_pool()
print(f"GPU memory used: {mempool.used_bytes() / 1e6:.2f} MB")
```

## Common Issues

### Issue: Out of GPU Memory

**Solution**: Reduce board size or MCTS searches:
```python
runner = GameRunner(
    max_time=5,
    max_turns=15,
    mcts_args={'num_searches': 10, 'C': 1.41}
)
```

### Issue: Slow Performance

**Solution**:
1. Ensure CuPy is using GPU (not CPU fallback)
2. Reduce MCTS search iterations
3. Use smaller board dimensions

### Issue: Import Errors

**Solution**: Install from repository root:
```bash
pip install -e .
```

## Next Steps

1. Read [API Documentation](API.md) for detailed API reference
2. Check [Architecture Guide](ARCHITECTURE.md) to understand system design
3. Review [Optimization Guide](OPTIMIZATION.md) for performance tuning
4. Explore example code in tests for more usage patterns

## Getting Help

- Check documentation in `docs/` directory
- Review test files for usage examples
- See `README.md` for project overview
