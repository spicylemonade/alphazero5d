# API Documentation

## Table of Contents

- [Chess Engine](#chess-engine)
- [Chess State](#chess-state)
- [MCTS Algorithm](#mcts-algorithm)
- [Game Runner](#game-runner)
- [Configuration](#configuration)
- [Exceptions](#exceptions)

---

## Chess Engine

### `Chess5D`

Main game engine for 5D Chess.

#### Constructor

```python
Chess5D(max_time: int, max_turns: int)
```

**Parameters**:
- `max_time`: Maximum number of timelines
- `max_turns`: Maximum number of turns per timeline

**Example**:
```python
from src.chess_engine import Chess5D

game = Chess5D(max_time=11, max_turns=30)
```

#### Methods

##### `get_initial_state() -> ChessState`

Initialize a new game state.

**Returns**: `ChessState` - Initial game state

**Example**:
```python
state = game.get_initial_state()
```

##### `make_move(state: ChessState, move: str) -> None`

Execute a move on the game state.

**Parameters**:
- `state`: Current game state (modified in place)
- `move`: Move string in format "(0T1)a2>>(0T1)a4"

**Raises**:
- `Checkmate`: If move results in checkmate
- `Stalemate`: If move results in stalemate
- `DrawLoss`: If move results in draw/loss

**Example**:
```python
game.make_move(state, "(0T1)a2>>(0T1)a4")
```

##### `convert_timeline(num: int) -> int`

Convert timeline number to internal representation.

**Parameters**:
- `num`: Timeline number

**Returns**: `int` - Internal timeline representation

##### `convert_timeline_opposite(num: int) -> int`

Convert internal timeline back to timeline number.

**Parameters**:
- `num`: Internal timeline number

**Returns**: `int` - Original timeline number

---

## Chess State

### `ChessState`

Represents the state of a 5D chess game.

#### Constructor

```python
ChessState()
```

**Example**:
```python
from src.chess_state import ChessState

state = ChessState()
```

#### Attributes

- `chess`: JavaScript chess engine instance
- `value`: Numeric game value (1 = win, -1 = loss, 0 = draw)
- `player`: Current player ('white' or 'black')
- `prev_player`: Previous player
- `is_terminal`: Whether game is over
- `moves`: List of available moves
- `board`: Board tensor representation
- `choices_start`: Valid start positions tensor
- `choices_end`: Valid end positions tensor

#### Methods

##### `copy() -> ChessState`

Create a deep copy of the state.

**Returns**: `ChessState` - Independent copy

**Example**:
```python
new_state = state.copy()
```

##### `is_game_over() -> bool`

Check if game is over.

**Returns**: `bool` - True if terminal

**Example**:
```python
if state.is_game_over():
    print("Game finished!")
```

##### `get_winner() -> Optional[str]`

Get the winner of the game.

**Returns**: `Optional[str]` - 'white', 'black', or None

**Example**:
```python
winner = state.get_winner()
print(f"Winner: {winner}")
```

---

## MCTS Algorithm

### `MCTS`

Monte Carlo Tree Search algorithm.

#### Constructor

```python
MCTS(game: Chess5D, args: Dict[str, Any])
```

**Parameters**:
- `game`: Game engine instance
- `args`: Configuration dictionary with:
  - `num_searches`: Number of MCTS iterations
  - `C`: Exploration constant

**Example**:
```python
from src.mcts import MCTS

args = {'num_searches': 20, 'C': 1.41}
mcts = MCTS(game, args)
```

#### Methods

##### `search(state: ChessState) -> Tuple[cp.ndarray, cp.ndarray]`

Perform MCTS search from given state.

**Parameters**:
- `state`: Current game state

**Returns**: Tuple of action probability distributions (start, end)

**Example**:
```python
probs_start, probs_end = mcts.search(state)
```

##### `get_best_move(state: ChessState) -> Optional[Tuple[str, Tuple, Tuple]]`

Get the best move using MCTS.

**Parameters**:
- `state`: Current game state

**Returns**: Tuple of (move_string, start_position, end_position)

**Example**:
```python
result = mcts.get_best_move(state)
if result:
    move, start, end = result
    print(f"Best move: {move}")
```

### `Node`

MCTS tree node.

#### Constructor

```python
Node(
    game: Chess5D,
    args: Dict[str, Any],
    state: ChessState,
    parent: Optional[Node] = None,
    action_taken_s: Optional[Tuple] = None,
    action_taken_e: Optional[Tuple] = None
)
```

#### Methods

##### `is_fully_expanded() -> bool`

Check if all moves explored.

##### `select() -> Node`

Select best child using UCB1.

##### `expand() -> Node`

Create new child node.

##### `simulate() -> float`

Run random playout.

##### `backpropagate(value: float) -> None`

Update statistics up the tree.

---

## Game Runner

### `GameRunner`

High-level game orchestration.

#### Constructor

```python
GameRunner(
    max_time: int = 11,
    max_turns: int = 30,
    mcts_args: Optional[dict] = None
)
```

**Parameters**:
- `max_time`: Maximum timelines
- `max_turns`: Maximum turns
- `mcts_args`: MCTS configuration

**Example**:
```python
from src.game_runner import GameRunner

runner = GameRunner()
```

#### Methods

##### `start_new_game() -> ChessState`

Initialize a new game.

**Returns**: `ChessState` - Initial state

**Example**:
```python
state = runner.start_new_game()
```

##### `make_ai_move() -> Optional[str]`

Make an AI move.

**Returns**: `Optional[str]` - Move string or None

**Example**:
```python
move = runner.make_ai_move()
print(f"AI played: {move}")
```

##### `make_move(move_str: str) -> bool`

Execute a move.

**Parameters**:
- `move_str`: Move in string notation

**Returns**: `bool` - Success status

**Example**:
```python
success = runner.make_move("(0T1)a2>>(0T1)a4")
```

##### `play_ai_vs_ai(max_moves: int = 100, verbose: bool = True) -> dict`

Run full AI vs AI game.

**Parameters**:
- `max_moves`: Maximum moves before stopping
- `verbose`: Print moves

**Returns**: `dict` - Game results with:
  - `winner`: Winner name or None
  - `moves`: List of moves
  - `move_count`: Total moves
  - `game_string`: Final game state
  - `final_state`: Final ChessState

**Example**:
```python
results = runner.play_ai_vs_ai(max_moves=50, verbose=True)
print(f"Winner: {results['winner']}")
print(f"Moves: {results['move_count']}")
```

##### `is_game_over() -> bool`

Check if game is finished.

##### `get_winner() -> Optional[str]`

Get game winner.

##### `get_game_string() -> Optional[str]`

Get game state string.

---

## Configuration

### `GameConfig`

Production configuration.

#### Class Attributes

- `MAX_TIMELINES = 11`
- `MAX_TURNS = 30`
- `BOARD_SIZE = 8`
- `MCTS_NUM_SEARCHES = 20`
- `MCTS_EXPLORATION_CONSTANT = 1.41`
- `DEFAULT_PLAYER = 'white'`
- `WIN_VALUE = 1.0`
- `LOSS_VALUE = -1.0`
- `DRAW_VALUE = 0.0`

#### Methods

##### `get_mcts_config() -> Dict[str, Any]`

Get MCTS configuration.

**Example**:
```python
from config.game_config import GameConfig

mcts_config = GameConfig.get_mcts_config()
```

##### `get_game_config() -> Dict[str, Any]`

Get game configuration.

### `TestConfig`

Testing configuration with smaller values.

#### Class Attributes

- `TEST_MAX_TIMELINES = 3`
- `TEST_MAX_TURNS = 10`
- `TEST_MCTS_SEARCHES = 5`

---

## Exceptions

### `GameException`

Base exception for all game errors.

### `Checkmate`

Raised when checkmate occurs.

### `Stalemate`

Raised when stalemate occurs.

### `DrawLoss`

Raised when draw/loss by timeline exceeded.

### `InvalidMoveError`

Raised for invalid moves.

**Example**:
```python
from src.exceptions import Checkmate, Stalemate

try:
    game.make_move(state, move)
except Checkmate:
    print("Checkmate!")
except Stalemate:
    print("Stalemate!")
```

---

## Complete Usage Example

```python
from src.game_runner import GameRunner
from config.game_config import GameConfig

# Create game runner
runner = GameRunner(
    max_time=GameConfig.MAX_TIMELINES,
    max_turns=GameConfig.MAX_TURNS,
    mcts_args=GameConfig.get_mcts_config()
)

# Play a game
results = runner.play_ai_vs_ai(max_moves=50, verbose=True)

# Check results
print(f"\nGame Statistics:")
print(f"Winner: {results['winner']}")
print(f"Total Moves: {results['move_count']}")
print(f"Final State: {results['game_string']}")
```

## Advanced Usage

### Custom MCTS Configuration

```python
custom_mcts = {
    'num_searches': 50,  # More searches = stronger play
    'C': 2.0            # Higher C = more exploration
}

runner = GameRunner(mcts_args=custom_mcts)
```

### Manual Move Entry

```python
runner.start_new_game()

while not runner.is_game_over():
    if runner.state.player == 'white':
        # Human move
        move = input("Enter move: ")
        runner.make_move(move)
    else:
        # AI move
        move = runner.make_ai_move()
        print(f"AI plays: {move}")
```

### State Analysis

```python
state = runner.start_new_game()

print(f"Current player: {state.player}")
print(f"Available moves: {len(state.moves)}")
print(f"Board shape: {state.board.shape}")
print(f"Game value: {state.value}")
```
