# 5D Chess MCTS Architecture

## Overview

This project implements a 5D Chess game with an AI player using Monte Carlo Tree Search (MCTS). The architecture is designed with modularity, testability, and performance in mind.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Game Runner                          │
│  (Orchestrates game flow and player interactions)       │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────┐      ┌──────────┐
│  MCTS   │      │ Chess5D  │
│Algorithm│◄─────┤  Engine  │
└────┬────┘      └─────┬────┘
     │                 │
     ▼                 ▼
┌─────────┐      ┌───────────┐
│  Node   │      │ChessState │
│  Tree   │      │  Manager  │
└─────────┘      └───────────┘
```

## Core Components

### 1. Chess State (`src/chess_state.py`)

**Purpose**: Represents the complete state of a 5D chess game at any point in time.

**Key Features**:
- Immutable state copies for tree search
- Board representation in multiple formats
- Player tracking and move history
- Terminal state detection

**Methods**:
- `copy()`: Deep copy for tree search
- `is_game_over()`: Terminal state check
- `get_winner()`: Winner determination

### 2. Chess Engine (`src/chess_engine.py`)

**Purpose**: Core game logic and move generation for 5D chess.

**Key Features**:
- Multi-timeline and multi-turn support
- Move validation and execution
- Board tensor representation for neural networks
- Timeline/turn boundary checking

**Methods**:
- `get_initial_state()`: Initialize game
- `make_move()`: Execute moves
- `convert_moves()`: Legal move generation
- `raw_board_to_tensor()`: Board encoding

**Performance Optimizations**:
- CuPy GPU acceleration for tensor operations
- Efficient move filtering
- Vectorized board operations

### 3. MCTS Algorithm (`src/mcts.py`)

**Purpose**: Monte Carlo Tree Search implementation for move selection.

**Key Classes**:

#### Node
- Tree node representing game state
- UCB1 value calculation
- Visit count and value tracking
- Child node management

#### MCTS
- Main search algorithm
- Selection, expansion, simulation, backpropagation
- Action probability distribution generation

**Algorithm Flow**:
1. **Selection**: Traverse tree using UCB1
2. **Expansion**: Add new child node
3. **Simulation**: Random playout to terminal state
4. **Backpropagation**: Update statistics up the tree

**Parameters**:
- `num_searches`: Number of MCTS iterations
- `C`: Exploration constant (typically √2 ≈ 1.41)

### 4. Game Runner (`src/game_runner.py`)

**Purpose**: High-level game orchestration and player interface.

**Key Features**:
- Game lifecycle management
- AI vs AI gameplay
- Move validation and execution
- Game statistics tracking

**Methods**:
- `start_new_game()`: Initialize new game
- `make_ai_move()`: AI move generation
- `play_ai_vs_ai()`: Full game automation

### 5. Configuration (`config/game_config.py`)

**Purpose**: Centralized configuration management.

**Configurations**:
- `GameConfig`: Production settings
- `TestConfig`: Testing settings

**Settings**:
- Board dimensions (timelines, turns)
- MCTS parameters
- Game values and constants

### 6. Exceptions (`src/exceptions.py`)

**Purpose**: Custom exception hierarchy for game events.

**Exception Types**:
- `GameException`: Base exception
- `Checkmate`: Checkmate condition
- `Stalemate`: Stalemate condition
- `DrawLoss`: Draw/loss by timeline exceeded
- `InvalidMoveError`: Invalid move attempt

## Data Flow

### Game Initialization
```
GameRunner → Chess5D.get_initial_state() → ChessState
```

### Move Selection
```
GameRunner → MCTS.get_best_move()
          → MCTS.search() (runs iterations)
          → Node.select/expand/simulate/backpropagate
          → Returns best move
```

### Move Execution
```
GameRunner → Chess5D.make_move()
          → Update ChessState
          → Check terminal conditions
```

## Performance Considerations

### GPU Acceleration
- CuPy used for all tensor operations
- Batch processing of move probabilities
- Vectorized board representations

### Memory Management
- Deep copying only when necessary
- Reuse of probability tensors
- Efficient tree node allocation

### Optimization Techniques
1. **Move Filtering**: Remove invalid moves early
2. **Tensor Operations**: Batch probability calculations
3. **Tree Reuse**: Reuse subtrees between moves
4. **Lazy Evaluation**: Compute values only when needed

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock dependencies
- Edge case coverage

### Integration Tests
- End-to-end game workflows
- AI vs AI games
- Configuration variations

### Test Coverage
- Aiming for >80% code coverage
- Critical paths 100% covered
- Edge cases thoroughly tested

## Design Patterns

### Strategy Pattern
- Different MCTS configurations
- Pluggable evaluation functions

### Template Method
- MCTS algorithm structure
- Game state management

### Factory Pattern
- State creation and copying
- Node instantiation

### Observer Pattern
- Game event tracking
- Move history recording

## Extension Points

### Future Enhancements
1. **Neural Network Integration**: Replace random simulations
2. **Opening Book**: Fast early game moves
3. **Endgame Tablebase**: Perfect endgame play
4. **Parallel MCTS**: Multi-threaded tree search
5. **Reinforcement Learning**: Self-play training

### Configuration Extensions
- Custom piece rules
- Variable board sizes
- Different time/turn limits
- Alternative victory conditions

## Dependencies

### Core
- `cupy`: GPU-accelerated array operations
- `javascript`: Bridge to 5d-chess-js library
- `jsrun`: JavaScript runtime integration

### Testing
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting

### Optional
- `torch`: Neural network support
- `numpy`: CPU fallback operations

## File Organization

```
chess-5d-mcts/
├── src/
│   ├── __init__.py
│   ├── chess_engine.py      # Core game logic
│   ├── chess_state.py       # State management
│   ├── mcts.py              # MCTS algorithm
│   ├── game_runner.py       # Game orchestration
│   └── exceptions.py        # Custom exceptions
├── config/
│   ├── __init__.py
│   └── game_config.py       # Configuration
├── tests/
│   ├── unit/                # Unit tests
│   │   ├── test_chess_engine.py
│   │   ├── test_chess_state.py
│   │   ├── test_mcts.py
│   │   └── test_exceptions.py
│   └── integration/         # Integration tests
│       └── test_game_workflow.py
├── docs/
│   ├── ARCHITECTURE.md      # This file
│   └── API.md              # API documentation
├── requirements.txt
├── setup.py
└── pytest.ini
```

## Code Quality

### Type Hints
- Full type annotation coverage
- MyPy static type checking
- Clear interfaces

### Documentation
- Comprehensive docstrings
- Usage examples
- Architecture documentation

### Testing
- Unit test coverage >80%
- Integration tests for workflows
- Performance benchmarks

## Best Practices

1. **Separation of Concerns**: Each module has single responsibility
2. **Immutability**: States are copied, not modified
3. **Type Safety**: Full type hints throughout
4. **Documentation**: Clear docstrings and comments
5. **Testing**: Comprehensive test coverage
6. **Performance**: GPU acceleration where beneficial
7. **Extensibility**: Easy to add new features

## Conclusion

This architecture provides a solid foundation for 5D Chess with MCTS AI, balancing performance, maintainability, and extensibility. The modular design allows for easy testing and future enhancements while maintaining clean separation of concerns.
