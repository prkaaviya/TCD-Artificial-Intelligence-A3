# CS7IS2: Artificial Intelligence - Assignment 3

@author: Kaaviya Paranji Ramkumar

@date: 4.4.2024

This repository contains the source code that implements and compares different AI algorithms for playing Tic Tac Toe and Connect 4 games. The algorithms implemented are:

1. Minimax (with and without alpha-beta pruning)
2. Q-Learning (Reinforcement Learning)

## Project Structure

```
project/
├── games/
│   ├── __init__.py
│   ├── tic_tac_toe.py        # Tic Tac Toe implementation
│   └── connect4.py           # Connect 4 implementation
├── agents/
│   ├── __init__.py
│   ├── default_agent.py      # Default opponent (better than random)
│   ├── minimax_agent.py      # Minimax implementation
│   └── qlearning_agent.py    # Q-learning implementation
├── evaluation/
│   ├── __init__.py
│   └── tournament.py         # Tournament functions to evaluate agents
├── results/                  # Directory for storing results
├── main.py                   # Main entry point
└── README.md                 # This file
```

## Requirements

- Python 3.6 or higher
- NumPy
- Matplotlib
- pandas
- tqdm

Install the required packages:

```bash
pip install numpy matplotlib pandas tqdm
```

## Running the Project

### Tournament Mode

Run a tournament to compare all algorithms against each other:

```bash
# For Tic Tac Toe
python main.py --game tictactoe --mode tournament --num_games 100

# For Connect 4
python main.py --game connect4 --mode tournament --num_games 50
```

### Training Q-Learning Agent

Train the Q-learning agent against the default opponent:

```bash
# For Tic Tac Toe
python main.py --game tictactoe --mode train --train_episodes 10000

# For Connect 4
python main.py --game connect4 --mode train --train_episodes 5000
```

### Playing a Single Game

Play a single game with visualization:

```bash
# For Tic Tac Toe
python main.py --game tictactoe --mode play --render

# For Connect 4
python main.py --game connect4 --mode play --render
```

### Testing Minimax Depth Effect

Test how the search depth affects Minimax performance:

```bash
# For Tic Tac Toe
python main.py --game tictactoe --mode depth_test --num_games 10

# For Connect 4
python main.py --game connect4 --mode depth_test --num_games 5
```

## Tournament with Pre-trained Q-Learning

To run a tournament using a pre-trained Q-learning agent:

```bash
python main.py --game tictactoe --mode tournament --load_qtable --num_games 100
```

## Command Line Arguments

- `--game`: Game to play (`tictactoe` or `connect4`)
- `--mode`: Mode to run (`tournament`, `train`, `play`, or `depth_test`)
- `--num_games`: Number of games to play in tournament or depth_test mode
- `--render`: Render games in play mode or the final game in tournament mode
- `--train_episodes`: Number of episodes for Q-learning training
- `--load_qtable`: Load previously trained Q-table for Q-learning

## Implementation Details

### Games

- **Tic Tac Toe**: Standard 3x3 grid game
- **Connect 4**: 6x7 grid game with gravity mechanics

### Agents

- **Default Agent**: Better-than-random opponent that:
  - Takes winning moves when available
  - Blocks opponent's winning moves
  - Makes random moves otherwise

- **Minimax Agent**:
  - Can be configured with/without alpha-beta pruning
  - Support for depth-limited search
  - Includes heuristic evaluation functions for non-terminal states

- **Q-Learning Agent**:
  - Tabular Q-learning implementation
  - Exploration-exploitation balance with decaying exploration rate
  - State representation optimized for game complexity

### Evaluation

The tournament system collects comprehensive metrics:
- Win/loss/draw statistics
- Game length statistics
- Execution time measurements
- Nodes visited (for Minimax agents)

Results are visualized as bar charts and exported to CSV files for further analysis.

## Notes on Connect 4 Scalability

Due to the large state space of Connect 4, some optimizations are applied:
- Depth-limited search for Minimax (default depth=3)
- Optimized state representation for Q-learning
- Efficient win-checking that only examines lines affected by the last move
