# Block Blast ML Environment

A headless, state-based Python environment designed for training Reinforcement Learning (RL) agents to master *Block Blast*. 

Currently in the foundational stage, this repository provides a highly optimized, `numpy`-based game engine that simulates the core mechanics of Block Blast. The environment is specifically structured to output mathematical state spaces rather than pixels, making it ideal for deep learning and agent training.

## Core Features

* **Matrix-Based State:** The 8x8 game board is represented as a 2D numpy array (1s for solid blocks, 0s for empty space), easily flattened into a tensor for neural networks.
* **Complex Shape Logic:** Fully supports irregular structural silhouettes (like L-shapes) using transparent bounding box matrices.
* **3-Shape Hand Mechanic:** Accurately replicates the official game's logic where the agent is dealt a hand of 3 random shapes and must place them all before a refill.
* **Line Clearing & Reward Signal:** Automatically detects and clears full rows and columns, calculating a structured reward signal (exponential scaling for combo clears) to guide agent behavior.
* **CLI Test Mode:** Includes a fully playable command-line interface to manually verify gravity, overlaps, and game-over states.

## Current Architecture

* `main.py`: The core environment script containing the `BlockBlastEnv` class and the CLI game loop.

## Installation

The environment requires `numpy` for matrix operations.

```bash
pip install numpy
```

## Usage

### Testing via CLI
Before hooking up an agent, you can play the game in your terminal to ensure the environment mechanics feel correct:

```bash
python main.py
```
**Controls:**
When prompted, enter your move as three numbers separated by spaces: `[hand_index] [row] [col]`
* `hand_index`: 0, 1, or 2 (which shape from your current hand you want to play).
* `row` & `col`: The target coordinate for the **top-left corner** of the shape's bounding box.

*Example:* `0 4 2` (Plays the first shape in your hand at row 4, column 2).

### Integrating with an ML Agent
The environment follows a standard step-based architecture:

```python
from main import BlockBlastEnv

env = BlockBlastEnv()
state = env.reset() 
# state is a tuple: (board_matrix, hand_list)

# Take an action: play shape 0 at row 4, col 2
next_state, reward, done = env.step(hand_index=0, row=4, col=2)
```

## Roadmap

- [ ] Map out and add the remaining standard Block Blast shapes to the `self.shapes` dictionary.
- [ ] Refactor state representation to flatten the board and hand into a single 1D tensor.
- [ ] Wrap `BlockBlastEnv` into the standard Gymnasium (`gym`) API for compatibility with major RL libraries.
- [ ] Implement a baseline RL agent (e.g., DQN) to begin the training loop.
