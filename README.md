# Block Blast ML Environment

A headless, state-based Python environment designed for training Reinforcement Learning (RL) agents to master *Block Blast*. 

Currently in the foundational stage, this repository provides a highly optimized, `numpy`-based game engine that simulates the core mechanics of Block Blast. The environment is specifically structured to output mathematical state spaces rather than pixels, making it ideal for deep learning and agent training.

## Core Features

* **Matrix-Based State:** The 8x8 game board is represented as a 2D numpy array (1s for solid blocks, 0s for empty space), easily flattened into a tensor for neural networks.
* **Comprehensive Shape Library:** Fully supports all 33 standard Block Blast shapes and their rotational variants (including irregular structural silhouettes like L, T, and Z shapes).
* **Weighted Shape Generation:** Accurately simulates real gameplay drop rates by assigning a rare 1% probability to smaller pieces (1x1, 1x2, 1x3).
* **3-Shape Hand Mechanic:** Replicates the official game's logic where the agent is dealt a hand of 3 random shapes and must place them all before a refill.
* **Dual-Mode Execution:** Built with a flexible `ai_mode` parameter. Enforces strict game-over termination for agent training to prevent infinite loops, but allows lenient continuation for human CLI testing.
* **Dynamic Reward Signal:** Automatically detects and clears full rows and columns. Calculates rewards based on the exact area of the block placed, the number of distinct squares cleared, and a scaling multiplier for back-to-back streak clears.

## Current Architecture

* `main.py`: The core mathematical environment containing the `BlockBlastEnv` class.
* `block_list.py`: A dedicated configuration file storing the library of 33 standard shape matrices and their structural logic.
* `play.py`: A separate command-line interface (CLI) script for manual human testing.
* `trainer.py`: A Deep Q-Network (DQN) training loop using PyTorch, featuring a dual-stream architecture, vectorized experience replay for GPU acceleration, and dynamic action masking.

## Installation

The environment requires `numpy` for matrix operations and `torch` for the neural network.

```bash
pip install numpy torch

## Usage

### Testing via CLI
Before hooking up an agent, you can play the game in your terminal to ensure the environment mechanics feel correct:

```bash
python play.py
```
**Controls:**
When prompted, enter your move as three numbers separated by spaces: `[hand_index] [row] [col]`
* `hand_index`: 0, 1, or 2 (which shape from your current hand you want to play).
* `row` & `col`: The target coordinate for the **top-left corner** of the shape's bounding box.

*Example:* `0 4 2` (Plays the first shape in your hand at row 4, column 2).

## Training the agent
To start training the RL agent, run the `trainer.py` script. It accepts command-line arguments to easily customize the hyperparameter tuning without editing the code.

```python
python trainer.py --episodes 10000 --batch_size 128 --lr 0.005
```

### Available arguments:

* `--episodes`: Total number of games the AI will play (default: 1000).
* `--lr`: Learning rate for the optimizer (default: 0.001).

* `--batch_size`: Number of experiences to train on at once (default: 64).

* `--gamma`: Discount factor for future rewards (default: 0.99).

* `--target_update`: How often to update the target network (default: 10).

## Integrating custom logic
The environment follows a standard step-based architecture. Ensure `ai_mode` is set to `True` (default) during training so invalid moves terminate the episode.

```python
from main import BlockBlastEnv

# Initialize environment for AI training
env = BlockBlastEnv(ai_mode=True)
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
