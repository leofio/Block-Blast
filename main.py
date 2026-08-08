import numpy as np

class BlockBlastEnv:
    def __init__(self):
        self.grid_size = 8
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        
        # A subset of standard Block Blast shapes (represented as 2D arrays)
        # 1 means solid block, 0 means empty space in the bounding box
        self.shapes = [
            np.array([[1]]),                    # 1x1 Dot
            np.array([[1, 1], [1, 1]]),         # 2x2 Square
            np.array([[1, 1, 1]]),              # 3x1 Horizontal line
            np.array([[1], [1], [1]]),          # 1x3 Vertical line
            np.array([[1, 0], [1, 1]])          # Small L-shape
            # TODO: Add the rest of the 21+ structural silhouettes here
        ]

        self.hand = [None, None, None]

    def reset(self):
        """Resets the board for a new game episode."""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self._refill_hand() # Fill the hand on reset
        
        # Make sure it returns BOTH the board and hand as a tuple!
        return self.board.copy(), self.hand.copy()

    def _refill_hand(self):
        """Fills the 3 slots with random shape indices."""
        # We store the indices of the shapes, not the arrays themselves, 
        # so it's easier to pass as a state to the ML agent later.
        self.hand = [np.random.randint(len(self.shapes)) for _ in range(3)]

    def is_valid_move(self, shape, row, col):
        """Checks if a shape can be placed at the given row/col."""
        h, w = shape.shape
        
        # Check out of bounds
        if row + h > self.grid_size or col + w > self.grid_size:
            return False
            
        # Check overlaps (if the sum of the board slice and shape has a value > 1, they overlap)
        board_slice = self.board[row:row+h, col:col+w]
        return np.max(board_slice + shape) <= 1

    def step(self, hand_index, row, col):
        # 1. Validate the chosen hand slot
        if hand_index not in [0, 1, 2] or self.hand[hand_index] is None:
            # Agent tried to play a shape that doesn't exist or is already played
            return self.board.copy(), -10, True 

        shape_idx = self.hand[hand_index]
        shape = self.shapes[shape_idx]

        # 2. Check if the placement is valid
        if not self.is_valid_move(shape, row, col):
            return self.board.copy(), -10, True 

        # 3. Place the shape
        h, w = shape.shape
        self.board[row:row+h, col:col+w] += shape

        # 4. Remove the shape from the hand (set slot to None)
        self.hand[hand_index] = None

        # 5. Clear lines & score
        lines_cleared = self._clear_lines()
        reward = 1 + (lines_cleared ** 2 * 10)  
        self.score += reward

        # 6. Refill hand if completely empty
        if all(s is None for s in self.hand):
            self._refill_hand()

        # 7. Check if game is over
        done = self._check_game_over()

        # We now return the hand alongside the board as part of the state
        return (self.board.copy(), self.hand.copy()), reward, done

    def _clear_lines(self):
        """Finds full rows/columns, clears them, and returns the count."""
        # Find which rows and columns sum to 8 (completely full)
        full_rows = np.where(self.board.sum(axis=1) == self.grid_size)[0]
        full_cols = np.where(self.board.sum(axis=0) == self.grid_size)[0]

        # Clear them by setting those indices to 0
        for r in full_rows:
            self.board[r, :] = 0
        for c in full_cols:
            self.board[:, c] = 0

        return len(full_rows) + len(full_cols)

    def _check_game_over(self):
        """Returns True if NONE of the remaining shapes can be placed."""
        for shape_idx in self.hand:
            if shape_idx is not None:
                shape = self.shapes[shape_idx]
                h, w = shape.shape
                
                # Scan the entire board for a valid placement for this shape
                can_place_this_shape = False
                for r in range(self.grid_size - h + 1):
                    for c in range(self.grid_size - w + 1):
                        if self.is_valid_move(shape, r, c):
                            can_place_this_shape = True
                            break # Found a spot, stop checking columns
                    if can_place_this_shape:
                        break # Found a spot, stop checking rows
                
                # If even ONE shape in the hand can be placed, the game continues
                if can_place_this_shape:
                    return False 
                    
        # If we checked every remaining shape and found 0 valid spots
        return True

def print_board(board):
    print("\nBoard:")
    # Print column numbers
    print("   " + " ".join([str(i) for i in range(len(board))]))
    for r, row in enumerate(board):
        # Print row number and then the grid cells
        row_str = " ".join(["██" if cell == 1 else " ." for cell in row])
        print(f"{r}  {row_str}")

def print_hand(env):
    print("\nHand:")
    for i, shape_idx in enumerate(env.hand):
        if shape_idx is None:
            print(f"[{i}]: (Empty)")
        else:
            print(f"[{i}]:")
            shape = env.shapes[shape_idx]
            for row in shape:
                # Use solid blocks for 1s and empty spaces for 0s
                print("     " + "".join(["██" if cell == 1 else "  " for cell in row]))

def play():
    env = BlockBlastEnv()
    state = env.reset()
    board = state[0]
    
    print("Welcome to CLI Block Blast!")
    
    while True:
        print_board(board)
        print_hand(env)
        print(f"\nScore: {env.score}")
        
        try:
            user_input = input("Enter 'hand_index row col' (e.g., '0 3 4') or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                print("Quitting game.")
                break
                
            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input! Please enter exactly three numbers.")
                continue
                
            hand_idx, row, col = map(int, parts)
            
            # Take a step in the environment
            state, reward, done = env.step(hand_idx, row, col)
            board = state[0]
            
            if reward == -10:
                print("\n*** INVALID MOVE! You tried to overlap or went out of bounds. ***")
                print(f"Final Score: {env.score}")
                break
                
            if done:
                print("\n*** GAME OVER! No more valid moves. ***")
                print_board(board)
                print(f"Final Score: {env.score}")
                break
                
        except ValueError:
            print("Invalid format! Use numbers separated by spaces (e.g., 1 5 2).")

if __name__ == "__main__":
    play()
