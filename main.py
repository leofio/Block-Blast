import numpy as np
from block_list import block_list

class BlockBlastEnv:
    def __init__(self, ai_mode=True):
        self.ai_mode = ai_mode
        self.grid_size = 8
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self.current_streak = 0
        
        self.shapes = block_list

        self.hand = [None, None, None]

        self.shape_weights = np.zeros(len(self.shapes))
        
        rare_indices = [0, 1, 2, 3, 4] 
        
        self.shape_weights[rare_indices] = 0.01
        
        remaining_prob = 1.0 - np.sum(self.shape_weights)
        normal_indices = [i for i in range(len(self.shapes)) if i not in rare_indices]
        
        self.shape_weights[normal_indices] = remaining_prob / len(normal_indices)

    def reset(self):
        """Resets the board for a new game episode."""
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.score = 0
        self._refill_hand()
        
        return self.board.copy(), self.hand.copy()

    def _can_place_all(self, board_state, remaining_hand):
        """
        Recursively checks if there is at least one valid sequence to place 
        all shapes in the remaining_hand on the given board_state.
        """
        if not remaining_hand:
            return True
            
        for i, shape_idx in enumerate(remaining_hand):
            shape = self.shapes[shape_idx]
            h, w = shape.shape
            
            for r in range(self.grid_size - h + 1):
                for c in range(self.grid_size - w + 1):
                    if np.max(board_state[r:r+h, c:c+w] + shape) <= 1:
                        
                        new_board = board_state.copy()
                        new_board[r:r+h, c:c+w] += shape
                        
                        full_rows = np.where(new_board.sum(axis=1) == self.grid_size)[0]
                        full_cols = np.where(new_board.sum(axis=0) == self.grid_size)[0]
                        for fr in full_rows:
                            new_board[fr, :] = 0
                        for fc in full_cols:
                            new_board[:, fc] = 0
                            
                        next_hand = remaining_hand[:i] + remaining_hand[i+1:]
                        if self._can_place_all(new_board, next_hand):
                            return True
                            
        return False

    def _refill_hand(self):
        """Fills the 3 slots with random shape indices, ensuring they can all be placed."""
        max_attempts = 50 
        
        for attempt in range(max_attempts):
            chosen_indices = np.random.choice(
                len(self.shapes), 
                size=3, 
                replace=True, 
                p=self.shape_weights
            ).tolist()
            
            if np.sum(self.board) == 0:
                self.hand = chosen_indices
                return
                
            if self._can_place_all(self.board, chosen_indices):
                self.hand = chosen_indices
                return
                
        # If we couldn't find a playable hand after max_attempts, 
        # the board is effectively dead. Deal the last generated hand 
        # and let the standard game-over logic handle the termination.
        self.hand = chosen_indices

    def is_valid_move(self, shape, row, col):
        """Checks if a shape can be placed at the given row/col."""
        h, w = shape.shape
        
        if row + h > self.grid_size or col + w > self.grid_size:
            return False
            
        board_slice = self.board[row:row+h, col:col+w]
        return np.max(board_slice + shape) <= 1

    def step(self, hand_index, row, col):
        if hand_index not in [0, 1, 2] or self.hand[hand_index] is None:
            return (self.board.copy(), self.hand.copy()), -10, self.ai_mode 

        shape_idx = self.hand[hand_index]
        shape = self.shapes[shape_idx]
        
        if not self.is_valid_move(shape, row, col):
            return (self.board.copy(), self.hand.copy()), -10, self.ai_mode 

        h, w = shape.shape
        self.board[row:row+h, col:col+w] += shape
        area_of_block = np.sum(shape)

        self.hand[hand_index] = None

        no_lines_cleared, blocks_cleared = self._clear_lines()
        multiplier = 1 + self.current_streak + no_lines_cleared
        reward = (area_of_block + (10 * blocks_cleared)) * multiplier 
        self.score += reward

        if no_lines_cleared > 0:
            self.current_streak += 1
        else:
            self.current_streak = 0

        if all(s is None for s in self.hand):
            self._refill_hand()

        done = self._check_game_over()

        return (self.board.copy(), self.hand.copy()), reward, done
    
    def _clear_lines(self):
        """Finds full rows/columns, clears them, and returns (lines_cleared, blocks_cleared)."""
        full_rows = np.where(self.board.sum(axis=1) == self.grid_size)[0]
        full_cols = np.where(self.board.sum(axis=0) == self.grid_size)[0]

        r_count = len(full_rows)
        c_count = len(full_cols)
        no_lines_cleared = r_count + c_count
        
        blocks_cleared = (r_count * self.grid_size) + (c_count * self.grid_size) - (r_count * c_count)

        for r in full_rows:
            self.board[r, :] = 0
        for c in full_cols:
            self.board[:, c] = 0

        return no_lines_cleared, blocks_cleared

    def _check_game_over(self):
        """Returns True if NONE of the remaining shapes can be placed."""
        for shape_idx in self.hand:
            if shape_idx is not None:
                shape = self.shapes[shape_idx]
                h, w = shape.shape
                
                can_place_this_shape = False
                for r in range(self.grid_size - h + 1):
                    for c in range(self.grid_size - w + 1):
                        if self.is_valid_move(shape, r, c):
                            can_place_this_shape = True
                            break
                    if can_place_this_shape:
                        break
                
                if can_place_this_shape:
                    return False 
                    
        return True
