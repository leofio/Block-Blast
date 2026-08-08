import numpy as np

block_list = [
            # 1x1 block (1)
            np.array([[1]]),
            
            # 2x1 block - both orientations (2)
            np.array([[1, 1]]),
            np.array([[1], 
                      [1]]),
            
            # 3x1 block - both orientations (2)
            np.array([[1, 1, 1]]),
            np.array([[1], 
                      [1], 
                      [1]]),
            
            # 4x1 block - both orientations (2)
            np.array([[1, 1, 1, 1]]),
            np.array([[1], 
                      [1], 
                      [1], 
                      [1]]),
            
            # 5x1 block - both orientations (2)
            np.array([[1, 1, 1, 1, 1]]),
            np.array([[1], 
                      [1], 
                      [1], 
                      [1], 
                      [1]]),
            
            # 2x2 square (1)
            np.array([[1, 1], 
                      [1, 1]]),
            
            # 2x3 block - both orientations (2)
            np.array([[1, 1, 1], 
                      [1, 1, 1]]),
            np.array([[1, 1], 
                      [1, 1], 
                      [1, 1]]),
            
            # 3x3 square (1)
            np.array([[1, 1, 1], 
                      [1, 1, 1], 
                      [1, 1, 1]]),
            
            # 2x3 L shape - all 8 possibilities (8)
            # Fits in 3x2 boxes
            np.array([[1, 0], 
                      [1, 0], 
                      [1, 1]]),
            np.array([[0, 1], 
                      [0, 1], 
                      [1, 1]]),
            np.array([[1, 1], 
                      [1, 0], 
                      [1, 0]]),
            np.array([[1, 1], 
                      [0, 1], 
                      [0, 1]]),
            # Fits in 2x3 boxes
            np.array([[1, 1, 1], 
                      [1, 0, 0]]),
            np.array([[1, 1, 1], 
                      [0, 0, 1]]),
            np.array([[1, 0, 0], 
                      [1, 1, 1]]),
            np.array([[0, 0, 1], 
                      [1, 1, 1]]),
            
            # 3x3 L shape - all 4 possibilities (4)
            np.array([[1, 0, 0], 
                      [1, 0, 0], 
                      [1, 1, 1]]),
            np.array([[0, 0, 1], 
                      [0, 0, 1], 
                      [1, 1, 1]]),
            np.array([[1, 1, 1], 
                      [1, 0, 0], 
                      [1, 0, 0]]),
            np.array([[1, 1, 1], 
                      [0, 0, 1], 
                      [0, 0, 1]]),
            
            # 2x3 T shape - all 4 possibilities (4)
            # Fits in 2x3 boxes
            np.array([[1, 1, 1], 
                      [0, 1, 0]]),
            np.array([[0, 1, 0], 
                      [1, 1, 1]]),
            # Fits in 3x2 boxes
            np.array([[1, 0], 
                      [1, 1], 
                      [1, 0]]),
            np.array([[0, 1], 
                      [1, 1], 
                      [0, 1]]),
            
            # 2x3 Z/S shape - all 4 possibilities (4)
            # Fits in 2x3 boxes
            np.array([[1, 1, 0], 
                      [0, 1, 1]]),
            np.array([[0, 1, 1], 
                      [1, 1, 0]]),
            # Fits in 3x2 boxes
            np.array([[0, 1], 
                      [1, 1], 
                      [1, 0]]),
            np.array([[1, 0], 
                      [1, 1], 
                      [0, 1]])
        ]
