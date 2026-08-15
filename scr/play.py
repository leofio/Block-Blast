from main import BlockBlastEnv

def print_board(board):
    print("\nBoard:")
    print("   " + " ".join([str(i) for i in range(len(board))]))
    for r, row in enumerate(board):
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
                print("     " + "".join(["██" if cell == 1 else "  " for cell in row]))

def play():
    env = BlockBlastEnv(ai_mode=False)
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
            
            state, reward, done = env.step(hand_idx, row, col)
            board = state[0]
            
            if reward == -10:
                print("\n*** INVALID MOVE! You tried to overlap or went out of bounds. ***")
                continue
                
            if done:
                print("\n*** GAME OVER! No more valid moves. ***")
                print_board(board)
                print(f"Final Score: {env.score}")
                break
                
        except ValueError:
            print("Invalid format! Use numbers separated by spaces (e.g., 1 5 2).")

if __name__ == "__main__":
    play()
