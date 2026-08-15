"""
Live visualizer for trained Block Blast RL agent.
Renders an in-place updating terminal view of the board, hand, and moves.
"""

import argparse
import sys
import time
import numpy as np
import torch

from main import BlockBlastEnv
from trainer import (
    BlockBlastDQN,
    encode_board,
    encode_hand,
    valid_action_mask,
    decode_action,
)


def format_cell(cell_val):
    """Returns styled string for a single cell."""
    return "██" if cell_val == 1 else " ·"


def render_ui(board, hand, shapes, score, streak, step_num, last_action_info=None):
    """
    Constructs and prints a unified dashboard to the terminal.
    Uses ANSI escape codes to reset the cursor to the top without screen flicker.
    """
    lines = []
    lines.append("=" * 48)
    lines.append("           BLOCK BLAST - AI AGENT LIVE")
    lines.append("=" * 48)
    lines.append(f" Step: {step_num:<6} | Score: {score:<6} | Streak: {streak}")
    
    if last_action_info:
        lines.append(f" Action: Placed hand [{last_action_info['hand_idx']}] at "
                     f"({last_action_info['row']}, {last_action_info['col']}) "
                     f"-> Reward: +{last_action_info['reward']}")
    else:
        lines.append(" Action: Starting game...")
    lines.append("-" * 48)

    # Render Board
    lines.append("     0  1  2  3  4  5  6  7")
    lines.append("   ┌────────────────────────┐")
    for r_idx, row in enumerate(board):
        row_str = " ".join(format_cell(c) for c in row)
        lines.append(f" {r_idx} │ {row_str} │")
    lines.append("   └────────────────────────┘")

    # Render Current Hand
    lines.append("\n Hand Available:")
    hand_representations = []
    for slot_idx, shape_idx in enumerate(hand):
        if shape_idx is None:
            hand_representations.append([f"[{slot_idx}] (Empty)"])
        else:
            shape = shapes[shape_idx]
            rendered_shape = [f"[{slot_idx}]"]
            for row in shape:
                rendered_shape.append(" " + "".join("██" if c == 1 else "  " for c in row))
            hand_representations.append(rendered_shape)

    # Format hand slots side by side
    max_height = max(len(h) for h in hand_representations)
    for row_i in range(max_height):
        row_chunks = []
        for h in hand_representations:
            if row_i < len(h):
                row_chunks.append(f"{h[row_i]:<14}")
            else:
                row_chunks.append(" " * 14)
        lines.append("   " + "   ".join(row_chunks))

    lines.append("=" * 48)

    # \033[H moves cursor to row 1, col 1; \033[J clears down to end of screen
    sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
    sys.stdout.flush()


def run_agent(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Initialize Environment
    env = BlockBlastEnv(ai_mode=True)
    shapes = env.shapes

    # Load Model
    model = BlockBlastDQN().to(device)
    print(f"Loading checkpoint from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Handle checkpoint format saved by trainer.py
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Unrecognized checkpoint format.")
    
    model.eval()

    # Clear terminal once before starting
    sys.stdout.write("\033[2J")
    sys.stdout.flush()

    for game_idx in range(1, args.episodes + 1):
        board, hand = env.reset()
        done = False
        step_count = 0
        last_action = None

        while not done:
            mask = valid_action_mask(shapes, board, hand)
            if not mask.any():
                break

            # Render current state
            render_ui(
                board=board,
                hand=hand,
                shapes=shapes,
                score=env.score,
                streak=env.current_streak,
                step_num=step_count,
                last_action_info=last_action
            )
            time.sleep(args.delay)

            # Model inference
            board_t = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
            hand_t = torch.from_numpy(encode_hand(hand)).unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = model(board_t, hand_t).squeeze(0)
            
            # Mask out illegal moves
            q_values[~torch.from_numpy(mask).to(device)] = -float("inf")
            action = int(q_values.argmax().item())

            h_idx, r, c = decode_action(action)
            (next_board, next_hand), reward, done = env.step(h_idx, r, c)

            last_action = {
                "hand_idx": h_idx,
                "row": r,
                "col": c,
                "reward": int(reward)
            }
            board, hand = next_board, next_hand
            step_count += 1

        # Final Game Over Render
        render_ui(
            board=board,
            hand=hand,
            shapes=shapes,
            score=env.score,
            streak=env.current_streak,
            step_num=step_count,
            last_action_info=last_action
        )
        print(f"\nGame {game_idx}/{args.episodes} finished! Final Score: {env.score} (Total moves: {step_count})")
        if game_idx < args.episodes:
            time.sleep(2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a trained DQN play Block Blast in real time.")
    parser.add_argument("--model_path", type=str, default="blockblast_best.pt",
                        help="Path to the .pt checkpoint (default: blockblast_best.pt)")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="Delay in seconds between steps to control animation speed (default: 0.25)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of consecutive games to watch (default: 5)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    cli_args = parser.parse_args()
    try:
        run_agent(cli_args)
    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")