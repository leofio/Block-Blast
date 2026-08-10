"""
Corrected DQN trainer for BlockBlastEnv.

Key fixes vs trainer.py:
  1. Terminal transitions no longer produce NaN targets (-inf * 0).
  2. Valid-action computation is a pure function of (board, hand), not of env.board.
  3. next-state action masks are stored in the replay buffer, not recomputed
     against a stale live environment.
  4. Huber loss + reward scaling + gradient clipping.
  5. Double DQN target.
  6. Learning warmup, target sync by step count, checkpointing, moving-average logs.
"""

import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from main import BlockBlastEnv

NUM_SHAPES = 33
GRID = 8
N_ACTIONS = 3 * GRID * GRID          # 192
REWARD_SCALE = 100.0                  # keeps Q-targets ~O(1)


class BlockBlastDQN(nn.Module):
    def __init__(self, num_shapes=NUM_SHAPES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.hand_fc = nn.Sequential(
            nn.Linear(3 * (num_shapes + 1), 128), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * GRID * GRID + 128, 512), nn.ReLU(),
            nn.Linear(512, N_ACTIONS),
        )

    def forward(self, board, hand):
        return self.fc(torch.cat((self.cnn(board), self.hand_fc(hand)), dim=1))


# ---------------------------------------------------------------- encoding

def encode_board(board):
    """(8,8) int array -> (1,8,8) float32 array."""
    return np.asarray(board, dtype=np.float32)[None, :, :]


def encode_hand(hand, num_shapes=NUM_SHAPES):
    """hand list -> flat one-hot, with an explicit 'empty slot' class."""
    out = np.zeros(3 * (num_shapes + 1), dtype=np.float32)
    for i, shape_idx in enumerate(hand):
        val = num_shapes if shape_idx is None else int(shape_idx)
        out[i * (num_shapes + 1) + val] = 1.0
    return out


# ------------------------------------------------------- action masking

def valid_action_mask(shapes, board, hand):
    """
    Pure function: returns a bool array of length 192 given ONLY the board and
    hand passed in. The original version read env.board, so replayed states were
    validated against whatever the live environment happened to look like.
    """
    board = np.asarray(board)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    for h_idx, shape_idx in enumerate(hand):
        if shape_idx is None:
            continue
        shape = shapes[shape_idx]
        h, w = shape.shape
        for r in range(GRID - h + 1):
            for c in range(GRID - w + 1):
                if np.max(board[r:r + h, c:c + w] + shape) <= 1:
                    mask[h_idx * 64 + r * GRID + c] = True
    return mask


def decode_action(action):
    return action // 64, (action % 64) // GRID, action % GRID


# ------------------------------------------------------------- training

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = BlockBlastEnv(ai_mode=True)
    shapes = env.shapes

    policy_net = BlockBlastDQN().to(device)
    target_net = BlockBlastDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    memory = deque(maxlen=args.buffer)

    eps_start, eps_end = 1.0, 0.05
    decay_steps = max(1, int(args.episodes * 0.7))

    global_step = 0
    returns, lengths = [], []
    best_avg = -float("inf")

    for episode in range(args.episodes):
        board, hand = env.reset()
        mask = valid_action_mask(shapes, board, hand)
        epsilon = max(eps_end, eps_start - (eps_start - eps_end) * episode / decay_steps)

        ep_return, ep_len, done = 0.0, 0, False

        while not done and mask.any():
            board_t = torch.from_numpy(encode_board(board)).unsqueeze(0).to(device)
            hand_t = torch.from_numpy(encode_hand(hand)).unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = int(np.random.choice(np.flatnonzero(mask)))
            else:
                with torch.no_grad():
                    q = policy_net(board_t, hand_t).squeeze(0)
                q[~torch.from_numpy(mask).to(device)] = -float("inf")
                action = int(q.argmax().item())

            h_idx, row, col = decode_action(action)
            (next_board, next_hand), reward, done = env.step(h_idx, row, col)

            # With correct masking an invalid move is impossible. Fail loudly
            # rather than silently absorbing the -10 penalty.
            assert reward != -10, f"masked action {action} was rejected by env"

            next_mask = (np.zeros(N_ACTIONS, dtype=bool) if done
                         else valid_action_mask(shapes, next_board, next_hand))

            memory.append((
                encode_board(board), encode_hand(hand), action,
                reward / REWARD_SCALE,
                encode_board(next_board), encode_hand(next_hand),
                float(done), next_mask,
            ))

            board, hand, mask = next_board, next_hand, next_mask
            ep_return += reward
            ep_len += 1
            global_step += 1

            # ------------------------------------------------ learn
            if len(memory) >= args.warmup and global_step % args.train_every == 0:
                batch = random.sample(memory, args.batch_size)
                (bb, bh, ba, br, nbb, nbh, bd, bnm) = zip(*batch)

                bb_t = torch.from_numpy(np.stack(bb)).to(device)
                bh_t = torch.from_numpy(np.stack(bh)).to(device)
                nbb_t = torch.from_numpy(np.stack(nbb)).to(device)
                nbh_t = torch.from_numpy(np.stack(nbh)).to(device)
                ba_t = torch.tensor(ba, device=device, dtype=torch.int64).unsqueeze(1)
                br_t = torch.tensor(br, device=device, dtype=torch.float32)
                bd_t = torch.tensor(bd, device=device, dtype=torch.float32)
                bnm_t = torch.from_numpy(np.stack(bnm)).to(device)

                q_pred = policy_net(bb_t, bh_t).gather(1, ba_t).squeeze(1)

                with torch.no_grad():
                    # Double DQN: policy net picks the argmax, target net values it.
                    online_next = policy_net(nbb_t, nbh_t).masked_fill(~bnm_t, -1e9)
                    best_a = online_next.argmax(dim=1, keepdim=True)
                    target_next = target_net(nbb_t, nbh_t).gather(1, best_a).squeeze(1)

                    # THE FIX: zero out terminals (and any all-masked rows)
                    # *before* they can be multiplied by (1 - done).
                    usable = (~bd_t.bool()) & bnm_t.any(dim=1)
                    max_next_q = torch.where(usable, target_next,
                                             torch.zeros_like(target_next))
                    target_q = br_t + args.gamma * max_next_q

                loss = nn.SmoothL1Loss()(q_pred, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                assert torch.isfinite(loss), "non-finite loss"

            if global_step % args.target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

        returns.append(ep_return)
        lengths.append(ep_len)

        if episode % args.log_every == 0:
            avg_r = float(np.mean(returns[-args.log_every:]))
            avg_l = float(np.mean(lengths[-args.log_every:]))
            has_nan = any(not torch.isfinite(p).all() for p in policy_net.parameters())
            print(f"Ep {episode:5d} | return(avg{args.log_every}) {avg_r:9.1f} "
                  f"| steps {avg_l:6.1f} | eps {epsilon:.3f} | finite: {not has_nan}")

            if avg_r > best_avg and episode > 0:
                best_avg = avg_r
                torch.save({"episode": episode,
                            "model": policy_net.state_dict(),
                            "avg_return": avg_r}, args.checkpoint)

    torch.save({"episode": args.episodes, "model": policy_net.state_dict()},
               "blockblast_final.pt")
    print(f"Done. Best {args.log_every}-episode average return: {best_avg:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the Block Blast RL agent")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=5_000)
    p.add_argument("--train_every", type=int, default=4)
    p.add_argument("--target_update_steps", type=int, default=2_000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--checkpoint", type=str, default="blockblast_best.pt")
    args = p.parse_args()
    print(f"Starting training with settings: {vars(args)}")
    train(args)
