import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
from collections import deque
import copy
from main import BlockBlastEnv

class BlockBlastDQN(nn.Module):
    def __init__(self, num_shapes=33):
        super(BlockBlastDQN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.hand_fc = nn.Sequential(
            nn.Linear(3 * (num_shapes + 1), 64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 192) # Action space: 3 hand slots * 8 rows * 8 cols = 192
        )

    def forward(self, board, hand):
        x_board = self.cnn(board)
        x_hand = self.hand_fc(hand)
        x = torch.cat((x_board, x_hand), dim=1)
        return self.fc(x)


def encode_state(state, num_shapes=33, device='cpu'):
    """Converts the environment state into PyTorch tensors."""
    board, hand = state
    
    board_t = torch.FloatTensor(board).unsqueeze(0).unsqueeze(0).to(device)
    
    hand_one_hot = np.zeros(3 * (num_shapes + 1))
    for i, shape_idx in enumerate(hand):
        val = num_shapes if shape_idx is None else shape_idx
        hand_one_hot[i * (num_shapes + 1) + val] = 1
        
    hand_t = torch.FloatTensor(hand_one_hot).unsqueeze(0).to(device)
    
    return board_t, hand_t


# --- 3. Action Masking Helper ---
def get_valid_actions(env, state):
    """Returns a list of valid action indices based on the current board and hand."""
    board, hand = state
    valid_actions = []
    
    for h_idx, shape_idx in enumerate(hand):
        if shape_idx is None:
            continue
            
        shape = env.shapes[shape_idx]
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                if env.is_valid_move(shape, r, c):
                    action_idx = h_idx * 64 + r * 8 + c
                    valid_actions.append(action_idx)
                    
    return valid_actions


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    EPISODES = 1000
    GAMMA = 0.99
    BATCH_SIZE = 64
    LR = 0.001
    TARGET_UPDATE = 10
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEP = (EPSILON_START - EPSILON_END) / (EPISODES * 0.8)

    env = BlockBlastEnv(ai_mode=True)
    policy_net = BlockBlastDQN().to(device)
    target_net = BlockBlastDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=10000)
    epsilon = EPSILON_START

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            board_t, hand_t = encode_state(state, device=device)
            valid_actions = get_valid_actions(env, state)
            
            if not valid_actions:
                break

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(board_t, hand_t)
                    
                    mask = torch.full((192,), float('-inf')).to(device)
                    mask[valid_actions] = 0 
                    
                    masked_q_values = q_values.squeeze() + mask
                    action = masked_q_values.argmax().item()

            hand_idx = action // 64
            rem = action % 64
            row = rem // 8
            col = rem % 8

            next_state, reward, done = env.step(hand_idx, row, col)
            total_reward += reward

            memory.append((state, action, reward, next_state, done, valid_actions))
            state = next_state

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_valid_actions_list = zip(*batch)

                b_boards = torch.cat([encode_state(s, device=device)[0] for s in b_states])
                b_hands = torch.cat([encode_state(s, device=device)[1] for s in b_states])

                b_next_boards = torch.cat([encode_state(s, device=device)[0] for s in b_next_states])
                b_next_hands = torch.cat([encode_state(s, device=device)[1] for s in b_next_states])

                b_actions_t = torch.tensor(b_actions, device=device, dtype=torch.int64).unsqueeze(1)
                b_rewards_t = torch.tensor(b_rewards, device=device, dtype=torch.float32)
                b_dones_t = torch.tensor(b_dones, device=device, dtype=torch.float32)

                q_values = policy_net(b_boards, b_hands).gather(1, b_actions_t).squeeze()

                with torch.no_grad():
                    next_q_values = target_net(b_next_boards, b_next_hands)
                    
                    next_masks = torch.full((BATCH_SIZE, 192), float('-inf'), device=device)
                    
                    for i in range(BATCH_SIZE):
                        if not b_dones[i]:
                            next_valid_acts = get_valid_actions(env, b_next_states[i])
                            if next_valid_acts:
                                next_masks[i, next_valid_acts] = 0
                                
                    max_next_q = (next_q_values + next_masks).max(1)[0]
                    
                    target_q = b_rewards_t + (GAMMA * max_next_q * (1 - b_dones_t))

                loss = nn.MSELoss()(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY_STEP)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Block Blast RL Agent")
    
    # Define the bash flags, their types, and default values
    parser.add_argument("--episodes", type=int, default=1000, help="Number of games to play")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of experiences to train on at once")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--target_update", type=int, default=10, help="How often to update the target network")
    
    # Parse the commands and pass them into the train function
    args = parser.parse_args()
    
    print(f"Starting training with settings: {vars(args)}")
    train(args)
