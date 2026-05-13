import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from Gridworld import Gridworld

# ==========================================
# 🌟 Assignment Toggle: Switch your DQN mode here 🌟
# ==========================================
USE_BUFFER = True  # Set to False for Naive DQN; True for Experience Replay Buffer
# ==========================================

# 1. Create Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 2. Build DQN Neural Network Architecture
class NaiveDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(NaiveDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# 3. Training Hyperparameters & Environment Setup
env = Gridworld(size=4, mode='static')
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

epochs = 500
max_steps = 50
batch_size = 32
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99
learning_rate = 1e-3

model = NaiveDQN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
buffer = ReplayBuffer(capacity=1000)

# 4. Main Training Loop
losses = []

for epoch in range(epochs):
    env.initGridStatic()
    state = env.board.render_np().flatten().astype(np.float32)
    
    status = 1 
    step = 0
    
    while status == 1 and step < max_steps:
        step += 1
        
        # (A) Select Action (Epsilon-Greedy)
        if random.random() < epsilon:
            action_idx = random.randint(0, 3)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
        action_str = action_set[action_idx]
        
        # (B) Interact with Environment
        env.makeMove(action_str)
        reward = env.reward()
        next_state = env.board.render_np().flatten().astype(np.float32)
        
        done = 0
        if reward == 1:
            status = 2
            done = 1
        elif reward == -1:
            status = 0
            done = 1
            
        # ==========================================
        # (C & D) Update Neural Network based on the toggle
        # ==========================================
        if not USE_BUFFER:
            # [Mode 1: Naive DQN]
            # Do not use Buffer. Convert the "current step" experience into Tensors and update the network immediately.
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            a_tensor = torch.LongTensor([action_idx]).unsqueeze(1)
            r_tensor = torch.FloatTensor([reward]).unsqueeze(1)
            ns_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            d_tensor = torch.FloatTensor([done]).unsqueeze(1)

            current_q = model(s_tensor).gather(1, a_tensor)
            with torch.no_grad():
                max_next_q = model(ns_tensor).max(1)[0].unsqueeze(1)
                target_q = r_tensor + (gamma * max_next_q * (1 - d_tensor))

            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        else:
            # [Mode 2: Experience Replay Buffer]
            # Push experience to Buffer first. Once enough data is collected, sample a random mini-batch for updates.
            buffer.push(state, action_idx, reward, next_state, done)
            
            if len(buffer) > batch_size:
                b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)
                
                b_states = torch.FloatTensor(b_states)
                b_actions = torch.LongTensor(b_actions).unsqueeze(1)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1)
                b_next_states = torch.FloatTensor(b_next_states)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1)
                
                current_q = model(b_states).gather(1, b_actions)
                with torch.no_grad():
                    max_next_q = model(b_next_states).max(1)[0].unsqueeze(1)
                    target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))
                    
                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
        # ==========================================
        
        state = next_state

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    if (epoch + 1) % 50 == 0:
        print(f"Epoch: {epoch + 1}/{epochs}, Epsilon: {epsilon:.2f}, Status: {'Win' if status == 2 else 'Lose'}")

print("Training Complete!")

# Plot and save the Loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
mode_name = "Buffer" if USE_BUFFER else "Naive"
plt.title(f"DQN Training Loss ({mode_name} Mode)")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig(f"dqn_loss_{mode_name.lower()}.png")
plt.show()