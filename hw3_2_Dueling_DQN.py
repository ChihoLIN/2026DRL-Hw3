import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from Gridworld import Gridworld

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

# 2. Build Dueling Neural Network Architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        
        # 共享特徵擷取層
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # 狀態價值分支 (Value Stream)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 動作優勢分支 (Advantage Stream)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

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
sync_target_freq = 10 

online_net = DuelingDQN()
target_net = DuelingDQN()
target_net.load_state_dict(online_net.state_dict()) 
target_net.eval() 

optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
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
                q_values = online_net(state_tensor)
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
        # (C) Store Experience in Buffer
        # ==========================================
        buffer.push(state, action_idx, reward, next_state, done)
        
        # ==========================================
        # (D) Sample and Update Network (D3QN Logic)
        # ==========================================
        if len(buffer) > batch_size:
            b_states, b_actions, b_rewards, b_next_states, b_dones = buffer.sample(batch_size)
            
            b_states = torch.FloatTensor(b_states)
            b_actions = torch.LongTensor(b_actions).unsqueeze(1)
            b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1)
            b_next_states = torch.FloatTensor(b_next_states)
            b_dones = torch.FloatTensor(b_dones).unsqueeze(1)
            
            # Online Network 計算目前的 Q 值
            current_q = online_net(b_states).gather(1, b_actions)
            
            with torch.no_grad():
                # Double DQN: Online 選動作，Target 算價值
                best_actions = online_net(b_next_states).argmax(dim=1, keepdim=True) 
                max_next_q = target_net(b_next_states).gather(1, best_actions)       
                target_q = b_rewards + (gamma * max_next_q * (1 - b_dones))
                
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        state = next_state

    # 衰減 Epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    # 定期同步 Target Network 權重
    if epoch % sync_target_freq == 0:
        target_net.load_state_dict(online_net.state_dict())
        
    if (epoch + 1) % 50 == 0:
        print(f"Epoch: {epoch + 1}/{epochs}, Epsilon: {epsilon:.2f}, Status: {'Win' if status == 2 else 'Lose'}")

print("Training Complete!")

# Plot and save the Loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss (Dueling DDQN with Replay Buffer)")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("loss_dueling_ddqn.png")
plt.show()