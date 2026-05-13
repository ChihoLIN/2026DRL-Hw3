import os
# 設定為空字串，讓 CUDA 看不到任何設備
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

# 接下來才是原本的 import
import torch
import torch.nn as nn
import pytorch_lightning as pl
# ... (你的其他 import)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from Gridworld import Gridworld

# ==========================================
# 1. Replay Buffer & Dataset
# ==========================================
class RLDataset(Dataset):
    """ 用於將 Replay Buffer 轉換為 PyTorch DataLoader 可讀取的格式 """
    def __init__(self, buffer, batch_size):
        self.buffer = buffer
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size # 每次訓練迭代抽樣的大小

    def __getitem__(self, item):
        state, action, reward, next_state, done = self.buffer.sample_1()
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.FloatTensor([done])
        )

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_1(self):
        return random.sample(self.buffer, 1)[0]

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 2. Dueling DQN Network Architecture
# ==========================================
class DuelingDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

# ==========================================
# 3. PyTorch Lightning Module (D3QN)
# ==========================================
class D3QNLightning(pl.LightningModule):
    def __init__(self, env, lr=1e-3, batch_size=32):
        super().__init__()
        self.env = env
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        
        # 建立 Online 與 Target 網路
        self.model = DuelingDQN()
        self.target_model = DuelingDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.buffer = ReplayBuffer(capacity=5000)
        self.populate(steps=100) # 預填一些隨機資料

    def populate(self, steps=100):
        """ 訓練前先讓 Agent 隨機跑幾步存入 Buffer """
        self.env.initGridRand()
        state = self.env.board.render_np().flatten().astype(np.float32)
        for _ in range(steps):
            action_idx = random.randint(0, 3)
            self.env.makeMove({0:'u', 1:'d', 2:'l', 3:'r'}[action_idx])
            reward = self.env.reward()
            next_state = self.env.board.render_np().flatten().astype(np.float32)
            done = 1 if reward == 1 or reward == -1 else 0
            self.buffer.push(state, action_idx, reward, next_state, done)
            state = next_state if not done else self.env.board.render_np().flatten().astype(np.float32)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        
        # 1. 目前 Q 值
        current_q = self.model(states).gather(1, actions)
        
        # 2. Target Q 值 (Double DQN 邏輯)
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_model(next_states).gather(1, next_actions)
            expected_q = rewards + (self.gamma * max_next_q * (1 - dones))
            
        loss = nn.MSELoss()(current_q, expected_q)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # 每個 Epoch 結束更新超參數與同步網路
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_model.load_state_dict(self.model.state_dict())

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # 【Bonus Tip: Learning Rate Scheduler】
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # 這裡定義如何從 Buffer 餵資料給 training_step
        dataset = RLDataset(self.buffer, batch_size=self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size)

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # 初始化環境 (Random Mode)
    grid_env = Gridworld(size=4, mode='random')
    
    # 初始化模型
    d3qn_model = D3QNLightning(env=grid_env)

    # 初始化 Trainer (包含 Bonus: Gradient Clipping)
    trainer = pl.Trainer(
        max_epochs=500,
        gradient_clip_val=1.0,  # 【Bonus Tip: Gradient Clipping】
        accelerator="cpu",     # 自動偵測 GPU/CPU
        devices=1,
        log_every_n_steps=10
    )

    # 開始訓練
    trainer.fit(d3qn_model)
    print("HW3-3 Training Complete with PyTorch Lightning!")