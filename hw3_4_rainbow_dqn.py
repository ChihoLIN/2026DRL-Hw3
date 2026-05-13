import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque
import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset
from Gridworld import Gridworld

# 1. Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. 定義可學習的參數 (Mu 和 Sigma)
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        
        self.reset_parameters(std_init)

    def reset_parameters(self, std_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            # 2. 在訓練時產生雜訊 (不需要 register_buffer)
            # 使用 torch.no_grad() 確保雜訊生成過程不參與梯度追蹤
            with torch.no_grad():
                epsilon_in = self._scale_noise(self.in_features)
                epsilon_out = self._scale_noise(self.out_features)
                weight_epsilon = epsilon_out.ger(epsilon_in)
                bias_epsilon = self._scale_noise(self.out_features)
            
            # 將雜訊注入權重
            weight = self.weight_mu + self.weight_sigma.mul(weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(bias_epsilon)
        else:
            # 推論時只使用 Mu
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

# 2. Dueling Architecture
class RainbowDQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(RainbowDQN, self).__init__()
        self.feature = nn.Sequential(NoisyLinear(input_dim, 128), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, 1))
        self.advantage = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, output_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))

# 3. Lightning Module
class RainbowLightning(pl.LightningModule):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.model = RainbowDQN()
        self.target_model = RainbowDQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.buffer = deque(maxlen=5000)
        self.gamma = 0.99
        self.batch_size = 32

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        current_q = self.model(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            max_next_q = self.target_model(next_states).gather(1, next_actions)
            expected_q = rewards + (self.gamma * max_next_q * (1 - dones))
            
        loss = nn.MSELoss()(current_q, expected_q)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        # 簡易產生資料供給訓練
        class SimpleDataset(IterableDataset):
            def __init__(self, env, buffer):
                self.env = env
                self.buffer = buffer
                self.env.initGridRand()
            def __iter__(self):
                state = self.env.board.render_np().flatten().astype(np.float32)
                while True:
                    action_idx = random.randint(0, 3)
                    self.env.makeMove({0:'u',1:'d',2:'l',3:'r'}[action_idx])
                    reward = self.env.reward()
                    next_state = self.env.board.render_np().flatten().astype(np.float32)
                    done = 1 if reward != 0 else 0
                    yield (torch.FloatTensor(state), torch.LongTensor([action_idx]), 
                           torch.FloatTensor([reward]), torch.FloatTensor(next_state), 
                           torch.FloatTensor([done]))
                    state = next_state if not done else self.env.board.render_np().flatten().astype(np.float32)
        
        return DataLoader(SimpleDataset(self.env, self.buffer), batch_size=32)

# 4. Main
if __name__ == "__main__":
    env = Gridworld(size=4, mode='random')
    model = RainbowLightning(env=env)
    
    print("開始 Rainbow DQN 訓練...")
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=1000, accelerator="cpu") # limit_train_batches 控制訓練長度
    trainer.fit(model)
    print("Rainbow DQN 訓練完成！")