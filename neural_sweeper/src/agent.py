from neural_network import MinesweeperDQN
from replay_buffer import ReplayBuffer
from hyperparameters import *

import torch
import torch.optim as optim
import torch.nn as nn
import random

class DQNAgent:
    def __init__(self):
        self.model = MinesweeperDQN()
        self.target_model = MinesweeperDQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = 1.0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = states.unsqueeze(1)
        next_states = next_states.unsqueeze(1)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        max_next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + GAMMA * max_next_q_value * (1 - dones)

        loss = nn.MSELoss()(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)