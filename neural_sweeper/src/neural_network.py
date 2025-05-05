import torch.nn as nn
from hyperparameters import *

class MinesweeperDQN(nn.Module):
    def __init__(self):
        super().__init__()

        # Identify Features of Board
        self.convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Neural Deep Q 
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 64 * NUM_ACTIONS, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_ACTIONS)
        )

    def forward(self, x):
        x = self.convolution(x)
        return self.fully_connected(x)