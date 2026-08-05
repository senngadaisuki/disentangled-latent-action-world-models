import torch
import torch.nn as nn

class ActionMLP(nn.Module):
    def __init__(self, num_actions: int = 4, action_dim: int = 32):
        super(ActionMLP, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(num_actions, action_dim),
            nn.SiLU(),
            nn.Linear(action_dim, action_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 2
        z = self.mapping(x)
        return z