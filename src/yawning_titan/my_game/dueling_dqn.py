import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, enable_dueling_dqn: bool = True):
        super().__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)

            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim)
            self.advantages = nn.Linear(hidden_dim, action_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + (A - A.mean(dim=1, keepdim=True))
        else:
            x = F.relu(self.fc2(x))
            Q = self.output(x)
        return Q