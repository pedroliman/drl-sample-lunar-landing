import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the Q-network.

        Parameters:
        - state_size (int): Size of the state space.
        - action_size (int): Size of the action space.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        - state (torch.Tensor): The input state.

        Returns:
        - torch.Tensor: The Q-values for the given state.
        """
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
