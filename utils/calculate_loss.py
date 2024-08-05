import torch
import torch.nn as nn

def calculate_loss(
    q_network: nn.Module,
    state: torch.Tensor,
    action: int,
    next_state: torch.Tensor,
    reward: float,
    done: bool,
    gamma: float
) -> torch.Tensor:
    """
    Calculate the loss for the Q-network using the Bellman equation.

    Parameters:
    - q_network (nn.Module): The Q-network model.
    - state (torch.Tensor): The current state.
    - action (int): The action taken.
    - next_state (torch.Tensor): The next state.
    - reward (float): The reward received.
    - done (bool): Whether the episode is done.
    - gamma (float): The discount factor for future rewards.

    Returns:
    - torch.Tensor: The calculated loss.
    """
    q_values = q_network(state)
    current_state_q_value = q_values[action]
    next_state_q_value = q_network(next_state).max()
    target_q_value = reward + gamma * next_state_q_value * (1 - done)
    loss = nn.MSELoss()(current_state_q_value, target_q_value)
    return loss
