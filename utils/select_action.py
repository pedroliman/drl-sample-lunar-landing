import torch

def select_action(q_network: nn.Module, state: torch.Tensor) -> int:
    """
    Select an action based on the Q-network's Q-values for the given state.

    Parameters:
    - q_network (nn.Module): The Q-network model.
    - state (torch.Tensor): The current state.

    Returns:
    - int: The selected action.
    """
    q_values = q_network(state)
    action = torch.argmax(q_values).item()
    print(f"Action selected: {action}, with q-value {q_values[action]:.2f}")
    return action
