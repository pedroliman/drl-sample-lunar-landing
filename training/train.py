import torch
import torch.optim as optim
from models.q_network import QNetwork
from utils.describe_episode import describe_episode
from utils.select_action import select_action
from utils.calculate_loss import calculate_loss
from environments.env_wrapper import initialize_environment
from config import STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, EPISODES

def train(state_size: int, action_size: int, learning_rate: float, gamma: float, episodes: int) -> None:
    """
    Main training loop for the DQN algorithm.

    Parameters:
    - state_size (int): The size of the state space.
    - action_size (int): The size of the action space.
    - learning_rate (float): The learning rate for the optimizer.
    - gamma (float): The discount factor for future rewards.
    - episodes (int): The number of training episodes.
    """
    env = initialize_environment()
    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), learning_rate)

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done:
            step += 1
            state = torch.tensor(state, dtype=torch.float32)
            action = select_action(q_network, state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32)
            loss = calculate_loss(q_network, state, action, next_state, reward, done, gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state = next_state
            episode_reward += reward

        describe_episode(episode, reward, episode_reward, step)

if __name__ == "__main__":
    train(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, EPISODES)
