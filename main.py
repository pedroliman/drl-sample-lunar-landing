from training.train import train

# Size of state and action space
STATE_SIZE = 8
ACTION_SIZE = 4

# Learning rate for the optimizer
LEARNING_RATE = 0.0001

# Discount factor for future rewards
GAMMA = 0.99

# Number of training episodes
EPISODES = 1000


if __name__ == "__main__":
    train(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, GAMMA, EPISODES)
