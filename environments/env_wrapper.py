import gym

def initialize_environment() -> gym.Env:
    """
    Initialize the Gym environment.

    Returns:
    - gym.Env: The initialized Gym environment.
    """
    return gym.make('LunarLander-v2')
