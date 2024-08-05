def describe_episode(episode: int, reward: float, episode_reward: float, t: int) -> None:
    """
    Print a summary of the episode.

    Parameters:
    - episode (int): The episode number.
    - reward (float): The reward received at the end of the episode.
    - episode_reward (float): The total reward accumulated during the episode.
    - t (int): The number of steps taken in the episode.
    """
    landed = reward == 100
    crashed = reward == -100
    status = (
        "Landed   |" if landed
        else "Crashed  |" if crashed
        else "Hovering |"
    )
    print(f"| Episode {episode + 1:4} | Duration: {t:4} steps | Return: {episode_reward:<7.2f} | {status}")
