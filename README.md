# DQN Project

This sample project implements a Deep Q-Network (DQN) for reinforcement learning using PyTorch and Gymnasium.

## Setup

1. Install the dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the training:
```bash
python main.py
```

## Project Structure

- `main.py`: Main script to run the training process.
- `config.py`: Configuration file for hyperparameters.
- `utils/`: Utility functions.
- `models/`: QNetwork class.
- `training/`: Training loop.
- `environments/`: Environment setup.
- `tests/`: Unit tests.
