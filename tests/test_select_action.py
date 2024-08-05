import unittest
import torch
from models.q_network import QNetwork
from utils.select_action import select_action

class TestSelectAction(unittest.TestCase):

    def setUp(self) -> None:
        self.q_network = QNetwork(state_size=8, action_size=4)
        self.state = torch.rand(8)

    def test_select_action(self) -> None:
        action = select_action(self.q_network, self.state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 4)

if __name__ == '__main__':
    unittest.main()
