import unittest
import torch
from models.q_network import QNetwork

class TestQNetwork(unittest.TestCase):

    def setUp(self) -> None:
        self.state_size = 8
        self.action_size = 4
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.state = torch.rand(self.state_size)

    def test_q_network_initialization(self) -> None:
        self.assertEqual(self.q_network.fc1.in_features, self.state_size)
        self.assertEqual(self.q_network.fc3.out_features, self.action_size)

    def test_forward_pass(self) -> None:
        q_values = self.q_network(self.state)
        self.assertEqual(q_values.shape, (self.action_size,))
        self.assertTrue(torch.is_tensor(q_values))

if __name__ == '__main__':
    unittest.main()
