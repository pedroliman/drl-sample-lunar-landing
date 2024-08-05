import unittest
import torch
from models.q_network import QNetwork
from utils.calculate_loss import calculate_loss

class TestCalculateLoss(unittest.TestCase):

    def setUp(self) -> None:
        self.q_network = QNetwork(state_size=8, action_size=4)
        self.state = torch.rand(8)
        self.next_state = torch.rand(8)
        self.action = 1
        self.reward = 1.0
        self.done = False
        self.gamma = 0.99

    def test_calculate_loss(self) -> None:
        loss = calculate_loss(self.q_network, self.state, self.action, self.next_state, self.reward, self.done, self.gamma)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

if __name__ == '__main__':
    unittest.main()
