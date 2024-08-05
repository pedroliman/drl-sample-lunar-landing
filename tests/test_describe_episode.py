import unittest
from io import StringIO
import sys
from utils.describe_episode import describe_episode

class TestDescribeEpisode(unittest.TestCase):

    def setUp(self) -> None:
        self.held_output = StringIO()
        sys.stdout = self.held_output

    def tearDown(self) -> None:
        sys.stdout = sys.__stdout__

    def test_describe_episode(self) -> None:
        describe_episode(1, 1.0, 10.0, 100)
        output = self.held_output.getvalue().strip()
        expected_output = "| Episode  2 | Duration: 100 steps | Return: 10.00  | Hovering |"
        self.assertIn(expected_output, output)

if __name__ == '__main__':
    unittest.main()
