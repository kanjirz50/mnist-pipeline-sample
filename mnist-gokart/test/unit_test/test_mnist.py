from logging import getLogger
import unittest
from unittest.mock import MagicMock
from mnist_gokart.model.mnist import GetMNISTDatasetTask

logger = getLogger(__name__)


class TestSample(unittest.TestCase):
    def setup(self):
        self.output_data = None

    def test_run(self):
        task = GetMNISTDatasetTask()
        task.dump = MagicMock(side_effect=self._dump)
        task.run()

        self.assertEqual(len(self.output_data.keys()), 7)

    def _dump(self, data):
        self.output_data = data
