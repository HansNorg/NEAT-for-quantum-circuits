import unittest
import logging

from quantumneat.configuration import QuantumNEATConfig

class TestConfiguration(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def test_QuantumNEATConfig(self):
        self.logger.debug("test_QuantumNEATConfig started")
        config = QuantumNEATConfig(0, 0)
        self.logger.info("test_QuantumNEATConfig passed")

if __name__ == '__main__':
    unittest.main()