import unittest
import logging

from quantumneat.configuration import QuantumNEATConfig

class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_quantumNEAT.configuration")
        self.logger.debug("setUp")

    def test_QuantumNEATConfig(self):
        self.logger.debug("test_QuantumNEATConfig started")
        QuantumNEATConfig(0, 100)
        self.logger.info("test_QuantumNEATConfig passed")

if __name__ == '__main__':
    unittest.main()