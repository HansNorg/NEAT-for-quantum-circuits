import unittest
import logging

from quantumneat.configuration import QuantumNEATConfig

class TestConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_quantumNEAT")
        self.logger.info("TestQuantumNEAT.setUp")

if __name__ == '__main__':
    unittest.main()