import unittest
import logging

from quantumNEAT.quantumneat import quantumneat
from quantumNEAT.quantumneat import logger

class TestQuantumNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("quantumNEAT.test")
        self.logger.info("TestQuantumNEAT.setUp")

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()