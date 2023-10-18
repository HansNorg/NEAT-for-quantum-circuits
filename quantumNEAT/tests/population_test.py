import unittest
import logging

from quantumNEAT.quantumneat import population
from quantumNEAT.quantumneat import logger

class TestPopulation(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("quantumNEAT.test")
        self.logger.info("TestPopulation.setUp")

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()