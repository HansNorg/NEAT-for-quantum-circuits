import unittest
import logging

from quantumNEAT.quantumneat import genome
from quantumNEAT.quantumneat import logger

class TestGenome(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("quantumNEAT.test")
        self.logger.info("TestGenome.setUp")

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()