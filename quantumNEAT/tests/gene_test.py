import unittest
import logging

from quantumNEAT.quantumneat import logger
from quantumNEAT.quantumneat import gene

class TestGate(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("quantumNEAT.test")
        self.logger.info("TestGene.setUp")

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()