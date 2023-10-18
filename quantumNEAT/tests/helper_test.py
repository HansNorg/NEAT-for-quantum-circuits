import unittest
import logging

from quantumNEAT.quantumneat import helper
from quantumNEAT.quantumneat import logger

class TestHelper(unittest.TestCase):
    def setUp(self):
         self.logger = logging.getLogger("quantumNEAT.test")
         self.logger.info("TestHelper.setUp")

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()