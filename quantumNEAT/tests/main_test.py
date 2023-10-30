import unittest
import logging

from quantumneat.main import QuantumNEAT
from quantumneat.logger import QuantumNEATLogger

class TestQuantumNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("quantumNEAT.test")
        self.logger.info("TestQuantumNEAT.setUp")

if __name__ == '__main__':
    QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()