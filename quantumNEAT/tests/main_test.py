import unittest
import logging

from quantumneat.main import QuantumNEAT

class TestQuantumNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_quantumNEAT.TestQuantumNEAT")
        self.logger.info("setUp")

if __name__ == '__main__':
    unittest.main()