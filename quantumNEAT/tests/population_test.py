import unittest
import logging

from quantumneat.population import *

class TestPopulation(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test_quantumNEAT.TestPopulation")
        self.logger.info("setUp")

if __name__ == '__main__':
    unittest.main()