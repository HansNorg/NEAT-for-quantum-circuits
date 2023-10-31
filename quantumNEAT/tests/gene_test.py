import unittest
import logging

from quantumneat.gene import *

class TestGate(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_quantumNEAT")
        self.logger.info("TestGene.setUp")

if __name__ == '__main__':
    unittest.main()