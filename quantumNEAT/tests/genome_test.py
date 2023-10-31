import unittest
import logging

from quantumneat.genome import *

class TestGenome(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_quantumNEAT.TestGenome")
        self.logger.debug("setUp")

if __name__ == '__main__':
    unittest.main()