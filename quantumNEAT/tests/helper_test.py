import unittest
import logging

from quantumneat.helper import *

class TestHelper(unittest.TestCase):
    def setUp(self):
         self.logger = logging.getLogger("test_quantumNEAT.TestHelper")
         self.logger.debug("setUp")

if __name__ == '__main__':
    unittest.main()