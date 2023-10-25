import unittest
import logging

from quantumNEAT.quantumneat import logger
from quantumNEAT.quantumneat.implementations import qneat

class TestGate(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("quantumNEAT.implementations.test")
        self.logger.info("QNEAT.setUp")
    
    def test_Layer_Gene(self):
        config = qneat.QNEAT_Config(5, 100)
        layer = qneat.LayerGene(config, 0)
        
        layer.add_gate()

if __name__ == '__main__':
    logger.QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()