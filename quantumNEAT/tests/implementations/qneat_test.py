import unittest
import logging
# import sys
# print(sys.path)
from quantumneat.logger import QuantumNEATLogger
from quantumneat.implementations import qneat

class TestGate(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("quantumNEAT.implementations.test")
        self.logger.info("setUp")
    
    def test_Layer_Gene(self):
        self.logger.info("test_Layer_Gene")
        config = qneat.QNEAT_Config(5, 100)
        layer = qneat.LayerGene(config, 0)
        self.assertEqual(len(layer.genes), 2)
        layer.genes[qneat.GateCNOT].append(qneat.GateCNOT(0, config, [0, 1]))
        self.assertEqual(len(layer.genes), 2)
        self.assertEqual(len([gate for gate in layer.gates()]), 1)
        layer.genes[qneat.GateROT].append(qneat.GateROT(0, config, [0, 1]))
        self.assertEqual(len(layer.genes), 2)
        self.assertEqual(len([gate for gate in layer.gates()]), 2)

if __name__ == '__main__':
    QuantumNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()