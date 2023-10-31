import unittest
import logging

from quantumneat.implementations import qneat

class TestQNEAT(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_quantumNEAT.implementations.qneat")
        self.logger.debug("setUp")
    
    def test_Layer_Gene(self):
        self.logger.debug("test_Layer_Gene started")
        config = qneat.QNEAT_Config(5, 100)
        layer = qneat.LayerGene(config, 0)
        self.assertEqual(len(layer.genes), 2)
        layer.genes[qneat.GateCNOT].append(qneat.GateCNOT(0, config, [0, 1]))
        self.assertEqual(len(layer.genes), 2)
        self.assertEqual(len([gate for gate in layer.gates()]), 1)
        layer.genes[qneat.GateROT].append(qneat.GateROT(0, config, [0, 1]))
        self.assertEqual(len(layer.genes), 2)
        self.assertEqual(len([gate for gate in layer.gates()]), 2)
        self.logger.info("test_Layer_Gene started passed")

if __name__ == '__main__':
    unittest.main()