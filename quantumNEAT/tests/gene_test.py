import unittest
import logging

import numpy as np

from quantumneat.gene import Gene, GateGene
from quantumneat.configuration import QuantumNEATConfig

class TestGene(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_quantumNEAT.TestGene")
        self.logger.debug("setUp")
        self.config = QuantumNEATConfig(3, 10)

    def test_get_distance(self):
        self.logger.debug("test_get_distance started")
        
        gene1 = Gene(0, self.config)
        gene2 = Gene(1, self.config)
        self.assertEqual(Gene.get_distance(gene1, gene2), (False, 0), 
                         "Distance between parameterless genes should equal 0 and not be included")
        
        gene1.n_parameters = 3
        gene2.n_parameters = 3
        gene1.parameters = [0, 0, 0]
        gene2.parameters = [0, 1, 1]
        self.assertEqual(Gene.get_distance(gene1, gene2), (True, np.sqrt(2)),
                         "Distance between parameter genes should equal euclidean distance and be included")

        gene1.parameters = [0.5, -0.5, -np.pi]
        gene2.parameters = [np.pi, 0, -2*np.pi]
        distance = np.sqrt((0.5-np.pi)**2 + (-0.5-0)**2 + (-np.pi+2*np.pi)**2)
        self.assertEqual(Gene.get_distance(gene1, gene2), (True, distance),
                         "Distance between parameter genes should equal euclidean distance and be included")
        self.logger.info("test_get_distance passed")

    def test_mutate_parameters(self):
        self.logger.debug("test_mutate_parameters started")
        gene = Gene(0, self.config)

        self.assertEqual(gene.mutate_parameters(), False)

        n_parameters = 3
        gene.n_parameters = n_parameters
        parameters = [0, 0, 0]
        gene.parameters = parameters.copy()
        self.assertEqual(gene.mutate_parameters(), True,
                            "Parameters should have changed after mutation")
        for i in range(n_parameters):
            self.assertNotEqual(gene.parameters[i], parameters[i], 
                            "Parameters should have changed after mutation")

        n_parameters = np.random.randint(1, 10)
        gene.n_parameters = n_parameters
        parameters = self.config.parameter_amplitude*np.random.random(n_parameters)
        gene.parameters = parameters.copy()
        self.assertEqual(gene.mutate_parameters(), True,
                            "Parameters should have changed after mutation")
        for i in range(n_parameters):
            self.assertNotEqual(gene.parameters[i], parameters[i], 
                            "Parameters should have changed after mutation")
        self.logger.info("test_mutate_parameters passed")


if __name__ == '__main__':
    unittest.main()