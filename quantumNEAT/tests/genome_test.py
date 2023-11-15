import unittest
import logging

import numpy as np

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.genome import CircuitGenome
from quantumneat.gene import GateGene

class ImplGateGene(GateGene):
    """GateGene with implemented abstract class for testing"""
    def add_to_circuit(self, circuit, n_parameters: int):
        self.logger.debug(f"{self.qubits[0]=}")
        circuit.add_H_gate(self.qubits[0])
        return super().add_to_circuit(circuit, n_parameters)
    
class TestCircuitGenome(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    logger = logging.getLogger(__name__)

    def setUp(self):
        self.logger.debug("setUp")
        self.config = QuantumNEATConfig(3, 10)
        self.config.simulator = 'qulacs' #Tests are only based on qulacs circuits atm
        self.config.excess_coefficient, self.config.disjoint_coefficient, self.config.weight_coefficient = np.random.uniform(size = 3)
        self.logger.debug(f"{self.config=}")
        self.logger.debug(f"{self.config.n_qubits=}")
        self.genome1 = CircuitGenome(self.config)
        self.genome2 = CircuitGenome(self.config)
        self.n_genes = 10
        self.genes = []
        for _ in range(self.n_genes):
            n_qubits = np.random.randint(0, self.config.n_qubits)+1
            qubits = list(np.random.randint(0, self.config.n_qubits, size=n_qubits))
            self.logger.debug(f"{_=}; {n_qubits=}; {qubits=}")
            new_gene = ImplGateGene(self.config.GlobalInnovationNumber.next(),self.config,qubits)
            new_gene.n_qubits = n_qubits
            self.genes.append(new_gene)

    def test_add_gene(self):
        self.logger.debug("test_add_gene started")
        for i in range(self.n_genes):
            self.assertEqual(len(self.genome1.genes), i)
            self.assertTrue(self.genome1.add_gene(self.genes[i]))
        self.assertEqual(len(self.genome1.genes), self.n_genes)
        self.logger.info("test_add_gene passed")
    
    def test_compatibility_distance_empty(self):
        self.logger.debug("test_compatibility_distance_empty started")
        self.check_compatibility(0, 0, 0, 0, "Both genomes empty")
        self.logger.info("test_compatibility_distance_empty passed")

    def test_compatibility_distance_equal_genomes(self):
        self.logger.debug("test_compatibility_distance_equal_genomes started")
        for i, j in enumerate([1, 3, 4, 5, 7, 9]):
            self.genome1.add_gene(self.genes[j])
            self.genome2.add_gene(self.genes[j])
            self.check_compatibility(i+1, 0, 0, 0, "Genomes equal")
        self.logger.info("test_compatibility_distance_equal_genomes passed")
        
    def test_compatibility_distance_excess(self):
        self.logger.debug("test_compatibility_distance_excess started")
        self.genome1.add_gene(self.genes[0])
        self.check_compatibility(1, 1, 0, 0, "One genome empty")
        self.genome1.add_gene(self.genes[2])
        self.check_compatibility(2, 2, 0, 0, "One genome empty")
        self.genome2.add_gene(self.genes[0])
        self.check_compatibility(2, 1, 0, 0, "One genome excess")
        self.genome1.add_gene(self.genes[4])
        self.check_compatibility(3, 2, 0, 0, "One genome excess")
        self.logger.info("test_compatibility_distance_excess passed")
        
    def test_compatibility_distance_disjoint(self):
        self.logger.debug("test_compatibility_distance_disjoint started")
        self.genome1.add_gene(self.genes[1])
        self.genome1.add_gene(self.genes[3])
        self.genome2.add_gene(self.genes[3])
        self.check_compatibility(2, 0, 1, 0, "Disjoint genomes")
        self.genome1.add_gene(self.genes[5])
        self.genome1.add_gene(self.genes[6])
        self.genome2.add_gene(self.genes[6])
        self.check_compatibility(4, 0, 2, 0, "Disjoint genomes")
        self.genome1.add_gene(self.genes[7])
        self.genome1.add_gene(self.genes[8])
        self.genome1.add_gene(self.genes[9])
        self.genome2.add_gene(self.genes[9])
        self.check_compatibility(7, 0, 4, 0, "Disjoint genomes")
        self.logger.info("test_compatibility_distance_disjoint passed")

    def test_compatibility_distance_excess_disjoint(self):
        self.logger.debug("test_compatibility_distance_excess_disjoint started")
        self.genome1.add_gene(self.genes[0])
        self.genome1.add_gene(self.genes[1])
        self.genome1.add_gene(self.genes[2])
        self.genome2.add_gene(self.genes[1])
        self.check_compatibility(3, 1, 1, 0, "Excess and disjoint genes")
        self.genome2.add_gene(self.genes[4])
        self.check_compatibility(3, 1, 2, 0, "Excess and disjoint genes")
        self.genome2.add_gene(self.genes[5])
        self.check_compatibility(3, 2, 2, 0, "Excess and disjoint genes")
        self.genome2.add_gene(self.genes[6])
        self.check_compatibility(4, 3, 2, 0, "Excess and disjoint genes")
        self.genome2.add_gene(self.genes[7])
        self.check_compatibility(5, 4, 2, 0, "Excess and disjoint genes")
        self.genome1.add_gene(self.genes[9])
        self.check_compatibility(5, 1, 6, 0, "Excess and disjoint genes")
        self.logger.info("test_compatibility_distance_excess_disjoint passed")

    def test_compatibility_distance_avg_distance(self):
        self.skipTest("Not_implemented")
        self.logger.debug("test_compatibility_distance_avg_distance started")
        self.logger.info("test_compatibility_distance_avg_distance passed")

    def test_compatibility_distance_excess_disjoint_avg_distance(self):
        self.skipTest("Not_implemented")
        self.logger.debug("test_compatibility_distance_excess_disjoint_avg_distance started")
        self.logger.info("test_compatibility_distance_excess_disjoint_avg_distance passed")

    @staticmethod
    def custom_gradient_function(circuit, n_parameters, parameters, config):
        """Gradient function for testing only."""
        TestCircuitGenome.logger.debug("custom_gradient_function")
        return circuit.get_gate_count()
    
    def test_crossover_unequal_fitness(self):
        self.logger.debug("test_crossover_unequal_fitness started")
        # Without parametrized gates all default gradients will equal 0, 
        # so can't test correct child generation as the fittest parent is not defined.
        # So instead a different gradient function is used for testing.
        self.config.gradient_function = self.custom_gradient_function
        
        correct_child = CircuitGenome(self.config)
        self.check_correct_child(correct_child, "empty genomes")
        
        self.genome1.add_gene(self.genes[0])
        correct_child.add_gene(self.genes[0])
        self.check_correct_child(correct_child, "one empty genome")

        self.genome1.add_gene(self.genes[2])
        correct_child.add_gene(self.genes[2])
        self.check_correct_child(correct_child, "one empty genome")

        self.genome2.add_gene(self.genes[1])
        self.check_correct_child(correct_child, "non-empty genomes, disjoint/excess")

        self.logger.info("test_crossover_unequal_fitness passed")
        
    def test_crossover_equal_fitness(self):
        self.logger.debug("test_crossover_unequal_fitness started")
        # Without parametrized gates all default gradients will equal 0, 
        # so can't test correct child generation as the fittest parent is not defined.
        # So instead a different gradient function is used for testing.
        self.config.gradient_function = self.custom_gradient_function
        
        correct_child = CircuitGenome(self.config)
        
        self.genome1.add_gene(self.genes[1])
        self.genome2.add_gene(self.genes[1])
        correct_child.add_gene(self.genes[1])
        self.genome1.add_gene(self.genes[2])
        self.genome2.add_gene(self.genes[2])
        correct_child.add_gene(self.genes[2])
        self.genome1.add_gene(self.genes[4])
        self.genome2.add_gene(self.genes[4])
        correct_child.add_gene(self.genes[4])
        self.check_correct_child(correct_child, "non-empty genomes, matching + disjoint/excess")

    def check_correct_child(self, correct_child:CircuitGenome, message):
        self.logger.debug("check_correct_child: "+message)
        child = CircuitGenome.crossover(self.genome1, self.genome2, self.config)
        with self.subTest("type"):
            self.assertEqual(type(child), type(correct_child), 
                "Child should match correct child: "+message)
        with self.subTest("len(genes)"):
            self.assertEqual(len(child.genes), len(correct_child.genes), 
                "Child should match correct child: "+message)
        with self.subTest("compatibility_distance"):
            self.assertEqual(CircuitGenome.compatibility_distance(
                child, correct_child, self.config), 0, 
                "Child should match correct child: "+message)

    def check_compatibility(self, n_genes, excess, disjoint, avg_dist, message):
        """
        Checks if the compatibility score is correctly calculated between self.genome1 and self.genome2.
        
        Parameters
        ----------
        - n_genes
        - excess
        - disjoint
        - avg_dist
        - message: Message to add to the assertion check
        """
        self.logger.debug("check_compatibility: "+message)
        if n_genes == 0:
            correct = 0
        else:
            correct = self.config.excess_coefficient*excess/n_genes \
                + self.config.disjoint_coefficient*disjoint/n_genes \
                + self.config.weight_coefficient*avg_dist
        self.assertEqual(CircuitGenome.compatibility_distance(
            self.genome1, self.genome2, self.config), correct, message)
        self.assertEqual(CircuitGenome.compatibility_distance(
            self.genome2, self.genome1, self.config), correct, message)
    
if __name__ == '__main__':
    unittest.main()