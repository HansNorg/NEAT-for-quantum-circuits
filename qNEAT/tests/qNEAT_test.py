import unittest
import qneat.qNEAT as q
import qneat.genome as gen
import qneat.logger as log
import numpy as np
import logging

class TestQNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("qNEAT.test")
        self.logger.info("TestQNEAT.setUp")
        self.population_size = 10
        self.qneat = q.QNEAT(self.population_size, 5)

    def test_init(self):
        self.logger.info("TestQNEAT.test_init")
        self.assertEqual(len(self.qneat.population), self.population_size)
        self.qneat.species[0].genomes.pop()
        self.assertEqual(len(self.qneat.population), self.population_size)

    def test_run(self):
        self.logger.info("TestQNEAT.test_run")
        self.qneat.run(2)
        compatibility_distances = []
        for ind, genome1 in enumerate(self.qneat.population):
            for genome2 in self.qneat.population[ind+1:]:
                compatibility_distances.append(gen.Genome.compatibility_distance(genome1, genome2))
        
        if self.logger.isEnabledFor(logging.DEBUG): 
            for genome in self.qneat.population:
                self.logger.debug(f"\n{genome.get_circuit(self.qneat.n_qubits)[0].draw(fold=-1)}")
            self.logger.debug(f"{len(compatibility_distances) =}")
            self.logger.debug(f"{compatibility_distances =}")
        
        self.logger.info(f"Compatibility distance. Mean: {np.mean(compatibility_distances)}, std: {np.std(compatibility_distances)}")

        # self.qneat.run(100)
        # compatibility_distances = []
        # for ind, genome1 in enumerate(self.qneat.population):
        #     for genome2 in self.qneat.population[ind+1:]:
        #         compatibility_distances.append(gen.Genome.compatibility_distance(genome1, genome2))
        
        # if self.population_size <= 20:
        #     for genome in self.qneat.population:
        #         print(genome.get_circuit(self.qneat.n_qubits)[0])
        #     print(len(compatibility_distances))
        #     print(compatibility_distances)
        
        # print(f"Compatibility distance. Mean: {np.mean(compatibility_distances)}, std: {np.std(compatibility_distances)}")
        
    def test_generate_new_population(self):
        self.logger.info("TestQNEAT.test_generate_new_population")
        self.qneat.generate_new_population("")

if __name__ == '__main__':
    log.QNEATLogger("test", file_level=logging.DEBUG, mode="w")
    unittest.main()