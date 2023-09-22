import sys
sys.path.append(sys.path[0]+"\\..")
sys.path.append(sys.path[0]+"\\..\qneat")
import unittest
import qneat.qNEAT as q
import qneat.genome as gen
import numpy as np

class TestQNEAT(unittest.TestCase):
    def setUp(self) -> None:
        self.population_size = 10
        self.qneat = q.QNEAT(self.population_size, 3)

    def test_init(self):
        self.assertEqual(len(self.qneat.population), self.population_size)
        self.qneat.species[0].genomes.pop()
        self.assertEqual(len(self.qneat.population), self.population_size)

    def test_run(self):
        self.qneat.run(10)
        compatibility_distances = []
        for ind, genome1 in enumerate(self.qneat.population):
            for genome2 in self.qneat.population[ind+1:]:
                compatibility_distances.append(gen.Genome.compatibility_distance(genome1, genome2))
        
        if self.population_size <= 20:
            for genome in self.qneat.population:
                print(genome.get_circuit(self.qneat.n_qubits)[0])
            print(len(compatibility_distances))
            print(compatibility_distances)
        
        print(f"Compatibility distance. Mean: {np.mean(compatibility_distances)}, std: {np.std(compatibility_distances)}")

        self.qneat.run(100)
        compatibility_distances = []
        for ind, genome1 in enumerate(self.qneat.population):
            for genome2 in self.qneat.population[ind+1:]:
                compatibility_distances.append(gen.Genome.compatibility_distance(genome1, genome2))
        
        if self.population_size <= 20:
            for genome in self.qneat.population:
                print(genome.get_circuit(self.qneat.n_qubits)[0])
            print(len(compatibility_distances))
            print(compatibility_distances)
        
        print(f"Compatibility distance. Mean: {np.mean(compatibility_distances)}, std: {np.std(compatibility_distances)}")
        
    def test_generate_new_population(self):
        self.qneat.generate_new_population("")

if __name__ == '__main__':
    unittest.main()