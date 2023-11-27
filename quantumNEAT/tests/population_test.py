import unittest
import logging

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.population import Population
from quantumneat.implementations.linear_growth import LinearGrowthConfig

class TestPopulation(unittest.TestCase):
    logger = logging.getLogger("quantumNEAT.test.population")

    def test_population_size(self):
        n_population = 100
        config = LinearGrowthConfig(5, n_population)
        population = Population(config)
        self.assertEqual(len(population.population), n_population)
        for _ in range(100):
            with self.subTest("check_during_generations"):
                population.next_generation()
                self.assertEqual(len(population.population), n_population)
        self.assertEqual(len(population.population), n_population)

if __name__ == '__main__':
    unittest.main()