import unittest
import logging
from time import time
import multiprocessing as mp

import numpy as np

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.population import Population
from quantumneat.implementations.linear_growth import LinearGrowthConfig
from quantumneat.problems.chemistry import GroundStateEnergySavedHamiltonian

class TestPopulation(unittest.TestCase):
    logger = logging.getLogger("quantumNEAT.test.population")

    def test_population_size(self):
        print("test_population_size")
        n_population = 100
        config = LinearGrowthConfig(2, n_population)
        problem = GroundStateEnergySavedHamiltonian(config, "h2", error_in_fitness=False)
        population = Population(config, problem)
        self.assertEqual(len(population.population), n_population)
        for _ in range(100):
            with self.subTest("check_during_generations"):
                population.next_generation()
                self.assertEqual(len(population.population), n_population)
        self.assertEqual(len(population.population), n_population)

    def test_fitness_updating_old(self):
        print("test_fitness_updating_old")
        for cores in [1, 2, 3, 4]:
            config = LinearGrowthConfig(8, 100)
            problem = GroundStateEnergySavedHamiltonian(config, "lih", error_in_fitness=False)
            population = Population(config, problem)
            new_population = population._generate_new_population()
            starttime = time()
            with mp.Pool(processes=cores) as p:
                chunks = len(new_population)/cores
                p.map(population._update_fitness, new_population, chunksize=int(np.ceil(chunks)))
                p.close()
                self.logger.info(f"{cores=} pool time = {time()-starttime}")
                print(f"\n{cores=} pool time = {time()-starttime}")
                p.join()
            self.logger.info(f"{cores=} update time = {time()-starttime}")
            print(f"{cores=} update time = {time()-starttime}")
            population.sort_genomes(new_population)
            self.logger.info(f"{cores=} total time = {time()-starttime}")
            print(f"{cores=} total time = {time()-starttime}")

    def test_fitness_updating_new(self):
        print("test_fitness_updating_new")
        for cores in [1, 2, 3, 4]:
            print()
            config = LinearGrowthConfig(8, 100)
            problem = GroundStateEnergySavedHamiltonian(config, "lih", error_in_fitness=False)
            population = Population(config, problem)
            new_population = population._generate_new_population()
            starttime = time()
            data = [(genome.get_circuit()[0], genome.get_parameters(), genome.get_circuit()[1]) for genome in new_population]
            self.logger.info(f"{cores=} prep time = {time()-starttime}")
            print(f"{cores=} prep time = {time()-starttime}")
            with mp.Pool(processes=cores) as p:
                energies = p.map(problem.energy_new, data)
                gradients = p.map(problem.gradient_new, data)
            self.logger.info(f"{cores=} pool time = {time()-starttime}")
            print(f"{cores=} pool time = {time()-starttime}")
            for genome, energy, gradient in zip(new_population, energies, gradients):
                genome._energy = energy
                genome._gradient = gradient
                genome._update_gradient = False
                genome.update_fitness()
            self.logger.info(f"{cores=} update time = {time()-starttime}")
            print(f"{cores=} update time = {time()-starttime}")
            population.sort_genomes(new_population)
            self.logger.info(f"{cores=} total time = {time()-starttime}")
            print(f"{cores=} total time = {time()-starttime}")
        for cores in [1, 2, 3, 4]:
            print()
            config = LinearGrowthConfig(8, 100)
            problem = GroundStateEnergySavedHamiltonian(config, "lih", error_in_fitness=False)
            population = Population(config, problem)
            new_population = population._generate_new_population()
            starttime = time()
            data = [(genome.get_circuit()[0], genome.get_parameters(), genome.get_circuit()[1]) for genome in new_population]
            chunks = int(np.ceil(len(new_population)/cores))
            self.logger.info(f"{cores=} prep time = {time()-starttime}")
            print(f"{cores=} prep time = {time()-starttime}")
            with mp.Pool(processes=cores) as p:
                energies = p.map(problem.energy_new, data, chunksize=chunks)
                gradients = p.map(problem.gradient_new, data, chunksize=chunks)
            self.logger.info(f"{cores=} pool time = {time()-starttime}")
            print(f"{cores=} pool time = {time()-starttime}")
            for genome, energy, gradient in zip(new_population, energies, gradients):
                genome._energy = energy
                genome._gradient = gradient
                genome._update_gradient = False
                genome.update_fitness()
            self.logger.info(f"chunks {cores=} update time = {time()-starttime}")
            print(f"chunks {cores=} update time = {time()-starttime}")
            population.sort_genomes(new_population)
            self.logger.info(f"{cores=} total time = {time()-starttime}")
            print(f"chunks {cores=} total time = {time()-starttime}")
# 30.9
if __name__ == '__main__':
    unittest.main()