import logging
import copy
from time import time

import numpy as np

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.problem import Problem

class QuantumNEAT:
    logger = logging.getLogger("quantumNEAT.quantumneat.main")
    def __init__(self, config:QuantumNEATConfig, problem:Problem):
        self.config = config
        self.problem = problem

        self.logger.info("QuantumNEAT Started")

        self.population = self.config.Population(self.config, self.problem)
        best_genome = self.population.get_best_genome()
        self.best_fitness = best_genome.get_fitness()
        self.best_length = best_genome.get_length()
        self.best_n_parameter = best_genome.get_n_parameters()

        # For experimenting only
        self.best_fitnesses = [self.best_fitness]
        self.best_lengths = [self.best_length]
        self.best_n_parameters = [self.best_n_parameter]
        if self.config.calculate_solution:
            self.optimal_energy = self.problem.solution()
            self.logger.info(f"{self.optimal_energy=:.2f}")
        else:
            self.optimal_energy = 0
        self.best_energies = [best_genome.get_energy()]
        energies = self.get_energies()
        self.number_of_solutions = [sum([abs(energy-self.optimal_energy) <= self.config.solution_margin for energy in energies])]
        self.min_energies = [min(energies)]
        self.best_genomes:list[tuple[int, self.config.Genome]] = [(self.population.generation, copy.deepcopy(best_genome))]
        self.average_fitnesses = [self.population.average_fitness]
        self.population_sizes = [len(self.population.population)]
        self.number_of_species = [len(self.population.species)]
        # self.specie_sizes = [[(specie.key, len(specie.genomes)) for specie in self.population.species]]
        # sizes = np.array([(specie.key, len(specie.genomes)) for specie in self.population.species], dtype=tuple)
        # self.specie_sizes = np.array(sizes, dtype=np.ndarray)
        self.species_data = []
        for specie in self.population.species:
            self.species_data.append((self.population.generation, specie.key, len(specie.genomes), specie.get_fitness(), specie.best_fitness))

    def run_generation(self):
        # if self.config.simulator == 'qiskit':
        #     self.logger.info(f"Best circuit: \n{self.population.get_best_genome().get_circuit()[0].draw(fold=-1)}")
        # elif self.config.simulator == 'qulacs':
        #     self.logger.info(f"Best circuit: \n{self.population.get_best_genome().get_circuit()[0].to_string()}")
        #TODO check stopping criterion
        self.population.next_generation()
        # self.best_fitness = max(self.best_fitness, self.population.get_best_genome().get_fitness())
        best_genome = self.population.get_best_genome()
        if best_genome.get_fitness() > self.best_fitness:
            self.best_fitness = best_genome.get_fitness()
            self.best_length = best_genome.get_length()
            self.best_n_parameter = best_genome.get_n_parameters()
            self.best_genomes.append((self.population.generation, copy.deepcopy(self.population.get_best_genome())))
        self.best_fitnesses.append(self.best_fitness)
        self.best_lengths.append(self.best_length)
        self.best_n_parameters.append(self.best_n_parameter)
        self.best_energies.append(best_genome.get_energy())
        self.average_fitnesses.append(self.population.average_fitness)
        self.population_sizes.append(len(self.population.population))
        self.number_of_species.append(len(self.population.species))
        # starttime = time()
        energies = self.get_energies()
        self.number_of_solutions.append(sum([abs(energy-self.optimal_energy) <= self.config.solution_margin for energy in energies]))
        self.min_energies.append(min(energies))
        # self.logger.debug(f"run_generation things {time()-starttime}")
        # self.specie_sizes.append([(specie.key, len(specie.genomes)) for specie in self.population.species])
        # sizes = np.array([(specie.key, len(specie.genomes)) for specie in self.population.species], dtype=tuple)
        # self.specie_sizes = np.vstack((self.specie_sizes, sizes))
        for specie in self.population.species:
            self.species_data.append((self.population.generation, specie.key, len(specie.genomes), specie.get_fitness(), specie.best_fitness))
        
    def run(self, max_generations:int = 10):
        self.logger.info(f"Started running for {max_generations-self.population.generation} generations.")

        while self.population.generation < max_generations:
            self.logger.info(f"Generation {self.population.generation:8}, population size: {len(self.population.population):8}, number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}, best length: {self.best_length:4}")
            self.run_generation()
        self.logger.info(f"Generation {self.population.generation:8}, population size: {len(self.population.population):8}, number of species: {len(self.population.species):4}, best fitness: {self.best_fitness:8.3f}, best length: {self.best_length:4}")
        best_circuit_performance = self.best_genomes[-1][1].evaluate(N=1000)
        self.logger.info(f"Best circuit performance: {best_circuit_performance}")
        self.logger.debug(f"Best circuit parameters: {self.best_genomes[-1][1].get_parameters()}")
        self.logger.info(f"Finished running.")

    def get_energies(self):
        energies = []
        for genome in self.population.population:
            energies.append(genome.get_energy())
        return energies