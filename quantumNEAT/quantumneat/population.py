from __future__ import annotations

import logging
import random
import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig

class Population():
    """Keep and update a population of genomes."""
    logger = logging.getLogger("quantumNEAT.quantumneat.population")
    
    def __init__(self, config:QuantumNEATConfig) -> None:
        """
        Initialise a population.

        Arguments:
        ----------
        config: class with all the configuration settings of the algorithm.
        """
        self.config = config
        self.generation:int = 0
        self.population = self.generate_initial_population()
        self.update_avg_fitness()
        self.species: list[QuantumNEATConfig.Species] = []
        self.speciate()

    def generate_initial_population(self) -> list[QuantumNEATConfig.Genome]:
        population = []
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config)
            gene_type = np.random.choice(self.config.gene_types)
            qubits = np.random.randint(self.config.n_qubits, size=gene_type.n_qubits).tolist()
            gate = gene_type(self.config.GlobalInnovationNumber.next(), self.config, qubits)
            genome.add_gene(gate)
            population.append(genome)
        return self.sort_genomes(population)

    @staticmethod
    def sort_genomes(genomes:list[QuantumNEATConfig.Genome]) -> list[QuantumNEATConfig.Genome]:
        """Sort the given genomes by fitness."""
        return sorted(genomes, key=lambda genome: genome.get_fitness(), reverse=True)
    
    def update_avg_fitness(self):
        self.average_fitness = np.mean([genome.get_fitness() for genome in self.population])
        self.logger.debug(f"{self.average_fitness=}")

    def generate_new_population(self) -> list[QuantumNEATConfig.Genome]:
        """Generate the next generation of the population by mutation and crossover."""
        self.logger.debug(f"{self.average_fitness =}")
        new_population:list[QuantumNEATConfig.Genome] = []
        for specie in self.species:
            total_specie_fitness = np.sum([genome.get_fitness() for genome in specie.genomes])
            n_offspring = round(total_specie_fitness/self.average_fitness)
            cutoff = int(np.ceil(self.config.percentage_survivors * len(specie.genomes)))
        
            sorted_genomes = self.sort_genomes(specie.genomes)[:cutoff]

            if len(specie.genomes) >= self.config.specie_champion_size:
                new_population.append(copy.deepcopy(sorted_genomes[0]))
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(sorted_genomes) > 1 and random.random() > self.config.prob_mutation_without_crossover:
                    self.logger.debug("if")
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(self.config.Genome.crossover(parent1, parent2))
                else:
                    self.logger.debug("else")
                    new_population.append(copy.deepcopy(random.choice(sorted_genomes))) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness()))
                self.logger.debug(f"{new_population[-1]=}")
                new_population[-1].mutate()#self.config.GlobalInnovationNumber, self.config.n_qubits)
            specie.empty()
        return self.sort_genomes(new_population)

    def speciate(self):
        """Devide the population in species by similarity"""
        for genome in self.population:
            found = False
            for specie in self.species:
                distance = self.config.Genome.compatibility_distance(genome, specie.representative, self.config)
                if distance < self.config.compatibility_threshold:
                    specie.add(genome)
                    found = True
                    break
            if not found:
                new_species = self.config.Species(self.generation, self.config.GlobalSpeciesNumber.next())
                new_species.update(genome, [genome])
                self.species.append(new_species)
        for ind, specie in enumerate(self.species):
            if not specie.update_representative():
                self.species.pop(ind) # Empty species

    def next_generation(self):
        self.population = self.generate_new_population()
        self.update_avg_fitness()
        self.speciate()
        self.generation += 1

    def get_best_genome(self) -> QuantumNEATConfig.Genome:
        return self.population[0]