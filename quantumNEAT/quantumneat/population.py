from __future__ import annotations

import logging
import random
import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig as C

class Population():
    """Keep and update a population of genomes."""
    def __init__(self, config:C) -> None:
        """
        Initialise a population.

        Arguments:
        ----------
        config: class with all the configuration settings of the algorithm.
        """
        self.logger = logging.getLogger("QuantumNEAT.Population")
        self.generation:int = 0
        self.population = self.generate_initial_population()
        self.species: list[C.Species] = []
        self.speciate()
        self.config = config

    def generate_initial_population(self) -> list[C.Genome]:
        population = []
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config)
            gate_type = np.random.choice(self.config.GateType)
            qubit = np.random.randint(self.config.n_qubits)
            gate = self.config.GateGene(self.config.global_innovation_number.next(),gate_type,qubit, self.config)
            genome.add_gate(gate)
            population.append(genome)
        return self.sort_genomes(population)

    @staticmethod
    def sort_genomes(genomes:list[C.Genome]) -> list[C.Genome]:
        """Sort the given genomes by fitness."""
        return sorted(genomes, key=lambda genome: genome.get_fitness(), reverse=True)

    def generate_new_population(self):
        """Generate the next generation of the population by mutation and crossover."""
        average_fitness = np.mean([genome.get_fitness(self.config.n_qubits, self.config.backend) for genome in self.population])
        self.logger.debug(f"{average_fitness =}")
        self.average_fitnesses.append(average_fitness)
        new_population:list[C.Genome] = []
        for specie in self.species:
            total_specie_fitness = np.sum([genome.get_fitness(self.config.n_qubits, self.config.backend) for genome in specie.genomes])
            n_offspring = round(total_specie_fitness/average_fitness)
            cutoff = int(np.ceil(self.config.percentage_survivors * len(specie.genomes)))
        
            sorted_genomes = self.sort_genomes(specie.genomes)[:cutoff]

            if len(specie.genomes) >= self.config.specie_champion_size:
                new_population.append(copy.deepcopy(sorted_genomes[0]))
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(sorted_genomes) > 1 and random.random() > self.config.prob_mutation_without_crossover:
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(self.config.Genome.crossover(parent1, parent2, self.config.n_qubits, self.config.backend))
                else:
                    new_population.append(copy.deepcopy(random.choice(sorted_genomes))) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness(self.n_qubits, backend)))
                new_population[-1].mutate(self.config.global_innovation_number, self.config.n_qubits)
            specie.empty()
        self.population = self.sort_genomes(new_population)
        self.speciate()

    def speciate(self):
        """Devide the population in species by similarity"""
        for genome in self.population:
            found = False
            for specie in self.species:
                distance = self.config.Genome.compatibility_distance(genome, specie.representative)
                if distance < self.config.compatibility_threshold:
                    specie.add(genome)
                    found = True
                    break
            if not found:
                new_species = self.config.Species(self.generation, self.config.global_species_number.next())
                new_species.update(genome, [genome])
                self.species.append(new_species)
        for ind, specie in enumerate(self.species):
            if not specie.update_representative():
                self.species.pop(ind) # Empty species

    def next_generation(self):
        self.generate_new_population()
        self.speciate()
        self.generation += 1

    def get_best_genome(self) -> C.Genome:
        return self.population[0]