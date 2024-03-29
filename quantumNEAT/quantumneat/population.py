from __future__ import annotations

import logging
import random
import copy
from typing import TYPE_CHECKING
import multiprocessing as mp
import time

import numpy as np

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig
    from quantumneat.problem import Problem

class Population():
    """Keep and update a population of genomes."""
    logger = logging.getLogger("quantumNEAT.quantumneat.population")
    
    def __init__(self, config:QuantumNEATConfig, problem:Problem) -> None:
        """
        Initialise a population.

        Arguments:
        ----------
        config: class with all the configuration settings of the algorithm.
        """
        self.config = config
        self.problem = problem
        self.logger.debug(f"{self.config.number_of_cpus=}")
        self.generation:int = 0
        self.population = self.generate_initial_population()
        self.update_avg_fitness()
        self.species: list[QuantumNEATConfig.Species] = []
        self.speciate()

    def generate_initial_population(self) -> list[QuantumNEATConfig.Genome]:
        population = []
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config, self.problem)
            gene_type = np.random.choice(self.config.gene_types)
            qubits = np.random.choice(range(self.config.n_qubits), size=gene_type.n_qubits, replace=False).tolist()
            # self.logger.debug(f"{qubits=}; {type(qubits)=}")
            gate = gene_type(self.config.GlobalInnovationNumber.next(), self.config, self.problem, qubits)
            genome.add_gene(gate)
            population.append(genome)
        
        starttime = time.time()
        if (self.config.number_of_cpus is None) or (self.config.number_of_cpus > 0):
            p = mp.Pool(processes=self.config.number_of_cpus)
            p.map_async(self._update_fitness, population)
            p.close()
        else:
            for pop in population:
                pop.get_fitness()
        difference = time.time() - starttime
        self.logger.debug(f"Time to update fitnesses = {difference}")
        return self.sort_genomes(population)

    @staticmethod
    def sort_genomes(genomes:list[QuantumNEATConfig.Genome]) -> list[QuantumNEATConfig.Genome]:
        """Sort the given genomes by fitness."""
        return sorted(genomes, key=lambda genome: genome.get_fitness(), reverse=True)
    
    def update_avg_fitness(self):
        self.average_fitness = np.mean([genome.get_fitness() for genome in self.population])
        fitnesses = [genome.get_fitness() for genome in self.population]
        fitnesses = self.normalise(fitnesses, True)
        self.average_normalised_fitness = np.mean(fitnesses)
        # self.logger.debug(f"{self.average_fitness=}")

    def normalise(self, fitnesses, update = False):
        # self.logger.debug(f"{fitnesses=}, {update=}")
        if update:
            self._min_fitness = min(fitnesses) - 1 # min(fitnesses) = fitnesses[-1] (sorted)
            # self._max_fitness = max(fitnesses) - self._min_fitness
        fitnesses -= self._min_fitness
        # fitnesses = fitnesses/self._max_fitness
        # self.logger.debug(f"{fitnesses=}")        
        return fitnesses
    
    def _generate_new_population(self) -> list[QuantumNEATConfig.Genome]:
        new_population:list[QuantumNEATConfig.Genome] = []
        n_offsprings:list[float] = []
        if self.config.normalise_fitness:
            average_fitness = self.average_normalised_fitness
            optional_normalisation = self.normalise
        else:
            average_fitness = self.average_fitness
            def no_normalisation(fitnesses, update:bool = False):
                return fitnesses
            optional_normalisation = no_normalisation
        if self.config.fitness_sharing:
            fitnesses = []
            for specie in self.species:
                fitnesses.extend(specie.get_fitnesses())
            average_fitness = np.mean(fitnesses)
        for specie in self.species:
            if self.config.fitness_sharing:
                specie_fitnesses = specie.get_fitnesses()
            else:
                specie_fitnesses = [genome.get_fitness() for genome in specie.genomes]
            specie_fitnesses = optional_normalisation(specie_fitnesses)
            specie_total_fitness = sum(specie_fitnesses)
            n_offspring = specie_total_fitness/average_fitness
            n_offsprings.append(n_offspring)
        
        if self.config.force_population_size:
            n_offsprings = n_offsprings/sum(n_offsprings)*self.config.population_size
        n_offsprings:list[int] = [int(x) for x in np.round(n_offsprings, decimals=0)]
        self.logger.debug(f"Sum n_offsprings = {sum(n_offsprings)}")


        # for specie in self.species:
        for n_offspring, specie in zip(n_offsprings, self.species):
            # total_specie_fitness = np.sum([genome.get_fitness() for genome in specie.genomes])
            # n_offspring = round(total_specie_fitness/self.average_fitness)
            # print(f"{total_specie_fitness=}, {self.average_fitness=}, {n_offspring=}")
            # specie_fitnesses = [genome.get_fitness() for genome in specie.genomes]
            # specie_normalised_fitnesses = self.normalise(specie_fitnesses)
            # total_specie_normalised_fitness = np.sum(specie_normalised_fitnesses)
            # n_offspring = round(total_specie_normalised_fitness/self.average_normalised_fitness)
            # print(f"{total_specie_fitness=:.2f}, {self.average_fitness=:.2f}, {total_specie_normalised_fitness=:.2f}, {self.average_normalised_fitness=:.2f}, {n_offspring}")
            # n_offsprings.append(n_offspring)
            cutoff = int(np.ceil(self.config.percentage_survivors * len(specie.genomes)))
        
            sorted_genomes = self.sort_genomes(specie.genomes)[:cutoff]

            if len(specie.genomes) >= self.config.specie_champion_size: # TODO: Check if sometimes champion is kept even when n_offspring == 0
                # self.logger.debug(f"new_population: {n_offspring=}; champion")
                new_population.append(copy.deepcopy(sorted_genomes[0]))
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(sorted_genomes) > 1 and random.random() > self.config.prob_mutation_without_crossover:
                    # self.logger.debug(f"new_population: {n_offspring=}; crossover")
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(self.config.Genome.crossover(parent1, parent2, self.config))
                else:
                    # self.logger.debug(f"new_population: {n_offspring=}; no crossover")
                    new_population.append(copy.deepcopy(random.choice(sorted_genomes))) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness()))
                # self.logger.debug(f"{new_population[-1]=}")
                new_population[-1].mutate()
            specie.empty()
        return new_population

    def generate_new_population(self) -> list[QuantumNEATConfig.Genome]:
        """Generate the next generation of the population by mutation and crossover."""
        # self.logger.debug(f"{self.average_fitness =}")
        new_population = self._generate_new_population()
        # print(n_offsprings)
        starttime = time.time()
        # self.logger.info(f"{starttime=}")
        if (self.config.number_of_cpus is None) or (self.config.number_of_cpus > 0):
            with mp.Pool(processes=self.config.number_of_cpus) as p:
                chunks = len(new_population)/self.config.number_of_cpus
                p.map(self._update_fitness, new_population, chunksize=int(np.ceil(chunks)))
        else:
            for pop in new_population:
                pop.get_fitness()
        difference = time.time() - starttime
        self.logger.debug(f"Time to update fitnesses = {difference}")
        starttime2 = time.time()
        sorted_genomes = self.sort_genomes(new_population)
        self.logger.debug(f"Time to sort genomes = {time.time() - starttime2}")
        return sorted_genomes

    def _update_fitness(self, i:QuantumNEATConfig.Genome):
        i.get_fitness()
    
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
                new_species = self.config.Species(self.generation, self.config, self.config.GlobalSpeciesNumber.next())
                new_species.update(genome, [genome], self.generation)
                self.species.append(new_species)
        for ind, specie in reversed(list(enumerate(self.species))):
            if not specie.update_representative(self.generation):
                self.logger.debug(f"Popping species at {ind}")
                self.species.pop(ind) # Empty species

    def remove_stagnant_species(self):
        update = False
        remove_inds = []
        for ind, specie in enumerate(self.species):
            if specie.check_stagnant(self.generation):
                remove_inds.append(ind)
                update = True
        
        if not update:
            return
        
        if len(remove_inds) == len(self.species):
            # All species are stagnant, refocus to two strongest species
            if len(self.species) <= self.config.all_stagnant_n_save:
                self.logger.debug("All species stagnant")
                return
            self.logger.warning("All species stagnant, selecting survivors")
            
            specie_fitnesses = []
            for ind, specie in enumerate(self.species):
                fitness = specie.get_fitness()
                specie_fitnesses.append((ind, fitness))
            specie_fitnesses.sort(key=lambda x: x[1], reverse=True)
            for i in range(self.config.all_stagnant_n_save):
                remove_inds.remove(specie_fitnesses[i][0])
                specie_fitnesses.remove(specie_fitnesses[i][1])
        for ind in reversed(remove_inds):
            self.species.pop(ind)
        self.update_avg_fitness()

    def next_generation(self):
        starttime = time.time()
        if self.config.remove_stagnant_species:
            self.remove_stagnant_species()
        self.population = self.generate_new_population()
        self.logger.debug(f"generate_new_population runtime = {time.time() - starttime}")
        self.update_avg_fitness()
        self.speciate()
        self.generation += 1

    def get_best_genome(self) -> QuantumNEATConfig.Genome:
        return self.population[0]