import copy
import numpy as np
import random
import logging
from . import logger as log

from . import configuration as c
from . import helper as h

class QuantumNEAT:
    def __init__(self, config:c.QuantumNEATConfig):
        self.config = config

        self.logger = logging.getLogger("QuantumNEAT")
        self.logger.info("QuantumNEAT Started")

        self.generation = 0
        self.best_fitness = None
        self.population = []
        self.species = []

        self.generate_initial_population()
        self.speciate()

        # For experimenting only
        self.average_fitnesses = []

    def generate_initial_population(self):
        for _ in range(self.config.population_size):
            genome = self.config.Genome(self.config)
            gate_type = np.random.choice(self.config.GateType)
            qubit = np.random.randint(self.config.n_qubits)
            gate = self.config.GateGene(self.config.global_innovation_number.next(),gate_type,qubit, self.config)
            genome.add_gate(gate)
            self.population.append(genome)
        self.population = self.sort_genomes(self.population)

    @staticmethod
    def sort_genomes(genomes):
        sorted(genomes, key=lambda genome: genome.get_fitness(), reverse=True)

    def generate_new_population(self):
        average_fitness = np.mean([genome.get_fitness(self.config.n_qubits, self.config.backend) for genome in self.population])
        self.logger.debug(f"{average_fitness =}")
        self.average_fitnesses.append(average_fitness)
        new_population = []
        for specie in self.species:
            total_specie_fitness = np.sum([genome.get_fitness(self.config.n_qubits, self.config.backend) for genome in specie.genomes])
            n_offspring = round(total_specie_fitness/average_fitness)
            cutoff = int(np.ceil(self.config.percentage_survivors * len(specie.genomes)))
        
            sorted_genomes = self.sort_genomes(specie.genomes)[:cutoff]

            if len(specie.genomes) >= self.specie_champion_size:
                new_population.append(copy.deepcopy(sorted_genomes[0]))
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(sorted_genomes) > 1 and random.random() > self.prob_mutation_without_crossover:
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(self.config.Genome.crossover(parent1, parent2, self.config.n_qubits, self.config.backend))
                else:
                    new_population.append(copy.deepcopy(random.choice(sorted_genomes))) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness(self.n_qubits, backend)))
                new_population[-1].mutate(self.config.global_innovation_number, self.config.n_qubits)
            specie.empty()
        self.population = new_population  

    def speciate(self):
        for genome in self.population:
            found = False
            for specie in self.species:
                distance = self.config.Genome.compatibility_distance(genome, specie.representative)
                if distance < self.compatibility_threshold:
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

    def run_generation(self, backend):
        self.population = sorted(self.population, key=lambda genome: genome.get_fitness(self.config.n_qubits, backend), reverse=True)
        self.logger.debug(f"Best circuit: \n{self.population[0].get_circuit(self.config.n_qubits)[0].draw(fold=-1)}")
        self.best_fitness = max(self.best_fitness, self.population[0].get_fitness(self.config.n_qubits, backend))
        #TODO check stopping criterion
        self.generate_new_population(backend)
        self.speciate()
        self.generation += 1
        
    def run(self, max_generations = 10, backend = "ibm_perth_fake"):
        self.logger.info(f"Started running for {max_generations-self.generation} generations.")
        if self.best_fitness == None:
            # Probably not best fitness, but need to configure a value before the first run
            self.best_fitness = self.population[0].get_fitness(self.config.n_qubits, backend)

        fitness_record, population_size, number_of_species = [], [], []
        while self.generation < max_generations:
            self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, number of species: {len(self.species):4}, best fitness: {self.best_fitness:8.3f}")
            self.run_generation(backend)    

            fitness_record.append(self.best_fitness)
            population_size.append(len(self.population))
            number_of_species.append(len(self.species))
        self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, number of species: {len(self.species):4}, best fitness: {self.best_fitness:8.3f}")
        self.logger.info(f"Finished running.")
        return fitness_record, population_size, number_of_species, self.average_fitnesses
    
    def get_best_circuit(self, backend = "ibm_perth_fake"):
        return sorted(self.population, key=lambda genome: genome.get_fitness(self.config.n_qubits, backend), reverse=True)[0].get_circuit(self.config.n_qubits)[0]

def main():
    log.QuantumNEATLogger("qNEAT_main")
    settings = c.QuantumNEATConfig(3, 10)
    quantum_neat = QuantumNEAT(settings)
    quantum_neat.run()

if __name__ == "__main__":
    main()