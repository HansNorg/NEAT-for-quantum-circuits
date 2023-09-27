import copy
import numpy as np
import helper as h
import genome as gen
import gate as g
import layer as l
import species as s
import random
import logging
import logger as log

class QNEAT:
    def __init__(self, population_size:int, n_qubits:int):
        self.logger = logging.getLogger("qNEAT")
        self.logger.info("QNEAT Started")

        self.global_innovation_number = h.GlobalInnovationNumber()
        self.global_layer_number = h.GlobalLayerNumber()
        self.global_species_number = h.GlobalSpeciesNumber()
        self.n_qubits = n_qubits
        self.compatibility_threshold = 3
        self.prob_mutation_without_crossover = 0.25
        self.specie_champion_size = 5
        self.percentage_survivors = 0.5
        self.generation = 0
        self.best_fitness = None

        self.population_size = population_size
        self.population = []
        for _ in range(population_size):
            genome = gen.Genome(self.global_layer_number)
            gate_type = np.random.choice(g.GateType)
            qubit = np.random.randint(n_qubits)
            gate = g.GateGene(self.global_innovation_number.next(),gate_type,qubit)
            genome.add_gate(gate)
            self.population.append(genome)

        species = s.Species(0, self.global_species_number.next())
        species.update(self.population[0], self.population.copy())
        self.species = [species]

    def generate_new_population(self, backend):
        average_fitness = np.mean([genome.get_fitness(self.n_qubits, backend) for genome in self.population])
        new_population = []
        for specie in self.species:
            total_specie_fitness = np.sum([genome.get_fitness(self.n_qubits, backend) for genome in specie.genomes])
            n_offspring = round(total_specie_fitness/average_fitness)
            cutoff = int(np.ceil(self.percentage_survivors * len(specie.genomes)))

            sorted_genomes = sorted(specie.genomes, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)
            sorted_genomes = sorted_genomes[:cutoff]

            if len(specie.genomes) >= self.specie_champion_size:
                new_population.append(copy.deepcopy(sorted_genomes[0]))
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(sorted_genomes) > 1 and random.random() > self.prob_mutation_without_crossover:
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(gen.Genome.crossover(parent1, parent2, self.n_qubits, backend))
                else:
                    new_population.append(copy.deepcopy(random.choice(sorted_genomes))) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness(self.n_qubits, backend)))
                new_population[-1].mutate(self.global_innovation_number, self.n_qubits)
            specie.empty()
        self.population = new_population  

    def speciate(self, generation):
        for genome in self.population:
            found = False
            for specie in self.species:
                distance = gen.Genome.compatibility_distance(genome, specie.representative)
                if distance < self.compatibility_threshold:
                    specie.add(genome)
                    found = True
                    break
            if not found:
                new_species = s.Species(generation, self.global_species_number.next())
                new_species.update(genome, [genome])
                self.species.append(new_species)
        for ind, specie in enumerate(self.species):
            if not specie.update_representative():
                self.species.pop(ind) # Empty species

    def run(self, max_generations = 10, backend = "ibm_perth_fake"):
        self.logger.info(f"Started running for {max_generations-self.generation} generations.")
        if self.best_fitness == None:
            # Probably not best fitness, but need to configure a value before the first run
            self.best_fitness = self.population[0].get_fitness(self.n_qubits, backend)

        fitness_record, population_size, number_of_species = [], [], []
        while self.generation < max_generations:
            self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, number of species: {len(self.species):4}, best fitness: {self.best_fitness:8.3f}")
            self.population = sorted(self.population, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)
            self.logger.debug(f"Best circuit: \n{self.population[0].get_circuit(self.n_qubits)[0].draw(fold=-1)}")
            self.best_fitness = max(self.best_fitness, self.population[0].get_fitness(self.n_qubits, backend))
            #TODO check stopping criterion
            self.generate_new_population(backend)
            self.speciate(self.generation)
            self.generation += 1

            fitness_record.append(self.best_fitness)
            population_size.append(len(self.population))
            number_of_species.append(len(self.species))
        self.logger.info(f"Generation {self.generation:8}, population size: {len(self.population):8}, number of species: {len(self.species):4}, best fitness: {self.best_fitness:8.3f}")
        self.logger.info(f"Finished running.")
        return fitness_record, population_size, number_of_species
    
    def get_best_circuit(self, backend = "ibm_perth_fake"):
        return sorted(self.population, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)[0].get_circuit(self.n_qubits)[0]

def main():
    log.QNEATLogger("qNEAT_main")
    qneat = QNEAT(10,3)
    qneat.run()

if __name__ == "__main__":
    main()