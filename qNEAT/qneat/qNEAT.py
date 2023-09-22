import numpy as np
import helper as h
import genome as gen
import gate as g
import layer as l
import species as s
import random

class QNEAT:
    def __init__(self, population_size:int, n_qubits:int):
        self.global_innovation_number = h.GlobalInnovationNumber()
        self.global_layer_number = h.GlobalLayerNumber()
        self.global_species_number = h.GlobalSpeciesNumber()
        self.n_qubits = n_qubits
        self.compatibility_threshold = 3
        self.prob_mutation_without_crossover = 0.25
        self.specie_champion_size = 5
        self.percentage_survivors = 0.2

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
            new_population = []
            cutoff = int(np.ceil(self.percentage_survivors * len(specie.genomes)))

            sorted_genomes = sorted(specie.genomes, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)
            sorted_genomes = sorted_genomes[:cutoff]

            if len(specie.genomes) >= self.specie_champion_size:
                new_population.append(sorted_genomes[0])
                n_offspring -= 1
            for _ in range(n_offspring):
                if len(specie.genomes) > 1 and random.random() > self.prob_mutation_without_crossover:
                    parent1, parent2 = random.sample(sorted_genomes, 2)
                    new_population.append(gen.Genome.crossover(parent1, parent2, self.n_qubits, backend))
                else:
                    new_population.append(random.choice(sorted_genomes)) # Possibility: choosing probability based on fitness , p = lambda genome: genome.get_fitness(self.n_qubits, backend)))
                new_population[-1].mutate(self.global_innovation_number, self.n_qubits)
            specie.empty()
        self.population = new_population
        print(len(new_population), len(self.species))

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
        for generation in range(max_generations):
            self.population = sorted(self.population, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)
            #TODO check stopping criterion
            self.generate_new_population(backend)
            self.speciate(generation)
        for genome in self.population:
            print(genome.get_circuit(self.n_qubits)[0])

def main():
    qneat = QNEAT()
    pass

if __name__ == "__main__":
    main()