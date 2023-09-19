import numpy as np
import helper as h
import genome as gen
import gate as g
import layer as l
import species as s

class QNEAT:
    def __init__(self, population_size:int, n_qubits:int):
        self.global_innovation_number = h.GlobalInnovationNumber()
        self.global_layer_number = h.GlobalLayerNumber()
        self.global_species_number = h.GlobalSpeciesNumber()
        self.n_qubits = n_qubits

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
        average_fitness = np.mean((genome.get_fitness(self.n_qubits, backend) for genome in self.population))
        for specie in self.species:
            total_specie_fitness = np.sum(genome.get_fitness(self.n_qubits, backend) for genome in specie.genomes)
            n_offspring = round(total_specie_fitness/average_fitness)
            ...

    def speciate(self):
        pass

    def run(self, max_generations = 10, backend = "ibm_perth"):
        for generation in range(max_generations):
            self.population = sorted(self.population, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)
            #TODO check stopping criterion
            self.generate_new_population(backend)
            self.speciate()

def main():
    qneat = QNEAT()
    pass

if __name__ == "__main__":
    main()