import numpy as np
import helper as h
import genome as gen
import gate as g
import layer as l

class QNEAT:
    def __init__(self, population_size:int, n_qubits:int):
        self.global_innovation_number = h.GlobalInnovationNumber()
        self.global_layer_number = h.GlobalLayerNumber()
        self.n_qubits = n_qubits

        self.population = []
        for _ in range(population_size):
            genome = gen.Genome(self.global_layer_number)
            gate_type = np.random.choice(g.GateType)
            qubit = np.random.randint(n_qubits)
            gate = g.GateGene(self.global_innovation_number.next(),gate_type,qubit)
            genome.add_gate(gate)
            self.population.append(genome)

    def run(self, backend = "ibm_perth"):
        
        sorted_population = sorted(self.population, key=lambda genome: genome.get_fitness(self.n_qubits, backend), reverse=True)

def main():
    qneat = QNEAT()
    pass

if __name__ == "__main__":
    main()