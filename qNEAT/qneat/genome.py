import helper as h
import gate as g
import layer as l
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np

class Genome(object):

    def __init__(self, global_layer_number) -> None:
        self.layers = {}
        self.global_layer_number = global_layer_number
        self._fitness = None
        self._update_fitness = True

    @classmethod
    def from_layers(cls, global_layer_numer, layers):
        genome = Genome(global_layer_numer)
        genome.layers = layers
        return genome

    def add_gate(self, gate:g.GateGene) -> bool:
        ind = np.random.randint(self.global_layer_number.current() + 1)
        if ind == self.global_layer_number.current():
            self.global_layer_number.next()
        if ind not in self.layers:
            new_layer = l.Layer(ind)
            self.layers[ind] = new_layer
        gate_added = self.layers[ind].add_gate(gate)
        if gate_added:
            self._update_fitness = True
        return gate_added

    def get_circuit(self, n_qubits, n_parameters = 0) -> (QuantumCircuit, int):
        circuit = QuantumCircuit(QuantumRegister(n_qubits))
        for layer_ind in self.layers:
            circuit, n_parameters = self.layers[layer_ind].add_to_circuit(circuit, n_parameters)
        return circuit, n_parameters
    
    def get_fitness(self, n_qubits, backend = "ibm_perth"):
        if not self._update_fitness: 
            # If the genome hasn't changed, return the already calculated fitness
            return self._fitness
        
        circuit, n_parameters = self.get_circuit(n_qubits)
        gradient = self.compute_gradient(circuit, n_parameters)
        configured_circuit, backend = h.configure_circuit_to_backend(circuit, backend)
        # if not type(backend) == str: # Don't know why
        #     CHIP_BACKEND = backend
        circuit_error = h.get_circuit_properties(configured_circuit, backend)
        self._fitness = 1/(1+circuit_error)*gradient
        self._update_fitness = False
        return self._fitness

    def compute_gradient(self, circuit, n_parameters, shots = 1024, epsilon = 10**-5):
        if n_parameters == 0:
            return 0 # Prevent division by 0
        total_gradient = 0
        parameters = np.array([])
        for layer in self.layers.values():
            for gate in np.ravel([layer.gates[key] for key in layer.gates.keys()]):
                parameters = np.append(parameters, gate.parameters)
        
        for ind in range(n_parameters):
            temp = parameters[ind]
            parameters[ind] += epsilon/2
            partial_gradient = h.energy_from_circuit(circuit, parameters, shots)
            parameters[ind] -= epsilon
            partial_gradient -= h.energy_from_circuit(circuit, parameters, shots)
            parameters[ind] = temp # Return the parameter to original value
            total_gradient += partial_gradient**2
        return total_gradient/n_parameters

    @staticmethod
    def line_up(genome1, genome2):
        #TODO Implement excess, atm excess is counted with disjoint
        n_layers = genome1.global_layer_number.current()
        matching = 0
        disjoint = 0
        excess = 0
        distances = []
        for layer_ind in range(n_layers):
            in_layer1 = layer_ind in genome1.layers
            in_layer2 = layer_ind in genome2.layers
            if (not in_layer1) and (not in_layer2):
                continue
            elif not in_layer1:
                disjoint += len(genome2.layers[layer_ind].gates)
            elif not in_layer2:
                disjoint += len(genome1.layers[layer_ind].gates)
            for type in g.GateType:
                type_in_layer1 = type.name in genome1.layers[layer_ind].gates
                type_in_layer2 = type.name in genome2.layers[layer_ind].gates
                if (not type_in_layer1) and (not type_in_layer2):
                    continue
                elif not type_in_layer1:
                    disjoint += len(genome2.layers[layer_ind].gates[type.name])
                    continue
                elif not type_in_layer2:
                    disjoint += len(genome1.layers[layer_ind].gates[type.name])
                    continue
                gates2 = genome2.layers[layer_ind].gates[type.name].copy()
                for gate1 in genome1.layers[layer_ind].gates[type.name]:
                    found = False
                    for gate2 in gates2:
                        if gate1.qubit != gate2.qubit:
                            continue
                        matching += 1
                        parametergate, distance = g.GateGene.get_distance(gate1, gate2)
                        if parametergate:
                            distances.append(distance)
                        gates2.remove(gate2)
                        found = True
                        break
                    if not found:
                        disjoint += 1
                disjoint += len(gates2)
        if len(distances) == 0:
            distances = 0
        return matching, disjoint, excess, np.average(distances)

    @staticmethod
    def compatibility_distance(genome1, genome2, c1=1, c2=1, c3=1):
        def get_n_genes(genome):
            n_genes = 0
            for layer_ind in genome.layers:
                gates = genome.layers[layer_ind].gates
                n_genes += np.sum([len(gates[key]) for key in gates.keys()])
            return n_genes
        n_genes = np.maximum(get_n_genes(genome1), get_n_genes(genome2))
        matching, disjoint, excess, avg_distance = Genome.line_up(genome1, genome2)
        return c1*excess/n_genes + c2*disjoint/n_genes + c3*avg_distance