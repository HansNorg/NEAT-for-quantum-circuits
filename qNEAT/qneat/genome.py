from gene import GenericGene, GateGene, NonChronologicalGateGene
import helper as h
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import numpy as np

class Genome(object):
    '''
    
    '''
    def __init__(self, n_qubits) -> None:
        self.genes = {}
        self.n_qubits = n_qubits

    def add_gene(self, gene:GenericGene) -> None:
        self.genes[gene.innovation_number] = gene

    def get_circuit(self, n_parameters = 0) -> QuantumCircuit:
        circuit = QuantumCircuit(QuantumRegister(self.n_qubits))
        key_order = self.get_gate_order()
        for key in key_order:
            gene = self.genes[key]
            if isinstance(gene, GateGene):
                circuit, n_parameters = gene.to_gate(circuit, n_parameters)
            else:
                raise NotImplementedError("Genetype not implemented")
        return circuit, n_parameters
    
    def get_gate_order(self):
        sorted_keys = np.sort(self.genes.keys)
        return sorted_keys
    
class NonChronologicalGenome(Genome):

    def get_gate_order(self):
        sorted_keys = np.sort(self.genes.keys)
        order = []
        for key in sorted_keys:
            gene = self.genes[key]
            if isinstance(gene, NonChronologicalGateGene):
                location = gene.location
                if location == 0:
                    order.append(location)
                elif location > 0:
                    order.insert(order.index(location)+1)
                elif location < 0:
                    order.insert(order.index(-location))
            else:
                raise NotImplementedError("NonChronological different genetype not implemented")
        return order