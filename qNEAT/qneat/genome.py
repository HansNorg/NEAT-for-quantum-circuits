import helper as h
import gate as g
import layer as l
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import numpy as np

class Genome(object):

    def __init__(self, global_layer_number) -> None:
        self.layers = {}
        self.global_layer_number = global_layer_number

    def add_gate(self, gate:g.GateGene) -> bool:
        ind = np.random.randint(self.global_layer_number.current() + 1)
        if ind == self.global_layer_number.current():
            self.global_layer_number.next()
        if ind not in self.layers:
            new_layer = l.Layer(ind)
            self.layers[ind] = new_layer
        return self.layers[ind].add_gate(gate)

    def get_circuit(self, n_qubits, n_parameters = 0) -> QuantumCircuit:
        circuit = QuantumCircuit(QuantumRegister(n_qubits))
        for layer_ind in self.layers:
            circuit, n_parameters = self.layers[layer_ind].add_to_circuit(circuit, n_parameters)
        return circuit, n_parameters