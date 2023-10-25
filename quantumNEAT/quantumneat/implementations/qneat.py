# from __future__ import annotations

import numpy as np
# from typing import TYPE_CHECKING
from qiskit.circuit import Parameter

from quantumNEAT.quantumneat.gene import GateGene, GeneTypes, Circuit
from quantumNEAT.quantumneat.helper import Singleton
from quantumNEAT.quantumneat.genome import CircuitGenome
from quantumNEAT.quantumneat.configuration import QuantumNEATConfig
# if TYPE_CHECKING:

class GlobalLayerNumber(Singleton):
    '''
    Class for keeping a global layer number.
    
    Layer number starts at 0.
    '''
    def __init__(self):
        self._layer_number:int = 0

    def next(self):
        '''
        Get the next layer number.

        Increments the layer number.
        '''
        self._layer_number += 1
        return self._layer_number
    
    def current(self):
        return self._layer_number
    
class GateCNOT(GateGene):
    n_qubits = 2

    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_CNOT_gate(self.qubits[0], self.qubits[1])
        elif self.config.simulator == 'qiskit':
            circuit.cnot(self.qubits[0], self.qubits[1])
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateROT(GateGene):
    n_qubits = 1
    n_parameters = 3
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RX_gate(self.qubits[0], self.parameters[0])
            circuit.add_parametric_RY_gate(self.qubits[0], self.parameters[1])
            circuit.add_parametric_RZ_gate(self.qubits[0], self.parameters[2])
            n_parameters += 3
        elif self.config.simulator == 'qiskit':
            circuit.rx(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
            circuit.ry(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
            circuit.rz(Parameter(n_parameters), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class QNEAT_GeneTypes(GeneTypes):
    ROT = GateROT
    CNOT = GateCNOT

class LayerGene(GateGene):
    #TODO Look at only adding gates in qubit order

    def __init__(self, config:QuantumNEATConfig, ind:int) -> None:
        super().__init__(self, config)#TODO HIER BEZIG
        self.genes:dict[GateGene] = {}
        self.ind = ind

    def add_gate(self, gate:GateGene) -> bool:
        if type(gate) in self.genes:
            for existing_gate in self.genes[gate.gatetype.name]:
                if gate.qubit == existing_gate.qubit:
                    # Don't add the same gate multiple times
                    return False
            self.genes[gate.gatetype.name].append(gate)
        else:
            self.genes[gate.gatetype.name] = [gate]
        return True

    def add_to_circuit(self, circuit, n_parameters):
        for genetype in GeneTypes:
            if genetype.name in self.genes:
                for gate in self.genes[genetype.name]:
                    circuit, n_parameters = gate.add_to_circuit(circuit, n_parameters)
        circuit.barrier()
        return circuit, n_parameters
    
    def get_gates_generator(self):
        for key in self.genes.keys():
            for gate in self.genes[key]:
                yield gate

class QNEAT_Genome(CircuitGenome):
    def __init__(self, config:QuantumNEATConfig) -> None:
        super().__init__(config)
        self.genes:dict[int,LayerGene] = {}

    def add_gene(self, gene) -> bool:
        ind = np.random.randint(self.config.global_layer_number.current() + 1)
        if ind == self.global_layer_number.current():
            self.global_layer_number.next()
        if ind not in self.genes.keys():
            new_layer = LayerGene(self.config, ind)
            self.genes[ind] = new_layer
        gene_added = self.genes[ind].add_gate(gene)
        if gene_added:
            self._update_fitness = True
        return gene_added

class QNEAT_Config(QuantumNEATConfig):
    Genome = QNEAT_Genome
    GeneTypes = QNEAT_GeneTypes
    global_layer_number = GlobalLayerNumber()