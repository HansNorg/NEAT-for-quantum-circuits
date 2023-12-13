from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qulacs import ParametricQuantumCircuit

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.gene import GateGene
from quantumneat.genome import CircuitGenome
from quantumneat.problems.fox_in_the_hole import energy as fith_energy
if TYPE_CHECKING:
    from quantumneat.configuration import Circuit

class GateCNOT(GateGene):
    n_qubits = 2

    def __init__(self, innovation_number: int, config: QuantumNEATConfig, qubits: list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, qubits, **kwargs)
        # self.qubits = [qubit%self.config.n_qubits for qubit in self.qubits]
        # self.logger.debug(f"{self.qubits=}")

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
            circuit.rx(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.ry(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
            circuit.rz(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters

class LinearGrowthGenome(CircuitGenome):        
    def update_circuit(self): 
        super().update_circuit()
        n_parameters = 0
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
            
        self.config.encoding_layer(circuit)
        for gene in self.genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        self._circuit = circuit
        self._n_circuit_parameters = n_parameters
    
    def update_gradient(self):
        super().update_gradient()
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        # self.logger.debug(f"{self.genes.values()=}")
        for gene in self.genes:
            # self.logger.debug(f"{gene.get_parameters()=}")
            parameters = np.append(parameters, gene.get_parameters())
        # self.logger.debug(f"{n_parameters==len(parameters)=}; {parameters=}")
        self._gradient = self.config.gradient_function(circuit, n_parameters, 
                                                       parameters, self.config)
        self._energy = self.config.energy_function(circuit, parameters, self.config)

    def evaluate(self, N = 100, **kwargs):
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        # self.logger.debug(f"{self.genes.values()=}")
        for gene in self.genes:
            # self.logger.debug(f"{gene.get_parameters()=}")
            parameters = np.append(parameters, gene.get_parameters())
        return fith_energy(self.config, self.get_circuit()[0], parameters, self.config, N=N)

@dataclass
class LinearGrowthConfig(QuantumNEATConfig):
    gene_types:list[GateGene] = field(default_factory=lambda:[GateROT, GateCNOT])
    Genome = LinearGrowthGenome

if __name__ == "__main__":
    config = LinearGrowthConfig(5, 10)
    LinearGrowthGenome(config)