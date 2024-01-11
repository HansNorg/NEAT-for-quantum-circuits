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
if TYPE_CHECKING:
    from quantumneat.configuration import Circuit
    from quantumneat.problem import Problem

class GateCNOT(GateGene):
    n_qubits = 2

    def __init__(self, innovation_number: int, config: QuantumNEATConfig, problem:Problem, qubits: list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, problem, qubits, **kwargs)
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

class GateRx(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RX_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
        elif self.config.simulator == 'qiskit':
            circuit.rx(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateRy(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RY_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
        elif self.config.simulator == 'qiskit':
            circuit.ry(Parameter(str(n_parameters)), self.qubits[0])
            n_parameters += 1
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateRz(GateGene):
    n_qubits = 1
    n_parameters = 1
    
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_parametric_RZ_gate(self.qubits[0], self.parameters[0])
            n_parameters += 1
        elif self.config.simulator == 'qiskit':
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
            
        self.problem.add_encoding_layer(circuit)
        for gene in self.genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        self._circuit = circuit
        self._n_circuit_parameters = n_parameters
    
    def update_gradient(self):
        super().update_gradient()
        circuit, n_parameters = self.get_circuit()
        parameters = self.get_parameters()
        self._gradient = self.problem.gradient(circuit,parameters,n_parameters)
        self._energy = self.problem.energy(circuit, parameters)

    def evaluate(self, N = 100, **kwargs):
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        for gene in self.genes:
            parameters = np.append(parameters, gene.get_parameters())
        return self.problem.energy(circuit, parameters)

@dataclass
class LinearGrowthConfig(QuantumNEATConfig):
    gene_types:list[GateGene] = field(default_factory=lambda:[GateROT, GateCNOT])
    Genome = LinearGrowthGenome

@dataclass
class LinearGrowthConfigSeparate(QuantumNEATConfig):
    gene_types:list[GateGene] = field(default_factory=lambda:[GateRx, GateRy, GateRz, GateCNOT])
    Genome = LinearGrowthGenome