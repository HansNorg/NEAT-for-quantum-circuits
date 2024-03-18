from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qulacs import ParametricQuantumCircuit

from quantumneat.configuration import QuantumNEATConfig
from quantumneat.implementations.gates import GateCNOT, GateROT, GateRx, GateRy, GateRz
from quantumneat.gene import GateGene
from quantumneat.genome import CircuitGenome

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