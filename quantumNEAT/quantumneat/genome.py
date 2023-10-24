from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qulacs import ParametricQuantumCircuit

from quantumNEAT.quantumneat import helper as h

if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig as C

class Genome(ABC):
    def __init__(self, config:C) -> None:
        super().__init__()
        self.config = config
        self._fitness = None
        self._update_fitness = True

        self.genes:list[C.Gene] = [] #TODO Onur: Discuss what datatype to use

    @abstractmethod
    def mutate(self):
        self._update_fitness = True

    def get_fitness(self) -> float:
        if not self._update_fitness: 
            # If the genome hasn't changed, return the already calculated fitness as fitness calculation can be costly.
            return self._fitness
        self.update_fitness()
        return self._fitness
    
    @abstractmethod
    def update_fitness(self):
        self._update_fitness = False

    @abstractmethod
    @staticmethod
    def compatibility_distance(self, genome1:Genome, genome2:Genome, config:C) -> float:
        return None

    @abstractmethod
    @staticmethod
    def crossover(genome1:Genome, genome2:Genome) -> C.Genome:
        return None

class CircuitGenome(Genome):
    
    def mutate(self):
        super().mutate()
        if np.random.random() < self.config.prob_add_gate_mutation:
            ... #TODO
        if np.random.random() < self.config.prob_weight_mutation:
            for gene in self.genes:
                gene.mutate_parameters()

    def get_circuit(self, n_parameters = 0) -> tuple[Union[QuantumCircuit, ParametricQuantumCircuit], int]:
        # TODO Look at changes when using ZX-calc
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
        for gene in self.genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        return circuit, n_parameters
    
    def update_fitness(self) -> float:
        super().update_fitness()
        circuit, n_parameters = self.get_circuit(self.config.n_qubits)
        gradient = self.compute_gradient(circuit, n_parameters)
        circuit_error = ... #TODO
        self._fitness = 1/(1+circuit_error)*gradient #TODO Update
        return self._fitness

    def compute_gradient(self, circuit:Union[QuantumCircuit, ParametricQuantumCircuit], n_parameters, shots = 10240, epsilon = 10**-5):
        #TODO Update/Change
        if n_parameters == 0:
            return 0 # Prevent division by 0
        total_gradient = 0
        parameters = np.array([])
        for layer in self.layers.values():
            for gate in layer.get_gates_generator():
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
    def compatibility_distance(genome1:Genome, genome2:Genome, config:C):
        def line_up(genome1:Genome, genome2:Genome):
            matching = 0
            disjoint = 0
            excess = 0
            distances = []
            ... #TODO
            if len(distances) == 0:
                distances = [0]
            return matching, disjoint, excess, np.mean(distances)
        n_genes = max(len(genome1.genes), len(genome2.genes))
        matching, disjoint, excess, avg_distance = line_up(genome1, genome2)
        return config.c1*excess/n_genes + config.c2*disjoint/n_genes + config.c3*avg_distance