from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union
from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from quantumneat.configuration import QuantumNEATConfig as C
from qulacs import ParametricQuantumCircuit

from quantumNEAT.quantumneat import helper as h

if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig as C
    from quantumNEAT.quantumneat.gene import Gene, GateGene, Circuit

class Genome(ABC):
    def __init__(self, config:C) -> None:
        super().__init__()
        self.config = config
        self._fitness = None
        self._update_fitness = True

        self.genes:list[Gene] = [] #TODO Onur: Discuss what datatype to use

    @abstractmethod
    def mutate(self):
        self._update_fitness = True

    def add_gene(self, gene:Gene) -> bool:
        """Try to add a gene, return True if successful, False otherwise."""
        self.genes.append(gene)
        return True

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
    def __init__(self, config: C) -> None:
        super().__init__(config)
        self.genes:list[GateGene] = []
    
    def mutate(self):
        super().mutate()
        if np.random.random() < self.config.prob_add_gene_mutation:
            for _ in range(self.config.max_add_gene_tries):
                new_gene:GateGene = np.random.choice(self.config.GeneTypes)
                qubits = np.random.choice(range(self.config.n_qubits), size = new_gene.n_qubits)
                new_gene = new_gene(self.config.global_innovation_number.next(), config = self.config, qubits = qubits)
                if self.add_gene(new_gene):
                    break
        if np.random.random() < self.config.prob_weight_mutation:
            for gene in self.genes:
                gene.mutate_parameters()

    def get_circuit(self, n_parameters:int = 0) -> tuple[Circuit, int]:
        # TODO Look at changes when using ZX-calc
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
        for gene in self.genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        return circuit, n_parameters
    
    def get_circuit_error(self, circuit:Circuit) -> float:
        return len(self.genes)*0.2 #TODO

    def update_fitness(self) -> float:
        super().update_fitness()
        circuit, n_parameters = self.get_circuit(self.config.n_qubits)
        gradient = self.compute_gradient(circuit, n_parameters)
        circuit_error = self.get_circuit_error(circuit)
        self._fitness = 1/(1+circuit_error)*gradient #TODO Update
        return self._fitness

    def compute_gradient(self, circuit:Circuit, n_parameters, shots = 10240, epsilon = 10**-5):
        #TODO Update/Change
        if n_parameters == 0:
            return 0 # Prevent division by 0
        total_gradient = 0
        parameters = np.array([])
        for gene in self.genes:
            parameters = np.append(parameters, gene.parameters)
        
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