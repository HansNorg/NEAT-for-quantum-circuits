from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qulacs import ParametricQuantumCircuit

if TYPE_CHECKING:
    from quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.gene import Gene, GateGene

class Genome(ABC):
    """
    Abstract base class for genomes.
    """
    def __init__(self, config:QuantumNEATConfig) -> None:
        """
        Initialise the Genome.
        
        Parameters
        ----------
        - config: class with all the configuration settings of the algorithm.
        """
        self.logger = logging.getLogger("quantumNEAT.Genome")
        self.config = config
        self._fitness = None
        self._update_fitness = True
        self._circuit = None
        self._n_circuit_parameters = 0
        self._update_circuit = True
        self._gradient = None
        self._update_gradient = True

        self.genes:list[Gene] = [] #TODO Onur: Discuss what datatype to use

    def _changed(self):
        """Signal the Genome has changed."""
        self._update_fitness = True
        self._update_circuit = True
        self._update_gradient = True

    @abstractmethod
    def mutate(self):
        """Mutate the Genome."""
        self._changed()

    def add_gene(self, gene:Gene) -> bool:
        """
        Add a gene to the Genome. 

        Parameters
        ----------
        - gene: Gene to be added.
        
        Returns
        -------
        - bool: Whether the addition was successfull.
        """
        self.genes.append(gene)
        self._changed()
        return True

    def get_fitness(self) -> float:
        """
        Get the fitness of the Genome.
        
        Returns
        -------
        - float: Fitness of the Genome.
        """
        if not self._update_fitness: 
            # Only update fitness if the Genome has changed,
            #  as fitness calculation can be costly.
            self.update_fitness()
        return self._fitness
    
    @abstractmethod
    def update_fitness(self):
        """Update the fitness of the Genome."""
        self._update_fitness = False

    def get_circuit(self) -> tuple[Circuit, int]:
        """
        Get the circuit the corresponding to the Genome.

        Returns
        -------
        - Circuit: Circuit representing the Genome.
        - int: Number of parameters in the Circuit
        """
        if self._update_circuit: 
            # Only update the circuit if the Genome has changed,
            #  as circuit composition can be costly.
            self.update_circuit()
        return self._circuit, self._n_circuit_parameters
    
    @abstractmethod
    def update_circuit(self):
        """Update the circuit of the Genome."""
        self._update_circuit = False

    def get_gradient(self) -> float:
        """
        Get the gradient of the Genome.
        
        Returns
        -------
        - float: Gradient of the Genome.
        """
        if not self._update_gradient: 
            # Only update fitness if the Genome has changed,
            #  as fitness calculation can be costly.
            self.update_gradient()
        return self._gradient
    
    @abstractmethod
    def update_gradient(self):
        """Update the gradient of the Genome."""
        self._update_gradient = False

    def get_circuit_error(self) -> float:
        """
        Get the error of the circuit of the Genome.
        """
        return len(self.genes)*0.2 #TODO

    @staticmethod
    @abstractmethod
    def compatibility_distance(genome1:Genome, genome2:Genome, 
                               config:QuantumNEATConfig) -> float:
        """
        Get the distance between two Genomes.

        Parameters
        ----------
        - genome1, genome2: Genomes between the distance is calculated.
        - config: class with all the configuration settings of the algorithm.

        Returns
        -------
        - float: Distance between the Genomes.
        """
        return None

    @staticmethod
    @abstractmethod
    def crossover(genome1:Genome, genome2:Genome) -> Genome:
        """
        Create a child by crossover between two parent Genomes.

        Parameters
        ----------
        - genome1, genome2: Parent Genomes, should be of the same type.

        Returns
        -------
        - Genome: Child Genome of the same type as the parents.
        """
        return None

class CircuitGenome(Genome):
    """Genome consisting of GateGenes acting on qubit wires in a defined order."""

    def __init__(self, config: QuantumNEATConfig) -> None:
        super().__init__(config)
        self.genes:list[GateGene] = []
    
    def mutate(self):
        super().mutate()
        if np.random.random() < self.config.prob_add_gene_mutation:
            innovation_number = self.config.GlobalInnovationNumber.next()
            for _ in range(self.config.max_add_gene_tries):
                new_gene:GateGene = np.random.choice(self.config.gene_types)
                qubits = np.random.choice(range(self.config.n_qubits), 
                                          size = new_gene.n_qubits)
                new_gene = new_gene(innovation_number, config = self.config, 
                                    qubits = qubits)
                if self.add_gene(new_gene):
                    break
            else:
                self.config.GlobalInnovationNumber.previous()
        if np.random.random() < self.config.prob_weight_mutation:
            for gene in self.genes:
                gene.mutate_parameters()

    def update_circuit(self) -> tuple[Circuit, int]:
        super().update_circuit()
        if self.config.simulator == "qiskit":
            circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
        elif self.config.simulator == "qulacs":
            circuit = ParametricQuantumCircuit(self.config.n_qubits)
        for gene in self.genes:
            circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
        self._circuit = circuit
        self._n_circuit_parameters = n_parameters

    def update_fitness(self, fitness_function = "Default", **fitness_function_kwargs):
        super().update_fitness()
        def default():
            gradient = self.get_gradient()
            circuit_error = self.get_circuit_error()
            return 1/(1+circuit_error)*gradient
        if fitness_function == "Default":
            self._fitness = default()
        self._fitness = fitness_function(self, **fitness_function_kwargs)

    def update_gradient(self) -> float:
        super().update_gradient()
        circuit, n_parameters = self.get_circuit()
        parameters = np.array([])
        for gene in self.genes:
            parameters = np.append(parameters, gene.parameters)
        self._gradient = self.config.gradient_function(circuit, n_parameters, 
                                                       parameters, self.config)

    @staticmethod
    def compatibility_distance(genome1:Genome, genome2:Genome, config:QuantumNEATConfig):
        def line_up(genome1:Genome, genome2:Genome):
            matching = 0
            disjoint = 0
            excess = 0
            distances = []
            
            n_genes1, n_genes2 = len(genome1.genes), len(genome2.genes)
            index1, index2 = 0, 0
            while index1 < n_genes1 or index2 < n_genes2:
                gene1 = genome1.genes[index1] if index1 < n_genes1 else None
                gene2 = genome2.genes[index2] if index2 < n_genes2 else None

                if gene1 and gene2:
                    if gene1.innovation_number < gene2.innovation_number:
                        disjoint += 1
                        index1 += 1
                    elif gene1.innovation_number > gene2.innovation_number:
                        disjoint += 1
                        index2 += 1
                    else:
                        distances.append(GateGene.get_distance(gene1, gene2))
                        matching += 1
                        index1 += 1
                        index2 += 1
                elif gene1:
                    excess += n_genes1 - index1
                    break
                elif gene2:
                    excess += n_genes2 - index2
                    break
                else:
                    genome1.logger.warning("Don't think this should ever occur.", exc_info=1)
                    break

            if len(distances) == 0:
                distances = [0]
            return matching, disjoint, excess, np.mean(distances)
        n_genes = max(len(genome1.genes), len(genome2.genes))
        matching, disjoint, excess, avg_distance = line_up(genome1, genome2)
        return config.c1*excess/n_genes + config.c2*disjoint/n_genes + config.c3*avg_distance
    
    @staticmethod
    def crossover(genome1: Genome, genome2: Genome) -> Genome:
        child = CircuitGenome(genome1.config)
        if genome1.get_fitness() > genome2.get_fitness():
            better = "genome1"
        elif genome1.get_fitness() < genome2.get_fitness():
            better = "genome2"
        else:
            better = np.random.choice(["genome1", "genome2"])

        n_genes1, n_genes2 = len(genome1.genes), len(genome2.genes)
        index1, index2 = 0, 0
        while index1 < n_genes1 or index2 < n_genes2:
            gene1 = genome1.genes[index1] if index1 < n_genes1 else None
            gene2 = genome2.genes[index2] if index2 < n_genes2 else None

            if gene1 and gene2:
                if gene1.innovation_number < gene2.innovation_number:
                    if "genome1" == better:
                        if not child.add_gene(copy.deepcopy(gene1)):
                            child.logger.error("Child did not add gene of parent.")
                    index1 += 1
                elif gene1.innovation_number > gene2.innovation_number:
                    if "genome2" == better:
                        if not child.add_gene(copy.deepcopy(gene2)):
                            child.logger.error("Child did not add gene of parent.")
                    index2 += 1
                else:
                    if "genome1" == better:
                        if not child.add_gene(copy.deepcopy(gene1)):
                            child.logger.warning("Child did not add gene of parent.")
                    elif "genome2" == better:
                        if not child.add_gene(copy.deepcopy(gene2)):
                            child.logger.warning("Child did not add gene of parent.")
                    else:
                        child.logger.error(f"better should have value 'genome1' or 'genome2' not {better}.", exc_info=1)
                        raise ValueError(f"better should have value 'genome1' or 'genome2' not {better}.")
                    index1 += 1
                    index2 += 1
            elif gene1:
                if not child.add_gene(copy.deepcopy(gene1)):
                    child.logger.warning("Child did not add gene of parent.")
                index1 += 1
            elif gene2:
                if not child.add_gene(copy.deepcopy(gene2)):
                    child.logger.warning("Child did not add gene of parent.")
                index2 += 1
            else:
                child.logger.warning("Don't think this should ever occur.", exc_info=1)
                break
        return child