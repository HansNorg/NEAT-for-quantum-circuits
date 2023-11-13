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
    logger = logging.getLogger("quantumNEAT.Genome")
    
    def __init__(self, config:QuantumNEATConfig) -> None:
        """
        Initialise the Genome.
        
        Parameters
        ----------
        - config: class with all the configuration settings of the algorithm.
        """
        self.config = config
        self._fitness = None
        self._update_fitness = True
        self._circuit = None
        self._n_circuit_parameters = 0
        self._update_circuit = True
        self._gradient = None
        self._energy = None
        self._update_gradient = True

        self.genes:list[Gene] = [] #TODO Onur: Discuss what datatype to use

    def _changed(self):
        """Signal the Genome has changed."""
        # self.logger.debug("changed")
        self._update_fitness = True
        self._update_circuit = True
        self._update_gradient = True

    @abstractmethod
    def mutate(self):
        """Mutate the Genome."""
        # self.logger.debug("mutate")
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
        # self.logger.debug("add_gene")
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
        # self.logger.debug("get_fitness")
        if self._update_fitness: 
            # Only update fitness if the Genome has changed,
            #  as fitness calculation can be costly.
            self.update_fitness()
        return self._fitness
    
    @abstractmethod
    def update_fitness(self):
        """Update the fitness of the Genome."""
        # self.logger.debug("update_fitness")
        self._update_fitness = False

    def get_circuit(self) -> tuple[Circuit, int]:
        """
        Get the circuit the corresponding to the Genome.

        Returns
        -------
        - Circuit: Circuit representing the Genome.
        - int: Number of parameters in the Circuit
        """
        # self.logger.debug("get_circuit")
        if self._update_circuit: 
            # Only update the circuit if the Genome has changed,
            #  as circuit composition can be costly.
            self.update_circuit()
        return self._circuit, self._n_circuit_parameters
    
    @abstractmethod
    def update_circuit(self):
        """Update the circuit of the Genome."""
        # self.logger.debug("update_circuit")
        self._update_circuit = False

    def get_gradient(self) -> float:
        """
        Get the gradient of the Genome.
        
        Returns
        -------
        - float: Gradient of the Genome.
        """
        # self.logger.debug("get_gradient")
        if self._update_gradient: 
            # Only update fitness if the Genome has changed,
            #  as fitness calculation can be costly.
            self.update_gradient()
        return self._gradient
    
    def get_energy(self):
        if self._update_gradient: 
            # Only update fitness if the Genome has changed,
            #  as fitness calculation can be costly.
            self.update_gradient()
        return self._energy
    
    @abstractmethod
    def update_gradient(self):
        """Update the gradient of the Genome."""
        # self.logger.debug("update_gradient")
        self._update_gradient = False

    def get_circuit_error(self) -> float:
        """
        Get the error of the circuit of the Genome.
        """
        # self.logger.debug("get_circuit_error")
        # self.logger.debug(len(self.genes)*0.2)
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
        Genome.logger.warning("Abstract compatibility_distance should not be called")
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
        # super().mutate() # Don't change the parameters in case of unsuccesful mutation
        if np.random.random() < self.config.prob_add_gene_mutation:
            self.mutate_add_gene()
        if np.random.random() < self.config.prob_weight_mutation:
            self._changed = True
            for gene in self.genes:
                gene.mutate_parameters()

    def mutate_add_gene(self):
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

    # def update_circuit(self):
    #     super().update_circuit()
    #     n_parameters = 0
    #     if self.config.simulator == "qiskit":
    #         circuit = QuantumCircuit(QuantumRegister(self.config.n_qubits))
    #     elif self.config.simulator == "qulacs":
    #         circuit = ParametricQuantumCircuit(self.config.n_qubits)
    #     for gene in self.genes:
    #         circuit, n_parameters = gene.add_to_circuit(circuit, n_parameters)
    #     self._circuit = circuit
    #     self._n_circuit_parameters = n_parameters

    def update_fitness(self, fitness_function = "Default", **fitness_function_kwargs):
        super().update_fitness()
        def default():
            # self.logger.debug("Default fitness function")
            gradient = self.get_gradient()
            circuit_error = self.get_circuit_error()
            energy = self.get_energy()
            # return 1/(1+circuit_error)*gradient
            return 1/(1+circuit_error)*(-energy)+gradient
        if fitness_function == "Default":
            self._fitness = default()
        else:
            self._fitness = fitness_function(self, **fitness_function_kwargs)

    # def update_gradient(self) -> float:
    #     super().update_gradient()
    #     circuit, n_parameters = self.get_circuit()
    #     parameters = np.array([])
    #     for gene in self.genes:
    #         parameters = np.append(parameters, gene.parameters)
    #     self._gradient = self.config.gradient_function(circuit, n_parameters, 
    #                                                    parameters, self.config)

    @staticmethod
    def compatibility_distance(genome1:Genome, genome2:Genome, config:QuantumNEATConfig):
        # Genome.logger.debug("compatibility_distance")
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
                        distances.append(gene1.get_distance(gene1, gene2))
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
                    Genome.logger.warning("Don't think this should ever occur.", exc_info=1)
                    break

            if len(distances) == 0:
                distances = [0]
            return matching, disjoint, excess, np.mean(distances)
        n_genes = max(len(genome1.genes), len(genome2.genes))
        if n_genes == 0:            
            # If both genomes are empty, distance == 0, prevent division by 0.
            return 0
        matching, disjoint, excess, avg_distance = line_up(genome1, genome2)
        return config.excess_coefficient*excess/n_genes + config.disjoint_coefficient*disjoint/n_genes + config.weight_coefficient*avg_distance
    
    @staticmethod
    def crossover(genome1: Genome, genome2: Genome) -> Genome:
        # Assumes genome1.genes, genome2.genes are sorted by innovation_number 
        # and equal genes have equal innovation_number.
        # Genome.logger.debug("crossover")
        child = CircuitGenome(genome1.config)
        if genome1.get_fitness() > genome2.get_fitness():
            better = "genome1"
        elif genome1.get_fitness() < genome2.get_fitness():
            better = "genome2"
        else:
            better = "equal"
        Genome.logger.debug(f"{genome1.get_fitness()=}, {genome2.get_fitness()=}, {better=}")

        n_genes1, n_genes2 = len(genome1.genes), len(genome2.genes)
        index1, index2 = 0, 0
        while index1 < n_genes1 or index2 < n_genes2:
            Genome.logger.debug(f"{index1=}:{n_genes1=}, {index2=}:{n_genes2=}")
            gene1 = genome1.genes[index1] if index1 < n_genes1 else None
            gene2 = genome2.genes[index2] if index2 < n_genes2 else None
            chosen_gene = None
            if gene1 and gene2:
                Genome.logger.debug("gene1 and gene2")
                if gene1.innovation_number < gene2.innovation_number: # disjoint
                    if better == "genome1":
                        chosen_gene = gene1
                    index1 += 1
                elif gene1.innovation_number > gene2.innovation_number: # disjoint
                    if better == "genome2":
                        chosen_gene = gene2
                    index2 += 1
                else: # matching (gene1.innovation_number == gene2.innovation_number)
                    # if better == "genome1":
                    #     chosen_gene = gene1
                    # elif better == "genome2":
                    #     chosen_gene = gene2
                    # else: # better == "equal"
                    #     chosen_gene = "random"
                    chosen_gene = "random"
                    index1 += 1
                    index2 += 1
            elif gene1 and better == "genome1": # excess
                Genome.logger.debug("gene1 and (not gene2) and better == 'genome1'")
                chosen_gene = gene1
                index1 += 1
            elif gene2 and better == "genome2": # excess
                Genome.logger.debug("(not gene1) and gene2 and better == 'genome2'")
                chosen_gene = gene2
                index2 += 1
            elif better == "equal":
                if np.random.random() < 0.5:
                    if gene1: chosen_gene = gene1
                    elif gene2: chosen_gene = gene2
            else: # excess
                Genome.logger.debug("not (gene1 and better == 'genome1') and not (gene2 and better == 'genome2')")
                break
            if chosen_gene: # not None
                if chosen_gene == "random":
                    chosen_gene = np.random.choice([gene1, gene2])
                if not child.add_gene(copy.deepcopy(chosen_gene)):
                    Genome.logger.error("Child did not add gene of parent.")

        return child