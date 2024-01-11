from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig, Circuit
    from quantumneat.problem import Problem
    
class Gene(ABC):
    """
    Abstract base class for genes.

    Variables
    ---------
    - n_parameters (int): The amount of parameters this gene has. (default = 0)
    """
    n_parameters:int = 0
    logger = logging.getLogger("quantumNEAT.quantumneat.Gene")
    
    def __init__(self, innovation_number: int, config:QuantumNEATConfig, problem:Problem, **kwargs) -> None:
        """
        Initialise the Gene.

        Parameters
        ----------
        - innovation_number: Chronological historical marking
        - config: class with all the configuration settings of the algorithm.
        """
        self.innovation_number = innovation_number
        self.config = config
        self.problem = problem
        self.parameters = config.parameter_amplitude*np.random.random(self.n_parameters)

    def mutate_parameters(self) -> bool:
        """
        Mutate the parameters of the Gene.
        
        Returns
        -------
        - bool: Whether the mutation was succesful.
        #TODO Reset parameters with probability
        """
        if self.n_parameters == 0:
            return False
        self.parameters += self.config.perturbation_amplitude*np.random.random(self.n_parameters)
        return True
    
    def get_parameters(self):
        return self.parameters
    
    @staticmethod
    def get_distance(gene1:Gene, gene2:Gene) -> tuple[bool, float]:
        """
        Calculate the distance between two genes.

        Parameters
        ----------
        - gene1, gene2: Genes between which the distance is calculated.

        Returns
        -------
        - bool: Whether this gene should be included in distance calculation (whether it has parameters)
        - float: Distance between the genes.

        Raises
        ------
        - ValueError: if the genes are of different types.
        """
        if type(gene1) != type(gene2):
            raise ValueError("Genes need to be the same")
        if gene1.n_parameters == 0:
            return False, 0
        dist = np.subtract(gene1.parameters, gene2.parameters)
        dist = np.square(dist)
        dist = np.sum(dist)
        return True, np.sqrt(dist)    

class GateGene(Gene):
    """
    Abstract base class for genes.

    Variables
    ---------
    - n_parameters (int): The amount of parameters this gene has. (default = 0)
    - n_qubits (int): The amount of qubits this gene acts on. (None)
    """
    n_qubits:int
    
    def __init__(self, innovation_number: int, config: QuantumNEATConfig, problem:Problem, qubits:list[int], **kwargs) -> None:
        """
        Initialise the Gene.

        Parameters
        ----------
        - innovation_number: Chronological historical marking.
        - config: class with all the configuration settings of the algorithm.
        - qubits: list of qubits the gene acts on. (should have length n_qubits)
        """
        super().__init__(innovation_number, config, problem, **kwargs)
        self.qubits = qubits
        # self.qubits = qubits%self.config.n_qubits

    @abstractmethod
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        """
        Add the gene to the given circuit.

        Parameters
        ----------
        - circuit: circuit the gate is added to.
        - n_parameters: amount of parameters in the circuit.

        Returns
        -------
        - Circuit: circuit the gate is added to.
        - int: amount of parameters in the circuit.
        """
        return circuit, n_parameters    

    def get_error(self) -> float:
        """
        Gives the error associated with the gate.

        Returns
        -------
        - float: error associated with the gate.
        """
        return 0.2    