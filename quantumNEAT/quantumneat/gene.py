from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import numpy as np

if TYPE_CHECKING:
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig, Circuit
    
class Gene(ABC):
    """
    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        n_qubits (int): Amount of qubits in the circuit
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
    """
    n_parameters = 0

    def __init__(self, innovation_number: int, config:QuantumNEATConfig, **kwargs) -> None:
        self.innovation_number = innovation_number
        self.config = config
        self.parameters = config.parameter_amplitude*np.random.random(self.n_parameters)

    def mutate_parameters(self) -> bool:
        """Mutate the parameters of the Gene, return a bool indicating if the mutation was succesfull."""
        if self.n_parameters == 0:
            return False
        self.parameters += self.config.perturbation_amplitude*np.random.random(self.n_parameters)
        return True
    
    @staticmethod
    def get_distance(gene1:Gene, gene2:Gene) -> tuple[bool, float]:
        """
        Returns:
        --------
            bool: Whether this gene should be included in distance calculation (whether it has parameters)
            float: Distance between the genes.
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
    n_qubits = 0
    
    def __init__(self, innovation_number: int, config: QuantumNEATConfig, qubits:list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, **kwargs)
        self.qubits = qubits
        # self.qubits = qubits%self.config.n_qubits

    @abstractmethod
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        """
        Add the gene to the given circuit.

        Parameters:
            circuit: circuit the gate is added to.
        """
        return circuit, n_parameters        