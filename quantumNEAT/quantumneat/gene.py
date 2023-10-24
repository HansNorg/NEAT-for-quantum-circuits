from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from qiskit.circuit import Parameter
from quantumneat.configuration import QuantumNEATConfig as C

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qulacs import ParametricQuantumCircuit
    from quantumNEAT.quantumneat.configuration import QuantumNEATConfig as C
    Circuit = TypeVar('Circuit', QuantumCircuit, ParametricQuantumCircuit)
    
class Gene(ABC):
    """
    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        n_qubits (int): Amount of qubits in the circuit
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
    """
    n_parameters = 0

    def __init__(self, innovation_number: int, config:C, **kwargs) -> None:
        self.innovation_number = innovation_number
        self.config = config
        self.parameters = config.parameter_amplitude*np.random.random(self.n_parameters)

    def mutate_parameters(self) -> bool:
        """Mutate the parameters of the Gene, return a bool indicating if the mutation was succesfull."""
        return False
    
    @abstractmethod
    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        """
        Add the gene to the given circuit.

        Parameters:
            circuit: circuit the gate is added to.
        """
        return circuit, n_parameters
    
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
    def __init__(self, innovation_number: int, config: C, qubits:list[int], **kwargs) -> None:
        super().__init__(innovation_number, config, **kwargs)
        self.qubits = qubits
        # self.qubits = qubits%self.config.n_qubits

    def mutate_parameters(self) -> bool:
        if self.n_parameters == 0:
            return False
        self.parameters += self.config.perturbation_amplitude*np.random.random(self.n_parameters)
        return True

class GateCNOT(GateGene):

    def add_to_circuit(self, circuit:Circuit, n_parameters:int) -> tuple[Circuit, int]:
        if self.config.simulator == 'qulacs':
            circuit.add_CNOT_gate(self.qubits[0], self.qubits[1])
        elif self.config.simulator == 'qiskit':
            circuit.cnot(self.qubits[0], self.qubits[1])
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit, n_parameters
    
class GateROT(GateGene):
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

class GeneTypes(Enum):
    """Define the possible gene types."""
    ROT = GateROT
    CNOT = GateCNOT