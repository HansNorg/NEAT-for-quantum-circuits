from typing import Union
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
import qiskit
import qiskit.circuit
# import qulacs
    
class GateGene(ABC):
    '''
    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        n_qubits (int): Amount of qubits in the circuit
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
    '''
    n_parameters = 0

    def __init__(self, innovation_number: int, config, qubit, **kwargs) -> None:
        self.innovation_number = innovation_number
        self.config = config
        self.qubit = qubit
        self.parameters = config.parameter_amplitude*np.random.random(self.n_parameters)

    @abstractmethod
    def add_to_circuit(self, circuit):
        '''
        Adds the gate to the given circuit.

        Parameters:
            circuit: circuit the gate is added to.
        '''
        return circuit
    
    @staticmethod
    def get_distance(gate1, gate2):
        if type(gate1) != type(gate2):
            raise ValueError("Gates need to be the same")
        if len(gate1.n_parameters) == 0:
            return False, 0
        dist = np.subtract(gate1.parameters, gate2.parameters)
        dist = np.square(dist)
        dist = np.sum(dist)
        return True, np.sqrt(dist)
    
class GateCNOT(GateGene):
    def __init__(self, innovation_number: int, config, qubit, target_qubit = None, **kwargs) -> None:
        super().__init__(innovation_number, config, qubit, **kwargs)
        if target_qubit == None:
            self.target = np.mod(qubit + 1, config.n_qubits)
        else:
            self.target = target_qubit

    def add_to_circuit(self, circuit):
        if self.config.simulator == 'qulacs':
            circuit.add_CNOT_gate(self.qubit, self.target)
        elif self.config.simulator == 'qiskit':
            circuit.cnot(self.qubit, self.target)
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit
    
class GateROT(GateGene):
    n_parameters = 3
    
    def add_to_circuit(self, circuit):
        if self.config.simulator == 'qulacs':
            circuit.add_RX_gate(self.qubit, self.parameters[0])
            circuit.add_RY_gate(self.qubit, self.parameters[1])
            circuit.add_RZ_gate(self.qubit, self.parameters[2])
        elif self.config.simulator == 'qiskit':
            circuit.rx(qiskit.circuit.Parameter(), self.qubit)
            circuit.ry(qiskit.circuit.Parameter(), self.qubit)
            circuit.rz(qiskit.circuit.Parameter(), self.qubit)
        else:
            raise NotImplementedError(f"Simulation method: {self.config.simulator} not implemented for {self.__class__}")
        return circuit

class GateTypes(Enum):
    '''
    Defines the possible gate types
    '''
    ROT = GateROT
    CNOT = GateCNOT