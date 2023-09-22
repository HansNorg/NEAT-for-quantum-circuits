from typing import Union
import numpy as np
from enum import Enum
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

class GateType(Enum):
    '''
    Defines the possible gate types
    '''
    ROT = 3
    CNOT = 0

    @classmethod
    def add_to_circuit(cls, circuit: QuantumCircuit, gate, qubit:int, n_parameters:int) -> Union[QuantumCircuit, int]:
        n_qubits = circuit.num_qubits
        if qubit >= n_qubits:
            raise ValueError("Given qubit exceeds number of qubits in the circuit.")
        
        if gate == cls.ROT:
            circuit.rx(Parameter(str(n_parameters)), qubit)
            circuit.ry(Parameter(str(n_parameters+1)), qubit)
            circuit.rz(Parameter(str(n_parameters+2)), qubit)
            n_parameters += 3
        elif gate == cls.CNOT:
            if n_qubits < 2:
                raise ValueError("CNOT cannot be used with less than 2 qubits")
            circuit.cnot(qubit, np.mod(qubit + 1, n_qubits))
            # if qubit == n_qubits - 1:
            #     # CNOT applied on last qubit
            #     circuit.cnot(qubit, 0)
            # else:
            #     circuit.cnot(qubit, qubit + 1)
        return circuit, n_parameters
    
class GateGene(object):
    '''
    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        n_qubits (int): Amount of qubits in the circuit
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
    '''
    def __init__(self, innovation_number: int, gatetype: GateType, qubit:int, parameter_amplitude = 1, **kwargs) -> None:
        self.innovation_number = innovation_number # Probaby unnecessary
        self.gatetype = gatetype
        self.parameter_amplitude = parameter_amplitude
        self.parameters = parameter_amplitude*np.random.random(gatetype.value)
        self.qubit = qubit

    def add_to_circuit(self, circuit:QuantumCircuit, n_parameters:int) -> Union[QuantumCircuit, int]:
        '''
        Adds the gate to the given circuit.

        Parameters:
            circuit (qiskit.QuantumCircuit): circuit the gate is added to.
        '''
        return GateType.add_to_circuit(circuit, self.gatetype, self.qubit, n_parameters)
    
    @staticmethod
    def get_distance(gate1, gate2):
        if gate1.gatetype.name != gate2.gatetype.name:
            raise ValueError("Gates need to be the same")
        if len(gate1.parameters) == 0:
            return False, 0
        dist = np.subtract(gate1.parameters, gate2.parameters)
        dist = np.square(dist)
        dist = np.sum(dist)
        return True, np.sqrt(dist)