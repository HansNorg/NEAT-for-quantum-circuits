import numpy as np
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit

#Inspired by QMS-Xenakis.
class GlobalInnovationNumber(object):
    '''
    Class for keeping a global innovation number.
    
    Innovation number starts at 0.
    '''
    def __init__(self):
        self._innovation_number:int = 0

    def next(self):
        '''
        Get the next innovation number.

        Increments the innovation number.
        '''
        self._innovation_number += 1
        return self._innovation_number
    
    # def __new__(cls):
    #     '''
    #     Make sure there cannot be multiple GlobalInnovationNumber instances (Singleton).
    #     '''
    #     if not hasattr(cls, 'instance'):
    #         cls.instance = super(InnovationNumber, cls).__new__(cls)
    #     return cls.instance


def gate_string_to_gate(circuit:QuantumCircuit, gate_string:str, n_qubits:int, qubit_seed:str, n_parameters:int = 0):
    '''
    Takes an encoded gate string and adds the corresponding gate to the given circuit.

    Parameters:
        circuit (qiskit.QuantumCircuit): circuit the gate is added to.
        gate_string (str): Encoded gate
        qubit_seed (str): Seed used for permuting the qubits to select the qubit(s) the gate acts on.
        n_parameters (int): TODO
    '''
    if n_qubits > circuit.num_qubits:
        raise ValueError("Number of qubits given exceeds number of qubits in the circuit.")
    
    qubits = np.random.RandomState(seed=int(qubit_seed,2)).permutation(n_qubits)
    ## Single Qubit gates
    if gate_string == "00000":
        # Pauli-X
        circuit.x(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00001":
        # originally u1 gate
        circuit.p(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "00010":
        # originally u2 gate
        circuit.u(np.pi/2,Parameter(str(n_parameters)),Parameter(str(n_parameters+1)),qubit=qubits[0])
        n_parameters += 2
        return circuit, n_parameters
    if gate_string == "00011":
        # originally u3 gate
        circuit.u(Parameter(str(n_parameters)),Parameter(str(n_parameters+1)),Parameter(str(n_parameters+2)),qubit=qubits[0])
        n_parameters += 3
        return circuit, n_parameters
    if gate_string == "00100":
        # Pauli-Y
        circuit.y(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00101":
        # Pauli-Z
        circuit.z(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00110":
        # Hadamard
        circuit.h(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "00111":
        # S gate
        circuit.s(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01000":
        # S conjugate gate
        circuit.sdg(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01001":
        # T gate
        circuit.t(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01010":
        # T conjugate gate
        circuit.tdg(qubit=qubits[0])
        return circuit, n_parameters
    if gate_string == "01011":
        # rx gate
        circuit.rx(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "01100":
        # ry gate
        circuit.ry(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "01101":
        # rz gate
        circuit.rz(Parameter(str(n_parameters)),qubit=qubits[0])
        n_parameters += 1
        return circuit, n_parameters
    ## Multi Qubit gates
    if gate_string == "10000":
        # Controlled NOT gate
        circuit.cx(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10001":
        # Controlled Y gate
        circuit.cy(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10011":
        # Controlled Z gate
        circuit.cz(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10100":
        # Controlled H gate
        circuit.ch(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "10101":
        # Controlled rotation Z gate
        circuit.crz(Parameter(str(n_parameters)),qubits[0],qubits[1])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "10110":
        # Controlled phase rotation gate
        circuit.cp(Parameter(str(n_parameters)),qubits[0],qubits[1])
        n_parameters += 1
        return circuit, n_parameters
    if gate_string == "10111":
        # SWAP gate
        circuit.swap(qubits[0],qubits[1])
        return circuit, n_parameters
    if gate_string == "11000":
        # Toffoli gate
        circuit.ccx(qubits[0],qubits[1],qubits[2])
        return circuit, n_parameters
    if gate_string == "11001":
        # controlled swap gate
        circuit.cswap(qubits[0],qubits[1],qubits[2])
        return circuit, n_parameters
    else:
        # Identity/no gate
        return circuit, n_parameters