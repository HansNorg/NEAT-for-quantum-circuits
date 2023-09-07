import helper as h

class GenericGene(object):
    '''
    A general representation of a gene.

    Parameters:
        innovation_number (int): Chronological historical marking
    '''
    def __init__(self, innovation_number:int, **kwargs) -> None:
        self.innovation_number = innovation_number
    
class GateGene(GenericGene):
    '''
    A chronological representation of a gene containing a quantum gate.

    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
    '''
    def __init__(self, innovation_number: int, gate_string:str, n_qubits:int, qubit_seed, **kwargs) -> None:
        super().__init__(innovation_number, **kwargs)
        self.gate_string = gate_string
        self.qubit_seed = qubit_seed
        self.n_qubits = n_qubits

    def to_gate(self, circuit, n_parameters = 0):
        '''
        Adds the gate in the gene to the given circuit.

        Parameters:
            circuit (qiskit.QuantumCircuit): circuit the gate is added to.
            n_parameters (int): TODO
        '''
        return h.gate_string_to_gate(circuit, self.gate_string, self.n_qubits, self.qubit_seed, n_parameters)

class NonChronologicalGateGene(GateGene):
    '''
    A chronological representation of a gene containing a quantum gate.
    
    Parameters:
        innovation_number (int): Chronological historical marking
        gate_string (str): Sequence of bits representing the gate
        qubit_seed (): Seed for the permutation of the qubits, representing the qubits the gate acts on.
        location (int): Before (negative) or after (positive) which innovation number this gate should be placed.
    '''
    def __init__(self, innovation_number: int, gate_string: str, n_qubits:int, qubit_seed, location:int, **kwargs) -> None:
        super().__init__(innovation_number, gate_string, qubit_seed, n_qubits, **kwargs)
        self.location = location